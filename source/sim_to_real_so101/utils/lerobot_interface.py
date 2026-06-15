# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import time
from collections import deque
from glob import glob
from pathlib import Path

import numpy as np
import torch
import uuid
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.robots.so101_follower import SO101FollowerConfig

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots import make_robot_from_config
from lerobot.processor import make_default_processors
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.policies.utils import make_robot_action
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# Ideally, we should make a base class for all robot interfaces,
# and inherit from it for each robot type.
class LeRobotSO101Interface:

    # B601 sim and hardware limits are not the same normalized space exposed by
    # LeRobot teleoperators.
    SO101_USD_MAPPING = {
        "shoulder_pan": {"joint_min": -145, "joint_max": 145},
        "shoulder_lift": {"joint_min": -170, "joint_max": 0},
        "elbow_flex": {"joint_min": -200, "joint_max": 0},
        "wrist_flex": {"joint_min": -80, "joint_max": 90},
        "wrist_yaw": {"joint_min": -90, "joint_max": 90},
        "wrist_roll": {"joint_min": -90, "joint_max": 90},
        "gripper": {"joint_min": 0, "joint_max": 100},
    }
    B601_GRIPPER_MAX_M = 0.0715

    # Logical LeRobot order. The B601 Isaac articulation expands gripper.pos
    # into both prismatic finger joints.
    SO101_JOINT_ORDER = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_yaw.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    ROBOT_TYPE_ALIASES = {
        "so101": "so101",
        "so101_leader": "so101",
        "so101_follower": "so101",
        "stararm102": "stararm102",
        "stararm102_leader": "stararm102",
        "lerobot_teleoperator_stararm102": "stararm102",
        "sim_to_real_stararm102": "stararm102",
        "b601": "seeed_b601_dm",
        "b601_dm": "seeed_b601_dm",
        "seeed_b601_dm": "seeed_b601_dm",
        "seeed_b601_dm_follower": "seeed_b601_dm",
    }

    # Star-Arm-102 and B601 are 6-DOF + gripper arms. The current Isaac task
    # still uses the SO-101 5-DOF + gripper articulation, so wrist_yaw is held
    # neutral and Motor_5/wrist_roll is mapped onto the simulated wrist_roll.
    DEFAULT_JOINT_ALIASES = {
        "so101": {
            "shoulder_pan.pos": "shoulder_pan.pos",
            "shoulder_lift.pos": "shoulder_lift.pos",
            "elbow_flex.pos": "elbow_flex.pos",
            "wrist_flex.pos": "wrist_flex.pos",
            "wrist_yaw.pos": "wrist_yaw.pos",
            "wrist_roll.pos": "wrist_roll.pos",
            "gripper.pos": "gripper.pos",
        },
        "stararm102": {
            "shoulder_pan.pos": "Motor_0.pos",
            "shoulder_lift.pos": "Motor_1.pos",
            "elbow_flex.pos": "Motor_2.pos",
            "wrist_flex.pos": "Motor_3.pos",
            "wrist_yaw.pos": "Motor_4.pos",
            "wrist_roll.pos": "Motor_5.pos",
            "gripper.pos": "gripper.pos",
        },
        "seeed_b601_dm": {
            "shoulder_pan.pos": "shoulder_pan.pos",
            "shoulder_lift.pos": "shoulder_lift.pos",
            "elbow_flex.pos": "elbow_flex.pos",
            "wrist_flex.pos": "wrist_flex.pos",
            "wrist_yaw.pos": "wrist_yaw.pos",
            "wrist_roll.pos": "wrist_roll.pos",
            "gripper.pos": "gripper.pos",
        },
    }

    def __init__(
        self,
        device: str,
        port: str,
        id: str,
        cameras: dict,
        fps: int,
        kind: str = "leader",
        rename_map: dict = None,
        robot_type: str = "so101",
        joint_aliases: dict | None = None,
        can_adapter: str = "damiao",
        use_degrees: bool = False,
        port_glob: str | None = None,
        alignment_path: str | None = None,
    ):

        self.port = port
        self.requested_port = port
        self.port_glob = port_glob
        self._last_valid_action = None
        self._last_reconnect_attempt_at = 0.0
        self._last_hold_warning_at = 0.0
        self._reconnect_interval_s = 1.0
        self.id = id
        self.cameras = cameras
        self.device = device
        self.fps = fps
        self.kind = kind
        self.rename_map = rename_map
        self.robot_type = self._canonical_robot_type(robot_type)
        self.joint_aliases = {
            **self.DEFAULT_JOINT_ALIASES[self.robot_type],
            **(joint_aliases or {}),
        }
        self.can_adapter = can_adapter
        self.use_degrees = use_degrees
        self.alignment_path = alignment_path

        self.joint_names = [joint.split(".")[0] for joint in self.SO101_JOINT_ORDER]
        self.joint_mins = torch.tensor(
            [self.SO101_USD_MAPPING[name]["joint_min"] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.joint_maxs = torch.tensor(
            [self.SO101_USD_MAPPING[name]["joint_max"] for name in self.joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self.joint_alignment_scales, self.joint_alignment_offsets = (
            self._load_leader_alignment()
        )
        self.joint_alignment_file_offsets = self.joint_alignment_offsets.clone()

    @classmethod
    def _canonical_robot_type(cls, robot_type: str) -> str:
        canonical = cls.ROBOT_TYPE_ALIASES.get(robot_type)
        if canonical is None:
            supported = ", ".join(sorted(cls.ROBOT_TYPE_ALIASES))
            raise ValueError(f"Unsupported robot_type '{robot_type}'. Supported values: {supported}")
        return canonical

    def _load_leader_alignment(self) -> tuple[torch.Tensor, torch.Tensor]:
        scales = torch.ones(len(self.SO101_JOINT_ORDER), dtype=torch.float32, device=self.device)
        offsets = torch.zeros(len(self.SO101_JOINT_ORDER), dtype=torch.float32, device=self.device)

        if not self.alignment_path:
            return scales, offsets

        alignment_path = Path(self.alignment_path).expanduser()
        if not alignment_path.is_file():
            raise FileNotFoundError(f"Leader alignment file not found: {alignment_path}")

        with alignment_path.open() as f:
            alignment = json.load(f)

        joints = alignment.get("joints", alignment)
        for index, sim_joint in enumerate(self.SO101_JOINT_ORDER):
            joint_name = sim_joint.removesuffix(".pos")
            joint_alignment = joints.get(sim_joint, joints.get(joint_name, {}))
            scales[index] = float(joint_alignment.get("scale", 1.0))
            if scales[index] == 0:
                raise ValueError(f"Leader alignment scale cannot be zero for {sim_joint}")
            offsets[index] = float(joint_alignment.get("offset_deg", 0.0))

        print(f"[INFO]: Loaded leader alignment from {alignment_path}")
        return scales, offsets

    def make_cameras_cfg(self):
        cameras = {}
        # need to rename cameras to match policy/feature config
        for camera in self.cameras.keys():
            camera_name = camera
            if self.rename_map:
                camera_name = self.rename_map[camera]
            cameras[camera_name] = OpenCVCameraConfig(
                index_or_path="null",  # we are in simulation :)
                fps=self.fps,
                width=self.cameras[camera]["width"],
                height=self.cameras[camera]["height"],
            )

        return cameras

    def make_cfg(self):
        if self.kind == "leader":
            if self.robot_type == "so101":
                return SO101LeaderConfig(port=self.port, id=self.id)
            if self.robot_type == "stararm102":
                self._normalize_stararm102_calibration()
                self.port = self._resolve_serial_port()
                try:
                    from sim_to_real_so101.adapters.stararm102 import (
                        SimToRealStararm102LeaderConfig,
                    )
                except ImportError as exc:
                    raise ImportError(
                        "Sim-to-real Star-Arm-102 adapter is unavailable. Ensure "
                        "sim_to_real_so101 and the vendor Star-Arm package are installed."
                    ) from exc

                return SimToRealStararm102LeaderConfig(
                    port=self.port,
                    id=self.id,
                    use_degrees=self.use_degrees,
                )
            raise ValueError(f"Robot type '{self.robot_type}' cannot be used as a leader")
        elif self.kind == "follower":
            cameras = self.make_cameras_cfg()
            if self.robot_type == "so101":
                return SO101FollowerConfig(port=self.port, id=self.id, cameras=cameras)
            if self.robot_type == "seeed_b601_dm":
                try:
                    from lerobot_robot_seeed_b601 import SeeedB601DMFollowerConfig
                except ImportError as exc:
                    raise ImportError(
                        "Seeed B601 support is not installed. Install "
                        "rebot/lerobot-robot-seeed-b601 inside the sim container."
                    ) from exc

                return SeeedB601DMFollowerConfig(
                    port=self.port,
                    id=self.id,
                    can_adapter=self.can_adapter,
                    cameras=cameras,
                )
            raise ValueError(f"Robot type '{self.robot_type}' cannot be used as a follower")

    def _normalize_stararm102_calibration(self) -> None:
        calibration_path = (
            HF_LEROBOT_CALIBRATION
            / TELEOPERATORS
            / "stararm102_leader"
            / f"{self.id}.json"
        )
        if not calibration_path.is_file():
            return

        with calibration_path.open() as f:
            calibration = json.load(f)

        changed = False
        for motor_calibration in calibration.values():
            for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
                value = motor_calibration.get(field)
                if isinstance(value, float):
                    motor_calibration[field] = int(round(value))
                    changed = True

        if changed:
            with calibration_path.open("w") as f:
                json.dump(calibration, f, indent=4)
                f.write("\n")

    def init_device(self, visualize: bool = False):
        self.cfg = self.make_cfg()
        self.robot = make_robot_from_config(self.cfg)

        random_session_name = f"eval_{uuid.uuid4().hex[:8]}"
        if visualize:
            init_rerun(session_name=random_session_name)

        print(f"[INFO]: Initialized {self.robot_type} {self.kind} at {self.port} with id {self.id}")

    def connect(self):
        self.robot.connect()

    def get_action(self):
        if self.robot_type != "stararm102" or self.kind != "leader":
            return self.robot.get_action()

        try:
            action = self.robot.get_action()
            action = self._sanitize_stararm102_action(action)
            self._last_valid_action = dict(action)
            return dict(action)
        except Exception as exc:
            print(f"[WARNING]: Star-Arm read failed ({exc}).")
            self._maybe_reconnect_stararm102()

        if self._last_valid_action is not None:
            now = time.monotonic()
            if now - self._last_hold_warning_at >= 2.0:
                print("[WARNING]: Holding last valid Star-Arm action until USB reconnects.")
                self._last_hold_warning_at = now
            return dict(self._last_valid_action)

        fallback_action = self._neutral_stararm102_action()
        self._last_valid_action = dict(fallback_action)
        print("[WARNING]: Using neutral Star-Arm action until valid servo readings arrive.")
        return fallback_action

    def _neutral_stararm102_action(self) -> dict[str, float]:
        return {
            self.joint_aliases[sim_joint]: 0.0 for sim_joint in self.SO101_JOINT_ORDER
        }

    def _sanitize_stararm102_action(self, action) -> dict[str, float]:
        if not isinstance(action, dict):
            raise RuntimeError(f"Star-Arm returned invalid action type: {type(action)}")

        sanitized = {}
        fallback_action = self._last_valid_action or self._neutral_stararm102_action()
        required_joints = [
            self.joint_aliases[sim_joint] for sim_joint in self.SO101_JOINT_ORDER
        ]
        missing = []
        invalid = []
        for joint in required_joints:
            value = action.get(joint)
            if joint not in action:
                missing.append(joint)
                value = fallback_action[joint]
            elif value is None:
                invalid.append(joint)
                value = fallback_action[joint]
            sanitized[joint] = value

        if missing or invalid:
            print(
                "[WARNING]: Star-Arm returned incomplete action; "
                f"using fallback values. Missing: {missing}; invalid values: {invalid}"
            )
        return sanitized

    def _resolve_serial_port(self) -> str:
        patterns = []
        if self.port_glob:
            patterns.append(self.port_glob)
        patterns.extend(
            [
                "/dev/serial/by-id/*",
                "/dev/ttyUSB*",
                "/dev/ttyACM*",
            ]
        )

        requested_realpath = (
            os.path.realpath(self.requested_port)
            if self.requested_port and os.path.exists(self.requested_port)
            else None
        )
        for pattern in patterns:
            ports = sorted(glob(pattern))
            if requested_realpath:
                for port in ports:
                    if os.path.realpath(port) == requested_realpath:
                        if port != self.port:
                            print(f"[INFO]: Resolved Star-Arm serial port: {port}")
                        return port
            if ports:
                resolved = ports[0]
                if resolved != self.port:
                    print(f"[INFO]: Resolved Star-Arm serial port: {resolved}")
                return resolved

        if self.requested_port and os.path.exists(self.requested_port):
            return self.requested_port
        return self.requested_port or self.port

    def _reconnect_stararm102(self) -> None:
        try:
            if getattr(self, "robot", None) is not None and self.robot.is_connected:
                self.robot.disconnect()
        except Exception:
            pass

        self.port = self._resolve_serial_port()
        self.cfg = self.make_cfg()
        robot = make_robot_from_config(self.cfg)
        robot.connect()
        self.robot = robot

    def _maybe_reconnect_stararm102(self) -> None:
        now = time.monotonic()
        if now - self._last_reconnect_attempt_at < self._reconnect_interval_s:
            return

        self._last_reconnect_attempt_at = now
        try:
            self._reconnect_stararm102()
            action = self.robot.get_action()
            action = self._sanitize_stararm102_action(action)
            self._last_valid_action = dict(action)
        except Exception as reconnect_exc:
            print(f"[WARNING]: Star-Arm reconnect failed: {reconnect_exc}")

    def get_raw_actions_tensor(self, real_action):
        values = []
        missing = []
        invalid = []
        for sim_joint in self.SO101_JOINT_ORDER:
            hardware_joint = self.joint_aliases[sim_joint]
            if hardware_joint not in real_action:
                missing.append(hardware_joint)
                continue
            if real_action[hardware_joint] is None:
                invalid.append(hardware_joint)
                continue
            values.append(real_action[hardware_joint])
        if missing or invalid:
            available = ", ".join(sorted(real_action))
            raise KeyError(
                f"Missing joint(s) from {self.robot_type} action: {missing}; "
                f"invalid values: {invalid}. "
                f"Available joints: {available}"
            )
        return torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    def get_mapped_degrees_vectorized(self, raw_values):
        normalized = torch.zeros_like(raw_values)
        normalized[:-1] = (
            raw_values[:-1] + 100
        ) / 200.0  # arm joints: -100-100 -> 0-1
        normalized[-1] = raw_values[-1] / 100.0  # gripper: 0-100 -> 0-1

        mapped_deg = self.joint_mins + normalized * (self.joint_maxs - self.joint_mins)
        joint_centers = (self.joint_mins + self.joint_maxs) / 2.0
        mapped_deg = (
            joint_centers
            + self.joint_alignment_scales * (mapped_deg - joint_centers)
            + self.joint_alignment_offsets
        )
        mapped_deg = torch.minimum(
            torch.maximum(mapped_deg, self.joint_mins),
            self.joint_maxs,
        )
        return mapped_deg

    def align_current_action_to_sim(self, real_action, sim_joint_positions):
        raw_values = self.get_raw_actions_tensor(real_action)
        mapped_deg = self.get_mapped_degrees_vectorized(raw_values)
        target_deg = sim_joint_positions[:6].to(self.device) * 180 / torch.pi
        offset_delta = target_deg - mapped_deg[:6]
        self.joint_alignment_offsets[:6] += offset_delta

        print("[INFO]: Applied startup leader alignment offsets (runtime only, JSON unchanged):")
        runtime_offsets = self.joint_alignment_offsets[:6] - self.joint_alignment_file_offsets[:6]
        for joint_name, offset in zip(self.joint_names[:6], runtime_offsets):
            print(f"        {joint_name}: {offset.item():.2f} deg")

    def format_alignment_debug(self, raw_values, sim_joint_positions=None) -> str:
        mapped_deg = self.get_mapped_degrees_vectorized(raw_values)
        sim_deg = None
        if sim_joint_positions is not None:
            sim_deg = sim_joint_positions[:6].to(self.device) * 180 / torch.pi

        lines = [
            "[ALIGN]: joint             raw    target_deg    sim_deg   scale  file_offset  runtime_offset",
        ]
        for index, joint_name in enumerate(self.joint_names[:6]):
            sim_value = "--" if sim_deg is None else f"{sim_deg[index].item():8.2f}"
            runtime_offset = (
                self.joint_alignment_offsets[index]
                - self.joint_alignment_file_offsets[index]
            )
            lines.append(
                f"[ALIGN]: {joint_name:<14}"
                f"{raw_values[index].item():8.2f}"
                f"{mapped_deg[index].item():12.2f}"
                f"{sim_value:>10}"
                f"{self.joint_alignment_scales[index].item():8.2f}"
                f"{self.joint_alignment_file_offsets[index].item():13.2f}"
                f"{runtime_offset.item():16.2f}"
            )
        return "\n".join(lines)

    def get_mapped_actions_vectorized(self, raw_values):
        mapped_deg = self.get_mapped_degrees_vectorized(raw_values)

        # Map arm joints to B601 joint ranges (degrees) and convert to radians.
        arm_actions = mapped_deg[:-1] * torch.pi / 180

        normalized = torch.zeros_like(raw_values)
        normalized[-1] = raw_values[-1] / 100.0  # gripper: 0-100 -> 0-1
        gripper_action = normalized[-1] * self.B601_GRIPPER_MAX_M
        return torch.cat([arm_actions, gripper_action.repeat(2).reshape(2)])

    def get_raw_actions_from_radians(self, raw_values):
        arm_values = raw_values[:6]
        gripper_values = raw_values[6:]

        # Convert from radians to degrees.
        mapped_deg = arm_values * 180 / torch.pi
        joint_centers = (self.joint_mins[:-1] + self.joint_maxs[:-1]) / 2.0
        mapped_deg = (
            joint_centers
            + (
                mapped_deg
                - joint_centers
                - self.joint_alignment_offsets[:-1]
            )
            / self.joint_alignment_scales[:-1]
        )

        # Reverse the joint range mapping
        normalized = (mapped_deg - self.joint_mins[:-1]) / (
            self.joint_maxs[:-1] - self.joint_mins[:-1]
        )

        # Reverse the normalization
        raw_degrees = normalized * 200.0 - 100
        gripper_raw = (
            gripper_values.mean().reshape(1) / self.B601_GRIPPER_MAX_M * 100.0
            if gripper_values.numel() > 0
            else torch.zeros(1, dtype=raw_values.dtype, device=raw_values.device)
        )

        return torch.cat([raw_degrees, gripper_raw])

    def make_policy(
        self,
        name_or_path: str,
    ):

        _, self.robot_action_processor, self.robot_observation_processor = (
            make_default_processors()
        )

        self.dataset_features = combine_feature_dicts(
            # Observation features (joint positions + camera images)
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_observation_processor,
                initial_features=create_initial_features(
                    observation=self.robot.observation_features
                ),
                use_videos=True,  # MUST be True to include cameras!
            ),
            # Action features (target joint positions)
            aggregate_pipeline_dataset_features(
                pipeline=self.robot_action_processor,
                initial_features=create_initial_features(
                    action=self.robot.action_features
                ),
                use_videos=True,
            ),
        )

        policy_config = PreTrainedConfig.from_pretrained(name_or_path)
        policy_config.pretrained_path = name_or_path
        policy_config.device = self.device

        self.dataset_meta = DummyDatasetMeta(self.dataset_features, self.robot.name)

        self.policy = make_policy(policy_config, ds_meta=self.dataset_meta)

        print(f"[INFO]: Policy loaded")

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=name_or_path,
            dataset_stats={},  # No normalization stats (policy handles it)
            preprocessor_overrides={
                "device_processor": {"device": policy_config.device},
            },
        )

        print(f"[INFO]: Preprocessor and postprocessor loaded")

    def sim_obs_to_policy_processor(self, sim_observation: torch.Tensor, visual_obs: dict) -> dict:
        # TODO: makes no sense to copy to host here, but this is whay predict_action expects

        state: torch.Tensor = self.get_raw_actions_from_radians(sim_observation)

        state_np = state.cpu().numpy()

        sim_observation = {}
        for index, joint_name in enumerate(self.SO101_JOINT_ORDER):
            sim_observation[self.joint_aliases[joint_name]] = state_np[index]

        for camera in self.cameras.keys():
            img: torch.Tensor = (
                visual_obs[f"rgb_{camera}"][0].clone()
            )
            # need to rename camera to match policy/feature config
            camera_name = camera
            if self.rename_map:
                camera_name = self.rename_map[camera]
            sim_observation[camera_name] = img.cpu().detach().numpy()

        obs_processed = self.robot_observation_processor(sim_observation)
        observation_frame = build_dataset_frame(
            self.dataset_features,  # Defines structure of observations/actions
            obs_processed,  # Raw observation data
            prefix=OBS_STR,  # Adds "observation." prefix to keys
        )

        return observation_frame

    def predict_action(self, observation_frame: dict) -> dict:
        action_values = predict_action(
            observation=observation_frame,
            policy=self.policy,
            device=get_safe_torch_device(self.policy.config.device),
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.policy.config.use_amp,
            task="Pick up the vial and place it in the tray",
            robot_type=self.robot.robot_type,
        )
        return action_values

    def prediction_to_sim_processor(
        self, action_values: dict, observation_frame: dict, log: bool = False
    ) -> dict:
        robot_action = make_robot_action(action_values, self.dataset_features)
        robot_action_to_send = self.robot_action_processor((robot_action, None))

        motor_actions = {
            k: v for k, v in robot_action_to_send.items() if k.endswith(".pos")
        }
        sim_motor_actions: torch.Tensor = self.get_raw_actions_tensor(motor_actions)

        if log:
            log_rerun_data(observation=observation_frame, action=robot_action)

        mapped_sim_motor_actions: torch.Tensor = self.get_mapped_actions_vectorized(
            sim_motor_actions
        )
        return mapped_sim_motor_actions

    # TODO: we should return a single object instead of a tuple
    def real_to_sim_obs_processor(
        self, sim_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_action: torch.Tensor = self.get_raw_actions_tensor(sim_obs)
        mapped_action: torch.Tensor = self.get_mapped_actions_vectorized(real_action)
        return real_action, mapped_action

    # TODO: we should return a single object instead of a tuple
    def sim_to_real_dataset_processor(
        self, policy_obs: torch.Tensor, visual_obs: dict
    ) -> tuple[torch.Tensor, dict]:
        real_obs = self.get_raw_actions_from_radians(policy_obs)

        visual_buffers = {}
        depth_buffers = {}
        instance_id_seg_buffers = {}

        for camera in self.cameras.keys():
            visual_buffers[camera] = visual_obs[f"rgb_{camera}"][0]
            depth_buffers[camera] = visual_obs[f"depth_{camera}"][0]
            instance_id_seg_buffers[camera] = visual_obs[f"instance_id_seg_{camera}"][0][..., :3]

        return real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers


class GR00TRemotePolicy:
    def __init__(
        self,
        robot_iface: LeRobotSO101Interface,
        host: str = "localhost",
        port: int = 5555,
        action_horizon: int = 8,
        lang_instruction: str = "Pick up the vial and place it in the tray",
    ):
        self._iface = robot_iface
        self._host = host
        self._port = port
        self._action_horizon = action_horizon
        self._lang_instruction = lang_instruction
        self._action_queue: deque = deque()
        self._client = None

    def connect(self):
        """Connect to the GR00T policy server."""
        from sim_to_real_so101.gr00t_client.server_client import PolicyClient

        print(f"[INFO]: Connecting to GR00T policy server at {self._host}:{self._port}...")
        self._client = PolicyClient(host=self._host, port=self._port)
        if not self._client.ping():
            raise RuntimeError("Cannot connect to GR00T policy server!")
        print("[INFO]: Policy server connected")

    def reset(self):
        """Reset server-side policy state and clear the local action buffer."""
        self._client.reset()
        self._action_queue.clear()

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _add_batch_time_dims(obs: dict) -> dict:
        """Recursively add one leading dimension to every leaf array/scalar.

        GR00T expects (batch=1, time=1, ...).  Call twice to get both dims.
        """
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                obs[key] = val[np.newaxis, ...]
            elif isinstance(val, dict):
                obs[key] = GR00TRemotePolicy._add_batch_time_dims(val)
            else:
                obs[key] = [val]
        return obs

    def _sim_obs_to_groot_inputs(
        self, joint_positions: torch.Tensor, visual_obs: dict
    ) -> dict:
        """Convert Isaac Lab sim observations into the GR00T VLA input format.

        Args:
            joint_positions: Joint positions in sim radians, shape (6,).
            visual_obs: Visual observation dict from env (e.g. rgb_ego, rgb_external_D455).

        Returns:
            Dict with ``video``, ``state``, and ``language`` keys,
            each with (B=1, T=1) leading dimensions.
        """
        state = self._iface.get_raw_actions_from_radians(joint_positions)
        state_np = state.cpu().numpy().astype(np.float32)

        rename = self._iface.rename_map
        model_obs = {}

        # Cameras → video dict
        model_obs["video"] = {}
        for camera in self._iface.cameras.keys():
            img = visual_obs[f"rgb_{camera}"][0].cpu().numpy()
            camera_name = rename[camera] if rename else camera
            model_obs["video"][camera_name] = img

        # Joint state (arm + gripper, matching GR00T convention)
        model_obs["state"] = {
            "single_arm": state_np[:6],
            "gripper": state_np[6:7],
        }

        # Language instruction
        model_obs["language"] = {
            "annotation.human.task_description": self._lang_instruction,
        }

        # Add (B=1, T=1) leading dimensions
        model_obs = self._add_batch_time_dims(model_obs)
        model_obs = self._add_batch_time_dims(model_obs)

        return model_obs

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_action_chunk(self, action_chunk: dict) -> list[dict[str, float]]:
        """Decode a GR00T action chunk into per-timestep joint-name dicts.

        Args:
            action_chunk: Dict with ``single_arm`` (B, T, 5) and
                ``gripper`` (B, T, 1) numpy arrays.

        Returns:
            List of dicts mapping joint name → float (real-robot degree space).
        """
        any_key = next(iter(action_chunk.keys()))
        T = action_chunk[any_key].shape[1]
        horizon = min(T, self._action_horizon)

        joint_order = self._iface.SO101_JOINT_ORDER
        actions_list = []
        for t in range(horizon):
            single_arm = action_chunk["single_arm"][0][t]
            gripper = action_chunk["gripper"][0][t]          # (1,)
            full = np.zeros(len(joint_order), dtype=np.float32)
            arm_count = min(len(joint_order) - 1, single_arm.shape[0])
            full[:arm_count] = single_arm[:arm_count]
            full[-1:] = gripper
            actions_list.append(
                {name: float(full[i]) for i, name in enumerate(joint_order)}
            )

        return actions_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(
        self, joint_positions: torch.Tensor, visual_obs: dict, log: bool = False
    ) -> torch.Tensor:
        """Return the next sim action as a radians tensor.

        Manages the action buffer internally — queries the server only
        when the buffer is empty, then pops one action per call.

        Args:
            joint_positions: Current joint positions in sim radians (6,).
            visual_obs: Visual observation dict from env.
            log: If True, send observation and action data to Rerun.

        Returns:
            Tensor of mapped joint actions in sim radians, ready
            for ``env.step()``.
        """
        if len(self._action_queue) == 0:
            model_input = self._sim_obs_to_groot_inputs(joint_positions, visual_obs)
            action_chunk, _info = self._client.get_action(model_input)
            decoded = self._decode_action_chunk(action_chunk)
            self._action_queue.extend(decoded)

        action_dict = self._action_queue.popleft()
        raw_tensor = self._iface.get_raw_actions_tensor(action_dict)
        sim_action = self._iface.get_mapped_actions_vectorized(raw_tensor)

        if log:
            state = self._iface.get_raw_actions_from_radians(joint_positions)
            state_np = state.cpu().numpy()
            rename = self._iface.rename_map

            obs_log = {}
            for camera in self._iface.cameras.keys():
                name = rename[camera] if rename else camera
                obs_log[name] = visual_obs[f"rgb_{camera}"][0].cpu().numpy()
            for i, joint in enumerate(self._iface.SO101_JOINT_ORDER):
                obs_log[joint] = float(state_np[i])

            log_rerun_data(observation=obs_log, action=action_dict)

        return sim_action


class DummyDatasetMeta:
    def __init__(self, features, robot_type):
        self.features = features
        self.stats = {}
        self.robot_type = robot_type


if __name__ == "__main__":
    lerobot_cfg = {"port": "/dev/ttyACM0", "id": "leader_arm_1"}
    lerobot_interface = LeRobotSO101Interface(cfg=lerobot_cfg)
    while True:
        real_action = lerobot_interface.teleop_dev.get_action()
        print(type(real_action))
        print(real_action)
        print(list(real_action.keys()))
