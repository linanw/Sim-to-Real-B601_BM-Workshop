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
import argparse
import json
import os
from sim_to_real_so101.utils.isaacsim_preflight import guard_known_bad_isaacsim_driver

guard_known_bad_isaacsim_driver()

from isaaclab.app import AppLauncher


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Teleop agent.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--port",
    type=str,
    default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"),
    help="Port of the robot.",
)
parser.add_argument(
    "--port_glob",
    type=str,
    default=os.getenv("TELEOP_PORT_GLOB", None),
    help="Glob used to rediscover the leader if its /dev/ttyUSB* path changes.",
)
parser.add_argument(
    "--robot_id",
    type=str,
    default=os.getenv("TELEOP_ID", "leader_arm_1"),
    help="ID of the robot.",
)
parser.add_argument(
    "--leader_type",
    type=str,
    default=os.getenv("TELEOP_TYPE", "so101"),
    help="Leader arm type: so101 or stararm102.",
)
parser.add_argument(
    "--follower_type",
    type=str,
    default=os.getenv("ROBOT_TYPE", "so101_follower"),
    help="Follower robot type to store in dataset metadata.",
)
parser.add_argument(
    "--leader_joint_aliases",
    type=str,
    default=os.getenv("TELEOP_JOINT_ALIASES", None),
    help="JSON mapping from sim joint names to leader joint names.",
)
parser.add_argument(
    "--leader_alignment",
    type=str,
    default=os.getenv("TELEOP_ALIGNMENT", None),
    help="Path to a JSON file with leader-to-sim joint alignment offsets and scales.",
)
parser.add_argument(
    "--align_on_start",
    action=argparse.BooleanOptionalAction,
    default=env_flag("TELEOP_ALIGN_ON_START", False),
    help="Offset the current leader pose to the initial simulated B601 pose at startup.",
)
parser.add_argument(
    "--debug_alignment",
    action=argparse.BooleanOptionalAction,
    default=env_flag("TELEOP_DEBUG_ALIGNMENT", False),
    help="Print leader raw values, mapped targets, sim joint positions, and alignment gains.",
)
parser.add_argument(
    "--repo_id", type=str, default=None, help="Repository ID to store the dataset."
)
parser.add_argument(
    "--repo_root", type=str, default=None, help="Repository root to store the dataset."
)
parser.add_argument(
    "--save_mp4",
    action="store_true",
    default=False,
    help="Save depth and RGB as mp4 videos.",
)
parser.add_argument(
    "--depth", action="store_true", default=False, help="Save depth as mp4 video."
)
parser.add_argument(
    "--instance_id_seg",
    action="store_true",
    default=False,
    help="Save instance id segmentation as mp4 video.",
)
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=101, help="Environment seed")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import time


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder


def main():

    keyboard_control = KeyboardControl()
    leader_joint_aliases = (
        json.loads(args_cli.leader_joint_aliases)
        if args_cli.leader_joint_aliases
        else None
    )

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")
    print(f"[INFO]: Click 'S' to start/stop recording; 'R' will also stop recording")
    print(f"[INFO]: Click 'L' to reload leader alignment JSON")

    # reset environment
    reset_obs, _ = env.reset()

    # cameras
    cameras = {}
    for obj in env.unwrapped.scene.keys():
        if obj.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
            cameras[obj.replace("camera_", "")] = {
                "height": camera_cfg.height,
                "width": camera_cfg.width,
            }
            print(f"[INFO]: Found Camera: {obj.replace('camera_', '')}")
    if len(cameras) == 0:
        print(f"[Info]: No cameras found - videos will not be recorded")

    robot_iface = LeRobotSO101Interface(
        device=env.unwrapped.device,
        port=args_cli.port,
        id=args_cli.robot_id,
        cameras=cameras,
        fps=30,
        kind="leader",
        robot_type=args_cli.leader_type,
        joint_aliases=leader_joint_aliases,
        port_glob=args_cli.port_glob,
        alignment_path=args_cli.leader_alignment,
    )
    robot_iface.init_device()
    robot_iface.connect()
    if args_cli.align_on_start:
        try:
            real_action = robot_iface.get_action()
            robot_iface.align_current_action_to_sim(
                real_action,
                reset_obs["policy"]["joint_pos_obs"][0],
            )
        except KeyError:
            print("[WARNING]: Startup alignment skipped; joint_pos_obs was not found.")

    # Allocate action tensor
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    last_alignment_debug_t = 0.0

    # simulate environment

    # Recording dataset
    if all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name]):
        recording_mode = True
    else:
        recording_mode = False

    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            cameras=cameras,
            save_mp4=args_cli.save_mp4,
            depth=args_cli.depth,
            instance_id_seg=args_cli.instance_id_seg,
            robot_type=args_cli.follower_type,
            action_names=robot_iface.SO101_JOINT_ORDER,
        )
        try:
            recorder.init_dataset()
        except ValueError:
            print(f"[ERROR]: Failed to initialize dataset. folder already exists")
            env.close()
            simulation_app.close()

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            leader_action = robot_iface.get_action()
            real_action, mapped_action = robot_iface.real_to_sim_obs_processor(
                leader_action
            )
            actions[:] = mapped_action

            obs, _, _, _, _ = env.step(actions)
            if args_cli.debug_alignment:
                now = time.monotonic()
                if now - last_alignment_debug_t >= 1.0:
                    sim_joint_pos = None
                    try:
                        sim_joint_pos = obs["policy"]["joint_pos_obs"][0]
                    except KeyError:
                        pass
                    print(robot_iface.format_alignment_debug(real_action, sim_joint_pos))
                    last_alignment_debug_t = now

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

            if keyboard_control.reload_alignment:
                keyboard_control.reload_alignment = False
                try:
                    robot_iface.reload_leader_alignment()
                except Exception as exc:
                    print(f"[WARNING]: Leader alignment reload failed: {exc}")

            if recording_mode and keyboard_control.recording:
                visual_obs = obs.get("visual", None)
                if visual_obs is None:
                    print(
                        "[WARNING]: No 'visual' observation group - recording requires a task with cameras"
                    )
                    keyboard_control.recording = False
                    continue
                # Extract joint positions from policy observation dict
                joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                visual_obs = obs["visual"]
                real_obs, visual_buffers, depth_buffers, instance_id_seg_buffers = (
                    robot_iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)
                )
                recorder.push_frame_to_buffer(
                    real_action,
                    real_obs,
                    visual_buffers,
                    depth_buffers,
                    instance_id_seg_buffers,
                )

    env.close()


if __name__ == "__main__":

    main()

    while True:
        simulation_app.update()

    simulation_app.close()
