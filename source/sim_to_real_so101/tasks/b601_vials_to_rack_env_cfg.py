# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101.assets.b601 import (
    B601_DM_CONTACT_GRASP_CFG,
    B601_JOINT_NAMES,
)
from sim_to_real_so101.mdp import (
    JointPositionActionCfg,
    randomize_robot_color,
    reset_joints_by_offset,
)

from .task_env_cfg import camera_object
from .vials_to_rack_env_cfg import (
    VialsToRackDREnvCfg,
    VialsToRackDRSceneCfg,
    VialsToRackEnvCfg,
    VialsToRackEvalDREnvCfg,
    VialsToRackEvalEnvCfg,
    VialsToRackEventDRCfg,
    VialsToRackEventCfg,
    VialsToRackSceneCfg,
    VialsToRackTerminationsCfg,
)


@configclass
class B601ActionsCfg:
    """Action specifications for the B601 articulation."""

    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=B601_JOINT_NAMES,
        scale=1,
        use_default_offset=False,
    )


@configclass
class B601RobotEventCfg:
    """B601 robot reset events shared by the task variants."""

    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=B601_JOINT_NAMES,
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={
            "color_names": ["orange"],
        },
    )


@configclass
class B601VialsToRackSceneCfg(VialsToRackSceneCfg):
    robot: ArticulationCfg = B601_DM_CONTACT_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link", name="gripper"
            ),
        ],
    )

    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/gripper_link/gripper_cam"
    camera_ego.offset.pos = (0.0, 0.055, -0.06)
    camera_ego.offset.rot = euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True)

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_left",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Vial_1",
            "{ENV_REGEX_NS}/Vial_2",
            "{ENV_REGEX_NS}/Vial_3",
        ],
    )


@configclass
class B601VialsToRackDRSceneCfg(B601VialsToRackSceneCfg, VialsToRackDRSceneCfg):
    pass


@configclass
class B601VialsToRackEventCfg(B601RobotEventCfg, VialsToRackEventCfg):
    pass


@configclass
class B601VialsToRackEventDRCfg(B601RobotEventCfg, VialsToRackEventDRCfg):
    pass


@configclass
class B601VialsToRackEnvCfg(VialsToRackEnvCfg):
    """B601/Star-Arm adaptation of the vials-to-rack task."""

    scene: B601VialsToRackSceneCfg = B601VialsToRackSceneCfg()
    actions: B601ActionsCfg = B601ActionsCfg()
    events: B601VialsToRackEventCfg = B601VialsToRackEventCfg()


@configclass
class B601VialsToRackDREnvCfg(VialsToRackDREnvCfg):
    """B601/Star-Arm adaptation of the domain-randomized vials-to-rack task."""

    scene: B601VialsToRackDRSceneCfg = B601VialsToRackDRSceneCfg()
    actions: B601ActionsCfg = B601ActionsCfg()
    events: B601VialsToRackEventDRCfg = B601VialsToRackEventDRCfg()


@configclass
class B601VialsToRackEvalEnvCfg(VialsToRackEvalEnvCfg):
    scene: B601VialsToRackSceneCfg = B601VialsToRackSceneCfg()
    actions: B601ActionsCfg = B601ActionsCfg()
    events: B601VialsToRackEventCfg = B601VialsToRackEventCfg()
    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()


@configclass
class B601VialsToRackEvalDREnvCfg(VialsToRackEvalDREnvCfg):
    scene: B601VialsToRackDRSceneCfg = B601VialsToRackDRSceneCfg()
    actions: B601ActionsCfg = B601ActionsCfg()
    events: B601VialsToRackEventDRCfg = B601VialsToRackEventDRCfg()
    terminations: VialsToRackTerminationsCfg = VialsToRackTerminationsCfg()
