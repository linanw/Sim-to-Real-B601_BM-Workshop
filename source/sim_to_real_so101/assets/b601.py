# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from xml.etree import ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

import numpy as np


here = Path(__file__).resolve().parent
repo_root = here.parents[2]
b601_description_root = repo_root / "rebot/reBotArmController_ROS2/src/rebotarm_bringup/description"
b601_source_urdf = b601_description_root / "urdf/reBot_B601_DM_with_gripper.urdf"
b601_mesh_root = b601_description_root / "meshes_b601_gripper"
b601_generated_root = Path(os.getenv("B601_ISAAC_ASSET_CACHE", "/tmp/sim_to_real_so101_b601"))
b601_generated_urdf = b601_generated_root / "reBot_B601_DM_with_gripper.urdf"
b601_generated_usd_root = b601_generated_root / "usd"


B601_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
]


def _write_isaac_resolved_urdf() -> Path:
    """Write a generated URDF with package mesh references resolved to absolute paths."""
    b601_generated_root.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(b601_source_urdf)
    root = tree.getroot()
    package_prefix = "package://rebotarm_bringup/description/meshes_b601_gripper/"

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith(package_prefix):
            mesh_name = filename.removeprefix(package_prefix)
            mesh.attrib["filename"] = (b601_mesh_root / mesh_name).as_posix()

    xml_content = ET.tostring(root, encoding="unicode")
    if not b601_generated_urdf.exists() or b601_generated_urdf.read_text() != xml_content:
        b601_generated_urdf.write_text(xml_content)
    return b601_generated_urdf


def _convert_b601_urdf_to_usd() -> str:
    urdf_path = _write_isaac_resolved_urdf()
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=urdf_path.as_posix(),
            usd_dir=b601_generated_usd_root.as_posix(),
            usd_file_name="reBot_B601_DM_with_gripper_preserve_links.usd",
            fix_base=True,
            root_link_name="base_link",
            merge_fixed_joints=False,
            make_instanceable=False,
            self_collision=False,
            collision_from_visuals=False,
            collider_type="convex_hull",
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=100.0,
                    damping=10.0,
                ),
            ),
        )
    )
    return converter.usd_path


B601_DM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_convert_b601_urdf_to_usd(),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -1.1,
            "joint3": -0.8,
            "joint4": 0.6,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_joint1": 0.0,
            "gripper_joint2": 0.0,
        },
        pos=(-0.05, 0.0, 0),
        rot=euler_angles_to_quat(np.array([0, 0, 90]), degrees=True),
    ),
    actuators={
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=27,
            stiffness=55,
            damping=0.7,
        ),
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit_sim=27,
            stiffness=45,
            damping=0.8,
        ),
        "elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=27,
            stiffness=35,
            damping=0.7,
        ),
        "wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=7,
            stiffness=15,
            damping=0.5,
        ),
        "wrist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            effort_limit_sim=7,
            stiffness=10,
            damping=0.5,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit_sim=7,
            stiffness=10,
            damping=0.5,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint1", "gripper_joint2"],
            effort_limit_sim=100,
            stiffness=80,
            damping=2.0,
        ),
    },
)


B601_DM_CONTACT_GRASP_CFG = B601_DM_CFG.copy()
B601_DM_CONTACT_GRASP_CFG.spawn.activate_contact_sensors = True
