"""Fetch robot configuration.

The full configuration vector is 11-DOF:
    [0:3]   mobile base (x, y, theta) — nonholonomic differential-drive
    [3:4]   torso_lift_joint
    [4:11]  7-DOF arm (shoulder_pan → wrist_roll)

The base is nonholonomic: motion is restricted to forward/backward + in-place
rotation (differential drive). The planning state space uses a
CompoundStateSpace = ReedsSheppStateSpace(turning_radius) + RealVectorStateSpace
whenever base joints are active; otherwise a plain RealVectorStateSpace is used.
"""

import os

import numpy as np

from fetch_planning.types.robot import CameraConfig, ChainConfig, RobotConfig

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCES_DIR = os.path.join(_PKG_ROOT, "resources", "robot", "fetch")

# Atomic joint groups — indices into the full 11-DOF configuration array.
# Order must match VAMP's URDF tree traversal: base → torso → arm.
JOINT_GROUPS = {
    "base": slice(0, 3),  # base_x, base_y, base_theta (nonholonomic)
    "torso": slice(3, 4),  # torso_lift_joint
    "arm": slice(4, 11),  # shoulder_pan → wrist_roll (7 DOF)
}

# Nonholonomic parameters for the Fetch mobile base.
# turning_radius=0.0 ⇒ true differential drive (in-place rotation allowed).
# Fetch's real minimum turning radius is ~0 (it's a holonomic-base diff-drive);
# 0.2 m is a comfortable compromise that keeps Reeds-Shepp curves smooth.
BASE_TURNING_RADIUS: float = 0.2
BASE_REVERSE_PENALTY: float = 20.0  # multiplier to discourage backing up

CHAIN_CONFIGS: dict[str, ChainConfig] = {
    # 7-DOF arm from shoulder to gripper (torso + base fixed)
    "arm": ChainConfig(
        base_link="torso_lift_link",
        ee_link="gripper_link",
        num_joints=7,
        urdf_path=os.path.join(_RESOURCES_DIR, "fetch.urdf"),
    ),
    # 8-DOF arm with torso (base fixed). This is the canonical chain that
    # the bundled ikfast solver targets (2 free params: torso + upperarm_roll).
    "arm_with_torso": ChainConfig(
        base_link="base_link",
        ee_link="gripper_link",
        num_joints=8,
        urdf_path=os.path.join(_RESOURCES_DIR, "fetch.urdf"),
    ),
    # 11-DOF whole body (base + torso + 7-DOF arm). The IK backend must be
    # aware of the planar root encoding — use the Pink backend for this chain.
    "whole_body": ChainConfig(
        base_link="base_link",
        ee_link="gripper_link",
        num_joints=11,
        urdf_path=os.path.join(_RESOURCES_DIR, "fetch.urdf"),
    ),
}

VIZ_URDF_PATH = os.path.join(_RESOURCES_DIR, "fetch.urdf")

# URDF with the three virtual base joints (prismatic x, prismatic y,
# revolute z) prepended above base_link, giving an 11-DOF kinematic tree.
# Used by SymbolicContext for CasADi-backed manifold constraints and by
# cricket for whole-body FK codegen.
WHOLE_BODY_URDF_PATH = os.path.join(_RESOURCES_DIR, "fetch_whole_body.urdf")

fetch_robot_config = RobotConfig(
    urdf_path=os.path.join(_RESOURCES_DIR, "fetch.urdf"),
    joint_names=[
        # [0:3]   mobile base (virtual planar joints)
        "base_x_joint",
        "base_y_joint",
        "base_theta_joint",
        # [3:4]   torso
        "torso_lift_joint",
        # [4:11]  7-DOF arm
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "upperarm_roll_joint",
        "elbow_flex_joint",
        "forearm_roll_joint",
        "wrist_flex_joint",
        "wrist_roll_joint",
    ],
    camera=CameraConfig(
        link_name="head_camera_rgb_optical_frame",
        width=640,
        height=480,
        fov=54.0,
        near=0.1,
        far=10.0,
    ),
)

# VAMP subgroup robot names for planning.
_BASE_JOINTS = ["base_x_joint", "base_y_joint", "base_theta_joint"]
_TORSO_JOINTS = ["torso_lift_joint"]
_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# Subgroups whose "joints" list contains any of _BASE_JOINTS are planned in a
# CompoundStateSpace(ReedsShepp + RealVector). See ext/ompl_vamp/planner.hpp.
PLANNING_SUBGROUPS = {
    # Mobile base only (3 DOF, nonholonomic)
    "fetch_base": {"dof": 3, "joints": _BASE_JOINTS},
    # 7-DOF arm (torso + base fixed)
    "fetch_arm": {"dof": 7, "joints": _ARM_JOINTS},
    # 8-DOF arm_with_torso (base fixed) — matches the ikfast solver chain
    "fetch_arm_with_torso": {"dof": 8, "joints": _TORSO_JOINTS + _ARM_JOINTS},
    # 10-DOF base + arm (torso fixed) — useful for navigate-and-reach
    "fetch_base_arm": {"dof": 10, "joints": _BASE_JOINTS + _ARM_JOINTS},
    # 11-DOF whole body — base + torso + arm (full)
    "fetch_whole_body": {
        "dof": 11,
        "joints": _BASE_JOINTS + _TORSO_JOINTS + _ARM_JOINTS,
    },
}

# Fetch "ready" pose — torso raised, arm tucked for manipulation reach.
HOME_JOINTS = np.array(
    [
        # [0:3]   base
        0.0,
        0.0,
        0.0,
        # [3:4]   torso (raised)
        0.35,
        # [4:11]  7-DOF arm (shoulder_pan → wrist_roll)
        1.32,
        1.40,
        -0.20,
        1.72,
        0.0,
        1.66,
        0.0,
    ]
)
