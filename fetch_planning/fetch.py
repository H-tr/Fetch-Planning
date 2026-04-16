"""The Fetch robot's bundled description.

This module collects every concrete value that describes the one
robot this project ships: joint groupings, the home pose, the URDF
chains TRAC-IK and Pinocchio operate on, the VAMP planning subgroups,
the Reeds-Shepp base parameters, and the top-level ``RobotConfig``
instance.

The dataclass *types* themselves live in :mod:`fetch_planning.types.robot`
— this file re-exports the *values* from
:mod:`fetch_planning.config.robot_config` under a flat import path
that mirrors ``autolife_planning.autolife`` in the sister project.

The full configuration vector is 11-DOF:

* ``[0:3]`` — mobile base ``(x, y, theta)``, nonholonomic diff-drive
* ``[3:4]`` — ``torso_lift_joint``
* ``[4:11]`` — 7-DOF arm (shoulder_pan → wrist_roll)

When the active subgroup contains any base joint the planner switches
to OMPL multilevel planning (fiber bundles) with the hierarchy
``RS → RS × R^N``.  See :mod:`fetch_planning.planning.motion_planner`.
"""

from fetch_planning.config.robot_config import (
    BASE_REVERSE_PENALTY,
    BASE_TURNING_RADIUS,
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
    PLANNING_SUBGROUPS,
    VIZ_URDF_PATH,
    WHOLE_BODY_URDF_PATH,
    fetch_robot_config,
)

__all__ = [
    "BASE_REVERSE_PENALTY",
    "BASE_TURNING_RADIUS",
    "CHAIN_CONFIGS",
    "HOME_JOINTS",
    "JOINT_GROUPS",
    "PLANNING_SUBGROUPS",
    "VIZ_URDF_PATH",
    "WHOLE_BODY_URDF_PATH",
    "fetch_robot_config",
]
