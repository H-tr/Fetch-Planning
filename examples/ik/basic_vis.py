"""IK solver example using TRAC-IK with PyBullet visualization."""

import time

import numpy as np
import pybullet as pb

from fetch_planning.fetch import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    JOINT_GROUPS,
    fetch_robot_config,
)
from fetch_planning.envs.pybullet_env import PyBulletEnv
from fetch_planning.kinematics import create_ik_solver
from fetch_planning.types import IKConfig, SE3Pose, SolveType

# Home configuration subsets matching each chain's joint ordering
G = JOINT_GROUPS
HOME_ARM = HOME_JOINTS[G["arm"]]
HOME_ARM_WITH_TORSO = np.concatenate(
    [
        HOME_JOINTS[G["torso"]],
        HOME_JOINTS[G["arm"]],
    ]
)

# Mapping from chain solution indices to body joint indices.
# Body joints = HOME_JOINTS[3:] (torso + 7-arm = 8 joints).
# Index 0 = torso_lift_joint, indices 1..7 = arm joints.
CHAIN_TO_BODY = {
    "arm": list(range(1, 8)),  # 7 arm joints → body[1:8]
    "arm_with_torso": list(range(0, 8)),  # torso + arm → body[0:8]
}

CHAIN_SEEDS = {
    "arm": HOME_ARM,
    "arm_with_torso": HOME_ARM_WITH_TORSO,
}


def get_ee_link_index(env, link_name):
    """Find PyBullet link index by name."""
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    return -1


def draw_frame_at_link(env, link_index, length=0.08, width=3):
    """Draw RGB axes at a link's world pose. Returns debug line IDs."""
    client = env.sim.client
    state = client.getLinkState(env.sim.skel_id, link_index)
    pos = np.array(state[0])
    rot = np.array(client.getMatrixFromQuaternion(state[1])).reshape(3, 3)

    line_ids = []
    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        axis = np.zeros(3)
        axis[axis_idx] = length
        end = (pos + rot @ axis).tolist()
        line_ids.append(
            client.addUserDebugLine(pos.tolist(), end, color, lineWidth=width)
        )
    return line_ids


def draw_frame_at_pose(env, pos, rot, length=0.08, width=3):
    """Draw RGB axes at a given world pose. Returns debug line IDs."""
    client = env.sim.client
    origin = pos.tolist()
    line_ids = []
    for axis_idx, color in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        axis = np.zeros(3)
        axis[axis_idx] = length
        end = (pos + rot @ axis).tolist()
        line_ids.append(client.addUserDebugLine(origin, end, color, lineWidth=width))
    return line_ids


def wait_key(env, key, msg):
    """Wait for a key press in the PyBullet GUI."""
    client = env.sim.client
    text_id = client.addUserDebugText(
        msg, [0, 0, 1.5], textColorRGB=[0, 0, 0], textSize=1.5
    )
    print(msg)
    while True:
        keys = client.getKeyboardEvents()
        if key in keys and keys[key] & pb.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.01)
    client.removeUserDebugItem(text_id)


def test_chain(env, chain_name):
    """Solve IK for one chain and visualize."""
    print(f"\n{'='*60}")
    print(f"Chain: {chain_name}")
    print(f"{'='*60}")

    config = IKConfig(
        timeout=0.2,
        epsilon=1e-5,
        solve_type=SolveType.DISTANCE,
        max_attempts=10,
    )

    solver = create_ik_solver(chain_name, config=config)
    seed = CHAIN_SEEDS[chain_name]
    ee_link = CHAIN_CONFIGS[chain_name].ee_link
    ee_idx = get_ee_link_index(env, ee_link)

    print(f"  DOF: {solver.num_joints}")
    print(f"  base: {solver.base_frame}")
    print(f"  ee:   {solver.ee_frame}")

    # Show home config and draw current EE frame
    env.set_joint_states(HOME_JOINTS[3:])
    debug_lines = draw_frame_at_link(env, ee_idx, length=0.06, width=2)

    # FK to get current EE pose
    current_pose = solver.fk(seed)

    # Define target pose
    if "torso" in chain_name or "whole" in chain_name:
        offset = np.array([0.10, 0.08, 0.05])
        angle = np.deg2rad(20)
        rot_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        target_pose = SE3Pose(
            position=current_pose.position + offset,
            rotation=rot_z @ current_pose.rotation,
        )
    else:
        # Arm-only: keep orientation, small position offset toward the front
        target_pose = SE3Pose(
            position=current_pose.position + np.array([0.05, 0.0, -0.05]),
            rotation=current_pose.rotation,
        )

    wait_key(env, ord("n"), f"[{chain_name}] Home config. Press 'n' to solve IK.")

    # Solve IK
    result = solver.solve(target_pose, seed=seed)
    print(
        f"  IK: {result.status.value}, "
        f"pos_err={result.position_error:.6f}m, "
        f"ori_err={result.orientation_error:.6f}rad"
    )

    if result.joint_positions is not None:
        # Apply solution: overlay IK result onto body joints
        body_joints = HOME_JOINTS[3:].copy()
        for i, bi in enumerate(CHAIN_TO_BODY[chain_name]):
            body_joints[bi] = float(result.joint_positions[i])
        env.set_joint_states(body_joints)
        debug_lines += draw_frame_at_link(env, ee_idx, length=0.05, width=2)

    wait_key(env, ord("n"), f"[{chain_name}] Solution shown. Press 'n' for next.")

    # Clean up
    for lid in debug_lines:
        env.sim.client.removeUserDebugItem(lid)


def main():
    print("TRAC-IK Solver Example")
    print("=" * 60)

    env = PyBulletEnv(fetch_robot_config, visualize=True)

    for chain_name in ["arm", "arm_with_torso"]:
        try:
            test_chain(env, chain_name)
        except Exception as e:
            print(f"  ERROR on {chain_name}: {e}")
            import traceback

            traceback.print_exc()

    wait_key(env, ord("q"), "All chains done. Press 'q' to quit.")
    print("\nDone.")


if __name__ == "__main__":
    main()