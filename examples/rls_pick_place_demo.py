"""Long-horizon pick-and-place inside the full RLS-env room.

A single end-to-end showcase of the project's planning stack:

* **Subgroup planning** — navigation uses ``fetch_base`` with
  BiT\* (asymptotically optimal), grasping uses
  ``fetch_arm_with_torso`` with RRTConnect (feasibility).
* **Collision avoidance** — every planning call is handed the
  151k-point cloud concatenated from all seven ``pcd/*.ply`` files in
  ``assets/envs/rls_env``.  VAMP's SIMD checker keeps it real-time.
* **IK for world-frame grasps** — grasp poses are hard-coded top-down
  SE(3) transforms in the world frame.  TRAC-IK on the
  ``whole_body`` chain (11 DOF) turns them into arm joint goals;
  the base joints are tight-clamped to their current values so only
  the torso + arm moves.
* **Constrained planning** — every pregrasp->grasp and grasp->lift
  motion is planned on a CasADi straight-line manifold (TCP pinned
  to the approach axis) via ``ProjectedStateSpace``.

Storyline (one PyBullet window, one concatenated path):

    1. robot spawns inside the room next to the big table
    2. pick apple from the table top
    3. place apple back on the table (different spot)
    4. base-navigate across the room to the sofa (BiT\*)
    5. pick bottle from the sofa
    6. base-navigate back to the table
    7. place bottle beside the apple

Usage::

    pixi run python examples/rls_pick_place_demo.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pybullet as pb
import trimesh
from fire import Fire

from fetch_planning.config.robot_config import (
    CHAIN_CONFIGS,
    HOME_JOINTS,
    fetch_robot_config,
)
from fetch_planning.envs.pybullet_env import PyBulletEnv
from fetch_planning.kinematics.trac_ik_solver import TracIKSolver
from fetch_planning.planning import Constraint, SymbolicContext, create_planner
from fetch_planning.types import (
    ChainConfig,
    IKConfig,
    PlannerConfig,
    SE3Pose,
    SolveType,
)

# ── Constants ──────────────────────────────────────────────────────
# Subgroup names from fetch_planning.config.robot_config.
BASE_SUBGROUP = "fetch_base"
ARM_SUBGROUP = "fetch_arm_with_torso"  # 8 DOF: torso_lift + 7-DOF arm

# Full-DOF layout (11 DOF):
#   [0:3]  virtual base (x, y, theta)
#   [3:4]  torso_lift_joint
#   [4:11] 7-DOF arm (shoulder_pan -> wrist_roll)
#
# fetch_arm_with_torso subgroup indices inside the 11-DOF vector:
TORSO_ARM_IDX = np.array([3, 4, 5, 6, 7, 8, 9, 10])  # torso + arm

# The gripper body frame — used for IK target and line constraint.
GRIPPER_LINK = "gripper_link"
LEFT_FINGER_LINK = "l_gripper_finger_link"
RIGHT_FINGER_LINK = "r_gripper_finger_link"

# For Fetch, the gripper_link origin coincides with the finger
# midpoint, so this offset is zero.
FINGER_MIDPOINT_IN_GRIPPER = np.array([0.0, 0.0, 0.0])

# Asset paths.
RLS_ROOT = "assets/envs/rls_env"
MESH_DIR = f"{RLS_ROOT}/meshes"
PCD_DIR = f"{RLS_ROOT}/pcd"

# Scene props — these seven meshes and pcds share a single room frame
# so they load at identity.
SCENE_PROPS: list[tuple[str, str]] = [
    ("rls_2", "rls_2"),
    ("open_kitchen", "open_kitchen"),
    ("wall", "wall"),
    ("workstation", "workstation"),
    ("table", "table"),
    ("sofa", "sofa"),
    ("tea_table", "coffee_table"),
]

# ── Hard-coded poses ────────────────────────────────────────────────
# Robot base poses (base_x, base_y, base_theta).
BASE_NEAR_TABLE = np.array([-2.00, 0.70, np.pi / 2])
BASE_NEAR_SOFA = np.array([1.15, 0.30, 0.0])

# Graspable object positions (world xyz).
APPLE_ON_TABLE = np.array([-2.30, 1.35, 0.77])
APPLE_PLACE_ON_TABLE = np.array([-2.10, 1.45, 0.77])
BOTTLE_ON_SOFA = np.array([1.95, 0.30, 0.72])
BOTTLE_PLACE_ON_TABLE = np.array([-2.45, 1.45, 0.77])

# Vertical approach offset (pregrasp sits this far above the grasp in +z).
PREGRASP_OFFSET = 0.12

# Planning budgets.
ARM_FREE_TIME = 3.0
ARM_LINE_TIME = 6.0
BASE_NAV_TIME = 5.0

# The finger midpoint sits this far above the object centroid.
GRASP_Z_ABOVE_OBJECT = 0.18


# ── Scene setup ────────────────────────────────────────────────────

def load_room_meshes(env: PyBulletEnv) -> None:
    for mesh_name, _ in SCENE_PROPS:
        path = os.path.abspath(f"{MESH_DIR}/{mesh_name}/{mesh_name}.obj")
        env.add_mesh(path, position=np.zeros(3))


def load_room_pointcloud(stride: int = 1) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for _, pcd_name in SCENE_PROPS:
        p = os.path.abspath(f"{PCD_DIR}/{pcd_name}.ply")
        pc = trimesh.load(p)
        v = np.asarray(pc.vertices, dtype=np.float32)
        if stride > 1:
            v = v[::stride]
        chunks.append(v)
    return np.concatenate(chunks, axis=0)


def place_graspable(env: PyBulletEnv, mesh_name: str, xyz: np.ndarray) -> int:
    path = os.path.abspath(f"{MESH_DIR}/{mesh_name}/{mesh_name}.obj")
    return env.add_mesh(path, position=np.asarray(xyz, dtype=float))


# ── CasADi / SymbolicContext helpers ───────────────────────────────

def gripper_translation(ctx: SymbolicContext) -> ca.SX:
    return ctx.link_translation(GRIPPER_LINK)


def gripper_position_numeric(ctx: SymbolicContext, q_active: np.ndarray) -> np.ndarray:
    return np.asarray(ctx.evaluate_link_pose(GRIPPER_LINK, q_active))[:3, 3]


def make_line_constraint(
    ctx: SymbolicContext,
    p_from: np.ndarray,
    p_to: np.ndarray,
    name: str,
) -> Constraint:
    """Pin the gripper link to the line through *p_from* -> *p_to*."""
    p_from = np.asarray(p_from, dtype=float)
    p_to = np.asarray(p_to, dtype=float)
    d = p_to - p_from
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-9:
        raise ValueError("make_line_constraint: p_from and p_to coincide")
    d = d / d_norm
    seed = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, seed)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    gripper = gripper_translation(ctx)
    diff = gripper - ca.DM(p_from.tolist())
    residual: ca.SX = ca.vertcat(
        ca.dot(diff, ca.DM(u.tolist())),
        ca.dot(diff, ca.DM(v.tolist())),
    )
    return Constraint(residual=residual, q_sym=ctx.q, name=name)


# ── World-frame IK ─────────────────────────────────────────────────

@dataclass
class IKBundle:
    """Cached TRAC-IK solver for the whole_body chain."""
    solver: TracIKSolver
    lower: np.ndarray
    upper: np.ndarray


def build_ik_bundle() -> IKBundle:
    """TRAC-IK solver from ``base_link`` to ``gripper_link`` (11 DOF)."""
    chain = ChainConfig(
        base_link="base_link",
        ee_link=GRIPPER_LINK,
        num_joints=11,
        urdf_path=CHAIN_CONFIGS["whole_body"].urdf_path,
    )
    cfg = IKConfig(
        timeout=0.3,
        epsilon=1e-5,
        solve_type=SolveType.DISTANCE,
        max_attempts=40,
    )
    solver = TracIKSolver(chain, cfg)
    lo, hi = solver.joint_limits
    return IKBundle(solver=solver, lower=lo.copy(), upper=hi.copy())


def solve_world_arm_goal(
    ik: IKBundle,
    current_full: np.ndarray,
    world_pose: SE3Pose,
) -> np.ndarray:
    """Solve torso+arm joint values for a world-frame gripper pose.

    * ``current_full`` is the live 11-DOF body configuration.
    * The IK chain has 11 DOF: ``[base_x, base_y, base_theta,
      torso_lift, arm x 7]``.
    * We tight-clamp the base joints (0..2) to their current values
      so only the torso + arm (joints 3..10) move — matching the
      ``fetch_arm_with_torso`` subgroup that will plan around this goal.
    """
    seed = np.array(current_full, dtype=np.float64)
    eps = 1e-4
    lower = ik.lower.copy()
    upper = ik.upper.copy()
    for i in range(3):  # base (0, 1, 2)
        lower[i] = seed[i] - eps
        upper[i] = seed[i] + eps
    ik.solver.set_joint_limits(lower, upper)
    try:
        result = ik.solver.solve(world_pose, seed=seed)
    finally:
        ik.solver.set_joint_limits(ik.lower, ik.upper)

    if not result.success or result.joint_positions is None:
        raise RuntimeError(
            f"IK failed for target {world_pose.position.tolist()}: "
            f"status={result.status.value} pos_err={result.position_error:.4f}"
        )
    new_full = current_full.copy()
    new_full[:] = result.joint_positions
    # Re-force base exactly — the solver is within +/- eps.
    new_full[:3] = current_full[:3]
    return new_full


# ── Planning wrappers ──────────────────────────────────────────────

def _extract_torso_arm(full: np.ndarray) -> np.ndarray:
    return full[TORSO_ARM_IDX].copy()


def plan_arm_free(
    current_full: np.ndarray,
    goal_full: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """RRTConnect over the 8-DOF torso+arm subgroup.  Returns (N, 11)."""
    planner = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_FREE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
    )
    start = _extract_torso_arm(current_full)
    goal = _extract_torso_arm(goal_full)
    result = planner.plan(start, goal)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def plan_arm_line(
    current_full: np.ndarray,
    goal_full: np.ndarray,
    p_from: np.ndarray,
    p_to: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """Line-constrained plan on the torso+arm subgroup.  Returns (N, 11)."""
    ctx = SymbolicContext(ARM_SUBGROUP, base_config=current_full)
    constraint = make_line_constraint(
        ctx, p_from, p_to, name=f"line_{label.replace(' ', '_')}"
    )

    raw_start = _extract_torso_arm(current_full)
    raw_goal = _extract_torso_arm(goal_full)
    start = ctx.project(raw_start, constraint.residual)
    goal = ctx.project(raw_goal, constraint.residual)

    planner = create_planner(
        ARM_SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=ARM_LINE_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
        constraints=[constraint],
    )
    result = planner.plan(start, goal)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def plan_base(
    current_full: np.ndarray,
    base_goal: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """BiT\* base navigation (3 DOF: x, y, theta).  Returns (N, 11)."""
    planner = create_planner(
        BASE_SUBGROUP,
        config=PlannerConfig(planner_name="bitstar", time_limit=BASE_NAV_TIME),
        pointcloud=pointcloud,
        base_config=current_full,
    )
    start = current_full[:3].copy()
    result = planner.plan(start, np.asarray(base_goal, dtype=np.float64))
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"{label}: {result.status.value}")
    return planner.embed_path(result.path)


def _report(label: str, result) -> None:
    n = result.path.shape[0] if result.path is not None else 0
    ms = result.planning_time_ns / 1e6
    print(
        f"  [{label}] {result.status.value}: {n} wp in {ms:.0f} ms"
        + (f", cost {result.path_cost:.2f}" if result.success else "")
    )


# ── Grasp pose builder ─────────────────────────────────────────────

def side_grasp_rotation(base_yaw: float) -> np.ndarray:
    """Rotation for a horizontal side grasp aligned with the robot."""
    c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
    approach = np.array([c, s, 0.0])
    y_axis = -approach
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def build_grasp_pose(
    object_xyz: np.ndarray, base_yaw: float
) -> tuple[SE3Pose, SE3Pose, np.ndarray, np.ndarray]:
    """Return ``(grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos)``."""
    R = side_grasp_rotation(base_yaw)
    c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
    approach = np.array([c, s, 0.0])

    finger_target = np.array(object_xyz, dtype=float)
    finger_target[2] += GRASP_Z_ABOVE_OBJECT

    grasp_pos = finger_target - R @ FINGER_MIDPOINT_IN_GRIPPER
    pregrasp_pos = grasp_pos - PREGRASP_OFFSET * approach

    grasp_pose = SE3Pose(position=grasp_pos, rotation=R)
    pregrasp_pose = SE3Pose(position=pregrasp_pos, rotation=R)
    return grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos


# ── Path segment bookkeeping ───────────────────────────────────────

@dataclass
class Segment:
    path: np.ndarray           # (N, 11)
    attach_body_id: int | None
    attach_local_tf: np.ndarray | None   # (4, 4): mesh pose in gripper frame
    banner: str


# ── High-level actions ─────────────────────────────────────────────

def pick(
    ik: IKBundle,
    current_full: np.ndarray,
    object_xyz: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Plan current -> pregrasp -> grasp (line) -> pregrasp (line)."""
    base_yaw = float(current_full[2])
    grasp_pose, pregrasp_pose, grasp_pos, pregrasp_pos = build_grasp_pose(
        object_xyz, base_yaw
    )

    pregrasp_full = solve_world_arm_goal(ik, current_full, pregrasp_pose)
    grasp_full = solve_world_arm_goal(ik, pregrasp_full, grasp_pose)

    free_path = plan_arm_free(
        current_full, pregrasp_full, pointcloud, f"{label} free->pregrasp"
    )
    approach_path = plan_arm_line(
        pregrasp_full, grasp_full, pregrasp_pos, grasp_pos, pointcloud,
        f"{label} approach",
    )
    lift_path = plan_arm_line(
        grasp_full, pregrasp_full, grasp_pos, pregrasp_pos, pointcloud,
        f"{label} lift",
    )

    return [free_path, approach_path], [lift_path], pregrasp_full


def place(
    ik: IKBundle,
    current_full: np.ndarray,
    object_xyz: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Plan current -> pre-place (free) -> place (line) -> pre-place (line)."""
    base_yaw = float(current_full[2])
    place_pose, pre_place_pose, place_pos, pre_place_pos = build_grasp_pose(
        object_xyz, base_yaw
    )

    pre_place_full = solve_world_arm_goal(ik, current_full, pre_place_pose)
    place_full = solve_world_arm_goal(ik, pre_place_full, place_pose)

    carry_free = plan_arm_free(
        current_full, pre_place_full, pointcloud, f"{label} free->preplace"
    )
    lower = plan_arm_line(
        pre_place_full, place_full, pre_place_pos, place_pos, pointcloud,
        f"{label} lower",
    )
    retreat = plan_arm_line(
        place_full, pre_place_full, place_pos, pre_place_pos, pointcloud,
        f"{label} retreat",
    )

    return [carry_free, lower], [retreat], pre_place_full


def navigate(
    current_full: np.ndarray,
    base_goal: np.ndarray,
    pointcloud: np.ndarray,
    label: str,
) -> np.ndarray:
    """BiT\* base nav; returns the (N, 11) full-DOF path."""
    return plan_base(current_full, base_goal, pointcloud, label)


# ── Playback ───────────────────────────────────────────────────────

def find_link_index(env: PyBulletEnv, link_name: str) -> int:
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    raise RuntimeError(f"link {link_name!r} not found on robot")


def capture_local_transform(
    env: PyBulletEnv, link_idx: int, body_id: int
) -> np.ndarray:
    client = env.sim.client
    link_state = client.getLinkState(env.sim.skel_id, link_idx)
    link_pos = np.asarray(link_state[0])
    link_quat = np.asarray(link_state[1])
    link_R = np.asarray(client.getMatrixFromQuaternion(link_quat)).reshape(3, 3)

    obj_pos, obj_quat = client.getBasePositionAndOrientation(body_id)
    obj_pos = np.asarray(obj_pos)
    obj_R = np.asarray(client.getMatrixFromQuaternion(obj_quat)).reshape(3, 3)

    R_inv = link_R.T
    local = np.eye(4)
    local[:3, :3] = R_inv @ obj_R
    local[:3, 3] = R_inv @ (obj_pos - link_pos)
    return local


def apply_attachment(
    env: PyBulletEnv,
    link_idx: int,
    body_id: int,
    local_tf: np.ndarray,
) -> None:
    client = env.sim.client
    link_state = client.getLinkState(env.sim.skel_id, link_idx)
    link_pos = np.asarray(link_state[0])
    link_quat = np.asarray(link_state[1])
    link_R = np.asarray(client.getMatrixFromQuaternion(link_quat)).reshape(3, 3)

    world_R = link_R @ local_tf[:3, :3]
    world_pos = link_R @ local_tf[:3, 3] + link_pos
    m = world_R
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = float(np.sqrt(t + 1.0) * 2.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    client.resetBasePositionAndOrientation(body_id, world_pos.tolist(), [x, y, z, w])


def play_segments(
    env: PyBulletEnv,
    segments: list[Segment],
    gripper_link_idx: int,
    fps: float = 60.0,
) -> None:
    client = env.sim.client
    dt = 1.0 / fps

    frames: list[tuple[int, int]] = []
    for si, seg in enumerate(segments):
        for ri in range(seg.path.shape[0]):
            frames.append((si, ri))

    idx = 0
    n = len(frames)
    playing = False
    space = ord(" ")
    left = pb.B3G_LEFT_ARROW
    right = pb.B3G_RIGHT_ARROW

    print("\nControls: SPACE play/pause   left/right step   close window to exit\n")
    for s in segments:
        print(f"  -> {s.banner} ({s.path.shape[0]} wp)")

    last_banner = -1
    try:
        while client.isConnected():
            si, ri = frames[idx]
            seg = segments[si]
            env.set_configuration(seg.path[ri])
            if seg.attach_body_id is not None and seg.attach_local_tf is not None:
                apply_attachment(
                    env,
                    gripper_link_idx,
                    seg.attach_body_id,
                    seg.attach_local_tf,
                )
            if si != last_banner:
                print(f"[{si}] {seg.banner}")
                last_banner = si

            keys = client.getKeyboardEvents()
            if space in keys and keys[space] & pb.KEY_WAS_TRIGGERED:
                playing = not playing
            elif not playing and left in keys and keys[left] & pb.KEY_WAS_TRIGGERED:
                idx = (idx - 1) % n
            elif not playing and right in keys and keys[right] & pb.KEY_WAS_TRIGGERED:
                idx = (idx + 1) % n
            elif playing:
                idx = (idx + 1) % n

            time.sleep(dt)
    except pb.error:
        pass


# ── Main ───────────────────────────────────────────────────────────

def main(pcd_stride: int = 1, visualize: bool = True) -> None:
    env = PyBulletEnv(fetch_robot_config, visualize=visualize)
    print("── scene setup ──")
    load_room_meshes(env)
    cloud = load_room_pointcloud(stride=pcd_stride)
    print(f"  collision cloud: {len(cloud)} points (stride={pcd_stride})")
    env.add_pointcloud(cloud[::4], pointsize=2)

    if visualize:
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=3.5,
            cameraYaw=-90.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[-1.5, 0.5, 0.9],
        )

    current_full = HOME_JOINTS.copy()
    current_full[:3] = BASE_NEAR_TABLE
    env.set_configuration(current_full)

    apple_id = place_graspable(env, "apple", APPLE_ON_TABLE)
    bottle_id = place_graspable(env, "bottle", BOTTLE_ON_SOFA)

    gripper_link = find_link_index(env, GRIPPER_LINK)
    ik = build_ik_bundle()

    segments: list[Segment] = []
    client = env.sim.client

    # ── Stage 1: pick apple off the table ────────────────────────────
    print("\n── stage 1: pick apple off the table ──")
    pick_free, pick_lift, end_full = pick(
        ik, current_full, APPLE_ON_TABLE, cloud, "pick apple"
    )
    for p in pick_free:
        segments.append(
            Segment(path=p, attach_body_id=None, attach_local_tf=None,
                    banner="stage 1: approach apple")
        )
    grasp_frame = pick_free[-1][-1]
    env.set_configuration(grasp_frame)
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_ON_TABLE.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    apple_local_tf = capture_local_transform(env, gripper_link, apple_id)

    for p in pick_lift:
        segments.append(
            Segment(path=p, attach_body_id=apple_id,
                    attach_local_tf=apple_local_tf,
                    banner="stage 1: lift apple")
        )
    current_full = end_full

    # ── Stage 2: place apple on the table ────────────────────────────
    print("\n── stage 2: place apple on the table ──")
    place_carry, place_retreat, end_full = place(
        ik, current_full, APPLE_PLACE_ON_TABLE, cloud, "place apple"
    )
    for p in place_carry:
        segments.append(
            Segment(path=p, attach_body_id=apple_id,
                    attach_local_tf=apple_local_tf,
                    banner="stage 2: carry apple -> placement")
        )
    for p in place_retreat:
        segments.append(
            Segment(path=p, attach_body_id=None, attach_local_tf=None,
                    banner="stage 2: retreat from apple")
        )
    current_full = end_full

    # ── Stage 3: navigate to the sofa ────────────────────────────────
    print("\n── stage 3: base navigation -> sofa ──")
    nav_path = navigate(
        current_full, BASE_NEAR_SOFA, cloud, "nav table -> sofa"
    )
    segments.append(
        Segment(path=nav_path, attach_body_id=None, attach_local_tf=None,
                banner="stage 3: nav table -> sofa")
    )
    current_full = nav_path[-1]

    # ── Stage 4: pick bottle off the sofa ────────────────────────────
    print("\n── stage 4: pick bottle off the sofa ──")
    pick_free, pick_lift, end_full = pick(
        ik, current_full, BOTTLE_ON_SOFA, cloud, "pick bottle"
    )
    for p in pick_free:
        segments.append(
            Segment(path=p, attach_body_id=None, attach_local_tf=None,
                    banner="stage 4: approach bottle")
        )
    grasp_frame = pick_free[-1][-1]
    env.set_configuration(grasp_frame)
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_ON_SOFA.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    bottle_local_tf = capture_local_transform(env, gripper_link, bottle_id)
    for p in pick_lift:
        segments.append(
            Segment(path=p, attach_body_id=bottle_id,
                    attach_local_tf=bottle_local_tf,
                    banner="stage 4: lift bottle")
        )
    current_full = end_full

    # ── Stage 5: navigate back to the table ──────────────────────────
    print("\n── stage 5: base navigation -> table ──")
    nav_back = navigate(
        current_full, BASE_NEAR_TABLE, cloud, "nav sofa -> table"
    )
    segments.append(
        Segment(path=nav_back, attach_body_id=bottle_id,
                attach_local_tf=bottle_local_tf,
                banner="stage 5: nav sofa -> table")
    )
    current_full = nav_back[-1]

    # ── Stage 6: place bottle beside the apple ──────────────────────
    print("\n── stage 6: place bottle beside the apple ──")
    place_carry, place_retreat, end_full = place(
        ik, current_full, BOTTLE_PLACE_ON_TABLE, cloud, "place bottle"
    )
    for p in place_carry:
        segments.append(
            Segment(path=p, attach_body_id=bottle_id,
                    attach_local_tf=bottle_local_tf,
                    banner="stage 6: carry bottle -> placement")
        )
    for p in place_retreat:
        segments.append(
            Segment(path=p, attach_body_id=None, attach_local_tf=None,
                    banner="stage 6: retreat from bottle")
        )

    env.set_configuration(segments[0].path[0])
    client.resetBasePositionAndOrientation(
        apple_id, APPLE_ON_TABLE.tolist(), [0.0, 0.0, 0.0, 1.0]
    )
    client.resetBasePositionAndOrientation(
        bottle_id, BOTTLE_ON_SOFA.tolist(), [0.0, 0.0, 0.0, 1.0]
    )

    print(f"\n── ready: {sum(s.path.shape[0] for s in segments)} total frames ──")
    if not visualize:
        return
    play_segments(env, segments, gripper_link)


if __name__ == "__main__":
    Fire(main)