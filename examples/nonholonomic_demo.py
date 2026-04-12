"""Whole-body motion planning with non-holonomic base constraint.

Demonstrates **true whole-body planning** (11 DOF) where base and arm
move **simultaneously** in a single OMPL call:

* **Base (3 DOF)**: Reeds-Shepp curves enforce non-holonomic
  differential-drive constraints.
* **Arm (8 DOF)**: ``RealVectorStateSpace`` with linear interpolation.
* **Both planned together**: the start and goal differ in *both* base
  pose and arm configuration, so OMPL explores the full 11-DOF space
  and the robot visibly moves base + arm at the same time.

Key whole-body motions (base + arm change simultaneously):

    - Navigate to table while unfolding arm to pregrasp  (base+arm)
    - Navigate along table while carrying apple           (base+arm)
    - Navigate across room while transitioning arm        (base+arm)
    - Navigate back while tucking arm                     (base+arm)

Usage::

    pixi run python examples/nonholonomic_demo.py
    pixi run python examples/nonholonomic_demo.py --nav_time=0.05
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

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
from fetch_planning.planning import create_planner
from fetch_planning.types import (
    IKConfig,
    PlannerConfig,
    SE3Pose,
    SolveType,
)

# ── Constants ──────────────────────────────────────────────────────────

RLS_ROOT = "assets/envs/rls_env"
MESH_DIR = f"{RLS_ROOT}/meshes"
PCD_DIR = f"{RLS_ROOT}/pcd"

SCENE_PROPS: list[tuple[str, str]] = [
    ("rls_2", "rls_2"),
    ("open_kitchen", "open_kitchen"),
    ("wall", "wall"),
    ("workstation", "workstation"),
    ("table", "table"),
    ("sofa", "sofa"),
    ("tea_table", "coffee_table"),
]

ARM_SUBGROUP = "fetch_arm_with_torso"
TORSO_ARM_IDX = np.array([3, 4, 5, 6, 7, 8, 9, 10])
GRIPPER_LINK = "gripper_link"

# ── Robot base poses (x, y, theta) ────────────────────────────────────

BASE_START = np.array([-1.50, -0.50, 0.0])
BASE_TABLE = np.array([-2.00, 0.70, np.pi / 2])
BASE_TABLE_FAR = np.array([-1.60, 0.70, np.pi / 2])
BASE_SOFA = np.array([1.30, 0.30, 0.0])
BASE_MID = np.array([0.0, 0.0, 0.0])
BASE_TEA = np.array([1.0, 1.0, -np.pi / 2])

# ── Object positions ──────────────────────────────────────────────────

APPLE_ON_TABLE = np.array([-2.30, 1.35, 0.77])
APPLE_PLACE_SPOT = np.array([-2.10, 1.45, 0.77])

# ── Planning parameters ──────────────────────────────────────────────

NAV_TIME = 0.1
ARM_TIME = 1.0
PREGRASP_OFFSET = 0.12
GRASP_Z_ABOVE = 0.18

TUCK_ARM = HOME_JOINTS[3:].copy()
BASE_BOUNDS = dict(x_lo=-4.0, x_hi=4.0, y_lo=-2.0, y_hi=4.0)


# ── Scene helpers ─────────────────────────────────────────────────────

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


# ── IK ────────────────────────────────────────────────────────────────

@dataclass
class IKBundle:
    solver: TracIKSolver
    lower: np.ndarray
    upper: np.ndarray


def build_ik_bundle() -> IKBundle:
    chain = CHAIN_CONFIGS["arm_with_torso"]
    cfg = IKConfig(
        timeout=0.3, epsilon=1e-5,
        solve_type=SolveType.DISTANCE, max_attempts=40,
    )
    solver = TracIKSolver(chain, cfg)
    lo, hi = solver.joint_limits
    return IKBundle(solver=solver, lower=lo.copy(), upper=hi.copy())


def _world_to_base_frame(base_pose: np.ndarray, world_se3: SE3Pose) -> SE3Pose:
    bx, by, bth = float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
    c, s = np.cos(bth), np.sin(bth)
    R_wb = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    t_wb = np.array([bx, by, 0.0])
    return SE3Pose(
        position=R_wb @ (world_se3.position - t_wb),
        rotation=R_wb @ world_se3.rotation,
    )


def solve_world_arm_goal(
    ik: IKBundle, current_full: np.ndarray, world_pose: SE3Pose,
) -> np.ndarray:
    local_pose = _world_to_base_frame(current_full[:3], world_pose)
    result = ik.solver.solve(local_pose, seed=current_full[3:].copy())
    if not result.success or result.joint_positions is None:
        raise RuntimeError(
            f"IK failed for {world_pose.position.tolist()}: "
            f"{result.status.value}, pos_err={result.position_error:.4f}"
        )
    out = current_full.copy()
    out[3:] = result.joint_positions
    return out


# ── Grasp helpers ─────────────────────────────────────────────────────

def side_grasp_rotation(base_yaw: float) -> np.ndarray:
    c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
    approach = np.array([c, s, 0.0])
    y_axis = -approach
    z_axis = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def build_grasp_pose(
    object_xyz: np.ndarray, base_yaw: float,
) -> tuple[SE3Pose, SE3Pose, np.ndarray, np.ndarray]:
    R = side_grasp_rotation(base_yaw)
    c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
    approach = np.array([c, s, 0.0])
    grasp_pos = np.array(object_xyz, dtype=float)
    grasp_pos[2] += GRASP_Z_ABOVE
    pregrasp_pos = grasp_pos - PREGRASP_OFFSET * approach
    return (
        SE3Pose(position=grasp_pos, rotation=R),
        SE3Pose(position=pregrasp_pos, rotation=R),
        grasp_pos, pregrasp_pos,
    )


# ── Reporting ─────────────────────────────────────────────────────────

def _report(label: str, result) -> None:
    n = result.path.shape[0] if result.path is not None else 0
    ms = result.planning_time_ns / 1e6
    tag = "OK" if result.success else "FAIL"
    cost_str = f", cost {result.path_cost:.3f}" if result.success else ""
    print(f"  [{label}] {tag}: {n} wp in {ms:.1f} ms{cost_str}")


# ── Segment bookkeeping ──────────────────────────────────────────────

@dataclass
class Segment:
    path: np.ndarray
    attach_body_id: int | None
    attach_local_tf: np.ndarray | None
    banner: str


# ── Attachment helpers ────────────────────────────────────────────────

def find_link_index(env: PyBulletEnv, link_name: str) -> int:
    client = env.sim.client
    for i in range(client.getNumJoints(env.sim.skel_id)):
        info = client.getJointInfo(env.sim.skel_id, i)
        if info[12].decode("utf-8") == link_name:
            return i
    raise RuntimeError(f"link {link_name!r} not found")


def capture_local_transform(env: PyBulletEnv, link_idx: int, body_id: int) -> np.ndarray:
    client = env.sim.client
    ls = client.getLinkState(env.sim.skel_id, link_idx)
    lp, lq = np.asarray(ls[0]), np.asarray(ls[1])
    lR = np.asarray(client.getMatrixFromQuaternion(lq)).reshape(3, 3)
    op, oq = client.getBasePositionAndOrientation(body_id)
    oR = np.asarray(client.getMatrixFromQuaternion(oq)).reshape(3, 3)
    local = np.eye(4)
    local[:3, :3] = lR.T @ oR
    local[:3, 3] = lR.T @ (np.asarray(op) - lp)
    return local


def apply_attachment(env: PyBulletEnv, link_idx: int, body_id: int, local_tf: np.ndarray) -> None:
    client = env.sim.client
    ls = client.getLinkState(env.sim.skel_id, link_idx)
    lp, lq = np.asarray(ls[0]), np.asarray(ls[1])
    lR = np.asarray(client.getMatrixFromQuaternion(lq)).reshape(3, 3)
    wR = lR @ local_tf[:3, :3]
    wp = lR @ local_tf[:3, 3] + lp
    m = wR
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0:
        s = float(np.sqrt(tr + 1) * 2)
        w, x, y, z = .25*s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = float(np.sqrt(1+m[0,0]-m[1,1]-m[2,2]) * 2)
        w, x, y, z = (m[2,1]-m[1,2])/s, .25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s
    elif m[1, 1] > m[2, 2]:
        s = float(np.sqrt(1+m[1,1]-m[0,0]-m[2,2]) * 2)
        w, x, y, z = (m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, .25*s, (m[1,2]+m[2,1])/s
    else:
        s = float(np.sqrt(1+m[2,2]-m[0,0]-m[1,1]) * 2)
        w, x, y, z = (m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, .25*s
    client.resetBasePositionAndOrientation(body_id, wp.tolist(), [x, y, z, w])


# ── Whole-body planning ──────────────────────────────────────────────

def plan_whole_body(
    start_full: np.ndarray,
    goal_full: np.ndarray,
    cloud: np.ndarray,
    label: str,
    planner_name: str = "rrtc",
    time_limit: float = NAV_TIME,
) -> tuple[np.ndarray, float]:
    """11-DOF whole-body plan: non-holonomic base + linear arm, together.

    *start_full* and *goal_full* are both 11-DOF.  When the base AND
    arm portions differ, the robot moves both simultaneously.
    """
    planner = create_planner(
        "fetch_whole_body",
        config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
        pointcloud=cloud,
    )
    planner.set_base_bounds(**BASE_BOUNDS)
    result = planner.plan(start_full, goal_full)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"Whole-body planning failed: {label}: {result.status.value}")
    return result.path, result.planning_time_ns / 1e6


# ── Linear interpolation (grasp approach / retreat only) ─────────────

def _linear_path(start_full: np.ndarray, end_full: np.ndarray, steps: int = 30) -> np.ndarray:
    """Short linear interpolation for the final grasp approach / retreat.

    Base is kept fixed — only arm moves linearly.
    """
    t = np.linspace(0, 1, steps)
    path = np.empty((steps, 11))
    for i in range(steps):
        path[i] = start_full + t[i] * (end_full - start_full)
        path[i, :3] = start_full[:3]
    return path


# ── Playback ──────────────────────────────────────────────────────────

def play_segments(env, segments, gripper_link_idx, fps=60.0):
    client = env.sim.client
    dt = 1.0 / fps
    frames = []
    for si, seg in enumerate(segments):
        for ri in range(seg.path.shape[0]):
            frames.append((si, ri))
    idx, n, playing = 0, len(frames), False
    space, left, right = ord(" "), pb.B3G_LEFT_ARROW, pb.B3G_RIGHT_ARROW

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
                apply_attachment(env, gripper_link_idx, seg.attach_body_id, seg.attach_local_tf)
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


# ── Main ──────────────────────────────────────────────────────────────

def main(
    pcd_stride: int = 1,
    visualize: bool = True,
    nav_planner: str = "rrtc",
    nav_time: float = NAV_TIME,
) -> None:
    print("=" * 65)
    print("  Whole-Body Non-Holonomic Motion Planning Demo")
    print("  11 DOF: ReedsSheppStateSpace(3) + RealVectorStateSpace(8)")
    print(f"  Planner: {nav_planner}  |  Budget: {nav_time * 1000:.0f} ms")
    print("  Base + arm planned TOGETHER in CompoundStateSpace")
    print("=" * 65)

    env = PyBulletEnv(fetch_robot_config, visualize=visualize)

    # ── Load scene ────────────────────────────────────────────────────
    print("\n-- loading scene --")
    load_room_meshes(env)
    cloud = load_room_pointcloud(stride=pcd_stride)
    print(f"  collision cloud: {len(cloud):,} points (stride={pcd_stride})")
    env.add_pointcloud(cloud[::4], pointsize=2)

    if visualize:
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=3.5,
            cameraYaw=-90.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[-1.5, 0.5, 0.9],
        )

    apple_id = place_graspable(env, "apple", APPLE_ON_TABLE)

    current = HOME_JOINTS.copy()
    current[:3] = BASE_START
    env.set_configuration(current)

    gripper_link = find_link_index(env, GRIPPER_LINK)
    ik = build_ik_bundle()

    segments: list[Segment] = []
    client = env.sim.client
    wb_times: list[float] = []

    # ==================================================================
    # Stage 1: WHOLE-BODY start -> table + unfold arm to pregrasp
    #   Base moves from start to table, arm moves from tuck to pregrasp
    #   — both happen simultaneously in one plan call.
    # ==================================================================
    print("\n-- stage 1: whole-body nav to table + unfold arm to pregrasp --")
    base_yaw_table = float(BASE_TABLE[2])
    grasp_pose, pregrasp_pose, _, _ = build_grasp_pose(APPLE_ON_TABLE, base_yaw_table)

    # Build the 11-DOF goal: base at table, arm at pregrasp
    goal_at_table = current.copy()
    goal_at_table[:3] = BASE_TABLE
    pregrasp_full = solve_world_arm_goal(ik, goal_at_table, pregrasp_pose)
    # pregrasp_full now has base=TABLE, arm=pregrasp config

    path, ms = plan_whole_body(
        current, pregrasp_full, cloud,
        "wb: start->table + tuck->pregrasp", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=None, attach_local_tf=None,
        banner="stage 1: wb nav + unfold arm (base+arm simultaneous)",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 2: Grasp approach + grasp (short linear, arm only)
    # ==================================================================
    print("\n-- stage 2: grasp approach --")
    grasp_full = solve_world_arm_goal(ik, current, grasp_pose)
    approach = _linear_path(current, grasp_full, steps=30)
    segments.append(Segment(
        path=approach, attach_body_id=None, attach_local_tf=None,
        banner="stage 2: grasp approach (arm only)",
    ))

    # Capture apple attachment at grasp config.
    env.set_configuration(grasp_full)
    client.resetBasePositionAndOrientation(apple_id, APPLE_ON_TABLE.tolist(), [0, 0, 0, 1])
    apple_tf = capture_local_transform(env, gripper_link, apple_id)

    # Lift back to pregrasp.
    lift = _linear_path(grasp_full, pregrasp_full, steps=30)
    segments.append(Segment(
        path=lift, attach_body_id=apple_id,
        attach_local_tf=apple_tf, banner="stage 2: lift apple",
    ))
    current = pregrasp_full.copy()

    # ==================================================================
    # Stage 3: WHOLE-BODY carry apple to place spot
    #   Base moves table -> table_far, arm transitions carrying -> pre-place
    # ==================================================================
    print("\n-- stage 3: whole-body carry apple to place spot --")
    base_yaw_far = float(BASE_TABLE_FAR[2])
    place_pose, pre_place_pose, _, _ = build_grasp_pose(APPLE_PLACE_SPOT, base_yaw_far)

    goal_at_far = current.copy()
    goal_at_far[:3] = BASE_TABLE_FAR
    pre_place_full = solve_world_arm_goal(ik, goal_at_far, pre_place_pose)

    path, ms = plan_whole_body(
        current, pre_place_full, cloud,
        "wb: table->far + carry->preplace", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=apple_id,
        attach_local_tf=apple_tf,
        banner="stage 3: wb carry apple (base+arm simultaneous)",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 4: Place apple (short linear, arm only)
    # ==================================================================
    print("\n-- stage 4: place apple --")
    place_full = solve_world_arm_goal(ik, current, place_pose)
    lower = _linear_path(current, place_full, steps=30)
    segments.append(Segment(
        path=lower, attach_body_id=apple_id,
        attach_local_tf=apple_tf, banner="stage 4: lower apple",
    ))

    retreat = _linear_path(place_full, pre_place_full, steps=30)
    segments.append(Segment(
        path=retreat, attach_body_id=None, attach_local_tf=None,
        banner="stage 4: retreat",
    ))
    current = pre_place_full.copy()

    # ==================================================================
    # Stage 5: WHOLE-BODY tour: table_far -> mid room + tuck arm
    # ==================================================================
    print("\n-- stage 5: whole-body nav to mid room + tuck arm --")
    goal_mid = HOME_JOINTS.copy()
    goal_mid[:3] = BASE_MID
    # arm goes to tuck during navigation
    path, ms = plan_whole_body(
        current, goal_mid, cloud,
        "wb: table_far->mid + arm->tuck", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=None, attach_local_tf=None,
        banner="stage 5: wb nav + tuck arm (base+arm simultaneous)",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 6: WHOLE-BODY mid -> sofa (arm stays tucked)
    # ==================================================================
    print("\n-- stage 6: whole-body nav mid -> sofa --")
    goal_sofa = current.copy()
    goal_sofa[:3] = BASE_SOFA
    path, ms = plan_whole_body(
        current, goal_sofa, cloud,
        "wb: mid->sofa", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=None, attach_local_tf=None,
        banner="stage 6: wb nav mid -> sofa",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 7: WHOLE-BODY sofa -> tea table (arm stays tucked)
    # ==================================================================
    print("\n-- stage 7: whole-body nav sofa -> tea table --")
    goal_tea = current.copy()
    goal_tea[:3] = BASE_TEA
    path, ms = plan_whole_body(
        current, goal_tea, cloud,
        "wb: sofa->tea_table", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=None, attach_local_tf=None,
        banner="stage 7: wb nav sofa -> tea table",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 8: WHOLE-BODY tea table -> table_far + unfold arm to pregrasp
    #   Navigate back AND prepare arm for second apple pick simultaneously
    # ==================================================================
    print("\n-- stage 8: whole-body nav to table + unfold arm to pregrasp --")
    base_yaw_far2 = float(BASE_TABLE_FAR[2])
    grasp_pose2, pregrasp_pose2, _, _ = build_grasp_pose(APPLE_PLACE_SPOT, base_yaw_far2)

    goal_table2 = current.copy()
    goal_table2[:3] = BASE_TABLE_FAR
    pregrasp_full2 = solve_world_arm_goal(ik, goal_table2, pregrasp_pose2)

    path, ms = plan_whole_body(
        current, pregrasp_full2, cloud,
        "wb: tea->table + tuck->pregrasp", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=None, attach_local_tf=None,
        banner="stage 8: wb nav + unfold arm (base+arm simultaneous)",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 9: Pick apple again (arm only)
    # ==================================================================
    print("\n-- stage 9: pick apple again --")
    grasp_full2 = solve_world_arm_goal(ik, current, grasp_pose2)
    approach2 = _linear_path(current, grasp_full2, steps=30)
    segments.append(Segment(
        path=approach2, attach_body_id=None, attach_local_tf=None,
        banner="stage 9: grasp approach (arm only)",
    ))

    env.set_configuration(grasp_full2)
    client.resetBasePositionAndOrientation(apple_id, APPLE_PLACE_SPOT.tolist(), [0, 0, 0, 1])
    apple_tf2 = capture_local_transform(env, gripper_link, apple_id)

    lift2 = _linear_path(grasp_full2, pregrasp_full2, steps=30)
    segments.append(Segment(
        path=lift2, attach_body_id=apple_id,
        attach_local_tf=apple_tf2, banner="stage 9: lift apple",
    ))
    current = pregrasp_full2.copy()

    # ==================================================================
    # Stage 10: WHOLE-BODY carry apple back to original spot
    #   Base: table_far -> table, arm: carrying -> pre-place
    # ==================================================================
    print("\n-- stage 10: whole-body carry apple back --")
    base_yaw_orig = float(BASE_TABLE[2])
    place_pose_orig, pre_place_pose_orig, _, _ = build_grasp_pose(APPLE_ON_TABLE, base_yaw_orig)

    goal_orig = current.copy()
    goal_orig[:3] = BASE_TABLE
    pre_place_orig = solve_world_arm_goal(ik, goal_orig, pre_place_pose_orig)

    path, ms = plan_whole_body(
        current, pre_place_orig, cloud,
        "wb: far->table + carry->preplace", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=apple_id,
        attach_local_tf=apple_tf2,
        banner="stage 10: wb carry apple back (base+arm simultaneous)",
    ))
    current = path[-1].copy()

    # ==================================================================
    # Stage 11: Place apple back (arm only)
    # ==================================================================
    print("\n-- stage 11: place apple back --")
    place_full_orig = solve_world_arm_goal(ik, current, place_pose_orig)
    lower2 = _linear_path(current, place_full_orig, steps=30)
    segments.append(Segment(
        path=lower2, attach_body_id=apple_id,
        attach_local_tf=apple_tf2, banner="stage 11: lower apple",
    ))

    retreat2 = _linear_path(place_full_orig, pre_place_orig, steps=30)
    segments.append(Segment(
        path=retreat2, attach_body_id=None, attach_local_tf=None,
        banner="stage 11: retreat",
    ))
    current = pre_place_orig.copy()

    # ==================================================================
    # Stage 12: WHOLE-BODY return to start + tuck arm
    # ==================================================================
    print("\n-- stage 12: whole-body nav home + tuck arm --")
    goal_home = HOME_JOINTS.copy()
    goal_home[:3] = BASE_START
    path, ms = plan_whole_body(
        current, goal_home, cloud,
        "wb: table->start + arm->tuck", nav_planner, nav_time,
    )
    wb_times.append(ms)
    segments.append(Segment(
        path=path, attach_body_id=None, attach_local_tf=None,
        banner="stage 12: wb nav home + tuck (base+arm simultaneous)",
    ))

    # ==================================================================
    # Summary
    # ==================================================================
    total_wp = sum(s.path.shape[0] for s in segments)
    print("\n" + "=" * 65)
    print("  PLANNING SUMMARY  (whole-body 11-DOF, non-holonomic base)")
    print("=" * 65)
    print(f"  Whole-body calls     : {len(wb_times)}")
    print(f"  WB times (ms)        : {['%.1f' % t for t in wb_times]}")
    print(f"  Mean WB time         : {np.mean(wb_times):.1f} ms")
    print(f"  Max WB time          : {np.max(wb_times):.1f} ms")
    print(f"  Total segments       : {len(segments)}")
    print(f"  Total waypoints      : {total_wp:,}")
    print("=" * 65)

    if not visualize:
        return

    env.set_configuration(segments[0].path[0])
    client.resetBasePositionAndOrientation(apple_id, APPLE_ON_TABLE.tolist(), [0, 0, 0, 1])

    print(f"\n-- ready: {total_wp:,} total frames --")
    play_segments(env, segments, gripper_link)


if __name__ == "__main__":
    Fire(main)
