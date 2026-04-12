"""Whole-body motion planning with non-holonomic base constraint.

Demonstrates **true whole-body planning** (11 DOF) where base and arm
move **simultaneously** in a single OMPL call:

* **Base (3 DOF)**: Reeds-Shepp curves enforce non-holonomic
  differential-drive constraints.  Reverse penalty discourages backing up.
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

from fetch_planning.config.robot_config import HOME_JOINTS, fetch_robot_config
from fetch_planning.envs.pybullet_env import PyBulletEnv
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig

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

# ── Pre-computed IK solutions (torso + 7-arm) ────────────────────────
# Computed offline; each pair is collision-free at the corresponding
# base pose.  No runtime IK needed.

# Pick apple from BASE_TABLE
PICK1_PREGRASP = np.array([0.32445069, 0.57501428, 0.98200134, -2.09404862,
                           1.73717401, -1.82980346, 2.14599869, -1.05836049])
PICK1_GRASP = np.array([0.26168402, 0.38121553, 0.68414124, -2.30476954,
                        1.37761053, -2.02117134, 2.05609073, -1.11658312])

# Place apple at APPLE_PLACE_SPOT from BASE_TABLE_FAR
PLACE1_PREGRASP = np.array([0.23418852, -0.19039616, 0.30379228, 1.77299377,
                            1.31343351, -0.54517405, 0.50889398, -1.34479722])
PLACE1_GRASP = np.array([0.17713681, -0.00620586, 0.12912140, 1.88172030,
                         0.70991416, -0.39976887, 0.90754807, -1.44344095])

# Pick apple again from APPLE_PLACE_SPOT at BASE_TABLE_FAR
PICK2_PREGRASP = np.array([0.24107071, -0.18703498, 0.33452523, 1.79668031,
                           1.30990650, -0.59485903, 0.52228389, -1.32847864])
PICK2_GRASP = np.array([0.20053258, 0.00252659, 0.19020634, 1.95317591,
                        0.70473539, -0.48611653, 0.92348058, -1.45351085])

# Place apple back at APPLE_ON_TABLE from BASE_TABLE
PLACE2_PREGRASP = np.array([0.33130361, 0.57289324, 1.00113383, -2.10220687,
                            1.72771426, -1.82364582, 2.13266093, -1.07639876])
PLACE2_GRASP = np.array([0.26721660, 0.37859320, 0.69650104, -2.31088897,
                         1.37324950, -2.01473813, 2.04916452, -1.12681534])

# ── Planning parameters ──────────────────────────────────────────────

NAV_TIME = 0.1
ARM_TIME = 1.0
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


def make_full(base: np.ndarray, arm: np.ndarray) -> np.ndarray:
    """Build an 11-DOF config from base (3) + arm (8)."""
    full = HOME_JOINTS.copy()
    full[:3] = base
    full[3:] = arm
    return full


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
    """11-DOF whole-body plan: non-holonomic base + linear arm, together."""
    planner = create_planner(
        "fetch_whole_body",
        config=PlannerConfig(planner_name=planner_name, time_limit=time_limit),
        pointcloud=cloud,
    )
    planner.set_base_bounds(**BASE_BOUNDS)
    result = planner.plan(start_full, goal_full)
    _report(label, result)
    if not result.success or result.path is None:
        raise RuntimeError(f"Planning failed: {label}: {result.status.value}")
    return result.path, result.planning_time_ns / 1e6


def _linear_path(start: np.ndarray, end: np.ndarray, steps: int = 30) -> np.ndarray:
    """Linear interpolation (arm only, base fixed). For grasp approach/retreat."""
    t = np.linspace(0, 1, steps)
    path = np.empty((steps, 11))
    for i in range(steps):
        path[i] = start + t[i] * (end - start)
        path[i, :3] = start[:3]
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

    print("\n-- loading scene --")
    load_room_meshes(env)
    cloud = load_room_pointcloud(stride=pcd_stride)
    print(f"  collision cloud: {len(cloud):,} points (stride={pcd_stride})")
    env.add_pointcloud(cloud[::4], pointsize=2)

    if visualize:
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=3.5, cameraYaw=-90.0, cameraPitch=-25.0,
            cameraTargetPosition=[-1.5, 0.5, 0.9],
        )

    apple_id = place_graspable(env, "apple", APPLE_ON_TABLE)

    current = make_full(BASE_START, TUCK_ARM)
    env.set_configuration(current)

    gripper_link = find_link_index(env, GRIPPER_LINK)
    segments: list[Segment] = []
    client = env.sim.client
    wb_times: list[float] = []

    # ==================================================================
    # Stage 1: WHOLE-BODY start -> table + tuck -> pregrasp
    # ==================================================================
    print("\n-- stage 1: wb nav to table + unfold arm to pregrasp --")
    pregrasp1 = make_full(BASE_TABLE, PICK1_PREGRASP)
    path, ms = plan_whole_body(current, pregrasp1, cloud,
                               "wb: start->table + tuck->pregrasp", nav_planner, nav_time)
    wb_times.append(ms)
    segments.append(Segment(path=path, attach_body_id=None, attach_local_tf=None,
                            banner="stage 1: wb nav + unfold arm (base+arm)"))
    current = path[-1].copy()

    # ==================================================================
    # Stage 2: Grasp approach + lift (arm-only linear)
    # ==================================================================
    print("\n-- stage 2: grasp apple --")
    grasp1 = make_full(BASE_TABLE, PICK1_GRASP)
    approach = _linear_path(current, grasp1, steps=30)
    segments.append(Segment(path=approach, attach_body_id=None, attach_local_tf=None,
                            banner="stage 2: grasp approach (arm only)"))

    env.set_configuration(grasp1)
    client.resetBasePositionAndOrientation(apple_id, APPLE_ON_TABLE.tolist(), [0, 0, 0, 1])
    apple_tf = capture_local_transform(env, gripper_link, apple_id)

    lift = _linear_path(grasp1, pregrasp1, steps=30)
    segments.append(Segment(path=lift, attach_body_id=apple_id,
                            attach_local_tf=apple_tf, banner="stage 2: lift apple"))
    current = pregrasp1.copy()

    # ==================================================================
    # Stage 3: WHOLE-BODY carry apple table -> table_far + arm -> pre-place
    # ==================================================================
    print("\n-- stage 3: wb carry apple to place spot --")
    preplace1 = make_full(BASE_TABLE_FAR, PLACE1_PREGRASP)
    path, ms = plan_whole_body(current, preplace1, cloud,
                               "wb: table->far + carry->preplace", nav_planner, nav_time)
    wb_times.append(ms)
    segments.append(Segment(path=path, attach_body_id=apple_id, attach_local_tf=apple_tf,
                            banner="stage 3: wb carry apple (base+arm)"))
    current = path[-1].copy()

    # ==================================================================
    # Stage 4: Place apple (arm-only linear)
    # ==================================================================
    print("\n-- stage 4: place apple --")
    place1 = make_full(BASE_TABLE_FAR, PLACE1_GRASP)
    lower = _linear_path(current, place1, steps=30)
    segments.append(Segment(path=lower, attach_body_id=apple_id, attach_local_tf=apple_tf,
                            banner="stage 4: lower apple"))
    retreat = _linear_path(place1, preplace1, steps=30)
    segments.append(Segment(path=retreat, attach_body_id=None, attach_local_tf=None,
                            banner="stage 4: retreat"))
    current = preplace1.copy()

    # ==================================================================
    # Stage 5: WHOLE-BODY table_far -> mid + arm -> tuck
    # ==================================================================
    print("\n-- stage 5: wb nav to mid room + tuck arm --")
    goal_mid = make_full(BASE_MID, TUCK_ARM)
    path, ms = plan_whole_body(current, goal_mid, cloud,
                               "wb: far->mid + arm->tuck", nav_planner, nav_time)
    wb_times.append(ms)
    segments.append(Segment(path=path, attach_body_id=None, attach_local_tf=None,
                            banner="stage 5: wb nav + tuck arm (base+arm)"))
    current = path[-1].copy()

    # ==================================================================
    # Stages 6-7: WHOLE-BODY tour (arm stays tucked)
    # ==================================================================
    waypoints = [
        (BASE_SOFA, "wb: mid->sofa"),
        (BASE_TEA, "wb: sofa->tea_table"),
    ]
    for i, (base_goal, label) in enumerate(waypoints, start=6):
        print(f"\n-- stage {i}: {label} --")
        goal = make_full(base_goal, TUCK_ARM)
        path, ms = plan_whole_body(current, goal, cloud, label, nav_planner, nav_time)
        wb_times.append(ms)
        segments.append(Segment(path=path, attach_body_id=None, attach_local_tf=None,
                                banner=f"stage {i}: {label}"))
        current = path[-1].copy()

    # ==================================================================
    # Stage 8: WHOLE-BODY tea_table -> table_far + tuck -> pregrasp
    # ==================================================================
    print("\n-- stage 8: wb nav to table + unfold arm --")
    pregrasp2 = make_full(BASE_TABLE_FAR, PICK2_PREGRASP)
    path, ms = plan_whole_body(current, pregrasp2, cloud,
                               "wb: tea->table + tuck->pregrasp", nav_planner, nav_time)
    wb_times.append(ms)
    segments.append(Segment(path=path, attach_body_id=None, attach_local_tf=None,
                            banner="stage 8: wb nav + unfold arm (base+arm)"))
    current = path[-1].copy()

    # ==================================================================
    # Stage 9: Pick apple again (arm-only linear)
    # ==================================================================
    print("\n-- stage 9: pick apple again --")
    grasp2 = make_full(BASE_TABLE_FAR, PICK2_GRASP)
    approach2 = _linear_path(current, grasp2, steps=30)
    segments.append(Segment(path=approach2, attach_body_id=None, attach_local_tf=None,
                            banner="stage 9: grasp approach (arm only)"))

    env.set_configuration(grasp2)
    client.resetBasePositionAndOrientation(apple_id, APPLE_PLACE_SPOT.tolist(), [0, 0, 0, 1])
    apple_tf2 = capture_local_transform(env, gripper_link, apple_id)

    lift2 = _linear_path(grasp2, pregrasp2, steps=30)
    segments.append(Segment(path=lift2, attach_body_id=apple_id,
                            attach_local_tf=apple_tf2, banner="stage 9: lift apple"))
    current = pregrasp2.copy()

    # ==================================================================
    # Stage 10: WHOLE-BODY carry apple far -> table + arm -> pre-place
    # ==================================================================
    print("\n-- stage 10: wb carry apple back --")
    preplace2 = make_full(BASE_TABLE, PLACE2_PREGRASP)
    path, ms = plan_whole_body(current, preplace2, cloud,
                               "wb: far->table + carry->preplace", nav_planner, nav_time)
    wb_times.append(ms)
    segments.append(Segment(path=path, attach_body_id=apple_id, attach_local_tf=apple_tf2,
                            banner="stage 10: wb carry apple back (base+arm)"))
    current = path[-1].copy()

    # ==================================================================
    # Stage 11: Place apple back (arm-only linear)
    # ==================================================================
    print("\n-- stage 11: place apple back --")
    place2 = make_full(BASE_TABLE, PLACE2_GRASP)
    lower2 = _linear_path(current, place2, steps=30)
    segments.append(Segment(path=lower2, attach_body_id=apple_id, attach_local_tf=apple_tf2,
                            banner="stage 11: lower apple"))
    retreat2 = _linear_path(place2, preplace2, steps=30)
    segments.append(Segment(path=retreat2, attach_body_id=None, attach_local_tf=None,
                            banner="stage 11: retreat"))
    current = preplace2.copy()

    # ==================================================================
    # Stage 12: WHOLE-BODY return home + tuck arm
    # ==================================================================
    print("\n-- stage 12: wb nav home + tuck arm --")
    goal_home = make_full(BASE_START, TUCK_ARM)
    path, ms = plan_whole_body(current, goal_home, cloud,
                               "wb: table->start + arm->tuck", nav_planner, nav_time)
    wb_times.append(ms)
    segments.append(Segment(path=path, attach_body_id=None, attach_local_tf=None,
                            banner="stage 12: wb nav home + tuck (base+arm)"))

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