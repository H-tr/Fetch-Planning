"""Minimal single-plan whole-body visualization.

One short-horizon plan: move the base a small distance while changing
heading, with the arm also transitioning from tuck to a reach pose.
Renders the path so you can visually judge whether:

  * the base follows Reeds-Shepp curves (forward along heading, no
    sideways sliding)
  * the arm interpolates smoothly alongside the base
  * the reverse penalty keeps the base from backing up unnecessarily

    pixi run python examples/planning/multilevel_single_step.py
    pixi run python examples/planning/multilevel_single_step.py --dx=0.5 --dtheta=1.57
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from fire import Fire

import fetch_planning
from fetch_planning.config.robot_config import HOME_JOINTS, fetch_robot_config
from fetch_planning.envs.pybullet_env import PyBulletEnv
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig


def load_cloud(stride: int = 2) -> np.ndarray:
    """Load the same RLS scene cloud the big demo uses."""
    pcd_dir = "assets/envs/rls_env/pcd"
    parts = [
        "rls_2", "open_kitchen", "wall", "workstation",
        "table", "sofa", "coffee_table",
    ]
    chunks = []
    for name in parts:
        p = Path(pcd_dir) / f"{name}.ply"
        pc = trimesh.load(str(p))
        v = np.asarray(pc.vertices, dtype=np.float32)
        chunks.append(v[::stride])
    return np.concatenate(chunks, axis=0)


def main(
    dx: float = 1.0,           # base move along +x (metres)
    dy: float = 0.5,           # base move along +y (metres)
    dtheta: float = np.pi / 2, # base heading change (radians)
    time_limit: float = 0.5,   # planner budget (seconds)
    arm_reach: bool = True,    # also change the arm pose
    load_scene: bool = True,   # include the RLS point cloud
    pcd_stride: int = 2,
) -> None:
    env = PyBulletEnv(fetch_robot_config, visualize=True)

    if load_scene:
        cloud = load_cloud(stride=pcd_stride)
        print(f"collision cloud: {len(cloud):,} points")
        env.add_pointcloud(cloud[::2], pointsize=2)
        pointcloud_arg = cloud
    else:
        pointcloud_arg = None

    env.sim.client.resetDebugVisualizerCamera(
        cameraDistance=3.0, cameraYaw=-90.0, cameraPitch=-30.0,
        cameraTargetPosition=[0.5, 0.0, 0.8],
    )

    planner = create_planner(
        "fetch_whole_body",
        config=PlannerConfig(planner_name="rrtc", time_limit=time_limit),
        pointcloud=pointcloud_arg,
    )
    planner.set_base_bounds(-4.0, 4.0, -4.0, 4.0)

    start = HOME_JOINTS.copy()
    goal = HOME_JOINTS.copy()
    goal[0] += dx
    goal[1] += dy
    goal[2] += dtheta

    if arm_reach:
        # Rough "reaching forward" pose for the arm (torso + 7-DOF arm).
        goal[3:] = np.array([0.30, 0.30, 0.60, 0.00, 1.20, 0.00, 1.20, 0.00])

    env.set_configuration(start)

    result = planner.plan(start, goal)
    n = len(result.path) if result.path is not None else 0
    print(
        f"status  : {result.status.value}\n"
        f"time    : {result.planning_time_ns / 1e6:.1f} ms\n"
        f"waypts  : {n}\n"
        f"cost    : {result.path_cost:.3f}\n"
        f"start   : base=({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})\n"
        f"goal    : base=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})"
    )

    if result.path is None:
        env.wait_for_close()
        return

    # Non-holonomic diagnostic: how much of the motion is along heading
    # vs sideways vs backward?
    path = result.path
    dxs = np.diff(path[:, 0])
    dys = np.diff(path[:, 1])
    th_mid = (path[:-1, 2] + path[1:, 2]) / 2
    fwd = dxs * np.cos(th_mid) + dys * np.sin(th_mid)
    side = -dxs * np.sin(th_mid) + dys * np.cos(th_mid)
    motion = np.sqrt(dxs**2 + dys**2)
    moving = motion > 1e-4
    if moving.any():
        total = motion[moving].sum()
        fwd_only = fwd[moving].copy()
        rev = -fwd_only[fwd_only < 0].sum()
        fwd_pos = fwd_only[fwd_only > 0].sum()
        print(
            f"forward/total : {fwd_pos / total:.3f}\n"
            f"reverse/total : {rev / total:.3f}\n"
            f"side/total    : {np.abs(side[moving]).sum() / total:.3f}\n"
            f"  (RS expects ~1, 0, 0; any large 'side' means holonomic slide)"
        )

    print("\nSPACE = play/pause   ←/→ = step   close window to exit\n")
    env.animate_path(path, fps=30)


if __name__ == "__main__":
    Fire(main)