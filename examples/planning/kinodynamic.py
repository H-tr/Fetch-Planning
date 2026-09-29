"""Kinodynamic (FLASK) planning on the whole-body demo scene.

Runs the scene, poses and IK solutions of ``nonholonomic.py`` through
:meth:`MotionPlanner.plan_kinodynamic` and answers three questions with
numbers from *our* robot and scene rather than from the paper:

* **benchmark** — planning time / success of FLASK versus the existing
  planners, per whole-body leg (vs. multilevel QRRT) and for arm-only
  motions (vs. RRT-Connect + TOPP-RA, including how often the timed
  trajectory actually collides once its corners are rounded).
* **track** — does feed-forward + PID suffice?  Every FLASK leg is
  executed in closed loop on a simulated robot: velocity-controlled
  joints with actuator lag, and a diff-drive base with actuator lag and
  wheel slip under a Kanayama tracking law (feed-forward twist + pose
  feedback).  Reports tracking errors and collision-checks the motion
  the robot actually executed.
* **--visualize** — play the FLASK trajectories in PyBullet in real time.

FLASK trajectories are collision-free with zero clearance, so tracking
errors can graze obstacles; ``--clearance`` plans against obstacles
inflated by that many metres while execution is still checked against
the real scene.

Usage::

    pixi run python examples/planning/kinodynamic.py
    pixi run python examples/planning/kinodynamic.py --mode=track
    pixi run python examples/planning/kinodynamic.py --mode=track --clearance=0.05
    pixi run python examples/planning/kinodynamic.py --mode=track --visualize
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import nonholonomic as demo  # sibling example: scene, base poses, IK solutions
import numpy as np
from fire import Fire

from fetch_planning.fetch import (
    BASE_MAX_ACCELERATION,
    BASE_MAX_SPEED,
    BASE_MAX_YAW_RATE,
    JOINT_ACCELERATION_LIMITS,
    JOINT_VELOCITY_LIMITS,
    fetch_robot_config,
)
from fetch_planning.planning import create_planner
from fetch_planning.trajectory import TimeOptimalParameterizer
from fetch_planning.types import KinodynamicConfig, PlannerConfig

mf = demo.make_full

# Whole-body legs of the demo: base and arm move together.
WB_LEGS = [
    (
        "start->table",
        mf(demo.BASE_START, demo.TUCK_ARM),
        mf(demo.BASE_TABLE, demo.PICK1_PREGRASP),
    ),
    (
        "carry 1",
        mf(demo.BASE_TABLE, demo.PICK1_PREGRASP),
        mf(demo.BASE_TABLE_FAR, demo.PLACE1_PREGRASP),
    ),
    (
        "far->mid",
        mf(demo.BASE_TABLE_FAR, demo.PLACE1_PREGRASP),
        mf(demo.BASE_MID, demo.TUCK_ARM),
    ),
    ("mid->sofa", mf(demo.BASE_MID, demo.TUCK_ARM), mf(demo.BASE_SOFA, demo.TUCK_ARM)),
    ("sofa->tea", mf(demo.BASE_SOFA, demo.TUCK_ARM), mf(demo.BASE_TEA, demo.TUCK_ARM)),
    (
        "tea->table",
        mf(demo.BASE_TEA, demo.TUCK_ARM),
        mf(demo.BASE_TABLE_FAR, demo.PICK2_PREGRASP),
    ),
    (
        "carry 2",
        mf(demo.BASE_TABLE_FAR, demo.PICK2_PREGRASP),
        mf(demo.BASE_TABLE, demo.PLACE2_PREGRASP),
    ),
    (
        "table->home",
        mf(demo.BASE_TABLE, demo.PLACE2_PREGRASP),
        mf(demo.BASE_START, demo.TUCK_ARM),
    ),
]

# Arm-only motions (torso + 7 arm joints) with the base parked at the table.
ARM_BASE = demo.BASE_TABLE
ARM_LEGS = [
    ("tuck->pregrasp", demo.TUCK_ARM, demo.PICK1_PREGRASP),
    ("pregrasp->tuck", demo.PICK1_PREGRASP, demo.TUCK_ARM),
    ("pregrasp->grasp", demo.PICK1_PREGRASP, demo.PICK1_GRASP),
]


def summarize(ms: list[float]) -> str:
    t = np.asarray(ms)
    return f"median {np.median(t):7.2f}  p95 {np.percentile(t, 95):7.2f}  max {t.max():7.2f} ms"


# ── Planners ──────────────────────────────────────────────────────────


def make_planners(cloud: np.ndarray, time_limit: float, point_radius: float = 0.01):
    wb = create_planner(
        "fetch_whole_body",
        config=PlannerConfig(
            planner_name="rrtc", time_limit=time_limit, point_radius=point_radius
        ),
        pointcloud=cloud,
    )
    wb.set_base_bounds(**demo.BASE_BOUNDS)
    arm = create_planner(
        "fetch_arm_with_torso",
        config=PlannerConfig(
            planner_name="rrtc", time_limit=time_limit, interpolate=False
        ),
        pointcloud=cloud,
        base_config=mf(ARM_BASE, demo.PICK1_PREGRASP),
    )
    return wb, arm


def arm_toppra(planner) -> TimeOptimalParameterizer:
    names = planner.joint_names
    return TimeOptimalParameterizer(
        np.array([JOINT_VELOCITY_LIMITS[j] for j in names]),
        np.array([JOINT_ACCELERATION_LIMITS[j] for j in names]),
    )


# ── Benchmark ─────────────────────────────────────────────────────────


def benchmark(wb, arm, trials: int, time_limit: float) -> None:
    kcfg = KinodynamicConfig(time_limit=time_limit)

    print("\n== Whole body (11 DOF, nonholonomic base) ==")
    print(
        "  FLASK: time-parameterised, limits enforced.  Multilevel QRRT: geometric path only."
    )
    for name, s, g in WB_LEGS:
        k_ms, k_ok, k_dur, k_coll = [], 0, [], 0
        g_ms, g_ok = [], 0
        for _ in range(trials):
            r = wb.plan_kinodynamic(s, g, config=kcfg)
            k_ms.append(r.planning_time_ns / 1e6)
            if r.success:
                k_ok += 1
                k_dur.append(r.trajectory.duration)
                _, q, _, _ = r.trajectory.sample_uniform(0.005)
                k_coll += int(not wb.validate_batch(q).all())
            p = wb.plan(s, g)
            g_ms.append(p.planning_time_ns / 1e6)
            g_ok += int(p.success)
        print(
            f"  {name:12s} FLASK {k_ok:3d}/{trials} {summarize(k_ms)} | duration {np.median(k_dur):5.1f} s"
            f" | colliding {k_coll}"
        )
        print(f"  {'':12s} QRRT  {g_ok:3d}/{trials} {summarize(g_ms)}")

    print("\n== Arm only (torso + 7 joints, base parked at the table) ==")
    print(
        "  RRTC+TOPP-RA: geometric path, then time-optimal parameterisation (spline corners)."
    )
    toppra = arm_toppra(arm)
    for name, s, g in ARM_LEGS:
        k_ms, k_ok, k_dur, k_coll = [], 0, [], 0
        g_ms, g_dur, g_coll = [], [], 0
        for _ in range(trials):
            r = arm.plan_kinodynamic(s, g, config=kcfg)
            k_ms.append(r.planning_time_ns / 1e6)
            if r.success:
                k_ok += 1
                k_dur.append(r.trajectory.duration)
                _, q, _, _ = r.trajectory.sample_uniform(0.005)
                k_coll += int(not arm.validate_batch(q).all())
            t0 = time.perf_counter()
            p = arm.plan(s, g)
            if p.success:
                traj = toppra.parameterize(p.path)
                g_ms.append((time.perf_counter() - t0) * 1e3)
                g_dur.append(traj.duration)
                _, q, _, _ = traj.sample_uniform(0.005)
                g_coll += int(not arm.validate_batch(q).all())
        print(
            f"  {name:16s} FLASK        {k_ok:3d}/{trials} {summarize(k_ms)} | duration {np.median(k_dur):5.2f} s"
            f" | colliding {k_coll}/{k_ok}"
        )
        print(
            f"  {'':16s} RRTC+TOPP-RA {len(g_ms):3d}/{trials} {summarize(g_ms)} | duration {np.median(g_dur):5.2f} s"
            f" | colliding {g_coll}/{len(g_ms)}"
        )


# ── Closed-loop tracking (feed-forward + PID) ─────────────────────────


@dataclass
class Plant:
    """Simulated actuators and controller gains."""

    dt: float = 0.002
    joint_lag: float = 0.05  # s, first-order joint velocity response
    base_lag: float = 0.10  # s, first-order wheel velocity response
    wheel_slip: float = 0.05  # base covers 5% less distance than commanded
    joint_kp: float = 5.0  # 1/s
    joint_ki: float = 1.0  # 1/s^2
    base_kx: float = 1.5  # 1/s   along-track
    base_ky: float = 6.0  # 1/m^2 cross-track (Kanayama)
    base_kth: float = 3.0  # 1/s   heading


def track(traj, planner, plant: Plant, rng: np.random.Generator):
    """Execute ``traj`` in closed loop; return executed configs and errors."""
    t, q_ref, qd_ref, _ = traj.sample_uniform(plant.dt)
    b = 3 if traj.has_base else 0
    names = [j for j in planner.joint_names if not j.startswith("base_")]
    vmax = np.array([JOINT_VELOCITY_LIMITS[j] for j in names])
    twist_ref = traj.sample_base_twist(t) if b else None

    q = q_ref[0].copy()
    qd = np.zeros(len(q) - b)
    integ = np.zeros(len(q) - b)
    v = w = 0.0
    executed = np.empty_like(q_ref)
    for k in range(len(t)):
        executed[k] = q
        # Joints: feed-forward velocity + PI on position -> velocity command.
        e = q_ref[k, b:] - q[b:]
        integ += e * plant.dt
        u = np.clip(
            qd_ref[k, b:] + plant.joint_kp * e + plant.joint_ki * integ, -vmax, vmax
        )
        qd += (u - qd) * plant.dt / plant.joint_lag
        q[b:] += qd * plant.dt
        if not b:
            continue
        # Base: Kanayama law — feed-forward (v, w) + body-frame pose feedback.
        x, y, th = q[:3]
        dx, dy = q_ref[k, 0] - x, q_ref[k, 1] - y
        ex = np.cos(th) * dx + np.sin(th) * dy
        ey = -np.sin(th) * dx + np.cos(th) * dy
        eth = np.angle(np.exp(1j * (q_ref[k, 2] - th)))
        vr, wr = twist_ref[k]
        vc = np.clip(
            vr * np.cos(eth) + plant.base_kx * ex, -BASE_MAX_SPEED, BASE_MAX_SPEED
        )
        wc = np.clip(
            wr + vr * plant.base_ky * ey + plant.base_kth * np.sin(eth),
            -BASE_MAX_YAW_RATE,
            BASE_MAX_YAW_RATE,
        )
        dv = np.clip(
            (vc - v) * plant.dt / plant.base_lag,
            -BASE_MAX_ACCELERATION * plant.dt,
            BASE_MAX_ACCELERATION * plant.dt,
        )
        v += dv
        w += (wc - w) * plant.dt / plant.base_lag
        v_true = v * (1.0 - plant.wheel_slip) * (1.0 + 0.01 * rng.standard_normal())
        q[0] += v_true * np.cos(th) * plant.dt
        q[1] += v_true * np.sin(th) * plant.dt
        q[2] = np.angle(np.exp(1j * (th + w * plant.dt)))

    err = executed - q_ref
    out = {"joint": np.abs(err[:, b:]).max()}
    if b:
        out["base_pos"] = np.hypot(err[:, 0], err[:, 1]).max()
        out["base_yaw"] = np.abs(np.angle(np.exp(1j * err[:, 2]))).max()
    return t, executed, out


def run_tracking(
    wb, check, trials: int, time_limit: float, visualize: bool, env=None
) -> None:
    """Plan with ``wb``; collision-check executed motion with ``check``."""
    plant = Plant()
    rng = np.random.default_rng(0)
    print(
        "\n== Closed-loop tracking: feed-forward + PID (joints), Kanayama FF+feedback (base) =="
    )
    print(
        f"  plant: joint lag {plant.joint_lag * 1e3:.0f} ms, base lag {plant.base_lag * 1e3:.0f} ms, "
        f"wheel slip {plant.wheel_slip:.0%}"
    )
    kcfg = KinodynamicConfig(time_limit=time_limit)
    for name, s, g in WB_LEGS:
        if not (wb.validate(s) and wb.validate(g)):
            print(
                f"  {name:12s} start or goal lies inside the clearance margin — skipped"
            )
            continue
        errs, colls, ms = [], 0, []
        for i in range(trials):
            r = wb.plan_kinodynamic(s, g, config=kcfg)
            ms.append(r.planning_time_ns / 1e6)
            if not r.success:
                continue
            t, executed, e = track(r.trajectory, wb, plant, rng)
            errs.append(e)
            colls += int(not check.validate_batch(executed[::5]).all())
            if visualize and i == 0 and env is not None:
                play(env, r.trajectory)
        je = max(x["joint"] for x in errs)
        pe = max(x["base_pos"] for x in errs)
        ye = max(x["base_yaw"] for x in errs)
        print(
            f"  {name:12s} planned {len(errs)}/{trials} (median {np.median(ms):6.1f} ms) | "
            f"max err: joint {je:.3f} rad, base {pe * 100:4.1f} cm, yaw {np.degrees(ye):4.1f} deg | "
            f"executed motion colliding {colls}/{len(errs)}"
        )


def play(env, traj, speed: float = 1.0) -> None:
    t, q, _, _ = traj.sample_uniform(1.0 / 60.0)
    start = time.perf_counter()
    for k in range(len(t)):
        env.set_configuration(q[k])
        delay = t[k] / speed - (time.perf_counter() - start)
        if delay > 0:
            time.sleep(delay)


# ── Main ──────────────────────────────────────────────────────────────


def main(
    mode: str = "benchmark",
    trials: int = 20,
    time_limit: float = 1.0,
    visualize: bool = False,
    clearance: float = 0.0,
) -> None:
    cloud = demo.load_room_pointcloud()
    print(f"scene: {len(cloud):,} collision points")
    wb, arm = make_planners(cloud, time_limit)

    env = None
    if visualize:
        from fetch_planning.envs.pybullet_env import PyBulletEnv

        env = PyBulletEnv(fetch_robot_config, visualize=True)
        demo.load_room_meshes(env)
        env.sim.client.resetDebugVisualizerCamera(
            cameraDistance=4.5,
            cameraYaw=-90.0,
            cameraPitch=-45.0,
            cameraTargetPosition=[-0.5, 0.8, 0.5],
        )

    if mode == "benchmark":
        benchmark(wb, arm, trials, time_limit)
    elif mode == "track":
        planning = wb
        if clearance > 0.0:
            planning, _ = make_planners(
                cloud, time_limit, point_radius=0.01 + clearance
            )
            print(f"planning with {clearance * 100:.1f} cm obstacle inflation")
        run_tracking(planning, wb, trials, time_limit, visualize, env)
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'benchmark' or 'track'")


if __name__ == "__main__":
    Fire(main)
