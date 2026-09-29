"""Kinodynamic (FLASK) planning: closed-form math and end-to-end checks.

Every plan is verified independently of the planner's own validation:
the returned trajectory is densely resampled and checked for collisions
(``validate_batch``), joint / base limits, C^1 continuity, and — for the
mobile base — the nonholonomic constraint (zero lateral velocity).
"""

from pathlib import Path

import numpy as np
import pytest

from fetch_planning._ompl_vamp import flat_optimal_time
from fetch_planning.fetch import (
    BASE_MAX_ACCELERATION,
    BASE_MAX_SPEED,
    BASE_MAX_YAW_RATE,
    HOME_JOINTS,
    JOINT_ACCELERATION_LIMITS,
    JOINT_VELOCITY_LIMITS,
)
from fetch_planning.planning import create_planner
from fetch_planning.types import KinodynamicConfig, PlannerConfig, PlanningStatus

DT = 0.002
TOL = 1e-6


def lqmt_cost(y0, v0, y1, v1, w, rho, T):
    """J(T) of the Hermite cubic, by numerical quadrature."""
    y0, v0, y1, v1, w = map(np.asarray, (y0, v0, y1, v1, w))
    d = y1 - y0
    c2 = (3 * d - T * (2 * v0 + v1)) / T**2
    c3 = (-2 * d + T * (v0 + v1)) / T**3
    t = np.linspace(0.0, T, 20001)[:, None]
    acc = 2 * c2 + 6 * c3 * t
    return float(np.trapezoid((w * acc**2).sum(axis=1), t[:, 0]) + rho * T)


def arm_limits(planner):
    names = [j for j in planner.joint_names if not j.startswith("base_")]
    vel = np.array([JOINT_VELOCITY_LIMITS[j] for j in names])
    acc = np.array([JOINT_ACCELERATION_LIMITS[j] for j in names])
    return vel, acc


def assert_same_config(a, b, has_base, atol=1e-6):
    """Compare configurations, treating the base heading modulo 2 pi."""
    a, b = np.asarray(a, dtype=float).copy(), np.asarray(b, dtype=float).copy()
    if has_base:
        assert abs(np.angle(np.exp(1j * (a[2] - b[2])))) <= atol
        a[2] = b[2] = 0.0
    np.testing.assert_allclose(a, b, atol=atol)


def assert_executable(planner, result, start, goal, start_velocity=None):
    """Independent checks of a returned kinodynamic trajectory."""
    assert result.success, result.status
    traj = result.trajectory
    assert traj.duration > 0.0
    t, q, qd, qdd = traj.sample_uniform(DT)

    # Boundary conditions: exact start state, goal reached at rest.
    assert_same_config(q[0], start, traj.has_base)
    assert_same_config(q[-1], goal, traj.has_base)
    # The base yaw rate is acceleration-level for the flat outputs: it is
    # finite (not necessarily zero) at the instant the base starts or
    # stops, and a replanned trajectory matches the given x_dot / y_dot /
    # joint rates but may start with a different yaw rate — like at any
    # knot.  All flat rates must match exactly.
    flat = [i for i in range(q.shape[1]) if not (traj.has_base and i == 2)]
    np.testing.assert_allclose(qd[-1][flat], 0.0, atol=1e-6)
    if start_velocity is None:
        np.testing.assert_allclose(qd[0][flat], 0.0, atol=1e-6)
    else:
        np.testing.assert_allclose(
            qd[0][flat], np.asarray(start_velocity)[flat], atol=1e-6
        )

    # Collision-free along the executed trajectory (not just at nodes).
    assert planner.validate_batch(q).all()

    # C^1 in the flat outputs: positions and velocities continuous across
    # segment knots.  The base yaw rate depends on the flat acceleration,
    # which (order r = 2) jumps at knots, so only the heading itself is
    # required to be continuous there.
    eps = 1e-7
    for k in traj.knot_times[1:-1]:
        assert_same_config(
            traj.position(k - eps), traj.position(k + eps), traj.has_base, 1e-5
        )
        np.testing.assert_allclose(
            traj.velocity(k - eps)[flat], traj.velocity(k + eps)[flat], atol=1e-4
        )

    # Joint limits (non-base DOF).
    b = 3 if traj.has_base else 0
    vel, acc = arm_limits(planner)
    lo = np.array(planner._planner.lower_bounds())[b:]
    hi = np.array(planner._planner.upper_bounds())[b:]
    assert np.all(q[:, b:] >= lo - TOL) and np.all(q[:, b:] <= hi + TOL)
    assert np.all(np.abs(qd[:, b:]) <= vel * (1 + 1e-6) + TOL)
    assert np.all(np.abs(qdd[:, b:]) <= acc * (1 + 1e-6) + TOL)

    if traj.has_base:
        theta = q[:, 2]
        lateral = -np.sin(theta) * qd[:, 0] + np.cos(theta) * qd[:, 1]
        np.testing.assert_allclose(lateral, 0.0, atol=1e-9)
        speed = np.hypot(qd[:, 0], qd[:, 1])
        assert speed.max() <= BASE_MAX_SPEED * (1 + 1e-3)
        twist = traj.sample_base_twist(t)
        np.testing.assert_allclose(np.abs(twist[:, 0]), speed, atol=1e-9)
        # Sampled checks in the planner run on a 50 ms grid; allow a
        # small overshoot between grid points.
        assert np.abs(twist[:, 1]).max() <= BASE_MAX_YAW_RATE * 1.05
        v_signed = twist[:, 0]
        a_tan = np.gradient(v_signed, t)
        assert np.abs(a_tan[2:-2]).max() <= BASE_MAX_ACCELERATION * 1.05
        # Heading continuity (no teleporting base).
        dtheta = np.abs(np.angle(np.exp(1j * np.diff(theta))))
        assert dtheta.max() <= BASE_MAX_YAW_RATE * DT * 2 + 1e-3


# ── Closed-form LQMT ───────────────────────────────────────────────────


def test_optimal_time_rest_to_rest_closed_form():
    # r = 2, rest-to-rest, unit weight: T* = (36 d^2 / rho)^(1/4).
    for d, rho in [(1.0, 1.0), (0.3, 4.0), (2.5, 0.5)]:
        T, J = flat_optimal_time([0.0], [0.0], [d], [0.0], [1.0], rho)
        assert T == pytest.approx((36 * d**2 / rho) ** 0.25, rel=1e-9)
        assert J == pytest.approx(lqmt_cost([0], [0], [d], [0], [1], rho, T), rel=1e-6)


@pytest.mark.parametrize("seed", range(10))
def test_optimal_time_is_global_minimum(seed):
    rng = np.random.default_rng(seed)
    n = 5
    y0, y1 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    v0, v1 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    w = rng.uniform(0.2, 3.0, n)
    rho = float(rng.uniform(0.1, 10.0))
    T, J = flat_optimal_time(y0, v0, y1, v1, w, rho)
    grid = np.geomspace(1e-2, 1e2, 4000)
    d = y1 - y0
    a = np.sum(w * 12 * d**2)
    b = np.sum(w * 12 * (v0 + v1) * d)
    c = np.sum(w * 4 * (v0**2 + v0 * v1 + v1**2))
    costs = a / grid**3 - b / grid**2 + c / grid + rho * grid
    assert J <= costs.min() + 1e-9
    # Stationarity at T*.
    dJ = -3 * a / T**4 + 2 * b / T**3 - c / T**2 + rho
    assert abs(dJ) < 1e-6 * max(1.0, rho)
    assert J == pytest.approx(lqmt_cost(y0, v0, y1, v1, w, rho, T), rel=1e-5)


# ── Arm-only (paper formulation) ───────────────────────────────────────


@pytest.fixture
def arm_planner():
    planner = create_planner(
        "fetch_arm_with_torso", config=PlannerConfig(time_limit=1.0)
    )
    # A shelf of spheres in front of the robot, between tuck and reach.
    for x in (0.55, 0.65):
        for y in np.linspace(-0.4, 0.4, 5):
            planner._planner.add_sphere([x, y, 0.75], 0.06)
    return planner


ARM_TUCK = HOME_JOINTS[3:].copy()
ARM_REACH = np.array([0.30, 0.10, 0.20, 0.0, -1.2, 0.0, 1.2, 0.0])


def test_arm_plan_is_executable(arm_planner):
    assert arm_planner.validate(ARM_TUCK) and arm_planner.validate(ARM_REACH)
    cfg = KinodynamicConfig(time_limit=2.0, seed=1)
    result = arm_planner.plan_kinodynamic(ARM_TUCK, ARM_REACH, config=cfg)
    assert_executable(arm_planner, result, ARM_TUCK, ARM_REACH)


def test_arm_plan_is_deterministic_with_seed(arm_planner):
    cfg = KinodynamicConfig(time_limit=2.0, seed=3)
    a = arm_planner.plan_kinodynamic(ARM_TUCK, ARM_REACH, config=cfg)
    b = arm_planner.plan_kinodynamic(ARM_TUCK, ARM_REACH, config=cfg)
    assert a.success and b.success
    np.testing.assert_allclose(a.trajectory.knot_times, b.trajectory.knot_times)


def test_arm_replans_from_moving_state(arm_planner):
    cfg = KinodynamicConfig(time_limit=2.0, seed=5)
    first = arm_planner.plan_kinodynamic(ARM_TUCK, ARM_REACH, config=cfg)
    assert first.success
    t_mid = 0.4 * first.trajectory.duration
    q_mid = first.trajectory.position(t_mid)
    qd_mid = first.trajectory.velocity(t_mid)
    assert np.abs(qd_mid).max() > 1e-3, "replanning test needs a moving state"
    second = arm_planner.plan_kinodynamic(
        q_mid, ARM_TUCK, start_velocity=qd_mid, config=cfg
    )
    assert_executable(arm_planner, second, q_mid, ARM_TUCK, start_velocity=qd_mid)


def test_invalid_start_is_reported():
    planner = create_planner(
        "fetch_arm_with_torso", config=PlannerConfig(time_limit=1.0)
    )
    planner._planner.add_sphere([0.0, 0.0, 0.7], 0.25)  # inside the torso
    assert not planner.validate(ARM_TUCK)
    result = planner.plan_kinodynamic(ARM_TUCK, ARM_REACH)
    assert result.status == PlanningStatus.INVALID_START
    assert result.trajectory is None and not result.success


# ── Whole body with the nonholonomic base ──────────────────────────────


@pytest.fixture
def wb_planner():
    planner = create_planner("fetch_whole_body", config=PlannerConfig(time_limit=1.0))
    planner.set_base_bounds(-3.0, 3.0, -3.0, 3.0)
    # A pillar the base has to steer around.
    planner._planner.add_sphere([1.0, 0.0, 0.3], 0.35)
    return planner


def wb(x, y, theta, arm=None):
    q = HOME_JOINTS.copy()
    q[:3] = (x, y, theta)
    if arm is not None:
        q[3:] = arm
    return q


@pytest.mark.parametrize(
    "goal",
    [
        wb(2.0, 0.0, 0.0),  # straight ahead, pillar in the way
        wb(0.0, 1.5, np.pi),  # sideways + turn around
        wb(-1.0, -1.0, np.pi / 2),  # behind: needs a turn or reverse
        wb(0.0, 0.0, np.pi / 2, ARM_REACH),  # rotate in place + move arm
    ],
)
def test_whole_body_plan_is_executable(wb_planner, goal):
    start = wb(0.0, 0.0, 0.0)
    cfg = KinodynamicConfig(time_limit=3.0, seed=2)
    result = wb_planner.plan_kinodynamic(start, goal, config=cfg)
    assert_executable(wb_planner, result, start, goal)


def test_whole_body_forward_only(wb_planner):
    start, goal = wb(0.0, 0.0, 0.0), wb(-1.0, 0.0, 0.0)
    cfg = KinodynamicConfig(time_limit=3.0, seed=4, allow_reverse=False)
    result = wb_planner.plan_kinodynamic(start, goal, config=cfg)
    assert_executable(wb_planner, result, start, goal)
    assert all(g == 0 for g in result.trajectory.segment_gears)
    t = np.arange(0.0, result.trajectory.duration, DT)
    assert result.trajectory.sample_base_twist(t)[:, 0].min() >= -1e-9


def test_whole_body_replans_while_driving(wb_planner):
    start, goal = wb(0.0, 0.0, 0.0), wb(2.0, 0.0, 0.0)
    cfg = KinodynamicConfig(time_limit=3.0, seed=6)
    first = wb_planner.plan_kinodynamic(start, goal, config=cfg)
    assert first.success
    tr = first.trajectory
    # Pick a time where the base is actually driving.
    t = np.linspace(0.0, tr.duration, 200)
    speed = np.abs(tr.sample_base_twist(t)[:, 0])
    t_mid = float(t[np.argmax(speed)])
    q_mid, qd_mid = tr.position(t_mid), tr.velocity(t_mid)
    new_goal = wb(0.0, 1.5, np.pi / 2)
    second = wb_planner.plan_kinodynamic(
        q_mid, new_goal, start_velocity=qd_mid, config=cfg
    )
    assert_executable(wb_planner, second, q_mid, new_goal, start_velocity=qd_mid)


# ── The demo scene (examples/planning/nonholonomic.py) ─────────────────

REPO = Path(__file__).resolve().parents[1]
HAS_DEMO_SCENE = (REPO / "assets" / "envs" / "rls_env" / "pcd").is_dir()


@pytest.mark.skipif(
    not HAS_DEMO_SCENE, reason="demo scene assets missing (scripts/download_assets.sh)"
)
def test_demo_scene_legs(monkeypatch):
    """Every whole-body and arm leg of the demo, planned kinodynamically
    in the 151k-point scene, is executable."""
    pytest.importorskip("pybullet")  # the demo scripts import it at module level
    monkeypatch.chdir(REPO)
    monkeypatch.syspath_prepend(str(REPO / "examples" / "planning"))
    import kinodynamic as example
    import nonholonomic as demo

    wb, arm = example.make_planners(demo.load_room_pointcloud(), time_limit=1.0)
    for planner, legs in ((wb, example.WB_LEGS), (arm, example.ARM_LEGS)):
        for i, (_, start, goal) in enumerate(legs):
            cfg = KinodynamicConfig(time_limit=2.0, seed=i + 1)
            result = planner.plan_kinodynamic(start, goal, config=cfg)
            assert_executable(planner, result, start, goal)
