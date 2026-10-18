"""Geometric planning: what the collision checks and the solver statuses
guarantee."""

import os

import numpy as np
import pytest
from conftest import requires_pinocchio

import fetch_planning
from fetch_planning._ompl_vamp import OmplVampPlanner
from fetch_planning.fetch import BASE_TURNING_RADIUS, HOME_JOINTS
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig
from fetch_planning.types.planning import BASE_PLANNERS, GEOMETRIC_PLANNERS


def grid(lo, hi):
    return np.arange(lo, hi + 1e-9, 0.02)


def table_and_shelf():
    """A table top and a shelf board in front of the robot, 2 cm apart."""
    x, y = np.meshgrid(grid(0.5, 1.3), grid(-0.6, 0.6))
    top = np.stack([x.ravel(), y.ravel(), np.full(x.size, 0.72)], 1)
    x, y = np.meshgrid(grid(0.5, 1.0), grid(-0.6, 0.6))
    shelf = np.stack([x.ravel(), y.ravel(), np.full(x.size, 1.2)], 1)
    return np.concatenate([top, shelf]).astype(np.float32)


@pytest.mark.parametrize("allow_reverse", [False, True])
def test_base_motions_follow_the_curve(allow_reverse):
    """Turning around on the spot is collision-free as a straight
    joint-space line, but the Dubins / Reeds-Shepp curve the base drives
    hits the obstacle, so simplification keeps the detour."""
    planner = OmplVampPlanner(
        [0, 1, 2], HOME_JOINTS.tolist(), 3, BASE_TURNING_RADIUS, allow_reverse
    )
    planner.add_sphere([0.2, 0.4, 0.2], 0.05)
    a, detour, b = [0.0, 0.0, 0.0], [-1.0, -1.0, np.pi], [0.0, 0.0, np.pi]
    assert all(planner.validate_batch(np.linspace(a, b, 1000).tolist()))
    assert not all(planner.validate_batch(planner.interpolate_path([a, b], 1000, 0.0)))
    assert len(planner.simplify_path([a, detour, b], 1.0)) > 2


@pytest.mark.parametrize(
    "planner_name", ["pdst", "kpiece", "bkpiece", "lbkpiece", "stride"]
)
@pytest.mark.parametrize("subgroup", ["fetch_arm_with_torso", "fetch_base"])
def test_partial_motions_keep_valid_states(subgroup, planner_name):
    """These planners grow from the last valid state of a blocked motion,
    found along the base curve with the base."""
    for seed in range(3):
        np.random.seed(seed)
        planner = create_planner(
            subgroup,
            config=PlannerConfig(
                planner_name=planner_name,
                time_limit=2.0,
                simplify=False,
                interpolate=False,
            ),
            pointcloud=table_and_shelf(),
        )
        if planner.has_base:
            planner.set_base_bounds(-1.0, 2.0, -1.5, 1.5)
        result = planner.plan(
            planner.extract_config(HOME_JOINTS), planner.sample_valid()
        )
        assert result.success
        assert planner.validate_batch(result.path).all()


def test_rrtc_runs_rrt_connect_with_the_base(capfd):
    planner = create_planner(
        "fetch_whole_body", config=PlannerConfig(planner_name="rrtc", time_limit=1.0)
    )
    goal = HOME_JOINTS.copy()
    goal[:3] = [1.0, 1.0, np.pi / 2]
    assert planner.plan(HOME_JOINTS, goal).success
    out = capfd.readouterr().out
    assert "RRTConnect" in out and "QRRT" not in out


# OMPL's BLIT* throws "Asked to remove vertex from reverse children that is
# currently not a child" on short motions (and on about 1 in 20 random ones).
BLITSTAR_BUG = pytest.mark.xfail(raises=RuntimeError, reason="OMPL BLIT* bug")


@pytest.mark.parametrize(
    "planner_name",
    [
        pytest.param(name, marks=BLITSTAR_BUG) if name == "blitstar" else name
        for name in GEOMETRIC_PLANNERS
    ],
)
def test_every_planner_runs(planner_name):
    """Every accepted name builds and runs its planner."""
    planner = create_planner(
        "fetch_arm", config=PlannerConfig(planner_name=planner_name, time_limit=0.1)
    )
    start = planner.extract_config(HOME_JOINTS)
    goal = start.copy()
    goal[-1] += 0.5
    planner.plan(start, goal)


@pytest.mark.parametrize("planner_name", BASE_PLANNERS + ("kpiece",))
def test_planners_run_with_the_base(planner_name):
    planner = create_planner(
        "fetch_base_arm",
        config=PlannerConfig(
            planner_name=planner_name, time_limit=2.0, interpolate=False
        ),
    )
    start = planner.extract_config(HOME_JOINTS)
    goal = start.copy()
    goal[:3] = [1.0, 1.0, np.pi / 2]
    result = planner.plan(start, goal)
    assert result.success
    assert planner.validate_batch(result.path).all()


def test_multilevel_planners_need_the_base():
    planner = create_planner("fetch_arm", config=PlannerConfig(planner_name="qrrt"))
    start = planner.extract_config(HOME_JOINTS)
    with pytest.raises(ValueError):
        planner.plan(start, start)


@pytest.mark.parametrize("planner_name", ["rrt", "rrtc"])
def test_approximate_solutions_are_not_solved(planner_name):
    """With only the shoulder pan active, a pillar in the swing of the
    straight arm cuts the goal off: OMPL's approximate solution stops at the
    pillar."""
    frozen = HOME_JOINTS.copy()
    frozen[5:] = 0.0
    planner = OmplVampPlanner([4], frozen.tolist(), 0, BASE_TURNING_RADIUS, False)
    planner.add_sphere([0.8, 0.0, 1.1], 0.1)
    assert planner.validate([-1.2]) and planner.validate([1.2])
    result = planner.plan([-1.2], [1.2], planner_name, 0.5, False, False, 0, 0.0)
    assert not result.solved


def head_height_beam():
    """A beam across the whole workspace at x = 1, above the base."""
    y, z = np.meshgrid(grid(-3.0, 3.0), grid(1.0, 1.6))
    return np.stack([np.ones(y.size), y.ravel(), z.ravel()], 1).astype(np.float32)


@pytest.mark.parametrize("planner_name", ["qrrt", "rrtc"])
def test_base_subgroup_checks_the_whole_body(planner_name):
    """The base fits under the beam, the frozen torso and head do not:
    there is no path."""
    planner = create_planner(
        "fetch_base",
        config=PlannerConfig(planner_name=planner_name, time_limit=0.5),
        pointcloud=head_height_beam(),
    )
    planner.set_base_bounds(-1.0, 4.0, -2.0, 2.0)
    assert not planner.validate(np.array([1.0, 0.0, 0.0]))
    assert not planner.plan(np.zeros(3), np.array([2.0, 0.0, 0.0])).success


def test_subgroup_inputs_are_checked():
    home = HOME_JOINTS.tolist()
    with pytest.raises(ValueError):
        OmplVampPlanner([4, 5], home[:10])
    with pytest.raises(ValueError):
        OmplVampPlanner([4, 11], home)
    with pytest.raises(ValueError):
        OmplVampPlanner([0, 1], home, 3)
    planner = OmplVampPlanner([4, 5], home)
    with pytest.raises(ValueError):
        planner.set_subgroup([4, 5], home + [0.0], 0)


@requires_pinocchio
def test_bounds_are_the_joint_limits():
    """The bounds come from the compiled robot model: the URDF limits."""
    import pinocchio as pin

    urdf = os.path.join(
        os.path.dirname(fetch_planning.__file__),
        "resources/robot/fetch/fetch_whole_body.urdf",
    )
    model = pin.buildModelFromUrdf(urdf)
    planner = OmplVampPlanner(list(range(3, 11)), HOME_JOINTS.tolist())
    np.testing.assert_allclose(
        planner.lower_bounds(), model.lowerPositionLimit[3:], atol=1e-6
    )
    np.testing.assert_allclose(
        planner.upper_bounds(), model.upperPositionLimit[3:], atol=1e-6
    )
