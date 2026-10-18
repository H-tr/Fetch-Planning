"""Geometric planning: what the collision checks and the solver statuses
guarantee."""

import numpy as np
import pytest

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
