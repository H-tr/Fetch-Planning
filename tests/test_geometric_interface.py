"""The geometric planning interface shared with autolife_planning:
joint limits, bounds, point clouds, plan results and sampling."""

import numpy as np
import pytest

from fetch_planning.fetch import HOME_JOINTS, fetch_robot_config
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig, PlanningResult, PlanningStatus

NAMES = fetch_robot_config.joint_names


def test_joint_limits_default_to_the_robot_model():
    planner = create_planner("fetch_whole_body")
    limits = planner.joint_limits
    assert list(limits) == NAMES
    assert limits["base_x_joint"] == pytest.approx((-10.0, 10.0))
    lower, upper = planner.bounds
    np.testing.assert_allclose(lower, [limits[n][0] for n in NAMES])
    np.testing.assert_allclose(upper, [limits[n][1] for n in NAMES])


def test_set_joint_limits_by_name_and_restore():
    planner = create_planner("fetch_whole_body")
    default = planner.joint_limits
    planner.set_joint_limits(
        {"base_x_joint": (-4.0, 4.0), "torso_lift_joint": (0.1, 0.3)}
    )
    limits = planner.joint_limits
    assert limits["base_x_joint"] == (-4.0, 4.0)
    assert limits["torso_lift_joint"] == (0.1, 0.3)
    assert limits["base_y_joint"] == default["base_y_joint"]
    lower, upper = planner.bounds
    assert (lower[0], upper[0]) == (-4.0, 4.0)
    planner.set_joint_limits(None)
    assert planner.joint_limits == default


@pytest.mark.parametrize(
    "limits", [{"no_such_joint": (0.0, 1.0)}, {"base_x_joint": (1.0, -1.0)}]
)
def test_bad_joint_limits_raise_and_change_nothing(limits):
    planner = create_planner("fetch_whole_body")
    planner.set_joint_limits({"base_y_joint": (-2.0, 2.0)})
    before = planner.joint_limits
    with pytest.raises(ValueError):
        planner.set_joint_limits(limits)
    assert planner.joint_limits == before


def test_joint_limits_survive_set_subgroup():
    planner = create_planner("fetch_whole_body")
    planner.set_joint_limits({"shoulder_pan_joint": (-0.5, 0.5)})
    planner.set_subgroup("fetch_arm")
    lower, upper = planner.bounds
    pan = planner.joint_names.index("shoulder_pan_joint")
    assert (lower[pan], upper[pan]) == (-0.5, 0.5)


@pytest.mark.parametrize(
    "limits",
    [
        {"base_x_joint": 1.0},
        {"base_x_joint": (0.0, float("inf"))},
        {"base_x_joint": (2.0, 2.0)},
    ],
)
def test_planner_config_validates_joint_limits(limits):
    with pytest.raises(ValueError):
        PlannerConfig(joint_limits=limits)


def test_planner_config_joint_limits_apply_at_construction():
    config = PlannerConfig(joint_limits={"base_y_joint": (-1.0, 3.0)})
    planner = create_planner("fetch_base", config=config)
    assert planner.joint_limits["base_y_joint"] == (-1.0, 3.0)


def test_base_plan_stays_inside_the_base_limits():
    config = PlannerConfig(
        time_limit=1.0,
        joint_limits={"base_x_joint": (-0.5, 2.0), "base_y_joint": (-0.5, 1.0)},
    )
    planner = create_planner("fetch_base", config=config)
    start = planner.extract_config(HOME_JOINTS)
    goal = np.array([1.5, 0.5, np.pi / 2])
    result = planner.plan(start, goal)
    assert result.success
    assert np.all(result.path[:, 0] >= -0.5) and np.all(result.path[:, 0] <= 2.0)
    assert np.all(result.path[:, 1] >= -0.5) and np.all(result.path[:, 1] <= 1.0)


def test_add_pointcloud_replaces_the_cloud():
    planner = create_planner("fetch_arm")
    pose = planner.extract_config(HOME_JOINTS)
    assert planner.validate(pose) and not planner.has_pointcloud
    blocking = np.random.default_rng(0).uniform(
        [-0.3, -0.3, 0.0], [0.3, 0.3, 1.2], (2000, 3)
    )
    planner.add_pointcloud(blocking)
    assert planner.has_pointcloud and not planner.validate(pose)
    planner.add_pointcloud(blocking + [5.0, 5.0, 0.0])
    assert planner.validate(pose)
    assert planner.remove_pointcloud()
    assert not planner.has_pointcloud and not planner.remove_pointcloud()


def test_plan_result_fields():
    planner = create_planner("fetch_arm", config=PlannerConfig(time_limit=1.0))
    start = planner.extract_config(HOME_JOINTS)
    result = planner.plan(start, planner.sample_valid())
    assert isinstance(result, PlanningResult) and result.success
    assert result.status == PlanningStatus.SUCCESS
    assert result.path.shape[1] == planner.num_dof
    assert result.planning_time_ns > 0 and np.isfinite(result.path_cost)
    assert not hasattr(result, "iterations")


def test_sample_valid_is_valid_and_in_bounds():
    planner = create_planner("fetch_arm")
    q = planner.sample_valid()
    lower, upper = planner.bounds
    assert planner.validate(q) and np.all(q >= lower) and np.all(q <= upper)


def test_sample_valid_raises_when_nothing_is_valid():
    planner = create_planner("fetch_arm")
    planner._planner.add_sphere([0.0, 0.0, 0.7], 0.25)  # inside the torso
    with pytest.raises(RuntimeError):
        planner.sample_valid(max_samples=512)
