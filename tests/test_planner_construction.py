import pytest

from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig

ARM_SUBGROUPS = ["fetch_arm", "fetch_arm_with_torso"]
BASE_SUBGROUPS = ["fetch_base", "fetch_base_arm", "fetch_whole_body"]


@pytest.mark.parametrize("subgroup", ARM_SUBGROUPS + BASE_SUBGROUPS)
def test_planner_constructs(subgroup):
    planner = create_planner(
        subgroup,
        config=PlannerConfig(planner_name="rrtc", time_limit=0.1),
    )
    assert planner is not None


@pytest.mark.parametrize("subgroup", BASE_SUBGROUPS)
def test_nonholonomic_subgroups_use_compound_space(subgroup):
    planner = create_planner(
        subgroup,
        config=PlannerConfig(planner_name="rrtc", time_limit=0.1),
    )
    assert planner is not None
