import numpy as np
import pytest

from fetch_planning.fetch import HOME_JOINTS, JOINT_GROUPS, PLANNING_SUBGROUPS
from fetch_planning.kinematics import create_ik_solver
from fetch_planning.planning import available_robots

from conftest import requires_ikfast, requires_pinocchio

IK_CHAINS = ["arm", "arm_with_torso", "whole_body"]


def test_home_joints_has_eleven_dof():
    assert HOME_JOINTS.shape == (11,)


def test_joint_groups_cover_all_dof():
    idx = set()
    for group_slice in JOINT_GROUPS.values():
        idx.update(range(*group_slice.indices(HOME_JOINTS.shape[0])))
    assert idx == set(range(HOME_JOINTS.shape[0]))


def test_planning_subgroups_exposed():
    expected = {
        "fetch_base",
        "fetch_arm",
        "fetch_arm_with_torso",
        "fetch_base_arm",
        "fetch_whole_body",
    }
    assert expected.issubset(set(PLANNING_SUBGROUPS.keys()))
    assert expected.issubset(set(available_robots()))


@pytest.mark.parametrize("chain", IK_CHAINS)
def test_trac_ik_constructs(chain):
    solver = create_ik_solver(chain, backend="trac_ik")
    assert solver is not None


@requires_pinocchio
@pytest.mark.parametrize("chain", IK_CHAINS)
def test_pink_constructs(chain):
    solver = create_ik_solver(chain, backend="pink")
    assert solver is not None


@requires_ikfast
@pytest.mark.parametrize("chain", ["arm", "arm_with_torso"])
def test_ikfast_fk_roundtrip(chain):
    solver = create_ik_solver(chain, backend="ikfast")
    if chain == "arm_with_torso":
        seed = HOME_JOINTS[JOINT_GROUPS["torso"].start : JOINT_GROUPS["arm"].stop]
    else:
        seed = HOME_JOINTS[JOINT_GROUPS["arm"]]
    target = solver.fk(seed)
    result = solver.solve(target, seed=seed)
    assert result.joint_positions is not None
    np.testing.assert_allclose(
        solver.fk(result.joint_positions).position, target.position, atol=1e-3
    )


@requires_ikfast
def test_ikfast_rejects_whole_body():
    with pytest.raises(ValueError):
        create_ik_solver("whole_body", backend="ikfast")
