import importlib

import pytest


def _have(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _ikfast_solver_available() -> bool:
    """The ``fetch_planning.ikfast_fetch`` extension is only built when
    CMake finds LAPACK, so it may be missing from a given install.  Test
    by actually trying to construct the solver.
    """
    try:
        from fetch_planning.kinematics import create_ik_solver

        create_ik_solver("arm_with_torso", backend="ikfast")
    except Exception:
        return False
    return True


HAS_IKFAST = _ikfast_solver_available()
HAS_PINOCCHIO = _have("pinocchio")
HAS_TRAC_IK = _have("pytracik")

requires_ikfast = pytest.mark.skipif(
    not HAS_IKFAST, reason="ikfast_fetch backend unavailable in this install"
)
requires_pinocchio = pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="pinocchio not installed"
)
requires_trac_ik = pytest.mark.skipif(
    not HAS_TRAC_IK,
    reason="pytracik not built (needs orocos-kdl + NLopt; absent from PyPI wheels)",
)
