import importlib

import pytest


def _have(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _ikfast_solver_available() -> bool:
    """IKFastSolver imports ``ikfast_fetch`` as a top-level module.

    In wheel installs that module is top-level; in editable installs it
    lives under ``fetch_planning.ikfast_fetch`` and the solver's
    ``import_module('ikfast_fetch')`` call fails.  Test by actually
    trying to construct the solver.
    """
    try:
        from fetch_planning.kinematics import create_ik_solver

        create_ik_solver("arm_with_torso", backend="ikfast")
    except Exception:
        return False
    return True


HAS_IKFAST = _ikfast_solver_available()
HAS_PINOCCHIO = _have("pinocchio")

requires_ikfast = pytest.mark.skipif(
    not HAS_IKFAST, reason="ikfast_fetch backend unavailable in this install"
)
requires_pinocchio = pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="pinocchio not installed"
)
