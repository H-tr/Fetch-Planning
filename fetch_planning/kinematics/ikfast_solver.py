"""IKFast-based analytic inverse kinematics for the Fetch arm.

The bundled `ikfast_fetch` C extension is an OpenRAVE-generated analytic
solver for the 8-DOF arm_with_torso chain (torso_lift + 7 arm joints). The
solver uses `Transform6D` and has **two free parameters** — the torso_lift
joint and the upperarm_roll joint — so each call to `get_ik` evaluates the
analytic closed-form IK for a specific (torso, upperarm_roll) pair and may
return multiple discrete elbow-up / elbow-down / wrist-flipped branches.

To behave like the TracIKSolver, IKFastSolver wraps a random-restart loop
that samples (torso, upperarm_roll) values, calls `get_ik`, filters the
returned branches against joint limits, and picks the closest solution to
the seed.
"""

from __future__ import annotations

import importlib

import numpy as np
from scipy.spatial.transform import Rotation

from fetch_planning.kinematics.ik_solver_base import IKSolverBase
from fetch_planning.types import ChainConfig, IKConfig, IKResult, IKStatus, SE3Pose

# Joint order returned by ikfast_fetch.get_ik:
#   [torso_lift, shoulder_pan, shoulder_lift, upperarm_roll,
#    elbow_flex, forearm_roll, wrist_flex, wrist_roll]
_IKFAST_JOINT_NAMES = [
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# Joint limits from the Fetch URDF (match JOINT_LIMITS in ikfast_api.py
# from grasp_anywhere). The ikfast solver does not enforce limits internally;
# we filter returned branches against these bounds.
_LOWER_BOUNDS = np.array(
    [0.0, -1.6056, -1.221, -np.pi, -2.251, -np.pi, -2.16, -np.pi],
    dtype=np.float64,
)
_UPPER_BOUNDS = np.array(
    [0.38615, 1.6056, 1.518, np.pi, 2.251, np.pi, 2.16, np.pi],
    dtype=np.float64,
)

# Default free-parameter sampling for the random-restart loop.
_TORSO_FREE_IDX = 0  # index of torso_lift in the 8-DOF output
_UPPERARM_FREE_IDX = 3  # index of upperarm_roll in the 8-DOF output


class IKFastSolver(IKSolverBase):
    """Analytic IK solver wrapping the ikfast_fetch C extension.

    Only the 8-DOF ``arm_with_torso`` and 7-DOF ``arm`` chains are supported.
    For the 7-DOF ``arm`` chain, torso_lift is treated as a fixed free
    parameter (taken from the seed) and only the 7 arm joints are returned.
    """

    def __init__(
        self,
        chain_config: ChainConfig,
        config: IKConfig | None = None,
    ) -> None:
        if config is None:
            config = IKConfig()

        self._ikfast = importlib.import_module("ikfast_fetch")
        self._chain_config = chain_config
        self._config = config

        if chain_config.num_joints == 8:
            self._include_torso = True
            self._lower = _LOWER_BOUNDS.copy()
            self._upper = _UPPER_BOUNDS.copy()
        elif chain_config.num_joints == 7:
            self._include_torso = False
            self._lower = _LOWER_BOUNDS[1:].copy()
            self._upper = _UPPER_BOUNDS[1:].copy()
        else:
            raise ValueError(
                f"ikfast_fetch supports 7-DOF (arm) or 8-DOF (arm_with_torso) chains, "
                f"got {chain_config.num_joints} DOF."
            )

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # IKSolverBase interface
    # ------------------------------------------------------------------

    @property
    def base_frame(self) -> str:
        return self._chain_config.base_link

    @property
    def ee_frame(self) -> str:
        return self._chain_config.ee_link

    @property
    def num_joints(self) -> int:
        return self._chain_config.num_joints

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

    def fk(self, joint_positions: np.ndarray) -> SE3Pose:
        q8 = self._to_full_8dof(joint_positions)
        pos, rot = self._ikfast.get_fk(q8.tolist())
        return SE3Pose(
            position=np.asarray(pos, dtype=np.float64),
            rotation=np.asarray(rot, dtype=np.float64),
        )

    def solve(
        self,
        target_pose: SE3Pose,
        seed: np.ndarray | None = None,
        config: IKConfig | None = None,
    ) -> IKResult:
        cfg = config or self._config
        pos = target_pose.position.astype(np.float64).tolist()
        rot = target_pose.rotation.astype(np.float64).tolist()

        seed8 = self._to_full_8dof(seed) if seed is not None else self._neutral_seed()

        # Random restart loop over free parameters (torso, upperarm_roll).
        # First attempt uses the seed's free-parameter values so the
        # deterministic case resolves quickly.
        best: np.ndarray | None = None
        best_err = np.inf
        pos_err = np.inf
        ori_err = np.inf
        iterations = 0

        for attempt in range(max(1, cfg.max_attempts)):
            iterations = attempt + 1
            if attempt == 0:
                torso = float(seed8[_TORSO_FREE_IDX])
                upperarm = float(seed8[_UPPERARM_FREE_IDX])
            else:
                # If the arm-only chain is in use, torso is pinned — don't
                # sample it.
                torso = (
                    float(seed8[_TORSO_FREE_IDX])
                    if not self._include_torso
                    else float(self._rng.uniform(_LOWER_BOUNDS[0], _UPPER_BOUNDS[0]))
                )
                upperarm = float(
                    self._rng.uniform(
                        _LOWER_BOUNDS[_UPPERARM_FREE_IDX],
                        _UPPER_BOUNDS[_UPPERARM_FREE_IDX],
                    )
                )

            sols = self._ikfast.get_ik(rot, pos, [torso, upperarm])
            if sols is None:
                continue

            for sol in sols:
                q = np.asarray(sol, dtype=np.float64)
                if q.shape != (8,):
                    continue
                # Reject branches that violate limits.
                if np.any(q < _LOWER_BOUNDS - 1e-6) or np.any(
                    q > _UPPER_BOUNDS + 1e-6
                ):
                    continue

                # Score by L2 distance to the seed (closest continuation).
                err = float(np.linalg.norm(q - seed8))
                if err < best_err:
                    best_err = err
                    best = q

            if best is not None and attempt >= 0:
                # Validate via FK
                p_chk, r_chk = self._ikfast.get_fk(best.tolist())
                p_chk = np.asarray(p_chk, dtype=np.float64)
                r_chk = np.asarray(r_chk, dtype=np.float64)
                pos_err = float(np.linalg.norm(p_chk - target_pose.position))
                rot_err_mat = r_chk @ target_pose.rotation.T
                ori_err = float(
                    np.linalg.norm(
                        Rotation.from_matrix(rot_err_mat).as_rotvec()
                    )
                )
                if (
                    pos_err <= cfg.position_tolerance
                    and ori_err <= cfg.orientation_tolerance
                ):
                    break

        if best is None:
            return IKResult(
                status=IKStatus.FAILED,
                joint_positions=np.zeros(self.num_joints, dtype=np.float64),
                final_error=np.inf,
                iterations=iterations,
                position_error=np.inf,
                orientation_error=np.inf,
            )

        status = (
            IKStatus.SUCCESS
            if pos_err <= cfg.position_tolerance
            and ori_err <= cfg.orientation_tolerance
            else IKStatus.MAX_ATTEMPTS_REACHED
        )
        q_out = best if self._include_torso else best[1:]
        return IKResult(
            status=status,
            joint_positions=q_out.astype(np.float64),
            final_error=best_err,
            iterations=iterations,
            position_error=pos_err,
            orientation_error=ori_err,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_full_8dof(self, q: np.ndarray) -> np.ndarray:
        """Pad a 7-DOF arm-only seed with the neutral torso height."""
        if self._include_torso:
            return np.asarray(q, dtype=np.float64)
        out = np.zeros(8, dtype=np.float64)
        out[0] = 0.2  # mid-range torso_lift (arm chain pins it)
        out[1:] = np.asarray(q, dtype=np.float64)
        return out

    def _neutral_seed(self) -> np.ndarray:
        seed = 0.5 * (_LOWER_BOUNDS + _UPPER_BOUNDS)
        return seed
