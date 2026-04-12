"""OMPL + VAMP motion planner.

Uses OMPL for planning algorithms and VAMP for SIMD-accelerated
collision checking.  The entire planning pipeline runs in C++ via
the ``_ompl_vamp`` extension — Python only crosses the boundary
once per ``plan()`` call.

Supports subgroup planning: each subgroup operates on a reduced
state space while frozen joints are expanded to the full 11-DOF
Fetch whole-body config before collision checks.

Subgroups that include any of the base joints (``base_x_joint``,
``base_y_joint``, ``base_theta_joint``) are planned over an OMPL
``CompoundStateSpace(ReedsSheppStateSpace + RealVectorStateSpace)``
so the nonholonomic differential-drive constraint is enforced
natively by the sampler and distance/interpolate methods.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from fetch_planning.types import PlannerConfig, PlanningResult, PlanningStatus


@runtime_checkable
class MotionPlannerBase(Protocol):
    """Protocol for motion planner backends."""

    @property
    def robot_name(self) -> str:
        ...

    @property
    def num_dof(self) -> int:
        ...

    def plan(self, start: np.ndarray, goal: np.ndarray) -> PlanningResult:
        ...

    def validate(self, configuration: np.ndarray) -> bool:
        ...


class MotionPlanner:
    """Motion planner using OMPL + VAMP C++ backend.

    All internals are private.  The public API accepts and returns
    only numpy arrays.

    For subgroup planners the helper methods :meth:`extract_config`,
    :meth:`embed_config`, and :meth:`embed_path` convert between the
    reduced DOF space used by the planner and the full 11-DOF body
    configuration.
    """

    def __init__(
        self,
        robot_name: str,
        config: PlannerConfig | None = None,
        pointcloud: np.ndarray | None = None,
        base_config: np.ndarray | None = None,
        constraints: list | None = None,
    ) -> None:
        from fetch_planning._ompl_vamp import OmplVampPlanner
        from fetch_planning.config.robot_config import (
            BASE_TURNING_RADIUS,
            HOME_JOINTS,
            JOINT_GROUPS,
            PLANNING_SUBGROUPS,
            fetch_robot_config,
        )

        if config is None:
            config = PlannerConfig()

        self._config = config
        self._robot_name = robot_name

        # Frozen 11-DOF joint values for any joint not controlled by
        # this planner.  Defaults to HOME_JOINTS, but the caller can
        # pass any 11-DOF array — e.g. the live config from the env —
        # so the inactive joints are pinned wherever they currently are.
        if base_config is None:
            base_config = HOME_JOINTS
        self._base_config = np.asarray(base_config, dtype=np.float64).copy()
        if self._base_config.shape != HOME_JOINTS.shape:
            raise ValueError(
                f"base_config must have shape {HOME_JOINTS.shape}, "
                f"got {self._base_config.shape}"
            )

        full_names = fetch_robot_config.joint_names
        base_slice = JOINT_GROUPS["base"]
        base_indices = set(range(base_slice.start, base_slice.stop))
        sg = PLANNING_SUBGROUPS.get(robot_name)

        if robot_name in ("fetch", "fetch_whole_body") and sg is None:
            # "fetch" alias → full whole-body planner
            self._planner = OmplVampPlanner(BASE_TURNING_RADIUS)
            self._joint_names = list(full_names)
            self._subgroup_indices = None
            self._has_base = True
        elif sg is None:
            raise ValueError(
                f"Unknown robot name '{robot_name}'. "
                f"Use one of: {available_robots()}"
            )
        else:
            # Subgroup planner — frozen joints come from the supplied
            # base_config; the C++ checker injects them around the
            # active subset before every collision query.
            sg_joint_names = sg["joints"]
            active_indices = [full_names.index(j) for j in sg_joint_names]
            self._has_base = any(idx in base_indices for idx in active_indices)
            self._planner = OmplVampPlanner(
                active_indices,
                self._base_config.tolist(),
                BASE_TURNING_RADIUS,
            )
            self._joint_names = list(sg_joint_names)
            self._subgroup_indices = np.array(active_indices)

        self._ndof = self._planner.dimension()

        if pointcloud is not None:
            r_min, r_max = self._planner.min_max_radii()
            self._planner.add_pointcloud(
                np.asarray(pointcloud, dtype=np.float32).tolist(),
                r_min,
                r_max,
                config.point_radius,
            )

        if constraints:
            self._push_constraints(constraints)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def num_dof(self) -> int:
        return self._ndof

    @property
    def joint_names(self) -> list[str]:
        """Joint names controlled by this planner, in DOF order."""
        return self._joint_names

    @property
    def is_subgroup(self) -> bool:
        """True if this planner controls a subset of the full body."""
        return self._subgroup_indices is not None

    @property
    def subgroup_indices(self) -> np.ndarray | None:
        """Indices of this planner's joints in the full 11-DOF config."""
        return self._subgroup_indices

    @property
    def base_config(self) -> np.ndarray:
        """The 11-DOF stance frozen for joints outside this subgroup."""
        return self._base_config.copy()

    @property
    def has_base(self) -> bool:
        """True if any of the mobile base joints (x, y, theta) are active
        in this planner. When True, the underlying OMPL state space is a
        CompoundStateSpace(ReedsSheppStateSpace + RealVectorStateSpace)
        and the nonholonomic differential-drive constraint is enforced by
        construction."""
        return self._has_base

    def set_base_bounds(
        self,
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
        theta_lo: float = -np.pi,
        theta_hi: float = np.pi,
    ) -> None:
        """Tighten the base (x, y, theta) workspace bounds.

        Only meaningful when ``has_base`` is True. Rebuilds the underlying
        OMPL state space in place — call before ``plan()``.
        """
        if not self._has_base:
            raise RuntimeError(
                "set_base_bounds() is only valid for planners whose subgroup "
                "includes the mobile base joints (fetch, fetch_whole_body, "
                "fetch_base, fetch_base_arm)."
            )
        self._planner.set_base_bounds(x_lo, x_hi, y_lo, y_hi, theta_lo, theta_hi)

    # ── Constraint integration ────────────────────────────────────────

    def _push_constraints(self, constraints) -> None:
        """Push compiled CasADi constraints to the C++ planner."""
        from fetch_planning.planning.constraints import Constraint

        for c in constraints:
            if not isinstance(c, Constraint):
                raise TypeError(
                    f"constraints must be Constraint instances from "
                    f"fetch_planning.planning.constraints; "
                    f"got {type(c).__name__}"
                )
            if c.ambient_dim != self._ndof:
                raise ValueError(
                    f"Constraint ambient_dim ({c.ambient_dim}) does not match "
                    f"planner active dimension ({self._ndof}).  Build the "
                    f"Constraint with a SymbolicContext for the same subgroup."
                )
            self._planner.add_compiled_constraint(
                str(c.so_path),
                c.symbol_name,
                c.ambient_dim,
                c.co_dim,
            )

    # ── Subgroup helpers ──────────────────────────────────────────────

    def extract_config(self, full_config: np.ndarray) -> np.ndarray:
        """Extract this planner's joints from a full 11-DOF configuration."""
        full_config = np.asarray(full_config, dtype=np.float64)
        if self._subgroup_indices is None:
            return full_config.copy()
        return full_config[self._subgroup_indices].copy()

    def embed_config(
        self,
        config: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Embed a subgroup config into a full 11-DOF configuration.

        ``base_config`` defaults to the planner's stored base — the same
        11-DOF values the C++ collision checker injects for inactive
        joints — so the embedded config matches what was validated.
        """
        config = np.asarray(config, dtype=np.float64)
        if self._subgroup_indices is None:
            return config.copy()

        if base_config is None:
            base_config = self._base_config
        full = np.array(base_config, dtype=np.float64)
        full[self._subgroup_indices] = config
        return full

    def embed_path(
        self,
        path: np.ndarray,
        base_config: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a subgroup path ``(N, sub_dof)`` to ``(N, 11)``.

        ``base_config`` defaults to the planner's stored base — the same
        11-DOF values the C++ collision checker injects for inactive
        joints — so the embedded path matches what was validated.
        """
        path = np.asarray(path, dtype=np.float64)
        if self._subgroup_indices is None:
            return path.copy()

        if base_config is None:
            base_config = self._base_config
        n = path.shape[0]
        full_path = np.tile(np.array(base_config, dtype=np.float64), (n, 1))
        full_path[:, self._subgroup_indices] = path
        return full_path

    # ── Planning ──────────────────────────────────────────────────────

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> PlanningResult:
        """Plan a collision-free path from start to goal."""
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        if len(start) != self._ndof:
            raise ValueError(f"start has {len(start)} DOF, expected {self._ndof}")
        if len(goal) != self._ndof:
            raise ValueError(f"goal has {len(goal)} DOF, expected {self._ndof}")

        if not self._planner.validate(start.tolist()):
            return PlanningResult(
                status=PlanningStatus.INVALID_START,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )
        if not self._planner.validate(goal.tolist()):
            return PlanningResult(
                status=PlanningStatus.INVALID_GOAL,
                path=None,
                planning_time_ns=0,
                iterations=0,
                path_cost=float("inf"),
            )

        result = self._planner.plan(
            start.tolist(),
            goal.tolist(),
            self._config.planner_name,
            self._config.time_limit,
            self._config.simplify,
            self._config.interpolate,
        )

        if not result.solved:
            return PlanningResult(
                status=PlanningStatus.FAILED,
                path=None,
                planning_time_ns=result.planning_time_ns,
                iterations=0,
                path_cost=float("inf"),
            )

        path_np = np.array(result.path, dtype=np.float64)

        return PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=path_np,
            planning_time_ns=result.planning_time_ns,
            iterations=0,
            path_cost=result.path_cost,
        )

    def validate(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free."""
        configuration = np.asarray(configuration, dtype=np.float64)
        return self._planner.validate(configuration.tolist())

    def sample_valid(self) -> np.ndarray:
        """Sample a random collision-free configuration."""
        lo = np.array(self._planner.lower_bounds())
        hi = np.array(self._planner.upper_bounds())
        while True:
            config = np.random.uniform(lo, hi)
            if self._planner.validate(config.tolist()):
                return config


def available_robots() -> list[str]:
    """Return all available robot names for planning."""
    from fetch_planning.config.robot_config import PLANNING_SUBGROUPS

    return ["fetch"] + sorted(PLANNING_SUBGROUPS.keys())


def create_planner(
    robot_name: str = "fetch",
    config: PlannerConfig | None = None,
    pointcloud: np.ndarray | None = None,
    base_config: np.ndarray | None = None,
    constraints: list | None = None,
) -> MotionPlanner:
    """Create a motion planner for any robot or subgroup.

    Args:
        robot_name: Robot or subgroup name. Use :func:`available_robots`
            to list all names. The default ``"fetch"`` is an alias for
            the full whole-body planner (11 DOF, nonholonomic base).
        config: Planner configuration (uses defaults if None).
        pointcloud: ``(N, 3)`` obstacle point cloud (optional).
        base_config: 11-DOF values to inject for joints not controlled
            by this planner (i.e. the frozen joints of a subgroup).
            Defaults to ``HOME_JOINTS``.  Supply any 11-DOF array — for
            example the live configuration read from your env — to pin
            the rest of the body wherever it currently is.  Ignored for
            the full-body ``"fetch"`` / ``"fetch_whole_body"`` planners.
        constraints: Optional list of
            :class:`~fetch_planning.planning.constraints.Constraint`
            instances (CasADi-backed).  When non-empty, the planner
            switches to ``ProjectedStateSpace`` and projects every state
            onto the constraint manifold.  Both ``start`` and ``goal``
            passed to ``plan(...)`` must already lie on the manifold.

    Returns:
        A :class:`MotionPlanner` instance.
    """
    return MotionPlanner(robot_name, config, pointcloud, base_config, constraints)
