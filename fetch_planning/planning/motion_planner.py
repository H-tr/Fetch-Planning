"""OMPL + VAMP motion planner.

Uses OMPL for planning algorithms and VAMP for SIMD-accelerated
collision checking.  The entire planning pipeline runs in C++ via
the ``_ompl_vamp`` extension — Python only crosses the boundary
once per ``plan()`` call.

Supports subgroup planning: each subgroup operates on a reduced
state space while frozen joints are expanded to the full 11-DOF
Fetch whole-body config before collision checks.

Subgroups that include any of the base joints (``base_x_joint``,
``base_y_joint``, ``base_theta_joint``) use OMPL multilevel
planning (fiber bundles) with a hierarchy ``SE2 → SE2 × R^N``.
The SE(2) level is ``DubinsStateSpace`` (forward-only) or
``ReedsSheppStateSpace`` (reverse allowed), selected by
``BASE_REVERSE_ENABLE``.  Tree extension and rewire use the
selected car-like curves between samples, and the default
multilevel planner is QRRTStar so the nonholonomic distance
metric enters the path-cost objective.

Arm-only subgroups use standard OMPL geometric planning and
optionally support CasADi-compiled manifold constraints
(``ProjectedStateSpace``).
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
        costs: list | None = None,
    ) -> None:
        from fetch_planning._ompl_vamp import OmplVampPlanner
        from fetch_planning.fetch import (
            BASE_REVERSE_ENABLE,
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
        self._turning_radius = BASE_TURNING_RADIUS
        self._allow_reverse = BASE_REVERSE_ENABLE

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
        self._base_joint_names = set(full_names[base_slice])

        active_indices, base_dim = self._resolve_subgroup(
            robot_name, full_names, PLANNING_SUBGROUPS
        )

        self._planner = OmplVampPlanner(
            active_indices,
            self._base_config.tolist(),
            base_dim,
            BASE_TURNING_RADIUS,
            BASE_REVERSE_ENABLE,
        )
        self._apply_subgroup_state(full_names, active_indices, base_dim)

        if pointcloud is not None:
            self._add_pointcloud_impl(pointcloud, config.point_radius)

        if constraints:
            self._push_constraints(constraints)

        if costs:
            self._push_costs(costs)

    # ── Subgroup resolution ───────────────────────────────────────────

    def _resolve_subgroup(
        self,
        robot_name: str,
        full_names: list[str],
        planning_subgroups: dict,
    ) -> tuple[list[int], int]:
        """Translate a subgroup name into (active_indices, base_dim).

        Accepts both the explicit ``"fetch_whole_body"`` name and the
        shorter alias ``"fetch"`` for the full 11-DOF body.
        """
        if robot_name in ("fetch", "fetch_whole_body"):
            sg = planning_subgroups.get("fetch_whole_body")
            if sg is None:
                sg = {"dof": len(full_names), "joints": list(full_names)}
        else:
            sg = planning_subgroups.get(robot_name)
        if sg is None:
            raise ValueError(
                f"Unknown robot name '{robot_name}'. "
                f"Use one of: {available_robots()}"
            )
        sg_joint_names = sg["joints"]
        active_indices = [full_names.index(j) for j in sg_joint_names]
        base_dim = sum(1 for j in sg_joint_names if j in self._base_joint_names)
        return active_indices, base_dim

    def _apply_subgroup_state(
        self,
        full_names: list[str],
        active_indices: list[int],
        base_dim: int,
    ) -> None:
        """Update cached Python-side subgroup metadata after a C++ rebuild."""
        self._has_base = base_dim > 0
        self._base_dim = base_dim
        sg_joint_names = [full_names[i] for i in active_indices]
        self._joint_names = list(sg_joint_names)
        if set(active_indices) == set(range(len(full_names))):
            self._subgroup_indices = None
        else:
            self._subgroup_indices = np.array(active_indices)
        self._ndof = self._planner.dimension()

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
        """True if any of the mobile base joints (x, y, theta) are active.

        When True, the planner uses OMPL multilevel planning with
        Dubins / Reeds-Shepp curves (hierarchy SE2 → SE2×R^N) and
        QRRTStar as the default optimal tree planner.
        """
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

    def clear_constraints(self) -> None:
        """Remove all constraints from the planner."""
        self._planner.clear_constraints()

    def set_constraints(self, constraints: list) -> None:
        """Replace all constraints: clear existing, then push new ones."""
        self.clear_constraints()
        self._push_constraints(constraints)

    # ── Cost integration ──────────────────────────────────────────────

    def _push_costs(self, costs) -> None:
        """Push compiled CasADi costs to the C++ planner.

        Costs are soft per-state terms that shape the solution returned
        by asymptotically-optimal planners (``rrtstar``, ``bitstar``,
        ``qrrtstar``, …).  For whole-body subgroups (``fetch_whole_body``,
        ``fetch_base_arm``) the multilevel backend keeps the SE(2)
        path-length term (Dubins or Reeds-Shepp, per ``BASE_REVERSE_ENABLE``)
        alongside the user cost, so the non-holonomic shaping is preserved.
        """
        from fetch_planning.planning.costs import Cost

        for c in costs:
            if not isinstance(c, Cost):
                raise TypeError(
                    f"costs must be Cost instances from "
                    f"fetch_planning.planning.costs; "
                    f"got {type(c).__name__}"
                )
            if c.ambient_dim != self._ndof:
                raise ValueError(
                    f"Cost ambient_dim ({c.ambient_dim}) does not match "
                    f"planner active dimension ({self._ndof}).  Build the "
                    f"Cost with a SymbolicContext for the same subgroup."
                )
            self._planner.add_compiled_cost(
                str(c.so_path),
                c.symbol_name,
                c.ambient_dim,
                float(c.weight),
            )

    def clear_costs(self) -> None:
        """Remove all costs from the planner (falls back to path length)."""
        self._planner.clear_costs()

    def set_costs(self, costs: list) -> None:
        """Replace all costs: clear existing, then push new ones."""
        self.clear_costs()
        self._push_costs(costs)

    # ── Pointcloud environment ────────────────────────────────────────

    def _add_pointcloud_impl(
        self, pointcloud: np.ndarray, point_radius: float
    ) -> None:
        r_min, r_max = self._planner.min_max_radii()
        self._planner.add_pointcloud(
            np.asarray(pointcloud, dtype=np.float32).tolist(),
            r_min,
            r_max,
            point_radius,
        )

    def add_pointcloud(self, pointcloud: np.ndarray) -> None:
        """Add a point cloud to the scene after construction.

        The C++ planner keeps every cloud handed to it — multiple calls
        accumulate.  Uses ``config.point_radius`` as the per-point
        inflation radius.

        Args:
            pointcloud: ``(N, 3)`` array of obstacle positions in world
                frame.
        """
        self._add_pointcloud_impl(pointcloud, self._config.point_radius)

    def remove_pointcloud(self) -> bool:
        """Drop the most-recently-added pointcloud.

        Returns ``False`` if there was none registered.
        """
        return self._planner.remove_pointcloud()

    @property
    def has_pointcloud(self) -> bool:
        """``True`` if a pointcloud is currently registered."""
        return self._planner.has_pointcloud()

    def clear_environment(self) -> None:
        """Drop every registered obstacle (spheres and point clouds)."""
        self._planner.clear_environment()

    # ── Point cloud filtering ────────────────────────────────────────

    def filter_pointcloud(
        self,
        pointcloud: np.ndarray,
        min_dist: float,
        max_range: float,
        origin: np.ndarray | list[float],
        workspace_min: np.ndarray | list[float],
        workspace_max: np.ndarray | list[float],
        cull: bool = True,
    ) -> np.ndarray:
        """Spatially downsample a point cloud via Morton-curve sorting.

        Keeps one representative point per ``min_dist`` neighbourhood and
        discards points farther than ``max_range`` from ``origin`` or
        outside the ``[workspace_min, workspace_max]`` bounding box.

        Args:
            pointcloud: ``(N, 3)`` array of 3-D points.
            min_dist: Minimum distance between two retained points.
            max_range: Maximum distance from ``origin`` to keep a point.
            origin: ``(3,)`` reference position for range culling.
            workspace_min: ``(3,)`` lower corner of the workspace AABB.
            workspace_max: ``(3,)`` upper corner of the workspace AABB.
            cull: If ``True`` (default), apply range and AABB culling.

        Returns:
            ``(M, 3)`` filtered point cloud with ``M <= N``.
        """
        pts = np.asarray(pointcloud, dtype=np.float32).tolist()
        origin = [float(x) for x in origin]
        workspace_min = [float(x) for x in workspace_min]
        workspace_max = [float(x) for x in workspace_max]
        filtered = self._planner.filter_pointcloud(
            pts,
            float(min_dist),
            float(max_range),
            origin,
            workspace_min,
            workspace_max,
            cull,
        )
        return np.asarray(filtered, dtype=np.float32)

    def filter_self_from_pointcloud(
        self,
        pointcloud: np.ndarray,
        point_radius: float,
        config: np.ndarray,
    ) -> np.ndarray:
        """Remove points that collide with the robot body or environment.

        Computes forward kinematics at ``config``, then drops every
        point whose inflated sphere (radius ``point_radius``) overlaps
        any robot collision sphere or any registered obstacle.

        Args:
            pointcloud: ``(N, 3)`` array of 3-D points.
            point_radius: Inflation radius for each point.
            config: Active-DOF configuration (same space as ``plan``).

        Returns:
            ``(M, 3)`` filtered point cloud with ``M <= N``.
        """
        pts = np.asarray(pointcloud, dtype=np.float32).tolist()
        config = np.asarray(config, dtype=np.float64)
        if len(config) != self._ndof:
            raise ValueError(f"config has {len(config)} DOF, expected {self._ndof}")
        filtered = self._planner.filter_self_from_pointcloud(
            pts,
            float(point_radius),
            config.tolist(),
        )
        return np.asarray(filtered, dtype=np.float32)

    # ── Subgroup switching ───────────────────────────────────────────

    def set_subgroup(
        self,
        robot_name: str,
        base_config: np.ndarray | None = None,
    ) -> None:
        """Switch active joints without rebuilding the collision environment.

        Clears all constraints and costs.  The pointcloud is preserved.

        Args:
            robot_name: Subgroup name from ``PLANNING_SUBGROUPS``, or
                ``"fetch"`` / ``"fetch_whole_body"`` for the full 11-DOF
                body.
            base_config: 11-DOF frozen config for inactive joints.
                Defaults to the previously stored base config.
        """
        from fetch_planning.fetch import PLANNING_SUBGROUPS, fetch_robot_config

        if base_config is not None:
            self._base_config = np.asarray(base_config, dtype=np.float64).copy()
        self._robot_name = robot_name

        full_names = fetch_robot_config.joint_names
        active_indices, base_dim = self._resolve_subgroup(
            robot_name, full_names, PLANNING_SUBGROUPS
        )
        self._planner.set_subgroup(
            active_indices, self._base_config.tolist(), base_dim
        )
        self._apply_subgroup_state(full_names, active_indices, base_dim)

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
        time_limit: float | None = None,
    ) -> PlanningResult:
        """Plan a collision-free path from start to goal.

        Args:
            start: Start configuration (active DOF).
            goal: Goal configuration (active DOF).
            time_limit: Optional per-call override for the solver time
                limit.  Defaults to ``self._config.time_limit``.
        """
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

        if time_limit is None:
            time_limit = self._config.time_limit

        result = self._planner.plan(
            start.tolist(),
            goal.tolist(),
            self._config.planner_name,
            time_limit,
            self._config.simplify,
            self._config.interpolate,
            self._config.interpolate_count,
            self._config.resolution,
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

    def simplify_path(self, path: np.ndarray, time_limit: float = 1.0) -> np.ndarray:
        """Run OMPL's shortcut-based path simplifier on ``path``.

        Same pipeline ``plan(..., simplify=True)`` uses internally
        (``reduceVertices`` + ``collapseCloseVertices`` + ``shortcutPath``
        + B-spline smoothing), but detached so you can apply it to any
        path you already have — e.g. replay an old plan with a
        different collision environment.

        Shortcuts only consult the motion validator.  Custom soft
        costs (:class:`Cost`) are ignored; for cost-driven plans, run
        :meth:`plan` with ``simplify=False`` and leave the path
        untouched unless you've explicitly decided shortcut shaping
        is acceptable.

        Args:
            path: ``(N, ndof)`` array of waypoints in the planner's
                active DOF space.
            time_limit: Wall-clock budget for the simplifier, seconds.

        Returns:
            ``(M, ndof)`` simplified waypoint array with ``M <= N``.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != self._ndof:
            raise ValueError(
                f"path must have shape (N, {self._ndof}), got {path.shape}"
            )
        simp = self._planner.simplify_path(path.tolist(), float(time_limit))
        return np.array(simp, dtype=np.float64)

    def interpolate_path(
        self,
        path: np.ndarray,
        count: int = 0,
        resolution: float = 64.0,
    ) -> np.ndarray:
        """Densify ``path`` with uniform waypoints along every edge.

        Three modes (pick one; the other must be zero):

            * ``count > 0``        — exactly that many total waypoints
              distributed proportionally to edge length.
            * ``resolution > 0.0`` — ``ceil(edge_length * resolution)``
              waypoints per edge (uniform density in state-space
              distance — the default).
            * both ``0``           — OMPL's default longest-valid-segment
              fraction.

        Uses ``StateSpace::interpolate`` internally, so the inserted
        states stay on the constraint manifold for projected state
        spaces and on the Dubins / Reeds-Shepp curve for compound
        base+arm paths.  No collision check is performed — the
        densification only lifts points along the existing edges.

        Args:
            path: ``(N, ndof)`` waypoint array.
            count: Exact total waypoint count if ``> 0``.
            resolution: Waypoints per unit state-space distance if
                ``> 0.0``.

        Returns:
            ``(M, ndof)`` densified waypoint array with ``M >= N``.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != self._ndof:
            raise ValueError(
                f"path must have shape (N, {self._ndof}), got {path.shape}"
            )
        dense = self._planner.interpolate_path(
            path.tolist(), int(count), float(resolution)
        )
        return np.array(dense, dtype=np.float64)

    def validate(self, configuration: np.ndarray) -> bool:
        """Check if a configuration is collision-free."""
        configuration = np.asarray(configuration, dtype=np.float64)
        return self._planner.validate(configuration.tolist())

    def validate_batch(self, configurations: np.ndarray) -> np.ndarray:
        """Batched collision check — one SIMD block per ``rake`` configs.

        Packs ``rake`` distinct configurations into a single VAMP
        ``ConfigurationBlock<rake>`` and runs one ``fkcc<rake>`` call
        per block, so ``N`` queries cost ``ceil(N / rake)`` SIMD
        sweeps in the common case.  When a packed block fails we fall
        back to per-lane single-state checks for that block only, so
        the returned array is always exactly one bool per input.

        Args:
            configurations: ``(N, ndof)`` array of active-DOF
                configurations.

        Returns:
            ``(N,)`` boolean array; ``True`` at index ``i`` iff
            ``configurations[i]`` is collision-free.
        """
        configurations = np.asarray(configurations, dtype=np.float64)
        if configurations.ndim != 2 or configurations.shape[1] != self._ndof:
            raise ValueError(
                f"configurations must have shape (N, {self._ndof}), "
                f"got {configurations.shape}"
            )
        valid = self._planner.validate_batch(configurations.tolist())
        return np.asarray(valid, dtype=bool)

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
    from fetch_planning.fetch import PLANNING_SUBGROUPS

    return ["fetch"] + sorted(PLANNING_SUBGROUPS.keys())


def create_planner(
    robot_name: str = "fetch",
    config: PlannerConfig | None = None,
    pointcloud: np.ndarray | None = None,
    base_config: np.ndarray | None = None,
    constraints: list | None = None,
    costs: list | None = None,
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
            Not supported for subgroups that include mobile-base joints.
        costs: Optional list of
            :class:`~fetch_planning.planning.costs.Cost` instances
            (CasADi-backed).  Soft per-state terms summed with their
            weights and trapezoidally integrated along every motion —
            the asymptotically-optimal planners (``rrtstar``,
            ``bitstar``, ``aitstar``, ``qrrtstar``, …) minimise this
            objective.  For whole-body planners the multilevel backend
            uses QRRTStar by default, whose cost aggregation already
            includes the Dubins / Reeds-Shepp distance metric — user
            costs add on top, preserving the non-holonomic shaping.
            Without any costs the planner uses OMPL's default path-length
            objective.

    Returns:
        A :class:`MotionPlanner` instance.
    """
    return MotionPlanner(
        robot_name, config, pointcloud, base_config, constraints, costs
    )
