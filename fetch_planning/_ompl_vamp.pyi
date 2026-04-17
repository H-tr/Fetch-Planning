"""Type stubs for the ``_ompl_vamp`` C++ extension.

The actual implementation is in ``ext/ompl_vamp/ompl_vamp_ext.cpp`` and
ships as a compiled ``_ompl_vamp.cpython-*.so`` next to this file in the
installed package.  This stub mirrors the nanobind bindings exactly so
type checkers can resolve ``import fetch_planning._ompl_vamp``.
"""

from collections.abc import Sequence

class PlanResult:
    """Result of a single ``OmplVampPlanner.plan`` call."""

    @property
    def solved(self) -> bool:
        """``True`` if OMPL returned a solution within the time limit."""
        ...
    @property
    def path(self) -> list[list[float]]:
        """Solution waypoints in the planner's active joint space.

        Each inner list is one configuration with length equal to
        :meth:`OmplVampPlanner.dimension`.  Empty when ``solved`` is
        false.
        """
        ...
    @property
    def planning_time_ns(self) -> int:
        """Wall-clock time spent inside ``ss.solve(...)``, in nanoseconds."""
        ...
    @property
    def path_cost(self) -> float:
        """Geometric length of the (possibly simplified) solution path.

        ``inf`` when ``solved`` is false.
        """
        ...

class OmplVampPlanner:
    """OMPL planner with VAMP SIMD-accelerated collision checking.

    Two construction modes:

    * ``OmplVampPlanner()`` — full body, 24 DOF (3 base + 21 joints).
    * ``OmplVampPlanner(active_indices, frozen_config)`` — subgroup
      planner over the joints listed in ``active_indices``; the C++
      collision checker injects ``frozen_config`` for every other slot
      in the 24-DOF body on every state and motion validity query.
    """

    def __init__(
        self,
        active_indices: Sequence[int],
        frozen_config: Sequence[float],
        base_dim: int = 0,
        turning_radius: float = 0.2,
        allow_reverse: bool = False,
    ) -> None:
        """Create a subgroup planner.

        Args:
            active_indices: Positions in the full-body config that this
                planner will plan over, in DOF order.
            frozen_config: Full-body stance to inject for every joint *not*
                in ``active_indices``.
            base_dim: How many leading active indices form the
                nonholonomic base (0 = arm-only, 3 = SE2 base).
            turning_radius: Minimum turning radius in metres (ignored
                when ``base_dim == 0``).
            allow_reverse: Select the SE(2) base state space.  ``False``
                (default) picks ``DubinsStateSpace`` (forward-only); ``True``
                picks ``ReedsSheppStateSpace`` (reverse permitted).
        """
        ...
    def add_pointcloud(
        self,
        points: Sequence[Sequence[float]],
        r_min: float,
        r_max: float,
        point_radius: float,
    ) -> None:
        """Add a point cloud obstacle to the collision environment.

        Multiple calls accumulate; the planner keeps every cloud handed
        to it.

        Args:
            points: ``(N, 3)`` array of obstacle positions in world frame.
            r_min: Minimum robot collision-sphere radius (from
                :meth:`min_max_radii`).
            r_max: Maximum robot collision-sphere radius (from
                :meth:`min_max_radii`).
            point_radius: Inflation radius applied to every cloud point.
        """
        ...
    def remove_pointcloud(self) -> bool:
        """Drop the most-recently-added pointcloud.

        Returns ``False`` if there was none to remove.
        """
        ...
    def has_pointcloud(self) -> bool:
        """``True`` if at least one pointcloud is currently registered."""
        ...
    def add_sphere(self, center: Sequence[float], radius: float) -> None:
        """Add a single sphere obstacle (centre + radius) to the environment."""
        ...
    def clear_environment(self) -> None:
        """Remove all obstacles from the collision environment."""
        ...
    def add_compiled_constraint(
        self,
        so_path: str,
        symbol_name: str,
        ambient_dim: int,
        co_dim: int,
    ) -> None:
        """Load a CasADi-generated shared library as an OMPL constraint.

        Args:
            so_path: Path to the compiled ``.so`` file.
            symbol_name: CasADi function symbol name inside the library.
            ambient_dim: Dimension of the joint space (must match
                :meth:`dimension`).
            co_dim: Number of constraint equations (rows of the residual).
        """
        ...
    def clear_constraints(self) -> None:
        """Drop every accumulated constraint."""
        ...
    def num_constraints(self) -> int:
        """Number of constraints currently registered."""
        ...
    def add_compiled_cost(
        self,
        so_path: str,
        symbol_name: str,
        ambient_dim: int,
        weight: float = 1.0,
    ) -> None:
        """Load a CasADi-generated shared library as a soft path cost.

        The cost is wrapped as an ``ompl::StateCostIntegralObjective`` —
        trapezoidally integrated along every motion — and drives the
        search of asymptotically-optimal planners (RRT*, BIT*, AIT*,
        QRRT*, …).  Multiple costs are summed with their ``weight``.

        For whole-body / base-including subgroups the cost is set on
        the multilevel top SpaceInformation and combined with the
        Reeds-Shepp reverse penalty (which lives in the state-space
        distance, not the objective) — non-holonomic shaping is
        preserved.

        Args:
            so_path: Path to the compiled ``.so`` file.
            symbol_name: CasADi function symbol name inside the library.
            ambient_dim: Dimension of the joint space (must match
                :meth:`dimension`).
            weight: Non-negative scalar multiplier applied to the cost.
        """
        ...
    def clear_costs(self) -> None:
        """Drop every accumulated cost."""
        ...
    def num_costs(self) -> int:
        """Number of costs currently registered."""
        ...
    def plan(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        planner_name: str = "rrtc",
        time_limit: float = 10.0,
        simplify: bool = True,
        interpolate: bool = True,
    ) -> PlanResult:
        """Plan a collision-free path from ``start`` to ``goal``.

        Args:
            start: Active-DOF start configuration (length :meth:`dimension`).
            goal: Active-DOF goal configuration (length :meth:`dimension`).
            planner_name: OMPL planner identifier (e.g. ``"rrtc"``,
                ``"rrtstar"``, ``"prm"``, ``"bitstar"``).
            time_limit: Solver time limit in seconds.
            simplify: If true, run ``SimpleSetup::simplifySolution`` on the
                returned path.
            interpolate: If true, densify the simplified path with
                ``PathGeometric::interpolate`` so the returned waypoints
                are spaced at the longest valid segment fraction.
        """
        ...
    def validate(self, config: Sequence[float]) -> bool:
        """Return ``True`` if ``config`` is collision-free.

        ``config`` must have length :meth:`dimension`.  Subgroup
        planners expand it to a full 24-DOF state with the stored
        ``frozen_config`` before checking.
        """
        ...
    def dimension(self) -> int:
        """Number of active joints — 24 for the full body, smaller for subgroups."""
        ...
    def lower_bounds(self) -> list[float]:
        """Per-joint lower bounds for the active DOFs."""
        ...
    def upper_bounds(self) -> list[float]:
        """Per-joint upper bounds for the active DOFs."""
        ...
    def min_max_radii(self) -> tuple[float, float]:
        """``(min_radius, max_radius)`` of the robot's collision spheres.

        Pass these to :meth:`add_pointcloud` so VAMP can index its
        broadphase correctly.
        """
        ...
