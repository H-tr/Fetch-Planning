"""Kinodynamic trajectory — the time-parameterised output of FLASK planning.

Thin Python veneer over the C++ ``FlatTrajectory`` handle returned by
:meth:`~fetch_planning.planning.MotionPlanner.plan_kinodynamic`.  The
trajectory is a chain of closed-form cubic segments in the flat output
space, so it can be sampled exactly at any time — no separate time
parameterisation step is needed.

It exposes the same sampling API as :class:`~fetch_planning.trajectory.Trajectory`
(``duration``, ``position``, ``velocity``, ``acceleration``, ``sample``,
``sample_uniform``) in the planner's active-DOF layout.  For subgroups
that include the mobile base the first three entries are ``(x, y, theta)``
with world-frame rates ``(x_dot, y_dot, theta_dot)``; :meth:`base_twist`
gives the body-frame command ``(v, omega)`` a diff-drive controller
expects.

Typical feed-forward + PID tracking::

    result = planner.plan_kinodynamic(start, goal)
    traj = result.trajectory
    times, q_ref, qd_ref, _ = traj.sample_uniform(dt=0.01)
    twist_ref = traj.sample_base_twist(times)      # (T, 2): v, omega
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fetch_planning._ompl_vamp import FlatTrajectory as _FlatTrajectory

_SEGMENT_KIND_NAMES = {0: "drive", 1: "spin"}


@dataclass(frozen=True)
class KinodynamicTrajectory:
    """A dynamically feasible trajectory produced by FLASK planning.

    Instances are immutable handles around a C++ piecewise-cubic
    trajectory; query them via :meth:`position`, :meth:`velocity`,
    :meth:`acceleration`, or one of the batch samplers.
    """

    _handle: "_FlatTrajectory"

    @property
    def duration(self) -> float:
        """Trajectory duration in seconds."""
        return float(self._handle.duration)

    @property
    def num_dof(self) -> int:
        """Active DOF of the planner that produced the trajectory."""
        return int(self._handle.active_dim)

    @property
    def has_base(self) -> bool:
        """True if the leading three DOF are the mobile base."""
        return int(self._handle.base_dim) > 0

    @property
    def num_segments(self) -> int:
        """Number of closed-form segments (after simplification)."""
        return int(self._handle.num_segments)

    @property
    def knot_times(self) -> np.ndarray:
        """Segment boundary times ``[0, t_1, ..., duration]``."""
        return np.asarray(self._handle.knot_times(), dtype=np.float64)

    @property
    def segment_kinds(self) -> list[str]:
        """Per-segment kind: ``"drive"`` or ``"spin"`` (rotate in place)."""
        return [_SEGMENT_KIND_NAMES[k] for k in self._handle.segment_kinds()]

    @property
    def segment_gears(self) -> list[int]:
        """Per-segment base gear: ``0`` forward, ``1`` reverse."""
        return list(self._handle.segment_gears())

    # ── Pointwise sampling ────────────────────────────────────────────

    def position(self, t: float) -> np.ndarray:
        """Configuration at time ``t`` (seconds), clamped to ``[0, duration]``."""
        return np.asarray(self._handle.position(float(t)), dtype=np.float64)

    def velocity(self, t: float) -> np.ndarray:
        """Active-DOF velocity at time ``t`` (world-frame for the base)."""
        return np.asarray(self._handle.velocity(float(t)), dtype=np.float64)

    def acceleration(self, t: float) -> np.ndarray:
        """Active-DOF acceleration at time ``t`` (world-frame for the base)."""
        return np.asarray(self._handle.acceleration(float(t)), dtype=np.float64)

    def base_twist(self, t: float) -> np.ndarray:
        """Body-frame base command ``(v, omega)`` at time ``t``.

        ``v`` is the signed forward speed (negative when reversing) and
        ``omega`` the yaw rate.  Zero for arm-only trajectories.
        """
        return np.asarray(self._handle.base_twist(float(t)), dtype=np.float64)

    # ── Batch sampling ────────────────────────────────────────────────

    def sample(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample at the user-supplied ``times`` grid.

        Args:
            times: ``(T,)`` array of sample times in seconds.

        Returns:
            ``(positions, velocities, accelerations)`` — each ``(T, ndof)``.
        """
        times = np.ascontiguousarray(times, dtype=np.float64).reshape(-1)
        positions, velocities, accelerations = self._handle.sample(times.tolist())
        return (
            np.asarray(positions, dtype=np.float64),
            np.asarray(velocities, dtype=np.float64),
            np.asarray(accelerations, dtype=np.float64),
        )

    def sample_uniform(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly-spaced rollout at step ``dt``.

        The returned ``times`` always start at ``0`` and end at
        :attr:`duration`, which may make the final step shorter than
        ``dt`` — this matches what a streaming controller expects.

        Args:
            dt: Sample interval in seconds (must be ``> 0``).

        Returns:
            ``(times, positions, velocities, accelerations)`` —
            ``times`` has shape ``(T,)``; state arrays have shape
            ``(T, ndof)``.
        """
        times, positions, velocities, accelerations = self._handle.sample_uniform(
            float(dt)
        )
        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(positions, dtype=np.float64),
            np.asarray(velocities, dtype=np.float64),
            np.asarray(accelerations, dtype=np.float64),
        )

    def sample_base_twist(self, times: np.ndarray) -> np.ndarray:
        """Body-frame base commands ``(v, omega)`` at each time, ``(T, 2)``."""
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        return np.array([self._handle.base_twist(float(t)) for t in times])
