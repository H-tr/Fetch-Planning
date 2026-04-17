"""Time-optimal trajectory generation (TOTG) — Python front-end.

Wraps the C++ ``_time_parameterization`` extension with a reusable
parameteriser object and a convenience one-shot function.  The
algorithm is Kunz & Stilman (2012) "Time-optimal trajectory generation
for path following with bounded acceleration and velocity", vendored
from MoveIt 2's stripped-to-Eigen adaptation — see
``ext/time_parameterization/LICENSE.TOTG`` for attribution.
"""

from __future__ import annotations

import numpy as np

from .trajectory import Trajectory


class TimeOptimalParameterizer:
    """Time-optimal trajectory parameteriser with fixed joint limits.

    Instantiate once per robot (limits are cached), then call
    :meth:`parameterize` on every path.  The parameteriser itself holds
    no state between calls — it is safe to reuse across threads so long
    as the limits do not change.

    Args:
        max_velocity: ``(ndof,)`` per-joint velocity bound, **strictly
            positive**.  Units must match the path (rad/s for revolute
            joints, m/s for prismatic).
        max_acceleration: ``(ndof,)`` per-joint acceleration bound,
            **strictly positive**.
        max_deviation: Radial tolerance (same unit as the path) for the
            circular blend inserted at every interior waypoint.  Larger
            values make cornering faster but deviate further from the
            original piecewise-linear path.  Defaults to ``0.1``.
        time_step: Forward-integration step along the path.  Smaller
            values are more accurate and slower; the MoveIt default of
            ``1e-3`` works for most manipulators.
    """

    def __init__(
        self,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_deviation: float = 0.1,
        time_step: float = 1e-3,
    ) -> None:
        max_velocity = np.ascontiguousarray(max_velocity, dtype=np.float64).reshape(-1)
        max_acceleration = np.ascontiguousarray(
            max_acceleration, dtype=np.float64
        ).reshape(-1)
        if max_velocity.shape != max_acceleration.shape:
            raise ValueError(
                f"max_velocity {max_velocity.shape} and max_acceleration "
                f"{max_acceleration.shape} must have the same shape"
            )
        if not np.all(max_velocity > 0):
            raise ValueError("max_velocity entries must be strictly positive")
        if not np.all(max_acceleration > 0):
            raise ValueError("max_acceleration entries must be strictly positive")
        if max_deviation <= 0:
            raise ValueError("max_deviation must be strictly positive")
        if time_step <= 0:
            raise ValueError("time_step must be strictly positive")

        self._max_velocity = max_velocity
        self._max_acceleration = max_acceleration
        self._max_deviation = float(max_deviation)
        self._time_step = float(time_step)

    @property
    def num_dof(self) -> int:
        return int(self._max_velocity.shape[0])

    @property
    def max_velocity(self) -> np.ndarray:
        return self._max_velocity.copy()

    @property
    def max_acceleration(self) -> np.ndarray:
        return self._max_acceleration.copy()

    def parameterize(
        self,
        path: np.ndarray,
        velocity_scaling: float = 1.0,
        acceleration_scaling: float = 1.0,
    ) -> Trajectory:
        """Convert a piecewise-linear joint-space path into a time-optimal
        trajectory.

        Args:
            path: ``(N, ndof)`` waypoint array.  Must have at least two
                waypoints and match the parameteriser's DOF.
            velocity_scaling: Scalar in ``(0, 1]`` applied to the
                velocity limit.  ``1.0`` uses the bound passed at
                construction time.
            acceleration_scaling: Scalar in ``(0, 1]`` applied to the
                acceleration limit.  Values below ``1`` trade speed for
                smoother motion.

        Returns:
            A :class:`Trajectory`.

        Raises:
            ValueError: If ``path`` is malformed, ``scaling`` factors
                are out of range, or the TOTG integrator fails (e.g.
                the path contains a 180-degree reversal).
        """
        path = np.ascontiguousarray(path, dtype=np.float64)
        if path.ndim != 2:
            raise ValueError(f"path must be 2D (N, ndof), got shape {path.shape}")
        if path.shape[0] < 2:
            raise ValueError(
                f"path must have at least 2 waypoints, got {path.shape[0]}"
            )
        if path.shape[1] != self.num_dof:
            raise ValueError(
                f"path has {path.shape[1]} DOF, parameteriser was built for "
                f"{self.num_dof} DOF"
            )
        if not (0.0 < velocity_scaling <= 1.0):
            raise ValueError(
                f"velocity_scaling must be in (0, 1], got {velocity_scaling}"
            )
        if not (0.0 < acceleration_scaling <= 1.0):
            raise ValueError(
                f"acceleration_scaling must be in (0, 1], got {acceleration_scaling}"
            )

        # Collapse adjacent duplicate waypoints — TOTG builds a circular
        # blend per interior corner and zero-length segments short-circuit
        # to degenerate blends that bloat the switching-point list without
        # contributing to the timing.
        path = _deduplicate_waypoints(path)
        if path.shape[0] < 2:
            raise ValueError(
                "path collapsed to fewer than 2 distinct waypoints after "
                "de-duplication"
            )

        from fetch_planning._time_parameterization import compute_trajectory

        handle = compute_trajectory(
            path,
            self._max_velocity * velocity_scaling,
            self._max_acceleration * acceleration_scaling,
            self._max_deviation,
            self._time_step,
        )
        if handle is None:
            raise ValueError(
                "TOTG failed to parameterize path — see stderr for the "
                "underlying reason (common cause: a 180-degree reversal "
                "between three consecutive waypoints)."
            )
        return Trajectory(handle)


def parameterize_path(
    path: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_deviation: float = 0.1,
    time_step: float = 1e-3,
) -> Trajectory:
    """One-shot helper that builds a :class:`TimeOptimalParameterizer`
    and immediately calls :meth:`parameterize` on ``path``.

    Prefer the class form when you are going to re-use the limits — it
    avoids revalidating them on every call.
    """
    return TimeOptimalParameterizer(
        max_velocity, max_acceleration, max_deviation, time_step
    ).parameterize(path)


def _deduplicate_waypoints(path: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate waypoints in place-safe fashion."""
    if path.shape[0] < 2:
        return path
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], diffs > eps))
    return path[keep]
