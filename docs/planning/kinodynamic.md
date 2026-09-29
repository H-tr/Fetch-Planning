# Kinodynamic Planning (FLASK)

`MotionPlanner.plan_kinodynamic` plans a **time-parameterised,
dynamically feasible trajectory** directly — no separate time
parameterization step. It implements FLASK (Duong et al., "Ultrafast
Sampling-based Kinodynamic Planning via Differential Flatness", T-RO
2026): RRT-Connect in the differentially flat output space, where every
tree edge is a closed-form cubic that is collision-checked with the same
SIMD VAMP kernels as the geometric planner.

```python
from fetch_planning.planning import create_planner
from fetch_planning.types import KinodynamicConfig

planner = create_planner("fetch_whole_body", pointcloud=cloud)
result = planner.plan_kinodynamic(start, goal, config=KinodynamicConfig(time_limit=1.0))

traj = result.trajectory                      # KinodynamicTrajectory
times, q, qd, qdd = traj.sample_uniform(dt=0.01)
twist = traj.sample_base_twist(times)         # (T, 2): v, omega for the base
```

It works for every subgroup (`fetch_arm`, `fetch_arm_with_torso`,
`fetch_base`, `fetch_base_arm`, `fetch_whole_body`).

## How it works

<div class="grid cards" markdown>

-   __Flat outputs__

    ---

    Arm and torso joints are their own flat outputs (fully actuated).
    The diff-drive base contributes its planar position `(x, y)`; the
    heading follows the direction of travel,
    `theta = atan2(y_dot, x_dot) + gear * pi`.

-   __Closed-form edges__

    ---

    Nodes are `(position, velocity)` states. An edge is the cubic
    Hermite between its endpoints — the minimum-effort (LQMT) motion —
    with the duration `T*` that minimises
    `sum_i int y_i''^2 / a_max_i^2 dt + rho T` (a quartic root).

-   __Limits__

    ---

    Joint position / velocity / acceleration bounds are checked exactly
    on each cubic; `T*` is stretched until they hold. Base speed,
    tangential acceleration and yaw rate are checked on a 50 ms grid.

-   __SIMD collision checks__

    ---

    Each edge is sampled at VAMP's resolution along its arc length and
    checked in interleaved `ConfigurationBlock` batches (FLASK Alg. 3),
    then the solution is simplified the way VAMP simplifies geometric
    paths: greedy shortcuts (Alg. 4) plus subdivide-and-cut-corners
    smoothing, keeping a change only if it lowers the cost.

</div>

### The base at rest

The unicycle model has no heading at zero velocity — exactly where a
mobile manipulator starts, stops and grasps. The planner makes parked
states first-class:

* A **parked node** keeps its heading. Edges leave and enter it along
  that heading (forward or reverse); when a tree grows out of a parked
  node, the sampled target's lateral velocity is projected so the cubic
  departs aligned.
* **Gears** only change at parked nodes, so every drive segment has a
  well-defined heading and no hidden cusp.
* **Rotate in place**: a connection into or out of a parked node whose
  heading does not match gets a spin segment (arm holding still). Between
  two parked nodes that is rotate–translate–rotate, which is why
  open-floor legs solve in about a millisecond.

The output is exactly nonholonomic: the lateral velocity is zero to
machine precision everywhere.

## Tracking the trajectory

The trajectory is C¹ in every flat output: positions and velocities are
continuous; accelerations — and the base yaw rate, which depends on
them — jump at segment knots. A feed-forward + feedback controller is
enough:

* **Joints**: `u = qd_ref + Kp (q_ref - q) + Ki ∫(q_ref - q)` as a
  velocity command (what a joint trajectory controller does).
* **Base**: feed-forward `(v, omega)` from `traj.base_twist(t)` plus
  pose feedback in the body frame (e.g. a Kanayama law).

Pass the current state to replan on the fly — the new trajectory starts
with the given velocity:

```python
q_now, qd_now = traj.position(t), traj.velocity(t)
result = planner.plan_kinodynamic(q_now, new_goal, start_velocity=qd_now)
```

!!! warning "Tracking error vs. clearance"

    Trajectories are collision-free at zero clearance (up to the
    sampling resolution, like every VAMP edge). In closed-loop
    simulation on the demo scene (50 ms joint lag, 100 ms base lag, 5 %
    wheel slip) feed-forward + PID kept joint errors below 0.015 rad but
    base errors reached 7–8 cm, and 1–3 % of executions grazed an
    obstacle — with or without a 1.5 cm planning margin
    (`PlannerConfig.point_radius`). The grasp poses sit within 3 cm of
    the table, so a margin cannot cover that error: tighten base
    tracking (odometry / localisation feedback) or replan from the
    measured state.

## Configuration

`KinodynamicConfig` holds the planner parameters; the limits come from
`fetch_planning/fetch.py` (`JOINT_VELOCITY_LIMITS`,
`JOINT_ACCELERATION_LIMITS`, `BASE_MAX_*`) and can be scaled with
`velocity_scale` / `acceleration_scale`.

| Field | Default | Meaning |
|---|---|---|
| `time_limit` | `1.0` | Planning budget (s) |
| `rho` | `1.0` | Time weight; rest-to-rest edges peak at `sqrt(rho)` × the acceleration limit |
| `max_extension_time` | `3.0` | Tree step: longer cubics are truncated (node stays on the cubic) |
| `rest_sample_probability` | `0.3` | Share of zero-velocity samples (straight-line edges, gear changes) |
| `spin_probability` | `0.3` | Share of rotate-in-place extensions from parked nodes |
| `allow_reverse` | `fetch.BASE_REVERSE_ENABLE` | Permit the reverse gear |
| `simplify_iterations` / `simplify_time_limit` | `5` / `0.05` | Rounds and time cap for the VAMP-style simplification |
| `seed` | `0` | RNG seed (`0` = nondeterministic) |

## Results on the demo scene

`examples/planning/kinodynamic.py` runs the scene and legs of
`examples/planning/nonholonomic.py` (151k-point cloud). Planning times
are per call on a desktop CPU, 20 trials each.

**Whole body** — FLASK vs. the multilevel geometric planner (which
returns an untimed path):

| Leg | FLASK success | FLASK median / p95 | Duration | QRRT success | QRRT median |
|---|---|---|---|---|---|
| start → table | 20/20 | 70 / 369 ms | 12.6 s | 18/20 | 6.7 ms |
| carry 1 | 20/20 | 74 / 394 ms | 14.2 s | 18/20 | 6.1 ms |
| far → mid | 20/20 | 0.7 / 0.9 ms | 9.1 s | 20/20 | 4.8 ms |
| mid → sofa | 20/20 | 0.4 / 0.4 ms | 5.9 s | 20/20 | 1.4 ms |
| sofa → tea | 20/20 | 0.3 / 0.3 ms | 6.5 s | 20/20 | 1.3 ms |
| tea → table | 20/20 | 22 / 231 ms | 9.8 s | 20/20 | 1.7 ms |
| carry 2 | 20/20 | 53 / 315 ms | 14.3 s | 18/20 | 7.5 ms |
| table → home | 20/20 | 63 / 187 ms | 13.5 s | 20/20 | 12.0 ms |

**Arm only** (base parked at the table) — FLASK vs. RRT-Connect + TOTG,
and how often the TOTG-timed trajectory collides once its corners are
blended:

| Motion | FLASK median | FLASK duration | RRTC+TOTG median | TOTG duration | TOTG colliding |
|---|---|---|---|---|---|
| tuck → pregrasp | 10.6 ms | 9.8 s | 25.5 ms | 7.2 s | 12/20 |
| pregrasp → tuck | 13.2 ms | 9.9 s | 35.9 ms | 7.1 s | 13/20 |
| pregrasp → grasp | 0.04 ms | 1.3 s | 3.3 ms | 0.8 s | 0/20 |

Takeaways:

* Open-floor legs plan in under a millisecond (the direct
  rotate-translate-rotate connection). Legs that manoeuvre the arm next
  to the table take tens of milliseconds, p95 200–400 ms including up to
  50 ms of simplification — slower than the geometric planner's median,
  but with no failures, whereas the geometric planner times out on 10 %
  of those legs at a 1 s budget.
* Simplification (greedy shortcuts + subdivide-and-cut-corners, after
  VAMP) shortens the table legs by 30–45 %; FLASK trajectories are now
  about 1.4× TOTG's duration on the arm motions.
* 2 of 220 FLASK trajectories touched an obstacle when re-checked every
  5 ms — tighter paths make contacts between collision samples more
  likely; TOTG-timed geometric paths collided in about 60 % of the long
  arm motions.
