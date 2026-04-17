# Motion Planning

<div class="grid cards" markdown>

-   __OMPL frontend__

    ---

    All of [OMPL](https://ompl.kavrakilab.org/)'s mature sampling-based
    planners (26 exposed), driven by one uniform Python call. No glue
    code to swap planners — change a string.

-   __VAMP backend__

    ---

    Collision checking runs through [VAMP](https://github.com/KavrakiLab/vamp)'s
    SIMD-vectorised sphere pipeline. Point-cloud obstacles are
    broadphase-indexed and checked in **~3 μs per config**.

-   __C++ hot path__

    ---

    Python calls `plan(start, goal)` once. Everything inside —
    sampling, NN queries, collision checks, path simplification —
    runs in C++ behind a single nanobind call.

</div>

## Architecture

```
         Python
           │  planner.plan(start, goal)
           ▼
    ┌─────────────────────────────────────────┐
    │     OMPL SimpleSetup  (C++)              │
    │  ┌────────────┐  ┌──────────────────┐    │
    │  │  Planner   │→ │ MotionValidator  │    │
    │  │  RRT-C,    │  │  checkMotion     │    │
    │  │  BIT*, …   │  └────────┬─────────┘    │
    │  └────────────┘           │              │
    │                           ▼              │
    │                  ┌─────────────────┐     │
    │                  │ VAMP SIMD       │     │
    │                  │  sphere-cloud   │     │
    │                  │  collision      │     │
    │                  └─────────────────┘     │
    └─────────────────────────────────────────┘
```

The frontend is vanilla OMPL — `ProjectedStateSpace`, `StateCostIntegralObjective`,
`SimpleSetup`. The collision checker is the only replaced component;
it swaps OMPL's per-state user callback for a VAMP pipeline that
batches many sphere/point checks into a single AVX2 kernel.

Fetch's mobile base is **nonholonomic** (differential-drive), so any
subgroup that includes a base joint is planned in a compound state
space `SE(2) + R^N` with car-like distance and interpolation. The
SE(2) level is [`DubinsStateSpace`](https://ompl.kavrakilab.org/classompl_1_1base_1_1DubinsStateSpace.html)
(forward-only) or [`ReedsSheppStateSpace`](https://ompl.kavrakilab.org/classompl_1_1base_1_1ReedsSheppStateSpace.html)
(reverse allowed), selected by `BASE_REVERSE_ENABLE` in
`fetch_planning/fetch.py`. Path simplification, interpolation, and
the AO objective all stay on the curve, so the output is directly
executable by a diff-drive controller.

## Supported planners

The `planner_name` field of
[`PlannerConfig`](../api/types.md#fetch_planning.types.planning.PlannerConfig)
accepts any of:

=== "Feasibility (single-query, returns on first solution)"

    | Family | Planners |
    |---|---|
    | RRT | `rrtc`, `rrt`, `trrt`, `bitrrt`, `strrtstar`, `lbtrrt` |
    | KPIECE | `kpiece`, `bkpiece`, `lbkpiece` |
    | PRM | `prm`, `lazyprm`, `spars`, `spars2` |
    | Exploration | `est`, `biest`, `sbl`, `stride`, `pdst` |
    | RRT\*-feasible | `rrtsharp`, `rrtxstatic` |

=== "Asymptotically optimal (keep refining until budget)"

    | Family | Planners |
    |---|---|
    | RRT\* | `rrtstar`, `informed_rrtstar` |
    | Informed trees | `bitstar`, `abitstar`, `aitstar`, `eitstar`, `blitstar` |
    | FMT | `fmt`, `bfmt` |
    | PRM\* | `prmstar`, `lazyprmstar` |

## Minimal example

<video controls loop muted playsinline width="100%">
  <source src="../../assets/motion_planning.mp4" type="video/mp4">
</video>

```python
import numpy as np

from fetch_planning.fetch import HOME_JOINTS
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig

planner = create_planner(
    "fetch_arm",
    config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
    base_config=HOME_JOINTS.copy(),
    pointcloud=obstacle_cloud,           # (N, 3) np.float32
)

start = planner.extract_config(HOME_JOINTS)
goal = planner.sample_valid()
result = planner.plan(start, goal)
print(result.success, result.planning_time_ns * 1e-6, "ms")
```

## Configuring a plan

Every plan is driven by a single
[`PlannerConfig`](../api/types.md#fetch_planning.types.planning.PlannerConfig)
dataclass. The defaults are tuned for interactive use — a
millisecond-scale single-query plan with a dense, simplified path
— so you rarely need to touch anything except `planner_name` and
`time_limit`. The knobs that matter in practice:

```python
PlannerConfig(
    # Which OMPL planner to run — see "Supported planners" above.
    planner_name="rrtc",

    # Wall-clock ceiling for the search.  Feasibility planners return
    # on the first solution, so this is just a safety cap.  AO planners
    # (rrtstar, bitstar, …) run until the budget to refine cost.
    time_limit=1.0,

    # Inflation applied to every obstacle point in the VAMP broadphase.
    # Larger = more conservative, but slows the search.  0.01 m is a
    # good default for ~1 cm-resolution scans.
    point_radius=0.01,

    # After the raw path, run OMPL's SimpleSetup shortcutter.  Removes
    # jagged detours the sampler produced without changing homotopy.
    # Turn this OFF for constrained / cost planners — the default
    # shortcutter takes straight-line shortcuts that ignore custom
    # constraints and costs.
    simplify=True,

    # Resample the (simplified) path densely.  Pick one knob:
    #
    #   interpolate_count > 0   → exactly this many total waypoints,
    #                             distributed proportionally to edge length
    #   resolution > 0.0        → ceil(edge_length * resolution) samples
    #                             per edge — cleanest "uniform density"
    #                             option, scales naturally with DOF
    #   both 0                  → OMPL's default longest-valid-segment
    #                             fraction, usually too sparse for control
    #
    # The default (resolution=64.0) gives ~64 waypoints per unit of
    # state-space distance, which is smooth enough for a 100 Hz control
    # loop and robust to low-frequency replanning.
    interpolate=True,
    resolution=64.0,
    interpolate_count=0,
)
```

Typical recipes:

=== "Interactive — fast feasibility"

    ```python
    PlannerConfig(
        planner_name="rrtc",          # returns on first solution
        time_limit=0.5,
        simplify=True,
        resolution=64.0,
    )
    ```

=== "High-quality, asymptotically optimal"

    ```python
    PlannerConfig(
        planner_name="bitstar",       # keeps refining until budget
        time_limit=2.0,
        simplify=True,
        resolution=128.0,             # denser output for smoother control
    )
    ```

=== "Constrained / cost planner"

    ```python
    PlannerConfig(
        planner_name="rrtstar",
        time_limit=5.0,
        simplify=False,               # the shortcutter ignores custom cost/constraint
        resolution=64.0,
    )
    ```

=== "Fixed waypoint count (e.g. 100 steps)"

    ```python
    PlannerConfig(
        planner_name="rrtc",
        time_limit=1.0,
        interpolate=True,
        interpolate_count=100,        # exactly 100 waypoints — mutually exclusive with resolution
        resolution=0.0,
    )
    ```

## Post-hoc simplify / interpolate

`simplify` and `interpolate` run inside `plan(...)` by default, but
the same pipeline is exposed as standalone methods on `MotionPlanner`
— handy when you want to:

- **Plan once with raw output** (`simplify=False, interpolate=False`)
  and apply them later.
- **Re-densify an old path** at a different `resolution` without
  replanning.
- **Keep the search raw, smooth only at display time** — the
  controller consumes the original waypoints, the visualiser gets
  a densely interpolated copy.
- **Drop simplification for constrained / cost plans** (which you
  want) but still apply it manually to specific segments where the
  shortcutting is known-safe.

```python
result = planner.plan(start, goal)                     # unsimplified, raw
smooth = planner.simplify_path(result.path, time_limit=1.0)
dense  = planner.interpolate_path(smooth, resolution=128.0)     # or count=200
```

Both methods reuse the planner's collision environment and
constraint set. `simplify_path` only consults the motion validator
(geometric shortcuts — not cost-aware, same caveat as the in-plan
flag). `interpolate_path` runs `StateSpace::interpolate` on the
existing edges, so it stays on the constraint manifold for projected
state spaces and on the Dubins / Reeds-Shepp curve for compound
base+arm paths, and does not perform collision checks itself.

## Standalone collision checking

The same VAMP SIMD pipeline that the planner drives per motion edge is
exposed directly — useful for filtering sampled goals, validating a
trajectory produced outside the planner, or scoring a whole candidate
batch of configurations at once.

```python
planner.validate(cfg)                   # single config → bool
planner.validate_batch(cfgs)            # (N, ndof) → (N,) bool array
```

`validate_batch` packs up to `rake` distinct configurations directly
into a VAMP `ConfigurationBlock<rake>` and runs **one** `fkcc<rake>`
call per block — the same primitive the motion validator uses for
interpolated samples along an edge, fed independent configs per lane
instead. Per-config result is preserved: when a packed block fails,
only that block falls back to per-lane checks.

```python
# Example: filter 1000 candidate goals by validity.
goals = np.random.uniform(lo, hi, size=(1000, planner.num_dof))
mask = planner.validate_batch(goals)
good_goals = goals[mask]
```

## More

<div class="grid cards" markdown>

-   [__Subgroup planning__](subgroup.md)

    ---

    Plan over a slice of the 11-DOF body — arm, arm + torso, base,
    base + arm, whole body — with the remaining joints pinned to any
    11-DOF stance. Base-including subgroups use nonholonomic
    multilevel planning.

-   [__Manifold planning__](manifold.md)

    ---

    Hard task-space equality constraints written as CasADi
    expressions (planes, rails, orientation locks, couplings).
    Compiled once, cached, and injected into OMPL's
    `ProjectedStateSpace`.

-   [__Cost-space planning__](cost.md)

    ---

    Soft path-integral costs — the same CasADi authoring model, but
    now the constraint becomes a preference. Drives OMPL's
    asymptotically-optimal planners.

-   [__Time parameterization__](trajectory.md)

    ---

    Convert geometric paths into time-optimal trajectories with
    per-joint velocity and acceleration limits. TOTG (Kunz-Stilman)
    in C++, one call from Python.

</div>
