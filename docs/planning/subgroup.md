# Subgroup Planning

The Fetch robot exposes 11 joints when you load the full URDF — a
3-DOF mobile base `(x, y, theta)`, a 1-DOF `torso_lift_joint`, and a
7-DOF arm (`shoulder_pan` → `wrist_roll`). Planning over all of
them is almost never what you want — for most tasks, only a slice of
the body is under active control while the rest stays put (or
follows its own controller).

**Subgroups are user-defined.** A subgroup is just an ordered list
of indices into the 11-DOF body. The OMPL planner reasons in the
reduced joint space; the C++ collision pipeline injects whatever
you choose as the frozen stance and still checks against the
**full** body. You can define a subgroup over any subset, in any
order — single joint, torso + one wrist, base + arm, anything.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/subgroup_planning.mp4" type="video/mp4">
</video>

## Built-in aliases

Convenience aliases ship for the common slicings, discoverable via
[`available_robots()`](../api/planning.md#fetch_planning.planning.motion_planner.available_robots):

| Alias | DOF | What it controls |
|---|---:|---|
| `fetch_base` | 3 | mobile base (x, y, yaw) |
| `fetch_arm` | 7 | 7-DOF arm, shoulder_pan → wrist_roll |
| `fetch_arm_with_torso` | 8 | torso_lift + 7-DOF arm |
| `fetch_base_arm` | 10 | mobile base + 7-DOF arm (torso fixed) |
| `fetch_whole_body` | 11 | full body — base + torso + arm |
| `fetch` | 11 | alias for `fetch_whole_body` |

They exist as starting points. The list is a dict in
`fetch_planning.fetch.PLANNING_SUBGROUPS` — add your own entry to
extend it, or use the low-level API below for truly one-off
slicings.

## Using a built-in alias

```python
from fetch_planning.fetch import HOME_JOINTS
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig

live_config = robot.get_joint_state()          # 11-D array from your system

planner = create_planner(
    "fetch_arm",
    config=PlannerConfig(time_limit=1.0),
    base_config=live_config,                    # frozen joints anchored here
    pointcloud=obstacle_cloud,
)

start = planner.extract_config(live_config)     # 11-D → 7-D
goal = planner.sample_valid()
result = planner.plan(start, goal)

full_path = planner.embed_path(result.path)     # 7-D path → 11-D path
```

`base_config` defaults to `HOME_JOINTS`. Pass the live joint reading
instead to pin every non-active joint exactly where the robot
currently is — collision checks see the real stance, not a synthetic
home pose.

## Nonholonomic base

Fetch has a differential-drive mobile base, so subgroups that
include any of the base joints (`fetch_base`, `fetch_base_arm`,
`fetch_whole_body` / `fetch`) cannot be planned as a plain
`RealVectorStateSpace` — the `(x, y, theta)` triple has to respect
the non-holonomic constraint. `MotionPlanner` handles this
automatically:

- The underlying state space becomes a `Compound(SE(2) + R^N)`
  where SE(2) is [`DubinsStateSpace`](https://ompl.kavrakilab.org/classompl_1_1base_1_1DubinsStateSpace.html)
  (forward-only) or [`ReedsSheppStateSpace`](https://ompl.kavrakilab.org/classompl_1_1base_1_1ReedsSheppStateSpace.html)
  (reverse allowed), selected by `BASE_REVERSE_ENABLE` in
  `fetch_planning/fetch.py`.
- The planner uses OMPL **multilevel planning** (fiber bundles)
  with the hierarchy `SE2 → SE2 × R^N`, and `QRRTStar` as the
  default asymptotically-optimal tree planner — its distance
  metric already includes the car-like arc length, so any user
  cost you add is composed with the non-holonomic shaping rather
  than replacing it.
- Tree extension, rewire, path simplification, and interpolation
  all stay on the curve. The output path is directly executable by
  a diff-drive controller.

Two planner helpers are specific to base-including subgroups:

```python
planner.has_base      # True iff any of (base_x, base_y, base_theta) is active

planner.set_base_bounds(
    x_lo=-4.0, x_hi=4.0,
    y_lo=-2.0, y_hi=4.0,
    theta_lo=-np.pi, theta_hi=np.pi,   # defaults shown
)
```

`has_base` is a property you can branch on; `set_base_bounds(...)`
tightens the `(x, y, theta)` workspace box after construction and
rebuilds the underlying state space in place — call it before
`plan()`. Calling `set_base_bounds` on an arm-only planner raises a
`RuntimeError`.

Constrained planning (CasADi manifold constraints — see
[Manifold planning](manifold.md)) is **not supported** for
base-including subgroups. Use `fetch_arm` or `fetch_arm_with_torso`
when you need the end-effector pinned to a manifold.

## Defining a custom subgroup

Any ordered list of joint indices is a valid subgroup. Drop straight
to the C++ binding when the alias list doesn't match your problem:

```python
from fetch_planning._ompl_vamp import OmplVampPlanner
from fetch_planning.fetch import (
    BASE_REVERSE_ENABLE, BASE_TURNING_RADIUS, HOME_JOINTS, fetch_robot_config,
)

# Name the joints you want to plan over; the rest stay frozen.
active_names = [
    "torso_lift_joint",
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
]
full_names = fetch_robot_config.joint_names
active_indices = [full_names.index(j) for j in active_names]

# base_dim = number of base joints (0, 1, 2, or 3) in the subgroup —
# here 0, so a plain RealVectorStateSpace.
planner = OmplVampPlanner(
    active_indices, HOME_JOINTS.tolist(), 0,
    BASE_TURNING_RADIUS, BASE_REVERSE_ENABLE,
)
planner.add_pointcloud(cloud.tolist(), *planner.min_max_radii(), 0.012)

start = [HOME_JOINTS[i] for i in active_indices]
goal = [...]                                    # your own sampling logic
result = planner.plan(start, goal, planner_name="rrtc", time_limit=1.0)
```

Same planner, same collision pipeline, same point-cloud broadphase
— just a different slice. You can also switch an existing planner
to a different subgroup on the fly with
[`set_subgroup(active_indices, frozen_config, base_dim)`](../api/planning.md#fetch_planning._ompl_vamp.OmplVampPlanner.set_subgroup),
which preserves the collision environment.

## When it matters

Planning in 11 DOF for a task that only needs an arm wastes almost
all of the search time on irrelevant configurations. For a 7-DOF
arm subgroup on the table-obstacle scene, `rrtc` returns in a
couple of milliseconds. The same plan over the 11-DOF whole body
runs noticeably slower for no practical reason — the base is
already where it should be.

Conversely, whole-body subgroups are the right choice when the task
needs the base to move alongside the arm — e.g. the
`examples/planning/nonholonomic.py` demo drives the base across the
room while the arm unfolds, carries an apple, and tucks back. Pick
the smallest subgroup that captures the joints your task actually
needs to move.
