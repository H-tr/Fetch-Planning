# Examples

The library exposes a small number of factories — `create_ik_solver`,
`create_planner`, `PyBulletEnv` — that cover every workflow below.
Each section pairs a short code snippet with a rendered clip of the
example in action.  All clips come from the offscreen renderer in
`scripts/render_videos/render_docs_videos.py`, so they are exactly
reproducible from a clean checkout.

---

## Inverse Kinematics (TRAC-IK)

TRAC-IK is a fast numerical IK solver — pass a 6-DOF target pose and
a joint seed, get either a closed-form-quality solution or a
`failure` status in under a millisecond.  The factory picks the right
kinematic chain and solve mode from the arguments alone.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/trac_ik.mp4" type="video/mp4">
</video>

```python
import numpy as np

from fetch_planning.fetch import HOME_JOINTS, JOINT_GROUPS
from fetch_planning.kinematics import create_ik_solver
from fetch_planning.types import IKConfig, SE3Pose, SolveType

G = JOINT_GROUPS
HOME_ARM = HOME_JOINTS[G["arm"]]

config = IKConfig(
    timeout=0.2,                  # seconds per TRAC-IK attempt
    epsilon=1e-5,                 # convergence tolerance
    solve_type=SolveType.DISTANCE,  # SPEED | DISTANCE | MANIP1 | MANIP2
    max_attempts=10,              # random restart attempts
    position_tolerance=1e-4,      # post-solve check (meters)
    orientation_tolerance=1e-4,   # post-solve check (radians)
)

solver = create_ik_solver("arm", config=config)

home_pose = solver.fk(HOME_ARM)
target = SE3Pose(
    position=home_pose.position + np.array([0.22, 0.15, 0.05]),
    rotation=home_pose.rotation,
)

result = solver.solve(target, seed=HOME_ARM)
if result.success:
    print(f"solution: {np.round(result.joint_positions, 4)}")
```

The interactive visualization lives in `examples/ik/basic_vis.py`:

```bash
pixi run -e dev python examples/ik/basic_vis.py
```

---

## Constrained IK (Pink QP)

The Pink backend solves IK as a weighted QP instead of a closed-form
Jacobian step — which means you can stack secondary objectives like
centre-of-mass stability, camera-frame stabilization, and (optionally)
self-collision avoidance on top of the primary task.  Same
`create_ik_solver` factory, different backend.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_ik.mp4" type="video/mp4">
</video>

```python
from fetch_planning.fetch import HOME_JOINTS, JOINT_GROUPS
from fetch_planning.kinematics import create_ik_solver
from fetch_planning.types import PinkIKConfig, SE3Pose

G = JOINT_GROUPS
SEED = np.concatenate([HOME_JOINTS[G["torso"]], HOME_JOINTS[G["arm"]]])

config = PinkIKConfig(
    lm_damping=1e-3,
    com_cost=0.1,                                    # CoM stability
    camera_frame="head_camera_rgb_optical_frame",    # head camera frame
    camera_cost=0.1,                                 # camera stabilization
    max_iterations=200,
)

solver = create_ik_solver("arm_with_torso", backend="pink", config=config)

home_pose = solver.fk(SEED)
target = SE3Pose(home_pose.position + np.array([0.30, 0.0, 0.0]), home_pose.rotation)

result = solver.solve_constrained(target, seed=SEED)
if result.success:
    print(f"Solved in {result.iterations} iterations")
    print(f"Position error: {result.position_error * 1000:.2f} mm")
```

```bash
pixi run -e dev python examples/ik/constrained_vis.py
```

---

## Motion Planning around Obstacles

The planner factory returns an OMPL-backed geometric planner with a
SIMD collision backend.  Pass a point cloud (or any supported
obstacle primitive) at construction time and the planner will treat
every point as a sphere of ``point_radius`` during edge validation.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/motion_planning.mp4" type="video/mp4">
</video>

```python
import numpy as np
import trimesh

from fetch_planning.fetch import HOME_JOINTS, fetch_robot_config
from fetch_planning.envs.pybullet_env import PyBulletEnv
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig

cloud = np.asarray(trimesh.load("table.ply").vertices, dtype=np.float32)

planner = create_planner(
    "fetch_arm",
    config=PlannerConfig(
        planner_name="rrtc",
        time_limit=2.0,
        point_radius=0.012,
    ),
    base_config=HOME_JOINTS.copy(),
    pointcloud=cloud,
)

start = planner.extract_config(HOME_JOINTS)
goal = planner.sample_valid()

result = planner.plan(start, goal)
if result.success:
    env = PyBulletEnv(fetch_robot_config, visualize=True)
    env.add_pointcloud(cloud)
    env.animate_path(planner.embed_path(result.path))
```

The runnable version lives at `examples/planning/motion.py`
and uses the bundled `table.ply` — no downloads needed.

```bash
pixi run -e dev python examples/planning/motion.py
pixi run -e dev python examples/planning/motion.py --planner_name bitstar --time_limit 3
```

---

## Subgroup Planning

Plan for just the joints you care about — arm, arm + torso, base,
base + arm, or the whole body — while the rest of the robot stays
pinned to whatever 11-DOF configuration you pass in as `base_config`.
The C++ collision checker injects the inactive joints on every state
and edge query, so any plan you get back is valid for the *whole*
robot, not just the subgroup.

Subgroups that include the mobile base joints (`fetch_base`,
`fetch_base_arm`, `fetch_whole_body`) automatically switch to
OMPL multilevel planning with Dubins (or Reeds-Shepp) curves on the
SE(2) base — so the returned base path respects the nonholonomic
differential-drive constraint.

### Available subgroups

| Category | Subgroup | DOF |
|---|---|---|
| Mobile base | `fetch_base` | 3 |
| Arm | `fetch_arm` | 7 |
| Arm + torso | `fetch_arm_with_torso` | 8 |
| Base + arm | `fetch_base_arm` | 10 |
| Whole body | `fetch_whole_body` (alias `fetch`) | 11 |

### Basic usage

```python
from fetch_planning.fetch import HOME_JOINTS
from fetch_planning.planning import create_planner
from fetch_planning.types import PlannerConfig

# Pin the rest of the body to whatever pose you like — HOME_JOINTS,
# a live robot state, or any custom 11-DOF array.
base_cfg = HOME_JOINTS.copy()

planner = create_planner(
    "fetch_arm",
    config=PlannerConfig(planner_name="rrtc"),
    base_config=base_cfg,
)

start = planner.extract_config(base_cfg)
goal = planner.sample_valid()

result = planner.plan(start, goal)
if result.success:
    # Map the subgroup path back to full-body configurations.
    full_path = planner.embed_path(result.path)
```

```bash
pixi run -e dev python examples/planning/subgroup.py
pixi run -e dev python examples/planning/subgroup.py --planner_name prm
```

---

## Nonholonomic Whole-Body Planning

For tasks that need the base and arm to move simultaneously, drive
the `fetch_whole_body` subgroup.  OMPL's multilevel framework plans
the SE(2) base pose on the lower level (Dubins / Reeds-Shepp) and
lifts the result into the 11-DOF compound space on the upper level,
so the returned path has a forward-only (or reverse-capable) base
trajectory with the arm changing shape alongside it.

```bash
pixi run -e dev python examples/planning/multilevel_single_step.py
pixi run -e dev python examples/planning/nonholonomic.py
```

`multilevel_single_step.py` is a single short-horizon plan you can
inspect visually.  `nonholonomic.py` is a multi-stage pick-and-place
demo that reuses a single planner across twelve navigation /
manipulation segments.

---

## Constrained Planning (Task-space Manifolds)

A constrained planner treats a user-supplied CasADi residual
`h(q) = 0` as a hard equality and runs OMPL's
`ProjectedStateSpace` — every sampled state is projected onto the
manifold before validation, so the returned path is guaranteed to
satisfy the constraint.  The residual can encode any task-space
equation you can express symbolically: a plane, a rail, an
orientation lock, a contact constraint, a handover coupling.

Constrained planning uses an arm-only subgroup (`fetch_arm`) — the
nonholonomic base is a different state space and cannot be combined
with a `ProjectedStateSpace` wrapper.

All five demos live in `examples/planning/constrained/` and share a
tiny scaffold (`_shared.py`) so each file only contains the one
interesting line: the residual itself.  Run any of them with:

```bash
pixi run -e dev python examples/planning/constrained/plane.py
pixi run -e dev python examples/planning/constrained/plane_with_obstacle.py
pixi run -e dev python examples/planning/constrained/line_horizontal.py
pixi run -e dev python examples/planning/constrained/line_vertical.py
pixi run -e dev python examples/planning/constrained/orientation_lock.py
```

```python
from fetch_planning.planning import Constraint, SymbolicContext, create_planner

ctx = SymbolicContext("fetch_arm")
p0 = ctx.evaluate_link_pose("gripper_link", start)[:3, 3]

plane = Constraint(
    residual=ctx.link_translation("gripper_link")[2] - float(p0[2]),
    q_sym=ctx.q,
    name="plane_z",
)

planner = create_planner("fetch_arm", constraints=[plane])
```

---

## Cost-space Planning

The soft counterpart: each demo under `examples/planning/cost/` mirrors
the constrained version but feeds the squared residual as a scalar
cost rather than a hard constraint.  RRT\* and the informed tree
family integrate the cost along every motion.

```bash
pixi run -e dev python examples/planning/cost/plane.py
pixi run -e dev python examples/planning/cost/plane_with_obstacle.py
pixi run -e dev python examples/planning/cost/line_horizontal.py
pixi run -e dev python examples/planning/cost/line_vertical.py
pixi run -e dev python examples/planning/cost/orientation_lock.py
```

---

## Time Parameterization

Given a planned path, produce a time-stamped trajectory respecting
per-joint velocity and acceleration limits via TOTG (Kunz-Stilman).

```bash
pixi run -e dev python examples/planning/time_parameterization.py
```

```python
from fetch_planning.trajectory import TimeOptimalParameterizer

param = TimeOptimalParameterizer(vel_limits, acc_limits)
traj = param.parameterize(path)
times, positions, velocities, accelerations = traj.sample_uniform(dt=0.01)
```

---

## Rendering the videos

Every clip on this page is rendered by one driver script that boots
a headless ``PyBulletEnv`` (``DIRECT`` mode), runs each demo
end-to-end, and pipes offscreen-rendered frames into ``ffmpeg`` via
:class:`fetch_planning.utils.video_recorder.VideoRecorder`.

```bash
# Render everything:
pixi run python scripts/render_videos/render_docs_videos.py

# Render just one clip:
pixi run python scripts/render_videos/render_docs_videos.py --only motion_planning
```

Output lands in ``docs/assets/*.mp4`` — idempotent, no external
download or GPU required.

---

## Reference

### Kinematic chains

| Chain | DOF | Description |
|-------|-----|-------------|
| `arm` | 7 | `torso_lift_link` → `gripper_link` |
| `arm_with_torso` | 8 | `base_link` → `gripper_link` (torso + arm) |
| `whole_body` | 11 | `base_link` → `gripper_link` (base + torso + arm) |

### Joint groups

Indices into the full 11-DOF configuration:

| Group | Indices | Joints |
|-------|---------|--------|
| `base` | 0–2 | `base_x_joint`, `base_y_joint`, `base_theta_joint` |
| `torso` | 3 | `torso_lift_joint` |
| `arm` | 4–10 | `shoulder_pan_joint` → `wrist_roll_joint` (7 DOF) |

### IK solve types

| Type | Description |
|------|-------------|
| `SolveType.SPEED` | Return first valid solution (fastest) |
| `SolveType.DISTANCE` | Minimize joint displacement from seed |
| `SolveType.MANIP1` | Maximize manipulability (product of singular values) |
| `SolveType.MANIP2` | Maximize isotropy (min/max singular value ratio) |
