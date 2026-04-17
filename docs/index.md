# Fetch Planning

A planning library for the **Fetch robot** — inverse kinematics
(TRAC-IK and Pink QP), motion planning (OMPL frontend, VAMP SIMD
collision backend) with a nonholonomic mobile base, and time-optimal
trajectory generation, all behind a unified Python API.

<video controls autoplay loop muted playsinline width="100%">
  <source src="assets/motion_planning.mp4" type="video/mp4">
</video>

<div class="grid cards" markdown>

-   :material-package-variant:{ .lg .middle } __Drop-in, few deps__

    ---

    For inference: **`numpy`, `scipy`, `pink`**. No conda required, no
    ROS, no MoveIt. Vendor it into any project with a single
    `pip install -e .`.

-   :material-source-branch:{ .lg .middle } __26 planners + multilevel__

    ---

    Full OMPL planner library, not a bespoke subset: RRT-Connect,
    RRT\*, BIT\*, AIT\*, EIT\*, KPIECE, PRM, FMT, SPARS, … plus OMPL
    multilevel planning (QRRT / QRRT\* / QMP / QMP\*) for the
    nonholonomic base hierarchy `SE(2) → SE(2) × R^N`.

-   :material-speedometer:{ .lg .middle } __Microsecond checks, millisecond plans__

    ---

    VAMP SIMD collision backend: **~3 μs per check**. RRT-Connect plans
    collision-free paths for the 7-, 8-, and 11-DOF subgroups with the
    full whole-body checker in the loop.

</div>

## Features

<div class="grid cards" markdown>

-   :material-robot-angry-outline:{ .lg .middle } __Inverse Kinematics__

    ---

    TRAC-IK for unconstrained numerical IK; Pink for QP-based
    constrained IK that composes end-effector tracking with
    centre-of-mass stability, camera-frame stabilization, and
    self-collision avoidance.

-   :material-map-marker-path:{ .lg .middle } [__Motion Planning__](planning/index.md)

    ---

    OMPL frontend + VAMP SIMD backend. Full-body and subgroup planning
    (arm, arm + torso, base, base + arm, whole body) with first-class
    support for point-cloud obstacles and a nonholonomic diff-drive
    base.

-   :material-vector-curve:{ .lg .middle } [__Manifold Planning__](planning/manifold.md)

    ---

    Hard task-space equality constraints — planes, rails, orientation
    locks, couplings — compiled from CasADi expressions and solved on
    OMPL's `ProjectedStateSpace`.

-   :material-chart-bell-curve:{ .lg .middle } [__Cost-space Planning__](planning/cost.md)

    ---

    Path-integral soft costs (orientation preference, pose
    stabilization, …) compiled from CasADi and driving OMPL's
    asymptotically-optimal planners (RRT\*, BIT\*, AIT\*, …).

-   :material-timer-outline:{ .lg .middle } [__Time Parameterization__](planning/trajectory.md)

    ---

    Time-optimal trajectory generation (TOTG) converts geometric paths
    into executable trajectories with per-joint velocity and acceleration
    limits — bridging the planner's output to hardware.

</div>

## Quick Install

For inference — running the planners and IK solvers — just pip install:

```bash
git clone --recursive https://github.com/H-tr/Fetch-Planning.git
cd Fetch-Planning
pip install -e .
```

Three runtime deps: `numpy`, `scipy`, `pink`. No conda, no ROS, no
MoveIt. See the [Getting Started](getting-started.md) guide for the
full development setup (pixi + conda-forge toolchain, URDF rebuilds,
FK codegen).
