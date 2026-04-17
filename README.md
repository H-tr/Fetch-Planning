# Fetch-Planning

A whole-body motion planning library for the **Fetch mobile manipulator**.
Mirrors the [Autolife-Planning](https://github.com/H-tr/Autolife-Planning)
interface (same public API, same module layout) but targets the 11-DOF
Fetch robot with a **nonholonomic differential-drive base**, an
**IKFast analytic IK backend**, and **whole-body VAMP collision checking**.

> Status: **early scaffold / under active development.** The public
> interface is stable; build steps and examples below describe the target
> state.

---

## Motivation

Autolife-Planning wraps OMPL over a VAMP collision backend for the Autolife
humanoid. Fetch-Planning is a sister library that ports the same stack to
Fetch. The key differences from a "default Fetch in VAMP" setup are:

1. **Whole-body** — the planner reasons over the **mobile base + torso + arm
   (11 DOF)**, not just the 8-DOF arm_with_torso chain.
2. **Nonholonomic base** — the base `(x, y, θ)` is constrained to
   differential-drive motion (forward/backward along the heading,
   in-place rotation, no lateral sliding). This is enforced in OMPL via a
   `CompoundStateSpace(ReedsSheppStateSpace + RealVectorStateSpace)` so
   every planner automatically respects the curvature constraint.
3. **IKFast backend** — in addition to TRAC-IK and Pink, Fetch-Planning
   ships a pre-generated OpenRAVE `ikfast` analytic solver for Fetch's
   8-DOF `arm_with_torso` chain. IKFast gives ~40 µs / solve and dense
   null-space sweeps, which the `grasp_anywhere` sampler exploits.
4. **`grasp_anywhere` module** — samples grasp poses on a sphere around a
   target object, solves IK from each, returns reachable grasps. Ported
   from [grasp_anywhere_v1](https://github.com/H-tr/mobile_grasping_uncertainty).

Everything else — the public API (`create_ik_solver`, `create_planner`,
`SE3Pose`, `PlannerConfig`, `IKConfig`, constrained planning via CasADi),
the overall module layout, the VAMP/OMPL stack — is byte-identical to
Autolife-Planning so downstream code can switch robots by changing a
single import.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    fetch_planning/ (Python)                    │
│                                                                │
│   create_planner(...)        create_ik_solver(...)             │
│        │                           │                          │
│        ▼                           ▼                           │
│   MotionPlanner            IKSolverBase                        │
│        │                    ├── TracIKSolver  (pytracik)       │
│        │                    ├── PinkIKSolver  (pinocchio+pink) │
│        │                    └── IKFastSolver  (ikfast_fetch)   │
└────────┼───────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────┐
│              _ompl_vamp  (C++/nanobind extension)              │
│                                                                │
│  ┌───────────── OMPL frontend ────────────┐                    │
│  │                                        │                    │
│  │  arm-only subgroups:                   │                    │
│  │    RealVectorStateSpace(N)             │                    │
│  │                                        │                    │
│  │  subgroups touching the base:          │                    │
│  │    CompoundStateSpace                  │                    │
│  │    ├── ReedsSheppStateSpace(ρ=0.2)     │   ← nonholonomic   │
│  │    └── RealVectorStateSpace(arm_dof)   │                    │
│  │                                        │                    │
│  │  Planners: RRT-Connect, KPIECE, PRM*,  │                    │
│  │  BIT*, RRT*, …                         │                    │
│  └────────────────┬───────────────────────┘                    │
│                   │                                            │
│                   ▼                                            │
│   StateValidityChecker + MotionValidator                       │
│                   │                                            │
└───────────────────┼────────────────────────────────────────────┘
                    │
┌───────────────────▼────────────────────────────────────────────┐
│              VAMP backend  (header-only, SIMD)                 │
│                                                                │
│   third_party/vamp  →  H-tr/vamp @ fetch-planning branch       │
│   vamp::robots::FetchWholeBody  (11 DOF spherized model)       │
└────────────────────────────────────────────────────────────────┘
```

---

## Robot layout (11 DOF)

| Index | Joint                    | Group  | Notes                          |
|------:|--------------------------|--------|--------------------------------|
|   0   | `base_x_joint`           | base   | world frame                    |
|   1   | `base_y_joint`           | base   | world frame                    |
|   2   | `base_theta_joint`       | base   | heading (nonholonomic with 0,1)|
|   3   | `torso_lift_joint`       | torso  | prismatic 0 – 0.386 m          |
|   4   | `shoulder_pan_joint`     | arm    |                                |
|   5   | `shoulder_lift_joint`    | arm    |                                |
|   6   | `upperarm_roll_joint`    | arm    | ikfast free-param #2           |
|   7   | `elbow_flex_joint`       | arm    |                                |
|   8   | `forearm_roll_joint`     | arm    |                                |
|   9   | `wrist_flex_joint`       | arm    |                                |
|  10   | `wrist_roll_joint`       | arm    |                                |

Planning subgroups (`fetch_planning.fetch.PLANNING_SUBGROUPS`):

- `fetch_base` — 3 DOF, nonholonomic
- `fetch_arm` — 7 DOF (arm only, torso+base fixed)
- `fetch_arm_with_torso` — 8 DOF (arm + torso, base fixed) **← ikfast chain**
- `fetch_base_arm` — 10 DOF (base + arm, torso fixed)
- `fetch_whole_body` — 11 DOF (everything)

---

## Implementation plan

This repo is being built up in phases. See the [issue tracker](https://github.com/H-tr/Fetch-Planning/issues) for task status.

### Phase 1 — scaffold & public API  ✅ in progress

- [x] Copy Autolife-Planning structure, rename to `fetch_planning/`
- [x] Rewrite `fetch.py` for the Fetch 11-DOF layout
- [x] Vendor Fetch URDFs + spherized meshes under
      `fetch_planning/resources/robot/fetch/`
- [x] `CMakeLists.txt` wired for the three extensions:
      `_ompl_vamp`, `pytracik`, `ikfast_fetch`
- [ ] Public entry points (`create_planner`, `create_ik_solver`)
      importable without errors (requires submodules)

### Phase 2 — VAMP submodule with whole-body Fetch

- [ ] On `H-tr/vamp`, create branch `fetch-planning` from `main`
- [ ] Add `src/impl/vamp/robots/fetch_whole_body.hh` — an 11-DOF
      spherized Fetch model whose `sphere_fk` applies the `(x, y, θ)`
      base transform before the arm FK. Mirrors the pattern used by
      `autolife_body_coupled.hh`.
- [ ] Push branch; wire as `third_party/vamp` submodule pinned to
      `fetch-planning`.

### Phase 3 — OMPL frontend adaptations

- [ ] `ext/ompl_vamp/validity.hpp`: swap `vamp::robots::Autolife` for
      `vamp::robots::FetchWholeBody`; extend `extract_real_state` to
      handle `CompoundStateType` when the base is part of the active
      subgroup.
- [ ] `ext/ompl_vamp/planner.hpp`: when `active_indices` include any
      base joint, construct `CompoundStateSpace(ReedsSheppStateSpace(ρ) +
      RealVectorStateSpace(N))` instead of a plain `RealVectorStateSpace`.
      ReedsSheppStateSpace's built-in distance/interpolate functions make
      any OMPL planner respect the nonholonomic constraint for free.
- [ ] Motion validator for the compound space: reads the base state via
      `state->as<CompoundState>()->as<SE2StateSpace::StateType>(0)`, the
      arm state via `...->as<RealVectorStateSpace::StateType>(1)`, packs
      both into a `FetchWholeBody::Configuration`, and calls VAMP.

### Phase 4 — IK backends

- [x] Vendor
      [`ikfast_fetch_module.cpp`](ext/ikfast_fetch/ikfast_fetch_module.cpp)
      from `grasp_anywhere_v1` (8-DOF `Transform6D`, free params =
      `torso_lift` + `upperarm_roll`)
- [x] `fetch_planning/kinematics/ikfast_solver.py` implementing
      `IKSolverBase` with a seed-aware random-restart sampler over the
      two free parameters
- [x] `create_ik_solver(backend="ikfast")` factory route
- [ ] Keep `trac_ik` + `pink` backends working on Fetch chains
      (`arm`, `arm_with_torso`, `whole_body`)

### Phase 5 — grasp_anywhere module

- [ ] Port `grasp_anywhere_v1/grasp_anywhere/samplers/prepose_sampler.py`
      as `fetch_planning/grasping/grasp_anywhere.py`: uniform sphere
      sampling around a target, IK check (ikfast-backed), collision
      check via the motion planner's validator
- [ ] Public entry point:
      `grasp_anywhere(target_position, *, num_samples, radius, ik_backend="ikfast") -> list[GraspCandidate]`
- [ ] Example: `examples/grasp_anywhere_example.py`

### Phase 6 — examples, build, ship

- [ ] `examples/ik_example.py` — Fetch arm FK/IK round-trip for each of
      the 3 backends
- [ ] `examples/motion_planning_example.py` — `fetch_arm` planning over a
      table pointcloud
- [ ] `examples/whole_body_planning_example.py` — `fetch_whole_body`
      nonholonomic base + arm motion around obstacles
- [ ] `pixi run build` succeeds end-to-end; examples run

---

## Usage (target interface)

```python
import numpy as np
from fetch_planning.kinematics import create_ik_solver
from fetch_planning.planning import create_planner
from fetch_planning.types import SE3Pose, IKConfig, PlannerConfig
from fetch_planning.fetch import HOME_JOINTS

# --- IK (three interchangeable backends) ---
solver = create_ik_solver("arm_with_torso", backend="ikfast")
target = solver.fk(HOME_JOINTS[3:])  # torso + 7 arm joints
target.position[0] += 0.10
result = solver.solve(target, seed=HOME_JOINTS[3:])
print(result.joint_positions)

# --- Whole-body motion planning with nonholonomic base ---
planner = create_planner(
    "fetch_whole_body",
    config=PlannerConfig(planner_name="rrtc", time_limit=5.0),
    pointcloud=table_points,
)

start = HOME_JOINTS.copy()
goal = HOME_JOINTS.copy()
goal[0], goal[1], goal[2] = 1.0, 0.5, 0.3  # drive to new base pose
goal[4] += 0.5                              # and rotate shoulder

result = planner.plan(start, goal)
if result.success:
    for waypoint in result.path:
        print(waypoint)
```

---

## Project layout

```
Fetch-Planning/
├── CMakeLists.txt              # three extensions: _ompl_vamp, pytracik, ikfast_fetch
├── pyproject.toml              # scikit-build-core
├── pixi.toml                   # conda env + build tasks
├── .gitmodules                 # vamp@fetch-planning, ompl
│
├── fetch_planning/             # Python package (mirrors autolife_planning/)
│   ├── fetch.py                # 11-DOF Fetch layout (joint groups, HOME, …)
│   ├── types/                  # SE3Pose, IKConfig, PlannerConfig, …
│   ├── trajectory/             # TOTG time parameterization
│   ├── kinematics/             # TracIK / Pink / IKFast solvers + factory
│   ├── planning/               # MotionPlanner, SymbolicContext, Constraint
│   ├── grasping/               # grasp_anywhere sampler  [Phase 5]
│   ├── envs/                   # PyBullet scene wrapper
│   └── resources/robot/fetch/  # fetch.urdf + fetch_spherized.urdf + meshes
│
├── ext/
│   ├── ompl_vamp/              # nanobind OMPL↔VAMP bridge
│   ├── trac_ik/                # vendored TRAC-IK C++ (pybind11)
│   └── ikfast_fetch/           # ikfast_fetch_module.cpp — OpenRAVE analytic IK
│
└── third_party/
    ├── vamp/                   # H-tr/vamp @ fetch-planning branch
    └── ompl/                   # ompl/ompl (upstream)
```

---

## Dependencies

- **Build**: cmake ≥ 3.20, ninja, C++17 compiler, nanobind, pybind11,
  Boost (serialization + filesystem + system), Eigen3, LAPACK
- **Runtime (conda)**: python ≥ 3.10, numpy, scipy, pinocchio, pink,
  orocos-kdl, nlopt
- **Optional**: pybullet (visualization), casadi (constrained planning)

Pixi handles all of this — `pixi install` then `pixi run build`.

---

## License & attribution

- The `ikfast_fetch` analytic solver is generated by OpenRAVE's `ikfast`
  (Apache 2.0, © Rosen Diankov) and originally produced for the
  `grasp_anywhere_v1` codebase.
- VAMP is from the KavrakiLab (BSD); we use the H-tr fork for the
  whole-body Fetch model on the `fetch-planning` branch.
- Everything else © 2026 H-tr, same license as Autolife-Planning.
