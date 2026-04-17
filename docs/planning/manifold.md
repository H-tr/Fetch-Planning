# Manifold Planning

Hard task-space equality constraints. Instead of sampling the full
joint space and hoping the planner threads a path along some
manifold, the planner **samples directly on the manifold** via
OMPL's `ProjectedStateSpace` — every state it ever sees satisfies
the constraint.

You define the constraint as a CasADi expression in terms of the
planner's active joint symbol. The library takes the symbolic
Jacobian, generates C, compiles to a `.so`, and caches it. On a
warm cache every constraint is a single `stat`. The C++ planner
`dlopen`'s the library and projects samples onto `residual = 0`
every time.

Five ready examples are laid out below — the math on the left, the
exact code that defines it on the right. You should be able to
write your own by analogy.

## Plane

Keep the gripper's world z equal to its home value \(z_0\):

<div class="grid" markdown>

=== "Math"

    $$
    r(q) \;=\; p_{\text{tcp},z}(q) \;-\; z_0 \;=\; 0
    $$

    One scalar residual. The planner projects every sampled joint
    vector \(q\) onto the zero set of \(r\).

=== "Code"

    ```python
    p0 = ee_position(ctx, start)

    plane = Constraint(
        residual=ee_translation(ctx)[2] - float(p0[2]),
        q_sym=ctx.q,
        name="plane_z",
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_plane.mp4" type="video/mp4">
</video>

## Plane + obstacle

Same plane constraint, now with a sphere obstacle straddling the
straight-line path. The planner still has to stay on the plane and
route around the obstacle — the collision check runs against a
pointcloud on the inflated sphere.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_plane_obstacle.mp4" type="video/mp4">
</video>

## Horizontal rail

Keep the gripper on a line parallel to world \(x\) and freeze its
two orientation axes (first two columns of \(R(q)\)):

<div class="grid" markdown>

=== "Math"

    $$
    r(q) \;=\;
    \begin{bmatrix}
      p_{\text{tcp},y}(q) - y_0 \\[2pt]
      p_{\text{tcp},z}(q) - z_0 \\[2pt]
      R_{:,0}(q) - R_{0;:,0} \\[2pt]
      R_{:,1}(q) - R_{0;:,1}
    \end{bmatrix} \;=\; 0
    $$

    One 8-row residual: two translation rows pin the rail, six
    rotation rows lock two columns of the gripper rotation matrix.

=== "Code"

    ```python
    p0 = ee_position(ctx, start)
    R0 = np.asarray(
        ctx.evaluate_link_pose(EE_LINK, start)
    )[:3, :3]
    tcp = ee_translation(ctx)
    rot = ctx.link_rotation(EE_LINK)

    line_h = Constraint(
        residual=ca.vertcat(
            tcp[1] - float(p0[1]),
            tcp[2] - float(p0[2]),
            rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        ),
        q_sym=ctx.q,
        name="line_h",
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_line_horizontal.mp4" type="video/mp4">
</video>

## Vertical rail

Same pattern, now the gripper slides along world \(z\) with \(x, y\)
and the orientation axes fixed:

<div class="grid" markdown>

=== "Math"

    $$
    r(q) \;=\;
    \begin{bmatrix}
      p_{\text{tcp},x}(q) - x_0 \\[2pt]
      p_{\text{tcp},y}(q) - y_0 \\[2pt]
      R_{:,0}(q) - R_{0;:,0} \\[2pt]
      R_{:,1}(q) - R_{0;:,1}
    \end{bmatrix} \;=\; 0
    $$

=== "Code"

    ```python
    p0 = ee_position(ctx, start)
    R0 = np.asarray(
        ctx.evaluate_link_pose(EE_LINK, start)
    )[:3, :3]
    tcp = ee_translation(ctx)
    rot = ctx.link_rotation(EE_LINK)

    line_v = Constraint(
        residual=ca.vertcat(
            tcp[0] - float(p0[0]),
            tcp[1] - float(p0[1]),
            rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        ),
        q_sym=ctx.q,
        name="line_v",
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_line_vertical.mp4" type="video/mp4">
</video>

## Orientation lock (translation free)

Let the gripper translate anywhere but fix its orientation exactly:

<div class="grid" markdown>

=== "Math"

    $$
    r(q) \;=\;
    \begin{bmatrix}
      R_{:,0}(q) - R_{0;:,0} \\[2pt]
      R_{:,1}(q) - R_{0;:,1}
    \end{bmatrix} \;=\; 0
    $$

    Two rotation columns pinned — the third is determined by
    orthogonality, so the whole orientation is fixed.

=== "Code"

    ```python
    R0 = np.asarray(
        ctx.evaluate_link_pose(EE_LINK, start)
    )[:3, :3]
    rot = ctx.link_rotation(EE_LINK)

    orient = Constraint(
        residual=ca.vertcat(
            rot[:, 0] - ca.DM(R0[:, 0].tolist()),
            rot[:, 1] - ca.DM(R0[:, 1].tolist()),
        ),
        q_sym=ctx.q,
        name="orient_lock",
    )
    ```

</div>

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_orientation_lock.mp4" type="video/mp4">
</video>

## How to plug it into a planner

```python
planner = create_planner(
    "fetch_arm",
    config=PlannerConfig(time_limit=5.0),
    constraints=[plane],          # or [line_h], [orient], [plane, orient], ...
)

# Both the start and the goal must already lie on the manifold.
goal = ctx.project(seed, plane.residual)
result = planner.plan(start, goal)
```

Multiple constraints are stacked — passing `[plane, orient]` pins
the gripper to a plane *and* freezes its orientation. The example
demos in `examples/planning/constrained/` use `EE_LINK =
"gripper_link"` for orientation and the midpoint of
`l_gripper_finger_link` / `r_gripper_finger_link` for the TCP
translation — see `examples/planning/constrained/_shared.py` for
the helpers.

Manifold constraints are only supported for arm-only subgroups
(`fetch_arm`, `fetch_arm_with_torso`); base-including subgroups
run on the multilevel nonholonomic backend, which does not compose
with `ProjectedStateSpace`.
