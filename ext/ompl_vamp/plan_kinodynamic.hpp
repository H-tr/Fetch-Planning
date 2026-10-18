/**
 * FLASK kinodynamic planner: RRT-Connect in the flat state space with
 * closed-form cubic edges and SIMD-batched collision checking.
 *
 * Follows Duong et al., "Ultrafast Sampling-based Kinodynamic Planning
 * via Differential Flatness" (T-RO 2026):
 *
 *   * Tree nodes are flat states ``z = (y, y_dot)``; see
 *     ``flat_trajectory.hpp`` for the flat outputs of a Fetch-style
 *     mobile manipulator.
 *   * Every edge is the closed-form LQMT motion — the cubic Hermite
 *     between its endpoint states — with the cost-optimal duration T*
 *     (Eq. 26/28), optionally stretched until the exact per-joint
 *     velocity / acceleration bounds hold ("limit-aware duration").
 *   * Edge validation evaluates the closed form at time samples, runs
 *     the kinematic checks (joint limits, base speed / yaw rate /
 *     tangential acceleration) and packs the configurations into VAMP
 *     ``ConfigurationBlock<rake>`` batches for ``fkcc`` (Alg. 3).
 *   * The solution is simplified like vamp::planning::simplify does for
 *     geometric paths: greedy shortcuts (Alg. 4) and subdivide-then-cut-
 *     corners smoothing, each change kept only if it lowers the LQMT cost.
 *
 * The paper's unicycle model leaves the base heading undefined at rest
 * (x_dot = y_dot = 0), which is exactly where a mobile manipulator
 * starts and stops.  We make rest states first-class:
 *
 *   * A *rest node* stores a parked heading.  Edges leaving (entering)
 *     it must depart (arrive) along that heading, forward or reverse.
 *     When a tree grows out of a rest node, the sampled target's lateral
 *     base velocity is projected so the cubic's initial (final)
 *     acceleration is aligned — a single linear condition,
 *     ``n . v_target = 3 n . dp / T``.
 *   * A *moving node* stores a gear (forward / reverse); gears can only
 *     change at rest nodes, so every drive segment has a well-defined
 *     heading ``atan2(y_dot, x_dot) + gear * pi``.
 *   * *Spin* edges rotate a parked base in place (diff-drive) while the
 *     arm follows its own cubic.  Trees occasionally extend a rest node
 *     by spinning to face the sampled target, and connections insert a
 *     spin at a parked end whose heading does not match the cubic —
 *     rotate-translate-rotate between two parked nodes.
 *
 * Arm-only subgroups use the paper's double-integrator formulation
 * unchanged (no rest / gear / spin logic).
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "flat_trajectory.hpp"
#include "validity.hpp"

namespace fetch_planning {

struct KinodynamicSettings {
  /// Per-joint limits for the non-base active DOF (length
  /// ``active_dim - base_dim``), in joint units per s / s^2.
  std::vector<double> max_velocity;
  std::vector<double> max_acceleration;

  /// Base limits (only used when the subgroup includes the base).
  double base_max_speed = 0.5;             ///< m/s
  double base_max_acceleration = 0.5;      ///< m/s^2, tangential
  double base_max_yaw_rate = 1.0;          ///< rad/s
  double base_max_yaw_acceleration = 1.5;  ///< rad/s^2, spin segments
  bool allow_reverse = true;               ///< permit reverse gear

  /// LQMT time weight.  Effort weights are ``1 / a_max^2`` per flat
  /// output, so for rest-to-rest edges the dominant joint peaks at
  /// ``sqrt(rho)`` times its acceleration limit.
  double rho = 1.0;
  /// Stretch T* until the exact joint velocity / acceleration bounds
  /// hold (a few multiplicative steps).  Off = the paper's pure T*.
  bool limit_aware_duration = true;

  /// Tree extension step (s): longer BVP solutions are truncated at
  /// this duration, keeping the node on the optimal cubic.
  double max_extension_time = 3.0;
  /// Fraction of the velocity limits used when sampling node velocities.
  double velocity_sample_scale = 0.5;
  /// Time constant (s) weighting velocity differences in the
  /// nearest-neighbour metric ``|dy|^2 + lambda^2 |dv|^2``.
  double velocity_metric_weight = 0.3;
  /// Probability of sampling a zero-velocity target (all joints at
  /// rest, base parked).  Rest-to-rest edges are straight lines in
  /// joint space; for the base they also enable cusps / gear changes.
  double rest_sample_probability = 0.3;
  /// Probability of extending a rest node by a spin that faces the target.
  double spin_probability = 0.3;
  /// Maximum departure / arrival misalignment accepted at a rest node
  /// whose arm is moving (it cannot rotate in place there).
  double heading_tolerance = 0.02;

  /// Post-processing after vamp::planning::simplify: repeat greedy
  /// shortcuts and subdivide-then-cut-corners smoothing until neither
  /// improves the trajectory, for at most ``simplify_iterations`` rounds.
  bool simplify = true;
  std::size_t simplify_iterations = 5;
  double simplify_time_limit = 0.05;  ///< s
  std::size_t max_iterations = 200000;
  std::uint64_t seed = 0;  ///< 0 = nondeterministic
};

struct KinodynamicResult {
  bool solved = false;
  flat::FlatTrajectory trajectory;
  int64_t planning_time_ns = 0;  ///< total, including simplification
  int64_t simplify_time_ns = 0;
  double cost = std::numeric_limits<double>::infinity();  ///< LQMT cost
  std::size_t iterations = 0;
  std::size_t start_tree_size = 0;
  std::size_t goal_tree_size = 0;
  std::size_t edges_checked = 0;
};

class FlaskPlanner {
 public:
  FlaskPlanner(const VampEnv& env, const std::vector<int>& active_indices,
               const std::vector<float>& frozen_config, int base_dim,
               const std::vector<double>& lower,
               const std::vector<double>& upper,
               const KinodynamicSettings& settings)
      : env_(env),
        active_indices_(active_indices),
        frozen_config_(frozen_config),
        active_dim_(static_cast<int>(active_indices.size())),
        base_dim_(base_dim),
        has_base_(base_dim > 0),
        n_flat_(has_base_ ? active_dim_ - 1 : active_dim_),
        arm_begin_(has_base_ ? 2 : 0),
        s_(settings) {
    // Flat-space bounds, limits and LQMT weights.
    lo_.resize(n_flat_);
    hi_.resize(n_flat_);
    vmax_.resize(n_flat_);
    amax_.resize(n_flat_);
    weight_.resize(n_flat_);
    for (int f = 0; f < n_flat_; ++f) {
      lo_[f] = lower[active_of(f)];
      hi_[f] = upper[active_of(f)];
      vmax_[f] =
          f < arm_begin_ ? s_.base_max_speed : s_.max_velocity[f - arm_begin_];
      amax_[f] = f < arm_begin_ ? s_.base_max_acceleration
                                : s_.max_acceleration[f - arm_begin_];
      weight_[f] = 1.0 / (amax_[f] * amax_[f]);
    }
    yaw_weight_ =
        1.0 / (s_.base_max_yaw_acceleration * s_.base_max_yaw_acceleration);
    scratch_prev_.resize(n_flat_);
    scratch_cur_.resize(n_flat_);
    rng_.seed(s_.seed ? s_.seed : std::random_device{}());
  }

  auto plan(const std::vector<double>& start_q,
            const std::vector<double>& start_qd,
            const std::vector<double>& goal_q, double time_limit)
      -> KinodynamicResult {
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           t0)
          .count();
    };
    KinodynamicResult result;
    result.trajectory = flat::FlatTrajectory(active_dim_, base_dim_);
    start_tree_ = {make_start(start_q, start_qd)};
    goal_tree_ = {make_goal(goal_q)};

    // Direct BVP first — the "fast path" of RRT-Connect.
    int conn_start = -1, conn_goal = -1;
    Connection conn;
    if (try_connect(0, 0, conn)) conn_start = conn_goal = 0;

    bool a_is_start = true;
    while (conn_start < 0 && elapsed() < time_limit &&
           result.iterations < s_.max_iterations) {
      ++result.iterations;
      auto& tree_a = a_is_start ? start_tree_ : goal_tree_;
      auto& tree_b = a_is_start ? goal_tree_ : start_tree_;

      const Node target = sample_state();
      const int near_a = nearest(tree_a, target);
      const int x = a_is_start ? extend_forward(near_a, target)
                               : extend_backward(near_a, target);
      if (x >= 0) {
        const Node& xn = tree_a[x];
        const int near_b = nearest(tree_b, xn);
        const int si = a_is_start ? x : near_b;
        const int gi = a_is_start ? near_b : x;
        if (try_connect(si, gi, conn)) {
          conn_start = si;
          conn_goal = gi;
          break;
        }
        // Grow the other tree one step toward the new node.
        if (a_is_start) {
          extend_backward(near_b, xn);
        } else {
          extend_forward(near_b, xn);
        }
      }
      // Balance: always grow the smaller tree next.
      a_is_start = start_tree_.size() <= goal_tree_.size();
    }

    result.start_tree_size = start_tree_.size();
    result.goal_tree_size = goal_tree_.size();
    result.edges_checked = edges_checked_;
    if (conn_start < 0) {
      result.planning_time_ns = ns_since(t0);
      return result;
    }

    // The solution path, VAMP style: a list of states, where node k also
    // carries the edge (kind, duration) that reaches it from node k - 1.
    std::vector<Node> path;
    for (int i = conn_start; i >= 0; i = start_tree_[i].parent)
      path.push_back(start_tree_[i]);
    std::reverse(path.begin(), path.end());
    path.insert(path.end(), conn.nodes.begin(), conn.nodes.end());
    for (int j = conn_goal; goal_tree_[j].parent >= 0;
         j = goal_tree_[j].parent) {
      Node next = goal_tree_[goal_tree_[j].parent];
      next.kind = goal_tree_[j].kind;  // goal-tree edges point to the parent
      next.T = goal_tree_[j].T;
      path.push_back(next);
    }

    // Like VAMP, a direct start -> goal connection needs no simplification.
    const bool direct = conn_start == 0 && conn_goal == 0;
    if (s_.simplify && !direct) {
      const auto ts = std::chrono::steady_clock::now();
      simplify(path, ts);
      result.simplify_time_ns = ns_since(ts);
    }

    result.cost = 0.0;
    for (std::size_t k = 1; k < path.size(); ++k) {
      const Edge e =
          rebuild_edge(path[k - 1], path[k], path[k].kind, path[k].T);
      result.cost += e.cost;
      result.trajectory.push(e.seg);
    }
    result.solved = true;
    result.edges_checked = edges_checked_;
    result.planning_time_ns = ns_since(t0);
    return result;
  }

 private:
  struct Node {
    std::vector<double> y;  ///< flat position
    std::vector<double> v;  ///< flat velocity
    int parent = -1;
    bool rest = true;      ///< base at rest (always true without a base)
    int gear = 0;          ///< moving base: 0 forward, 1 reverse
    double heading = 0.0;  ///< base heading (parked heading at rest)
    flat::SegmentKind kind = flat::SegmentKind::kDrive;  ///< edge to parent
    double T = 0.0;                                      ///< edge duration
  };

  struct Edge {
    flat::Segment seg;
    int gear = 0;
    double heading_a = 0.0;
    double heading_b = 0.0;
    double cost = 0.0;
  };

  /// Endpoint of an edge under construction.  ``fixed`` endpoints are
  /// existing tree nodes whose gear / heading constrain the edge; free
  /// endpoints receive them from the edge.
  struct End {
    const Node* node;
    bool fixed;
  };

  /// A validated connection a -> b: the nodes after ``a`` (parked
  /// rotate-in-place ends, then ``b``), each carrying its incoming edge.
  struct Connection {
    std::vector<Node> nodes;
    double cost = 0.0;
  };

  static constexpr double kPosEps = 1e-6;
  static constexpr double kAccEps = 1e-6;
  /// Below this speed a moving node's heading is ill-conditioned.
  static constexpr double kMinMovingSpeed = 1e-3;
  /// Headings closer than this are "equal" (rounding only).
  static constexpr double kHeadingEps = 1e-9;
  /// Grid for the sampled planar checks (s); the same pass measures the
  /// edge's config-space arc length, which sets the collision density.
  static constexpr double kKinematicDt = 0.05;
  /// VAMP's straight-edge density (``resolution`` samples per unit of
  /// config-space arc length; 1.1 covers chord sums underestimating the
  /// arc).  Like VAMP's own edges this is collision-free up to the
  /// sampling resolution: on the demo scene ~0.4% of whole-body
  /// trajectories graze an obstacle for ~10 ms between two samples.
  /// Inflate obstacles (``point_radius``) for a clearance margin.
  static constexpr double kCollisionDensity = 1.1;

  const VampEnv& env_;
  const std::vector<int>& active_indices_;
  const std::vector<float>& frozen_config_;
  int active_dim_;
  int base_dim_;
  bool has_base_;
  int n_flat_;
  int arm_begin_;
  KinodynamicSettings s_;
  std::vector<double> lo_, hi_, vmax_, amax_, weight_;
  double yaw_weight_ = 1.0;
  std::mt19937_64 rng_;
  std::uniform_real_distribution<double> unit_{0.0, 1.0};
  std::vector<Node> start_tree_, goal_tree_;
  std::vector<double> scratch_prev_, scratch_cur_;
  std::size_t edges_checked_ = 0;

  static auto ns_since(std::chrono::steady_clock::time_point t) -> int64_t {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - t)
        .count();
  }

  auto active_of(int f) const -> int {
    return (has_base_ && f >= 2) ? f + 1 : f;
  }

  auto arm_at_rest(const Node& n) const -> bool {
    for (int f = arm_begin_; f < n_flat_; ++f)
      if (n.v[f] != 0.0) return false;
    return true;
  }

  // ── Start / goal / sampling ────────────────────────────────────────

  /// The planar start velocity is projected onto the heading (a
  /// diff-drive cannot move sideways).
  auto make_start(const std::vector<double>& q,
                  const std::vector<double>& qd) const -> Node {
    Node n;
    n.y.resize(n_flat_);
    n.v.assign(n_flat_, 0.0);
    for (int f = 0; f < n_flat_; ++f) {
      n.y[f] = q[active_of(f)];
      if (!qd.empty()) n.v[f] = qd[active_of(f)];
    }
    if (!has_base_) return n;
    n.heading = flat::wrap_angle(q[2]);
    const double c = std::cos(n.heading), s = std::sin(n.heading);
    const double along = c * n.v[0] + s * n.v[1];
    n.rest = std::abs(along) < 1e-6;
    n.gear = along < 0.0 ? 1 : 0;
    n.v[0] = n.rest ? 0.0 : along * c;
    n.v[1] = n.rest ? 0.0 : along * s;
    return n;
  }

  auto make_goal(const std::vector<double>& q) const -> Node {
    Node n;
    n.y.resize(n_flat_);
    n.v.assign(n_flat_, 0.0);
    for (int f = 0; f < n_flat_; ++f) n.y[f] = q[active_of(f)];
    if (has_base_) n.heading = flat::wrap_angle(q[2]);
    return n;
  }

  auto sample_state() -> Node {
    Node n;
    n.y.resize(n_flat_);
    n.v.resize(n_flat_);
    for (int f = 0; f < n_flat_; ++f)
      n.y[f] = lo_[f] + (hi_[f] - lo_[f]) * unit_(rng_);
    // Zero-velocity targets make rest-to-rest edges straight lines in
    // joint space (every Hermite dim shares the 3s^2 - 2s^3 profile), so
    // the planner degrades gracefully to geometric RRT-Connect where
    // overshooting cubics keep colliding; for the base they also give
    // cusps (gear changes).
    const bool zero_velocity = unit_(rng_) < s_.rest_sample_probability;
    for (int f = arm_begin_; f < n_flat_; ++f) {
      if (zero_velocity) {
        n.v[f] = 0.0;
        continue;
      }
      const double v =
          (2.0 * unit_(rng_) - 1.0) * s_.velocity_sample_scale * vmax_[f];
      // A joint moving toward its limit must be able to brake before it:
      // |v| <= sqrt(2 a d).  Faster states overshoot the limit whatever
      // the next edge does.
      const double room = v > 0.0 ? hi_[f] - n.y[f] : n.y[f] - lo_[f];
      const double vcap = std::sqrt(2.0 * amax_[f] * room);
      n.v[f] = std::clamp(v, -vcap, vcap);
    }
    if (has_base_) {
      n.rest = zero_velocity;
      const double sp = n.rest ? 0.0
                               : s_.base_max_speed * s_.velocity_sample_scale *
                                     std::sqrt(unit_(rng_));
      const double dir = 2.0 * M_PI * unit_(rng_);
      n.v[0] = sp * std::cos(dir);
      n.v[1] = sp * std::sin(dir);
    }
    return n;
  }

  auto metric(const Node& a, const Node& b) const -> double {
    const double lam2 = s_.velocity_metric_weight * s_.velocity_metric_weight;
    double d = 0.0;
    for (int f = 0; f < n_flat_; ++f) {
      const double dy = a.y[f] - b.y[f];
      const double dv = a.v[f] - b.v[f];
      d += dy * dy + lam2 * dv * dv;
    }
    return d;
  }

  auto nearest(const std::vector<Node>& tree, const Node& target) const -> int {
    int best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tree.size(); ++i) {
      const double d = metric(tree[i], target);
      if (d < best_d) {
        best_d = d;
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  // ── Durations ──────────────────────────────────────────────────────

  auto lqmt_terms(const Node& a, const Node& b, bool spin, double heading_a,
                  double heading_b) const -> flat::LqmtTerms {
    flat::LqmtTerms k;
    for (int f = spin ? arm_begin_ : 0; f < n_flat_; ++f)
      k.add(weight_[f], a.y[f], a.v[f], b.y[f], b.v[f]);
    if (spin)
      k.add(yaw_weight_, 0.0, 0.0, flat::wrap_angle(heading_b - heading_a),
            0.0);
    return k;
  }

  /// Largest ratio "achieved / allowed" over the exactly-checkable
  /// bounds of an edge (arm joints; base per-axis speed / acceleration;
  /// spin yaw rate / acceleration).  Velocity ratios scale ~1/T and
  /// acceleration ratios ~1/T^2, so the returned value is the factor by
  /// which T should grow.
  auto duration_stretch(const Node& a, const Node& b, double T, bool spin,
                        double heading_a, double heading_b) const -> double {
    double ratio = 0.0;
    for (int f = arm_begin_; f < n_flat_; ++f) {
      const auto c = flat::Cubic::hermite(a.y[f], a.v[f], b.y[f], b.v[f], T);
      ratio = std::max(ratio, c.max_abs_vel(T) / vmax_[f]);
      ratio = std::max(ratio, std::sqrt(c.max_abs_acc(T) / amax_[f]));
    }
    if (!has_base_) return ratio;
    if (spin) {
      const auto th = flat::Cubic::hermite(
          0.0, 0.0, flat::wrap_angle(heading_b - heading_a), 0.0, T);
      ratio = std::max(ratio, th.max_abs_vel(T) / s_.base_max_yaw_rate);
      ratio = std::max(
          ratio, std::sqrt(th.max_abs_acc(T) / s_.base_max_yaw_acceleration));
    } else {
      // Per-axis maxima bound the planar ones from below: use the larger
      // axis as a cheap, optimistic proxy.
      const auto cx = flat::Cubic::hermite(a.y[0], a.v[0], b.y[0], b.v[0], T);
      const auto cy = flat::Cubic::hermite(a.y[1], a.v[1], b.y[1], b.v[1], T);
      ratio = std::max(ratio, std::max(cx.max_abs_vel(T), cy.max_abs_vel(T)) /
                                  s_.base_max_speed);
      ratio = std::max(
          ratio, std::sqrt(std::max(cx.max_abs_acc(T), cy.max_abs_acc(T)) /
                           s_.base_max_acceleration));
    }
    return ratio;
  }

  auto choose_duration(const Node& a, const Node& b, bool spin,
                       double heading_a, double heading_b) const -> double {
    double T = flat::optimal_time(lqmt_terms(a, b, spin, heading_a, heading_b),
                                  s_.rho);
    if (!s_.limit_aware_duration) return T;
    for (int it = 0; it < 6; ++it) {
      const double r = duration_stretch(a, b, T, spin, heading_a, heading_b);
      if (r <= 1.0) break;
      T *= std::min(std::max(r, 1.05), 3.0);
    }
    return T;
  }

  // ── Edge construction ──────────────────────────────────────────────

  /// Build the drive (cubic) edge ``a -> b`` of duration T, resolving
  /// the gear and rest-node headings.  Returns false when the endpoints
  /// admit no consistent gear / heading; does not run the sampled checks.
  auto build_drive(End ea, End eb, double T, Edge& out) const -> bool {
    const Node& a = *ea.node;
    const Node& b = *eb.node;
    flat::Segment& seg = out.seg;
    seg = flat::Segment{};
    seg.kind = flat::SegmentKind::kDrive;
    seg.T = T;
    seg.cubics.resize(n_flat_);
    for (int f = 0; f < n_flat_; ++f)
      seg.cubics[f] = flat::Cubic::hermite(a.y[f], a.v[f], b.y[f], b.v[f], T);
    out.gear = 0;
    out.cost = lqmt_terms(a, b, false, 0.0, 0.0).cost(T, s_.rho);
    if (!has_base_) return true;

    if (a.rest && b.rest &&
        std::hypot(b.y[0] - a.y[0], b.y[1] - a.y[1]) < kPosEps) {
      // Parked base: the arm moves, the heading cannot change.
      if (ea.fixed && eb.fixed &&
          std::abs(flat::wrap_angle(a.heading - b.heading)) >
              s_.heading_tolerance)
        return false;
      const double h = ea.fixed ? a.heading : b.heading;
      seg.base_stationary = true;
      seg.cubics[0] = flat::Cubic::constant(a.y[0]);
      seg.cubics[1] = flat::Cubic::constant(a.y[1]);
      seg.heading0 = seg.heading1 = out.heading_a = out.heading_b = h;
      return true;
    }

    int gear = -1;
    auto require = [&](int g) {
      if (gear == -1) gear = g;
      return gear == g;
    };
    // Gear implied by a fixed rest heading, or -1 if misaligned.
    auto rest_gear = [&](double dir, double heading) {
      const int g = std::cos(dir - heading) >= 0.0 ? 0 : 1;
      return std::abs(flat::wrap_angle(dir + g * M_PI - heading)) <=
                     s_.heading_tolerance
                 ? g
                 : -1;
    };
    const flat::Cubic& cx = seg.cubics[0];
    const flat::Cubic& cy = seg.cubics[1];
    double dep = 0.0, arr = 0.0;
    if (a.rest) {
      if (std::hypot(cx.acc(0.0), cy.acc(0.0)) < kAccEps) return false;
      dep = std::atan2(cy.acc(0.0), cx.acc(0.0));
      if (ea.fixed && !require(rest_gear(dep, a.heading))) return false;
    } else {
      if (std::hypot(a.v[0], a.v[1]) < kMinMovingSpeed) return false;
      if (ea.fixed && !require(a.gear)) return false;
    }
    if (b.rest) {
      if (std::hypot(cx.acc(T), cy.acc(T)) < kAccEps) return false;
      arr = std::atan2(-cy.acc(T), -cx.acc(T));
      if (eb.fixed && !require(rest_gear(arr, b.heading))) return false;
    } else {
      if (std::hypot(b.v[0], b.v[1]) < kMinMovingSpeed) return false;
      if (eb.fixed && !require(b.gear)) return false;
    }
    if (gear < 0) gear = 0;
    if (gear == 1 && !s_.allow_reverse) return false;

    seg.gear = out.gear = gear;
    out.heading_a =
        a.rest ? (ea.fixed ? a.heading : flat::wrap_angle(dep + gear * M_PI))
               : flat::heading_from_direction(a.v[0], a.v[1], gear);
    out.heading_b =
        b.rest ? (eb.fixed ? b.heading : flat::wrap_angle(arr + gear * M_PI))
               : flat::heading_from_direction(b.v[0], b.v[1], gear);
    seg.heading0 = out.heading_a;
    seg.heading1 = out.heading_b;
    return true;
  }

  /// Rotate-in-place edge between two rest nodes at the same (x, y).
  auto build_spin(const Node& a, const Node& b, double T) const -> Edge {
    Edge out;
    flat::Segment& seg = out.seg;
    seg.kind = flat::SegmentKind::kSpin;
    seg.T = T;
    seg.cubics.resize(n_flat_);
    seg.cubics[0] = flat::Cubic::constant(a.y[0]);
    seg.cubics[1] = flat::Cubic::constant(a.y[1]);
    for (int f = arm_begin_; f < n_flat_; ++f)
      seg.cubics[f] = flat::Cubic::hermite(a.y[f], a.v[f], b.y[f], b.v[f], T);
    seg.theta = flat::Cubic::hermite(
        a.heading, 0.0, a.heading + flat::wrap_angle(b.heading - a.heading),
        0.0, T);
    seg.heading0 = out.heading_a = a.heading;
    seg.heading1 = out.heading_b = b.heading;
    out.cost = lqmt_terms(a, b, true, a.heading, b.heading).cost(T, s_.rho);
    return out;
  }

  /// Rebuild a stored tree edge between two existing nodes.
  auto rebuild_edge(const Node& a, const Node& b, flat::SegmentKind kind,
                    double T) const -> Edge {
    if (kind == flat::SegmentKind::kSpin) return build_spin(a, b, T);
    Edge e;
    build_drive({&a, true}, {&b, true}, T, e);
    return e;
  }

  // ── Edge validation (Alg. 3) ───────────────────────────────────────

  auto validate(const Edge& e) -> bool {
    double length = 0.0;
    if (!validate_kinematics(e, length)) return false;
    const int n = static_cast<int>(std::ceil(kCollisionDensity * length *
                                             Robot::resolution)) +
                  2;
    return collision_free(e.seg, n);
  }

  /// Joint limits (exact on the cubics), spin rate limits (exact), and
  /// the sampled planar checks for drive segments: speed, tangential
  /// acceleration, yaw rate, and heading continuity (a cusp inside an
  /// edge would flip the heading).  Writes the arc length to ``length``.
  auto validate_kinematics(const Edge& e, double& length) -> bool {
    ++edges_checked_;
    const flat::Segment& seg = e.seg;
    constexpr double kTol = 1e-6;
    for (int f = arm_begin_; f < n_flat_; ++f) {
      const flat::Cubic& c = seg.cubics[f];
      const auto [pmin, pmax] = c.pos_range(seg.T);
      if (pmin < lo_[f] - kTol || pmax > hi_[f] + kTol) return false;
      if (c.max_abs_vel(seg.T) > vmax_[f] * (1.0 + kTol)) return false;
      if (c.max_abs_acc(seg.T) > amax_[f] * (1.0 + kTol)) return false;
    }

    const bool spin = has_base_ && seg.kind == flat::SegmentKind::kSpin;
    const bool drive_base = has_base_ &&
                            seg.kind == flat::SegmentKind::kDrive &&
                            !seg.base_stationary;
    if (spin &&
        (seg.theta.max_abs_vel(seg.T) > s_.base_max_yaw_rate * (1.0 + kTol) ||
         seg.theta.max_abs_acc(seg.T) >
             s_.base_max_yaw_acceleration * (1.0 + kTol)))
      return false;
    if (drive_base) {
      for (int f = 0; f < 2; ++f) {
        const auto [pmin, pmax] = seg.cubics[f].pos_range(seg.T);
        if (pmin < lo_[f] - kTol || pmax > hi_[f] + kTol) return false;
      }
    }

    const int n = static_cast<int>(std::ceil(seg.T / kKinematicDt)) + 2;
    const double dt = seg.T / (n - 1);
    const double vmax2 = std::pow(s_.base_max_speed * (1.0 + kTol), 2);
    const double wmax = s_.base_max_yaw_rate * (1.0 + kTol);
    const double amax = s_.base_max_acceleration * (1.0 + kTol);
    const double turn_tol =
        1.5 * s_.base_max_yaw_rate * dt + s_.heading_tolerance + 1e-6;
    const double gear_offset = seg.gear != 0 ? M_PI : 0.0;
    auto& prev = scratch_prev_;
    auto& cur = scratch_cur_;
    double prev_theta = seg.heading0;
    length = 0.0;
    for (int k = 0; k < n; ++k) {
      const double t = (k == n - 1) ? seg.T : k * dt;
      for (int f = 0; f < n_flat_; ++f) cur[f] = seg.cubics[f].pos(t);
      double theta = 0.0;
      if (drive_base) {
        const flat::Cubic& cx = seg.cubics[0];
        const flat::Cubic& cy = seg.cubics[1];
        const double vx = cx.vel(t), vy = cy.vel(t);
        const double ax = cx.acc(t), ay = cy.acc(t);
        const double v2 = vx * vx + vy * vy;
        if (v2 > vmax2) return false;
        const double speed = std::sqrt(v2);
        const double a2 = ax * ax + ay * ay;
        double omega, a_tan;
        if (speed > std::max(flat::kRestSpeed, 1e-7 * std::sqrt(a2) * seg.T)) {
          theta = flat::wrap_angle(std::atan2(vy, vx) + gear_offset);
          omega = (vx * ay - ax * vy) / v2;
          a_tan = (vx * ax + vy * ay) / speed;
        } else {
          // Only the endpoints of an edge may be at rest.
          if (k > 0 && k < n - 1) return false;
          theta = (k == 0) ? seg.heading0 : seg.heading1;
          omega = (ax * cy.jerk() - cx.jerk() * ay) / (2.0 * a2);
          a_tan = std::sqrt(a2);
        }
        if (std::abs(omega) > wmax || std::abs(a_tan) > amax) return false;
        if (std::abs(flat::wrap_angle(theta - prev_theta)) > turn_tol)
          return false;
      } else if (spin) {
        theta = seg.theta.pos(t);
      } else if (has_base_) {
        theta = seg.heading0;
      }
      if (k > 0) {
        double d2 = 0.0;
        for (int f = 0; f < n_flat_; ++f)
          d2 += (cur[f] - prev[f]) * (cur[f] - prev[f]);
        if (has_base_) d2 += std::pow(flat::wrap_angle(theta - prev_theta), 2);
        length += std::sqrt(d2);
      }
      std::swap(prev, cur);
      prev_theta = theta;
    }
    return !drive_base ||
           std::abs(flat::wrap_angle(prev_theta - seg.heading1)) <=
               s_.heading_tolerance + 1e-6;
  }

  /// Interleaved SIMD collision check of ``n`` time samples: block ``i``
  /// holds samples ``{i, i + m, i + 2m, ...}`` so early blocks already
  /// span the whole edge (Alg. 3's spatially distributed batches).
  auto collision_free(const flat::Segment& seg, int n) const -> bool {
    const int m = (n + static_cast<int>(kRake) - 1) / static_cast<int>(kRake);
    std::vector<double> q(active_dim_);
    alignas(vamp::FloatVectorAlignment)
        std::array<float, Robot::dimension * kRake>
            buf{};
    for (int blk = 0; blk < m; ++blk) {
      for (std::size_t lane = 0; lane < kRake; ++lane) {
        const int idx = std::min(static_cast<int>(lane) * m + blk, n - 1);
        flat::eval_segment(seg, seg.T * idx / (n - 1), base_dim_, q.data(),
                           nullptr, nullptr);
        for (std::size_t d = 0; d < Robot::dimension; ++d)
          buf[d * kRake + lane] = frozen_config_[d];
        for (int k = 0; k < active_dim_; ++k)
          buf[active_indices_[k] * kRake + lane] = static_cast<float>(q[k]);
      }
      typename Robot::template ConfigurationBlock<kRake> block(buf.data());
      const bool ok = env_.attachments
                          ? Robot::template fkcc_attach<kRake>(env_, block)
                          : Robot::template fkcc<kRake>(env_, block);
      if (!ok) return false;
    }
    return true;
  }

  // ── Tree growth ────────────────────────────────────────────────────

  /// Project the base velocity of ``target`` so an edge between it and
  /// the rest node ``rest`` departs (``forward``) / arrives aligned with
  /// the parked heading: ``n . v = 3 n . dp / T``.
  void align_with_rest(const Node& rest, Node& target, double T,
                       bool forward) const {
    const double nx = -std::sin(rest.heading), ny = std::cos(rest.heading);
    const double sign = forward ? 1.0 : -1.0;
    const double dpx = sign * (target.y[0] - rest.y[0]);
    const double dpy = sign * (target.y[1] - rest.y[1]);
    const double delta =
        3.0 * (nx * dpx + ny * dpy) / T - (nx * target.v[0] + ny * target.v[1]);
    target.v[0] += delta * nx;
    target.v[1] += delta * ny;
  }

  /// The aligned target velocity depends on the edge duration, and the
  /// (limit-aware) duration depends on the target velocity: iterate,
  /// then project once more for the final ``T`` so the edge built with
  /// exactly that duration is aligned to rounding error.
  void align_with_rest_fixed_point(const Node& rest, Node& target, double& T,
                                   bool forward) const {
    for (int it = 0; it < 10; ++it) {
      align_with_rest(rest, target, T, forward);
      const double T_new = forward
                               ? choose_duration(rest, target, false, 0.0, 0.0)
                               : choose_duration(target, rest, false, 0.0, 0.0);
      if (std::abs(T_new - T) <= 1e-9 * T) break;
      T = T_new;
    }
    align_with_rest(rest, target, T, forward);
  }

  /// Heading a parked base at ``from`` should face to drive toward
  /// ``to`` (``forward``) or to have arrived from ``to``; the reverse
  /// alternative is used when closer to ``current``.
  auto facing(const Node& from, const Node& to, double current,
              bool forward) const -> double {
    const double sign = forward ? 1.0 : -1.0;
    double h =
        std::atan2(sign * (to.y[1] - from.y[1]), sign * (to.y[0] - from.y[0]));
    if (s_.allow_reverse &&
        std::abs(flat::wrap_angle(h - current)) > M_PI / 2.0)
      h = flat::wrap_angle(h + M_PI);
    return h;
  }

  auto want_spin(const Node& n, const Node& target) -> bool {
    return has_base_ && n.rest &&
           std::hypot(target.y[0] - n.y[0], target.y[1] - n.y[1]) > 0.05 &&
           unit_(rng_) < s_.spin_probability;
  }

  /// Usually re-aim a moving target's base velocity within +-45 deg of
  /// the direction of travel ``from -> to``: a unicycle cannot follow a
  /// cubic whose velocity has to swing around faster than the yaw-rate
  /// limit, so uniformly random directions are mostly wasted samples.
  void steer_base_velocity(const Node& from, const Node& to, Node& target) {
    if (!has_base_ || target.rest || unit_(rng_) > 0.7) return;
    const double sp = std::hypot(target.v[0], target.v[1]);
    const double dir = std::atan2(to.y[1] - from.y[1], to.y[0] - from.y[0]) +
                       (unit_(rng_) - 0.5) * (M_PI / 2.0);
    target.v[0] = sp * std::cos(dir);
    target.v[1] = sp * std::sin(dir);
  }

  /// Start tree: grow ``tree[near]`` forward in time toward ``target``.
  auto extend_forward(int near, const Node& target_in) -> int {
    const Node from = start_tree_[near];
    if (want_spin(from, target_in)) {
      const double h = facing(from, target_in, from.heading, true);
      if (std::abs(flat::wrap_angle(h - from.heading)) > s_.heading_tolerance)
        return spin_extend(start_tree_, near, target_in, h, true);
    }
    Node target = target_in;
    steer_base_velocity(from, target, target);
    double T = choose_duration(from, target, false, 0.0, 0.0);
    if (has_base_ && from.rest) {
      if (target.rest) return -1;  // cannot align both ends of a free sample
      align_with_rest_fixed_point(from, target, T, true);
    }
    const double tau = std::min(T, s_.max_extension_time);
    Node x = state_on_edge(from, target, T, tau);
    x.rest = has_base_ && tau >= T && target.rest;
    Edge e;
    if (!build_drive({&from, true}, {&x, false}, tau, e) || !validate(e))
      return -1;
    x.gear = e.gear;
    x.heading = e.heading_b;
    x.parent = near;
    x.kind = flat::SegmentKind::kDrive;
    x.T = tau;
    start_tree_.push_back(std::move(x));
    return static_cast<int>(start_tree_.size()) - 1;
  }

  /// Goal tree: grow ``tree[near]`` backward in time toward ``target``
  /// (the new node precedes ``tree[near]``).
  auto extend_backward(int near, const Node& target_in) -> int {
    const Node to = goal_tree_[near];
    if (want_spin(to, target_in)) {
      const double h = facing(to, target_in, to.heading, false);
      if (std::abs(flat::wrap_angle(h - to.heading)) > s_.heading_tolerance)
        return spin_extend(goal_tree_, near, target_in, h, false);
    }
    Node target = target_in;
    steer_base_velocity(target, to, target);
    double T = choose_duration(target, to, false, 0.0, 0.0);
    if (has_base_ && to.rest) {
      if (target.rest) return -1;
      align_with_rest_fixed_point(to, target, T, false);
    }
    const double tau = std::min(T, s_.max_extension_time);
    Node x = state_on_edge(target, to, T, T - tau);
    x.rest = has_base_ && tau >= T && target.rest;
    Edge e;
    if (!build_drive({&x, false}, {&to, true}, tau, e) || !validate(e))
      return -1;
    x.gear = e.gear;
    x.heading = e.heading_a;
    x.parent = near;
    x.kind = flat::SegmentKind::kDrive;
    x.T = tau;
    goal_tree_.push_back(std::move(x));
    return static_cast<int>(goal_tree_.size()) - 1;
  }

  /// Spin a rest node to heading ``h`` while the arm moves toward the
  /// target's arm configuration (ending at rest).  ``forward``: the new
  /// node follows ``tree[near]``.
  auto spin_extend(std::vector<Node>& tree, int near, const Node& target,
                   double h, bool forward) -> int {
    const Node base = tree[near];
    Node far = base;  // parked at the same (x, y), arm at rest
    far.heading = h;
    for (int f = arm_begin_; f < n_flat_; ++f) {
      far.y[f] = target.y[f];
      far.v[f] = 0.0;
    }
    const Node& a = forward ? base : far;
    const Node& b = forward ? far : base;
    const double T = choose_duration(a, b, true, a.heading, b.heading);
    const double tau = std::min(T, s_.max_extension_time);
    // Truncate on the spin cubics (arm + heading).
    const Edge full = build_spin(a, b, T);
    const double s = forward ? tau : T - tau;
    Node x = far;
    for (int f = arm_begin_; f < n_flat_; ++f) {
      x.y[f] = full.seg.cubics[f].pos(s);
      x.v[f] = full.seg.cubics[f].vel(s);
    }
    x.heading = flat::wrap_angle(full.seg.theta.pos(s));
    // Parked nodes keep their arm at rest so a later residual rotation
    // is always possible; a truncated spin would leave the arm moving.
    if (tau < T && !arm_at_rest(x)) return -1;
    const Edge e =
        forward ? build_spin(base, x, tau) : build_spin(x, base, tau);
    if (!validate(e)) return -1;
    x.parent = near;
    x.kind = flat::SegmentKind::kSpin;
    x.T = tau;
    tree.push_back(std::move(x));
    return static_cast<int>(tree.size()) - 1;
  }

  /// Flat state at time ``s`` on the cubic from ``a`` to ``b`` (duration
  /// T); exactly ``a`` / ``b`` at the ends so rest states stay at rest.
  auto state_on_edge(const Node& a, const Node& b, double T, double s) const
      -> Node {
    Node x;
    x.rest = false;
    if (s >= T || s <= 0.0) {
      x.y = s >= T ? b.y : a.y;
      x.v = s >= T ? b.v : a.v;
      return x;
    }
    x.y.resize(n_flat_);
    x.v.resize(n_flat_);
    for (int f = 0; f < n_flat_; ++f) {
      const auto c = flat::Cubic::hermite(a.y[f], a.v[f], b.y[f], b.v[f], T);
      x.y[f] = c.pos(s);
      x.v[f] = c.vel(s);
    }
    return x;
  }

  // ── Connections ────────────────────────────────────────────────────

  /// Connect start-tree node ``si`` to goal-tree node ``gi``.
  auto try_connect(int si, int gi, Connection& out) -> bool {
    return connect_nodes(start_tree_[si], goal_tree_[gi], out);
  }

  /// Connect ``a -> b`` with the closed-form BVP.  For the base, a parked
  /// endpoint whose departure (arrival) direction does not match its
  /// heading gets a rotate-in-place segment first (last) — the diff-drive
  /// equivalent of the paper's single BVP edge.  Between two parked
  /// nodes this is rotate-translate-rotate.
  auto connect_nodes(const Node& a, const Node& b, Connection& out) -> bool {
    out.nodes.clear();
    out.cost = 0.0;
    Edge e;
    // Append ``n`` reached by the validated edge ``e``.
    auto push = [&](Node n, const Edge& edge) {
      n.kind = edge.seg.kind;
      n.T = edge.seg.T;
      out.cost += edge.cost;
      out.nodes.push_back(std::move(n));
    };
    const bool same_xy = has_base_ && a.rest && b.rest &&
                         std::hypot(b.y[0] - a.y[0], b.y[1] - a.y[1]) < kPosEps;
    if (!has_base_ || same_xy) {
      const bool spin =
          same_xy &&
          std::abs(flat::wrap_angle(b.heading - a.heading)) > kHeadingEps;
      const double T = choose_duration(a, b, spin, a.heading, b.heading);
      if (spin) {
        e = build_spin(a, b, T);
      } else if (!build_drive({&a, true}, {&b, true}, T, e)) {
        return false;
      }
      if (!validate(e)) return false;
      push(b, e);
      return true;
    }

    const double T = choose_duration(a, b, false, 0.0, 0.0);
    const auto cx = flat::Cubic::hermite(a.y[0], a.v[0], b.y[0], b.v[0], T);
    const auto cy = flat::Cubic::hermite(a.y[1], a.v[1], b.y[1], b.v[1], T);
    const double dep = std::atan2(cy.acc(0.0), cx.acc(0.0));
    const double arr = std::atan2(-cy.acc(T), -cx.acc(T));
    // Pick the gear needing the least rotation at the parked ends.
    int gear = -1;
    double best = std::numeric_limits<double>::infinity();
    double spin_a = 0.0, spin_b = 0.0;
    for (int g = 0; g < (s_.allow_reverse ? 2 : 1); ++g) {
      if ((!a.rest && a.gear != g) || (!b.rest && b.gear != g)) continue;
      const double sa =
          a.rest ? flat::wrap_angle(dep + g * M_PI - a.heading) : 0.0;
      const double sb =
          b.rest ? flat::wrap_angle(b.heading - (arr + g * M_PI)) : 0.0;
      if (std::abs(sa) + std::abs(sb) < best) {
        best = std::abs(sa) + std::abs(sb);
        gear = g;
        spin_a = sa;
        spin_b = sb;
      }
    }
    if (gear < 0) return false;

    // Any residual misalignment at a parked end becomes a (possibly tiny)
    // rotation, so the heading stays exactly continuous.  Rotating in
    // place needs the arm at rest at that node; otherwise only a
    // misalignment within ``heading_tolerance`` is accepted.
    auto needs_rotation = [&](const Node& n, double spin) {
      return n.rest && std::abs(spin) > kHeadingEps &&
             (arm_at_rest(n) || std::abs(spin) > s_.heading_tolerance);
    };
    const bool rot_a = needs_rotation(a, spin_a);
    const bool rot_b = needs_rotation(b, spin_b);
    if ((rot_a && !arm_at_rest(a)) || (rot_b && !arm_at_rest(b))) return false;
    Node a2 = a, b2 = b;
    if (rot_a) {
      a2.heading = flat::wrap_angle(dep + gear * M_PI);
      e = build_spin(a, a2,
                     choose_duration(a, a2, true, a.heading, a2.heading));
      if (!validate(e)) return false;
      push(a2, e);
    }
    if (rot_b) b2.heading = flat::wrap_angle(arr + gear * M_PI);
    if (!build_drive({&a2, true}, {&b2, true}, T, e) || !validate(e))
      return false;
    if (!rot_b) {
      push(b, e);
      return true;
    }
    push(b2, e);
    e = build_spin(b2, b, choose_duration(b2, b, true, b2.heading, b.heading));
    if (!validate(e)) return false;
    push(b, e);
    return true;
  }

  // ── Simplification (after vamp::planning::simplify) ──────────────
  //
  // Same structure as VAMP's geometric simplifier: the path is a list of
  // states and small operations each report whether they changed it —
  // greedy shortcuts (also the paper's Alg. 4) and subdivide-then-cut-
  // corners smoothing — repeated until neither does.  Edges are closed-
  // form cubics, so a change is kept only when it is valid and lowers
  // the LQMT cost.

  void simplify(std::vector<Node>& path,
                std::chrono::steady_clock::time_point ts) {
    for (std::size_t i = 0; i < s_.simplify_iterations; ++i) {
      bool any = shortcut_path(path, ts);
      any |= smooth_path(path, ts);
      if (!any) break;
    }
  }

  auto out_of_time(std::chrono::steady_clock::time_point ts) const -> bool {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - ts)
               .count() > s_.simplify_time_limit;
  }

  /// Connect node i to the farthest node j it reaches directly
  /// (vamp::planning::shortcut_path).
  auto shortcut_path(std::vector<Node>& path,
                     std::chrono::steady_clock::time_point ts) -> bool {
    bool changed = false;
    for (std::size_t i = 0; i + 2 < path.size(); ++i) {
      for (std::size_t j = path.size() - 1; j > i + 1; --j) {
        if (out_of_time(ts)) return changed;
        if (replace(path, i, j)) {
          changed = true;
          break;
        }
      }
    }
    return changed;
  }

  /// Split every drive edge at its time midpoint, then try to cut each
  /// original interior node by connecting its two neighbours directly
  /// (vamp::planning::smooth_bspline's subdivide + corner cut).
  auto smooth_path(std::vector<Node>& path,
                   std::chrono::steady_clock::time_point ts) -> bool {
    std::vector<Node> split{path.front()};
    std::vector<std::size_t> corners;  // original interior nodes in ``split``
    for (std::size_t k = 1; k < path.size(); ++k) {
      Node b = path[k];
      if (b.kind == flat::SegmentKind::kDrive) {
        split.push_back(midpoint(path[k - 1], b));
        b.T *= 0.5;
      }
      if (k + 1 < path.size()) corners.push_back(split.size());
      split.push_back(std::move(b));
    }
    path = std::move(split);
    // Walk backwards so a replacement never shifts a corner still to visit.
    bool changed = false;
    for (auto it = corners.rbegin(); it != corners.rend() && !out_of_time(ts);
         ++it)
      changed |= replace(path, *it - 1, *it + 1);
    return changed;
  }

  /// Replace nodes ``i+1 .. j-1`` by a direct connection ``i -> j`` if it
  /// is valid and cheaper than the current edges.
  auto replace(std::vector<Node>& path, std::size_t i, std::size_t j) -> bool {
    double old_cost = 0.0;
    for (std::size_t k = i + 1; k <= j; ++k)
      old_cost += edge_cost(path[k - 1], path[k]);
    Connection c;
    if (!connect_nodes(path[i], path[j], c) || c.cost >= old_cost - 1e-9)
      return false;
    path.erase(path.begin() + static_cast<long>(i) + 1,
               path.begin() + static_cast<long>(j) + 1);
    path.insert(path.begin() + static_cast<long>(i) + 1, c.nodes.begin(),
                c.nodes.end());
    return true;
  }

  auto edge_cost(const Node& a, const Node& b) const -> double {
    const bool spin = b.kind == flat::SegmentKind::kSpin;
    return lqmt_terms(a, b, spin, a.heading, b.heading).cost(b.T, s_.rho);
  }

  /// State at the time midpoint of the drive edge ``a -> b``.  Both halves
  /// are sub-arcs of that (validated) cubic.
  auto midpoint(const Node& a, const Node& b) const -> Node {
    const Edge e = rebuild_edge(a, b, b.kind, b.T);
    const double s = 0.5 * b.T;
    Node m;
    m.y.resize(n_flat_);
    m.v.resize(n_flat_);
    for (int f = 0; f < n_flat_; ++f) {
      m.y[f] = e.seg.cubics[f].pos(s);
      m.v[f] = e.seg.cubics[f].vel(s);
    }
    m.kind = flat::SegmentKind::kDrive;
    m.T = s;
    m.gear = e.seg.gear;
    m.rest = !has_base_ || e.seg.base_stationary;
    if (has_base_)
      m.heading =
          m.rest ? e.seg.heading0
                 : flat::heading_from_direction(m.v[0], m.v[1], e.seg.gear);
    return m;
  }
};

}  // namespace fetch_planning
