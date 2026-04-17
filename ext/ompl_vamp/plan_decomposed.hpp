/**
 * Decomposed whole-body planner: base roadmap + layered arm scheduler.
 *
 * Replaces OMPL's multilevel "single base path → fiber lift" pipeline
 * with two explicit phases:
 *
 *   Phase 1 — generate K diverse base paths in SE(2) (Dubins or
 *     Reeds-Shepp, depending on ``allow_reverse``) by running
 *     RRT-Connect K times with distinct RNG seeds.  Each base path is
 *     collision-checked against the dedicated ``FetchBase`` VAMP model
 *     (14 base-link spheres only), keeping Phase 1 cheap.
 *
 *   Phase 2 — for each base path, resample to M dense waypoints along
 *     the curve, build a layered graph where layer i carries J arm
 *     candidates sampled Gaussian-noised around the linear
 *     interpolation from ``arm_start`` to ``arm_goal``, and run
 *     Dijkstra from (0, arm_start) to (M-1, arm_goal).  Edges are
 *     full-body VAMP motion checks across the 11-DOF whole-body
 *     collision model.  The first base path whose layered graph has a
 *     goal-reaching route wins; remaining candidates are not explored.
 *
 * Why this over OMPL's ``plan_multilevel``:
 *
 *   - **Bounded, explicit retries.**  ``FindSection`` retries nothing
 *     useful when the first base path fails to lift — it falls back to
 *     vanilla QRRT in the 11-D compound space.  Here, each failed
 *     base-candidate attempt is immediately replaced by another diverse
 *     base path, and ``K`` is a hard upper bound on attempts.
 *   - **No coupled high-D sampling.**  Phase 2 is a finite graph
 *     search, not RRT in 11-D; its latency is predictable and its
 *     completeness depends only on J and M, not on OMPL's bundle
 *     sampler.
 *   - **Fast-path preserved.**  The first arm sample in each layer is
 *     the naive linear interpolation — exactly the candidate the
 *     existing multilevel lift tries first.  When that works the
 *     solver exits after a single Dijkstra sweep, matching the ~5 ms
 *     fast path of the existing pipeline.
 *
 * Completeness:
 *
 *   - PC as ``(samples, K, J, M) → ∞`` (PRM-class base roadmap would
 *     tighten this for Phase 1; the RRT-Connect-repeats strategy used
 *     here is sufficient for the demo scene and easy to upgrade later).
 *   - Not jointly AO in the coupled 11-D sense — Phase 2 is
 *     conditioned on Phase 1's choice.  Each phase is AO in isolation.
 */

#pragma once

#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/util/RandomNumbers.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "validity.hpp"

namespace fetch_planning {

namespace ob = ompl::base;
namespace og = ompl::geometric;

struct DecomposedConfig {
  int k_base_candidates = 4;     ///< Phase 1: distinct base-planner seeds to try
  int j_arm_samples = 24;        ///< Phase 2: arm candidates per intermediate layer
  int m_base_waypoints = 48;     ///< Phase 2: base path resample count
  double arm_sigma = 0.8;        ///< Gaussian std (rad) for arm candidate noise
  double base_rrt_range = 0.6;   ///< RRT-Connect extension range at Level 0 (m)
  std::uint32_t seed = 0;        ///< 0 = use default
};

struct DecomposedResult {
  bool solved = false;
  std::vector<std::vector<double>> path;
  int64_t planning_time_ns = 0;
  double path_cost = std::numeric_limits<double>::infinity();
  int base_attempts = 0;          ///< How many base candidates were tried
  int winning_candidate = -1;     ///< Index of the candidate that succeeded
};

// Build a SE(2) base state space (Dubins or Reeds-Shepp) — same builder
// as ``planner.hpp``.  Forward-declared here so the header stays
// standalone if ``plan_multilevel``'s helper is refactored later.
inline auto make_se2_base_space_for_decomposed(double turning_radius,
                                               bool allow_reverse)
    -> std::shared_ptr<ob::SE2StateSpace> {
  if (allow_reverse) {
    return std::make_shared<ob::ReedsSheppStateSpace>(turning_radius);
  }
  return std::make_shared<ob::DubinsStateSpace>(turning_radius,
                                                /*isSymmetric=*/false);
}

namespace decomposed_detail {

inline auto make_full_config(
    const std::array<double, 3>& base_pose, const std::vector<double>& arm,
    const std::vector<int>& active_indices,
    const std::vector<float>& frozen_config, int base_dim, int active_dim)
    -> Robot::Configuration {
  alignas(Robot::Configuration::S::Alignment)
      std::array<float, Robot::Configuration::num_scalars_rounded> buf{};
  std::copy(frozen_config.begin(), frozen_config.end(), buf.begin());
  buf[active_indices[0]] = static_cast<float>(base_pose[0]);
  buf[active_indices[1]] = static_cast<float>(base_pose[1]);
  buf[active_indices[2]] = static_cast<float>(base_pose[2]);
  for (int i = base_dim; i < active_dim; ++i) {
    buf[active_indices[i]] = static_cast<float>(arm[i - base_dim]);
  }
  return Robot::Configuration(buf.data());
}

inline auto validate_full_static(const Robot::Configuration& q,
                                 const VampEnv& env) -> bool {
  return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env);
}

inline auto validate_full_motion(const Robot::Configuration& q1,
                                 const Robot::Configuration& q2,
                                 const VampEnv& env) -> bool {
  return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
      q1, q2, env);
}

inline auto resample_base_path(og::PathGeometric& path, int m)
    -> std::vector<std::array<double, 3>> {
  // PathGeometric::interpolate(count) densifies along the underlying
  // SE(2) curve (Dubins/RS) — the added states follow the non-holonomic
  // local connector, not straight-line SE(2).  Linear motion checks
  // between adjacent waypoints therefore approximate the RS curve
  // closely when m is dense.
  path.interpolate(m);
  std::vector<std::array<double, 3>> out;
  out.reserve(path.getStateCount());
  for (std::size_t i = 0; i < path.getStateCount(); ++i) {
    const auto* se2 = path.getState(i)->as<ob::SE2StateSpace::StateType>();
    out.push_back({se2->getX(), se2->getY(), se2->getYaw()});
  }
  return out;
}

}  // namespace decomposed_detail

/// Run decomposed whole-body planning.  See header comment for the
/// architecture overview.  ``start_active`` / ``goal_active`` are
/// active-subgroup configurations with the base (x, y, theta) as the
/// leading ``base_dim`` entries.
inline auto plan_decomposed(
    bool allow_reverse, double turning_radius,
    const std::array<double, 4>& base_xy_bounds,
    const std::vector<double>& start_active,
    const std::vector<double>& goal_active,
    const std::vector<int>& active_indices,
    const std::vector<float>& frozen_config, int base_dim, int active_dim,
    const VampEnv& env, double time_limit,
    const DecomposedConfig& cfg = {}) -> DecomposedResult {
  using namespace decomposed_detail;

  const auto t0 = std::chrono::steady_clock::now();
  DecomposedResult result;

  const int arm_dim = active_dim - base_dim;
  if (arm_dim <= 0 || base_dim != 3) {
    // Decomposed planning only covers the SE(2)-base + R^N-arm case.
    // Callers for arm-only or base-only subgroups should stay on the
    // existing geometric / multilevel paths.
    return result;
  }

  // ── Phase 1 setup: SE(2) base SpaceInformation ─────────────────────
  auto base_space = make_se2_base_space_for_decomposed(turning_radius,
                                                       allow_reverse);
  {
    ob::RealVectorBounds bnd(2);
    bnd.setLow(0, base_xy_bounds[0]);
    bnd.setHigh(0, base_xy_bounds[1]);
    bnd.setLow(1, base_xy_bounds[2]);
    bnd.setHigh(1, base_xy_bounds[3]);
    base_space->setBounds(bnd);
  }
  base_space->setLongestValidSegmentFraction(0.02);

  auto base_si = std::make_shared<ob::SpaceInformation>(base_space);
  base_si->setStateValidityChecker(
      std::make_shared<BaseOnlyValidityChecker>(base_si, env));
  base_si->setup();

  ob::ScopedState<ob::SE2StateSpace> start_base(base_space);
  start_base->setX(start_active[0]);
  start_base->setY(start_active[1]);
  start_base->setYaw(start_active[2]);
  ob::ScopedState<ob::SE2StateSpace> goal_base(base_space);
  goal_base->setX(goal_active[0]);
  goal_base->setY(goal_active[1]);
  goal_base->setYaw(goal_active[2]);

  std::vector<double> arm_start(arm_dim), arm_goal(arm_dim);
  for (int i = 0; i < arm_dim; ++i) {
    arm_start[i] = start_active[base_dim + i];
    arm_goal[i] = goal_active[base_dim + i];
  }

  // ── Phase 1 + Phase 2 loop ─────────────────────────────────────────
  std::mt19937 seed_rng(cfg.seed ? cfg.seed : 12345u);
  const double per_base_budget =
      time_limit / std::max(1, cfg.k_base_candidates);

  auto elapsed = [&] {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t0)
        .count();
  };

  for (int k = 0; k < cfg.k_base_candidates; ++k) {
    if (elapsed() >= time_limit) break;
    ++result.base_attempts;

    // Reseeding OMPL's global RNG before building the planner so the
    // new RRT-Connect instance picks up distinct samples.  OMPL's RNG
    // has process-global state; any subsequent planners inherit this
    // seed, which is fine — downstream callers construct their own
    // anyway.
    ompl::RNG::setSeed(seed_rng());

    auto base_planner = std::make_shared<og::RRTConnect>(base_si);
    base_planner->setRange(cfg.base_rrt_range);
    auto pdef = std::make_shared<ob::ProblemDefinition>(base_si);
    pdef->setStartAndGoalStates(start_base, goal_base);
    base_planner->setProblemDefinition(pdef);
    base_planner->setup();

    const double budget_remaining = std::max(0.0, time_limit - elapsed());
    const double this_budget =
        std::min(per_base_budget, budget_remaining);
    auto ptc_time = ob::timedPlannerTerminationCondition(this_budget);
    auto ptc_soln = ob::exactSolnPlannerTerminationCondition(pdef);
    auto ptc = ob::plannerOrTerminationCondition(ptc_time, ptc_soln);
    if (!base_planner->solve(ptc)) continue;

    auto* raw = pdef->getSolutionPath()->as<og::PathGeometric>();
    og::PathGeometric path = *raw;  // copy: interpolate() mutates
    // Shortcut-smooth the base path before resampling.  Without this,
    // RRT-Connect with range=0.6 routinely returns paths that wander
    // through random orientations (e.g. flipping the heading by 180°
    // mid-leg), which pushes ``arm_start`` / ``arm_goal`` into
    // infeasible configurations at those waypoints.  Smoothing keeps
    // the path close to the RS-optimal curve between start and goal.
    og::PathSimplifier simplifier(base_si);
    simplifier.simplifyMax(path);
    auto base_wp = resample_base_path(path, cfg.m_base_waypoints);
    const int M = static_cast<int>(base_wp.size());
    if (M < 2) continue;

    // ── Phase 2: layered arm graph ───────────────────────────────────
    std::mt19937 arm_rng(seed_rng());
    std::normal_distribution<double> noise(0.0, cfg.arm_sigma);

    std::vector<std::vector<std::vector<double>>> layers(M);
    layers[0] = {arm_start};
    layers[M - 1] = {arm_goal};

    auto lerp_arm = [&](int i) {
      const double t = static_cast<double>(i) / (M - 1);
      std::vector<double> a(arm_dim);
      for (int d = 0; d < arm_dim; ++d) {
        a[d] = (1.0 - t) * arm_start[d] + t * arm_goal[d];
      }
      return a;
    };

    // Per-layer arm candidates = nominal linear-interp, plus J
    // correlated "detour trajectories".  A detour is parameterised by
    // a single random offset direction eta_j (8-D Gaussian) that is
    // reused across layers, scaled by a sine ramp which is 0 at
    // start/end and peaks at the midpoint.  Effect: the j-th candidate
    // at layer i is ``lerp_arm(i) + eta_j * sin(pi * t_i)`` — a smooth
    // arm trajectory that deviates most in the middle and matches the
    // endpoints.  Because adjacent layers of the same j differ only by
    // the small change in ramp value, ``validate_full_motion`` between
    // them almost always passes, so Dijkstra can follow whichever
    // detour is collision-free.
    //
    // This is the key difference from i.i.d. per-layer Gaussian
    // sampling: independent noise at each layer makes edges fail even
    // when individual layers are rich, because linear interpolation of
    // two jittery arm poses wanders off into collisions.  Correlated
    // ramps produce coherent detour families.
    auto push_if_valid = [&](std::vector<std::vector<double>>& lyr,
                             const std::array<double, 3>& base_pose,
                             std::vector<double> cand) {
      auto q = make_full_config(base_pose, cand, active_indices,
                                frozen_config, base_dim, active_dim);
      if (validate_full_static(q, env)) {
        lyr.push_back(std::move(cand));
      }
    };

    std::vector<std::vector<double>> etas(cfg.j_arm_samples);
    for (int j = 0; j < cfg.j_arm_samples; ++j) {
      etas[j].resize(arm_dim);
      for (int d = 0; d < arm_dim; ++d) {
        etas[j][d] = noise(arm_rng);
      }
    }

    auto ramp = [&](int i) {
      const double t = static_cast<double>(i) / (M - 1);
      return std::sin(M_PI * t);
    };

    bool empty_layer = false;
    for (int i = 1; i < M - 1; ++i) {
      auto nominal = lerp_arm(i);
      const double r = ramp(i);
      auto& lyr = layers[i];
      lyr.reserve(cfg.j_arm_samples + 3);
      // Fast-path + anchor candidates
      push_if_valid(lyr, base_wp[i], nominal);
      push_if_valid(lyr, base_wp[i], arm_start);
      push_if_valid(lyr, base_wp[i], arm_goal);
      // Correlated detour trajectories
      for (int j = 0; j < cfg.j_arm_samples; ++j) {
        std::vector<double> cand(arm_dim);
        for (int d = 0; d < arm_dim; ++d) {
          cand[d] = nominal[d] + etas[j][d] * r;
        }
        push_if_valid(lyr, base_wp[i], std::move(cand));
      }
      if (lyr.empty()) {
        empty_layer = true;
        break;
      }
    }
    // Optional per-candidate diagnostics.  Enable with
    // ``FETCH_DECOMPOSED_DEBUG=1`` in the environment.
    if (std::getenv("FETCH_DECOMPOSED_DEBUG") != nullptr) {
      std::size_t min_sz = layers[0].size(), max_sz = layers[0].size();
      std::size_t total = 0;
      int first_empty = -1;
      for (int i = 0; i < M; ++i) {
        total += layers[i].size();
        min_sz = std::min(min_sz, layers[i].size());
        max_sz = std::max(max_sz, layers[i].size());
        if (first_empty < 0 && layers[i].empty()) first_empty = i;
      }
      std::fprintf(stderr,
                   "[decomposed k=%d M=%d empty=%d] layer sizes "
                   "min/max/avg = %zu/%zu/%.1f\n",
                   k, M, first_empty, min_sz, max_sz,
                   static_cast<double>(total) / M);
    }
    if (empty_layer) continue;

    // ── Dijkstra on the layered DAG ─────────────────────────────────
    std::vector<std::vector<double>> dist(M);
    std::vector<std::vector<std::pair<int, int>>> prev(M);
    for (int i = 0; i < M; ++i) {
      dist[i].assign(layers[i].size(),
                     std::numeric_limits<double>::infinity());
      prev[i].assign(layers[i].size(), {-1, -1});
    }
    dist[0][0] = 0.0;

    using Entry = std::tuple<double, int, int>;  // (cost, layer, idx)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> pq;
    pq.emplace(0.0, 0, 0);

    auto arm_euclid = [&](const std::vector<double>& a,
                          const std::vector<double>& b) {
      double s = 0.0;
      for (int d = 0; d < arm_dim; ++d) {
        const double diff = a[d] - b[d];
        s += diff * diff;
      }
      return std::sqrt(s);
    };

    while (!pq.empty()) {
      auto [c, li, ii] = pq.top();
      pq.pop();
      if (c > dist[li][ii]) continue;
      if (li == M - 1) break;
      const int lj = li + 1;
      auto q1 = make_full_config(base_wp[li], layers[li][ii], active_indices,
                                 frozen_config, base_dim, active_dim);
      for (int ij = 0; ij < static_cast<int>(layers[lj].size()); ++ij) {
        auto q2 = make_full_config(base_wp[lj], layers[lj][ij],
                                   active_indices, frozen_config, base_dim,
                                   active_dim);
        if (!validate_full_motion(q1, q2, env)) continue;
        const double step = arm_euclid(layers[li][ii], layers[lj][ij]);
        const double new_cost = dist[li][ii] + step;
        if (new_cost < dist[lj][ij]) {
          dist[lj][ij] = new_cost;
          prev[lj][ij] = {li, ii};
          pq.emplace(new_cost, lj, ij);
        }
      }
    }

    if (!std::isfinite(dist[M - 1][0])) {
      if (std::getenv("FETCH_DECOMPOSED_DEBUG") != nullptr) {
        int last_reach = 0;
        for (int i = 0; i < M; ++i) {
          for (double d : dist[i]) {
            if (std::isfinite(d)) { last_reach = i; break; }
          }
        }
        std::fprintf(stderr,
                     "[decomposed k=%d] Dijkstra unreachable, last reached "
                     "layer = %d / %d\n",
                     k, last_reach, M - 1);
      }
      continue;
    }

    // ── Reconstruct full active-config path ──────────────────────────
    std::vector<std::pair<int, int>> rev;
    std::pair<int, int> cur = {M - 1, 0};
    while (cur.first != -1) {
      rev.push_back(cur);
      if (cur.first == 0 && cur.second == 0) break;
      cur = prev[cur.first][cur.second];
    }
    std::reverse(rev.begin(), rev.end());

    result.path.reserve(rev.size());
    for (auto [li, ii] : rev) {
      std::vector<double> full(active_dim);
      full[0] = base_wp[li][0];
      full[1] = base_wp[li][1];
      full[2] = base_wp[li][2];
      for (int d = 0; d < arm_dim; ++d) {
        full[base_dim + d] = layers[li][ii][d];
      }
      result.path.push_back(std::move(full));
    }
    result.solved = true;
    result.path_cost = dist[M - 1][0];
    result.winning_candidate = k;
    auto t1 = std::chrono::steady_clock::now();
    result.planning_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return result;
  }

  const auto t1 = std::chrono::steady_clock::now();
  result.planning_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  return result;
}

}  // namespace fetch_planning