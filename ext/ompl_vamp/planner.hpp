/**
 * Main planner class — OMPL frontend with VAMP collision backend.
 *
 * Single constructor: the Python side decides which joints are active
 * and how many of them form the nonholonomic base.  No robot-specific
 * constants are hardcoded here.
 *
 * When the active subgroup includes base joints (base_dim > 0), the
 * planner uses OMPL's **multilevel planning** framework (fiber bundles)
 * with a hierarchy RS → Compound(RS + R^N).  The lower RS level plans
 * the base in isolation; the upper level adds the arm as the fiber,
 * lifting the base path into the full configuration space.
 *
 * The base state space is selected by the ``allow_reverse`` flag:
 * ``ReedsSheppStateSpace`` (shortest-of-48 curves, may include reverse)
 * when reverse is allowed, else ``DubinsStateSpace`` (forward-only, 6
 * curve families).  Dubins is the default — it structurally forbids
 * reverse motion, eliminating the need for any reverse-cost penalty.
 *
 * When the subgroup is arm-only, the planner uses a standard OMPL
 * geometric SimpleSetup with a RealVectorStateSpace (optionally
 * projected onto a constraint manifold via CasADi-compiled
 * constraints).
 */

#pragma once

#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/SimpleSetup.h>
// OMPL — multilevel (fiber bundle) planners
#include <ompl/multilevel/datastructures/projections/XRN_X_SE2.h>
#include <ompl/multilevel/planners/qmp/QMP.h>
#include <ompl/multilevel/planners/qmp/QMPStar.h>
#include <ompl/multilevel/planners/qrrt/QRRT.h>
#include <ompl/multilevel/planners/qrrt/QRRTStar.h>
#include <cmath>

#include "compiled_constraint.hpp"
#include "compiled_cost.hpp"
#include "plan_decomposed.hpp"
#include "validity.hpp"
// OMPL — informed trees
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/EITstar.h>
#include <ompl/geometric/planners/lazyinformedtrees/BLITstar.h>
// OMPL — FMT
#include <ompl/geometric/planners/fmt/BFMT.h>
#include <ompl/geometric/planners/fmt/FMT.h>
// OMPL — KPIECE
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
// OMPL — PRM
#include <ompl/geometric/planners/prm/LazyPRM.h>
#include <ompl/geometric/planners/prm/LazyPRMstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>
// OMPL — RRT family
#include <ompl/geometric/planners/rrt/BiTRRT.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTXstatic.h>
#include <ompl/geometric/planners/rrt/RRTsharp.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/STRRTstar.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
// OMPL — exploration-based
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/stride/STRIDE.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vamp/collision/shapes.hh>
#include <vector>

namespace fetch_planning {

namespace og = ompl::geometric;
namespace om = ompl::multilevel;

/// Build the SE(2) base state space used by the multilevel planner.
///
/// * ``allow_reverse = false`` (default) — ``DubinsStateSpace`` with
///   asymmetric distance.  Every tree extension uses a forward-only
///   curve; the planner cannot produce reverse motion, so no penalty
///   is required anywhere downstream.
/// * ``allow_reverse = true`` — ``ReedsSheppStateSpace``.  Curves are
///   the unpenalized shortest-of-48 RS paths, which may include reverse
///   when it is geometrically shorter.
///
/// Both spaces store states in ``SE2StateSpace::StateType`` layout, so
/// downstream validators and the manual projection below can treat the
/// return value polymorphically via ``SE2StateSpace`` casts.
inline auto make_se2_base_space(double turning_radius, bool allow_reverse)
    -> std::shared_ptr<ob::SE2StateSpace> {
  if (allow_reverse) {
    return std::make_shared<ob::ReedsSheppStateSpace>(turning_radius);
  }
  return std::make_shared<ob::DubinsStateSpace>(turning_radius,
                                                /*isSymmetric=*/false);
}

// Projection for the multilevel hierarchy Compound(SE2 + R^N) → SE2.
//
// OMPL's ProjectionFactory only auto-detects plain SE2 — the RS/Dubins
// spaces carry STATE_SPACE_REEDS_SHEPP / STATE_SPACE_DUBINS tags so the
// factory refuses to build the projection.  Both subclasses inherit from
// SE2StateSpace with the same state layout, so OMPL's own
// Projection_SE2RN_SE2 works correctly when constructed manually — its
// project/lift only use ``SE2StateSpace::StateType`` casts, never the
// type tag.

struct PlanResult {
  bool solved;
  std::vector<std::vector<double>> path;
  int64_t planning_time_ns;
  double path_cost;
};

class OmplVampPlanner {
 public:
  /// Unified constructor — Python decides what is base vs. arm.
  ///
  /// @param active_indices  Joint indices into the full-body config
  ///     that this planner controls.
  /// @param frozen_config   Full-body joint values; joints not in
  ///     active_indices are pinned to these during collision checks.
  /// @param base_dim  How many of the *leading* active indices form
  ///     the nonholonomic base (0 = arm-only, 3 = SE2 base).
  ///     When > 0, the planner uses OMPL multilevel planning with a
  ///     hierarchy RS → RS × R^N.  When 0, a standard geometric
  ///     planner on RealVectorStateSpace is used.
  /// @param turning_radius  Minimum turning radius (m).  Ignored when
  ///     base_dim == 0.
  /// @param allow_reverse  Select the SE(2) base state space.  When
  ///     false (default), uses ``DubinsStateSpace`` (forward-only
  ///     curves, no reverse ever).  When true, uses
  ///     ``ReedsSheppStateSpace`` (shortest-of-48 curves, reverse
  ///     allowed when geometrically shorter).
  OmplVampPlanner(std::vector<int> active_indices,
                  std::vector<double> frozen_config,
                  int base_dim = 0,
                  double turning_radius = 0.2,
                  bool allow_reverse = false)
      : active_dim_(static_cast<int>(active_indices.size())),
        active_indices_(std::move(active_indices)),
        base_dim_(base_dim),
        turning_radius_(turning_radius),
        allow_reverse_(allow_reverse) {
    frozen_config_.resize(frozen_config.size());
    for (std::size_t i = 0; i < frozen_config.size(); ++i)
      frozen_config_[i] = static_cast<float>(frozen_config[i]);

    has_base_ = base_dim_ > 0;
    base_only_ = has_base_ && (active_dim_ == base_dim_);
    build_state_space();
  }

  /// Configure base workspace bounds (x, y, theta). Call before plan()
  /// to tighten the default limits.
  void set_base_bounds(double x_lo, double x_hi, double y_lo, double y_hi,
                       double theta_lo = -M_PI, double theta_hi = M_PI) {
    base_x_lo_ = x_lo;
    base_x_hi_ = x_hi;
    base_y_lo_ = y_lo;
    base_y_hi_ = y_hi;
    base_theta_lo_ = theta_lo;
    base_theta_hi_ = theta_hi;
    build_state_space();
  }

  void add_pointcloud(const std::vector<std::array<float, 3>> &points,
                      float r_min, float r_max, float point_radius) {
    std::vector<vamp::collision::Point> pts;
    pts.reserve(points.size());
    for (const auto &p : points) pts.push_back({p[0], p[1], p[2]});
    float_env_.pointclouds.emplace_back(pts, r_min, r_max, point_radius);
    sync_env();
  }

  /// Drop the most-recently-added pointcloud.  Returns false if there
  /// was none registered.
  auto remove_pointcloud() -> bool {
    if (float_env_.pointclouds.empty()) return false;
    float_env_.pointclouds.pop_back();
    sync_env();
    return true;
  }

  auto has_pointcloud() const -> bool {
    return !float_env_.pointclouds.empty();
  }

  void add_sphere(const std::array<float, 3> &center, float radius) {
    float_env_.spheres.emplace_back(vamp::collision::Sphere<float>{
        center[0], center[1], center[2], radius});
    float_env_.sort();
    sync_env();
  }

  void clear_environment() {
    float_env_ = FloatEnv{};
    env_ = VampEnv{};
  }

  // ── Constraint API ────────────────────────────────────────────────

  void add_compiled_constraint(const std::string &so_path,
                               const std::string &symbol_name,
                               unsigned int ambient_dim, unsigned int co_dim) {
    if (static_cast<int>(ambient_dim) != active_dim_) {
      throw std::invalid_argument(
          "add_compiled_constraint: ambient_dim (" +
          std::to_string(ambient_dim) +
          ") does not match planner active dimension (" +
          std::to_string(active_dim_) + ")");
    }
    constraints_.push_back(std::make_shared<CompiledConstraint>(
        ambient_dim, co_dim, so_path, symbol_name));
  }

  void clear_constraints() { constraints_.clear(); }

  std::size_t num_constraints() const { return constraints_.size(); }

  // ── Cost API ──────────────────────────────────────────────────────
  //
  // Costs are soft per-state terms integrated along every motion by
  // OMPL's StateCostIntegralObjective.  They do not constrain the
  // feasible set — collision checking still does that — but they
  // shape the solution returned by asymptotically-optimal planners
  // (RRT*, BIT*, AIT*, QRRT* …).  Without a user-supplied cost the
  // planner falls back to OMPL's default path-length objective.
  //
  // For multilevel plans the cost objective is set on the top
  // SpaceInformation's ProblemDefinition, added on top of the base
  // path-length term so the non-holonomic distance metric still
  // contributes to motionCost.
  //
  // Multiple costs are summed via MultiOptimizationObjective with
  // the weights supplied at add time.

  void add_compiled_cost(const std::string &so_path,
                         const std::string &symbol_name,
                         unsigned int ambient_dim, double weight) {
    if (static_cast<int>(ambient_dim) != active_dim_) {
      throw std::invalid_argument(
          "add_compiled_cost: ambient_dim (" + std::to_string(ambient_dim) +
          ") does not match planner active dimension (" +
          std::to_string(active_dim_) + ")");
    }
    if (weight < 0.0) {
      throw std::invalid_argument("add_compiled_cost: weight must be >= 0");
    }
    cost_libs_.push_back(
        std::make_shared<CostLibrary>(ambient_dim, so_path, symbol_name));
    cost_weights_.push_back(weight);
  }

  void clear_costs() {
    cost_libs_.clear();
    cost_weights_.clear();
  }

  std::size_t num_costs() const { return cost_libs_.size(); }

  auto plan(std::vector<double> start, std::vector<double> goal,
            const std::string &planner_name, double time_limit, bool simplify,
            bool interpolate) -> PlanResult {
    if (has_base_) {
      if (!constraints_.empty()) {
        throw std::invalid_argument(
            "Constrained planning is not supported for subgroups that "
            "include mobile-base joints.  Use an arm-only subgroup for "
            "constrained planning.");
      }
      if (planner_name == "decomposed") {
        return plan_decomposed_wrapper(start, goal, time_limit, simplify,
                                       interpolate);
      }
      return plan_multilevel(start, goal, planner_name, time_limit, simplify,
                             interpolate);
    }
    return plan_geometric(start, goal, planner_name, time_limit, simplify,
                          interpolate);
  }

  auto validate(std::vector<double> config) -> bool {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    std::copy(frozen_config_.begin(), frozen_config_.end(), buf.begin());
    for (std::size_t i = 0; i < active_indices_.size(); ++i)
      buf[active_indices_[i]] = static_cast<float>(config[i]);
    auto q = Robot::Configuration(buf.data());
    return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
  }

  auto dimension() const -> int { return active_dim_; }

  auto lower_bounds() const -> std::vector<double> {
    std::vector<double> lo(active_dim_);
    if (has_base_) {
      lo[0] = base_x_lo_;
      lo[1] = base_y_lo_;
      lo[2] = base_theta_lo_;
      if (arm_subspace_) {
        auto bounds = arm_subspace_->getBounds();
        for (int i = base_dim_; i < active_dim_; ++i)
          lo[i] = bounds.low[i - base_dim_];
      }
    } else {
      auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
      for (int i = 0; i < active_dim_; ++i) lo[i] = bounds.low[i];
    }
    return lo;
  }

  auto upper_bounds() const -> std::vector<double> {
    std::vector<double> hi(active_dim_);
    if (has_base_) {
      hi[0] = base_x_hi_;
      hi[1] = base_y_hi_;
      hi[2] = base_theta_hi_;
      if (arm_subspace_) {
        auto bounds = arm_subspace_->getBounds();
        for (int i = base_dim_; i < active_dim_; ++i)
          hi[i] = bounds.high[i - base_dim_];
      }
    } else {
      auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
      for (int i = 0; i < active_dim_; ++i) hi[i] = bounds.high[i];
    }
    return hi;
  }

  auto min_max_radii() const -> std::pair<float, float> {
    return {Robot::min_radius, Robot::max_radius};
  }

 private:
  int active_dim_;
  int base_dim_;
  bool has_base_ = false;
  bool base_only_ = false;
  double turning_radius_;
  bool allow_reverse_;

  double base_x_lo_ = -10.0;
  double base_x_hi_ = 10.0;
  double base_y_lo_ = -10.0;
  double base_y_hi_ = 10.0;
  double base_theta_lo_ = -M_PI;
  double base_theta_hi_ = M_PI;

  std::vector<int> active_indices_;
  std::vector<float> frozen_config_;
  ob::StateSpacePtr space_;
  std::shared_ptr<ob::RealVectorStateSpace> arm_subspace_;
  FloatEnv float_env_;
  VampEnv env_;
  std::vector<ob::ConstraintPtr> constraints_;
  std::vector<std::shared_ptr<CostLibrary>> cost_libs_;
  std::vector<double> cost_weights_;

  void sync_env() { env_ = VampEnv(float_env_); }

  void build_state_space() {
    Robot::Configuration lo, hi;
    std::array<float, Robot::dimension> zeros{}, ones{};
    ones.fill(1.0f);
    lo = Robot::Configuration(zeros.data());
    hi = Robot::Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);
    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    if (has_base_) {
      // SE(2) base space — Dubins (forward-only) or Reeds-Shepp
      // (reverse allowed), selected by ``allow_reverse_``.  Both provide
      // non-holonomic car-like distance() + interpolate() semantics; the
      // distinction is whether reverse segments are admissible.
      auto se2 = make_se2_base_space(turning_radius_, allow_reverse_);
      ob::RealVectorBounds se2_bounds(2);
      se2_bounds.setLow(0, base_x_lo_);
      se2_bounds.setHigh(0, base_x_hi_);
      se2_bounds.setLow(1, base_y_lo_);
      se2_bounds.setHigh(1, base_y_hi_);
      se2->setBounds(se2_bounds);
      // Car-like curves are longer than straight lines, so the default
      // 1%-of-maxextent segment fraction produces extremely fine
      // subdivision (thousands of collision checks per edge) and the
      // motion validator can spend seconds inside a single edge without
      // returning to check the planner-termination condition.  2% keeps
      // checks fine enough to catch obstacles on a 0.2 m turning radius
      // while letting the planner breathe.
      se2->setLongestValidSegmentFraction(0.02);

      const int arm_active_dim = active_dim_ - base_dim_;

      if (arm_active_dim == 0) {
        base_only_ = true;
        arm_subspace_ = nullptr;
        space_ = se2;
      } else {
        base_only_ = false;
        auto rv = std::make_shared<ob::RealVectorStateSpace>(arm_active_dim);
        ob::RealVectorBounds rv_bounds(arm_active_dim);
        for (int i = 0; i < arm_active_dim; ++i) {
          auto idx = active_indices_[i + base_dim_];
          rv_bounds.setLow(i, std::min(lo_arr[idx], hi_arr[idx]));
          rv_bounds.setHigh(i, std::max(lo_arr[idx], hi_arr[idx]));
        }
        rv->setBounds(rv_bounds);
        arm_subspace_ = rv;

        auto compound = std::make_shared<ob::CompoundStateSpace>();
        compound->addSubspace(se2, 1.0);
        compound->addSubspace(rv, 1.0);
        compound->lock();
        compound->setLongestValidSegmentFraction(0.02);
        space_ = compound;
      }
    } else {
      base_only_ = false;
      auto rv = std::make_shared<ob::RealVectorStateSpace>(active_dim_);
      ob::RealVectorBounds bounds(active_dim_);
      for (int i = 0; i < active_dim_; ++i) {
        auto idx = active_indices_[i];
        bounds.setLow(i, std::min(lo_arr[idx], hi_arr[idx]));
        bounds.setHigh(i, std::max(lo_arr[idx], hi_arr[idx]));
      }
      rv->setBounds(bounds);
      space_ = rv;
      arm_subspace_ = rv;
    }
  }

  void write_scoped_state(ob::ScopedState<> &state,
                          const std::vector<double> &config) const {
    if (base_only_) {
      auto *se2 = state.get()->as<ob::SE2StateSpace::StateType>();
      se2->setX(config[0]);
      se2->setY(config[1]);
      se2->setYaw(config[2]);
    } else if (has_base_) {
      auto *compound = state.get()->as<ob::CompoundStateSpace::StateType>();
      auto *se2 = compound->as<ob::SE2StateSpace::StateType>(0);
      auto *rv = compound->as<ob::RealVectorStateSpace::StateType>(1);
      se2->setX(config[0]);
      se2->setY(config[1]);
      se2->setYaw(config[2]);
      for (int i = base_dim_; i < active_dim_; ++i) {
        rv->values[i - base_dim_] = config[i];
      }
    } else {
      for (int i = 0; i < active_dim_; ++i) {
        state[i] = config[i];
      }
    }
  }

  auto read_state(const ob::State *state) const -> std::vector<double> {
    std::vector<double> out(active_dim_);
    if (base_only_) {
      const auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
      const auto *se2 =
          wrapper
              ? wrapper->getState()->as<ob::SE2StateSpace::StateType>()
              : state->as<ob::SE2StateSpace::StateType>();
      out[0] = se2->getX();
      out[1] = se2->getY();
      out[2] = se2->getYaw();
    } else if (has_base_) {
      const auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
      const auto *compound =
          wrapper
              ? wrapper->getState()->as<ob::CompoundStateSpace::StateType>()
              : state->as<ob::CompoundStateSpace::StateType>();
      const auto *se2 = compound->as<ob::SE2StateSpace::StateType>(0);
      const auto *rv =
          compound->as<ob::RealVectorStateSpace::StateType>(1);
      out[0] = se2->getX();
      out[1] = se2->getY();
      out[2] = se2->getYaw();
      for (int i = base_dim_; i < active_dim_; ++i) {
        out[i] = rv->values[i - base_dim_];
      }
    } else {
      const auto *rv = extract_real_state(state);
      for (int i = 0; i < active_dim_; ++i) out[i] = rv->values[i];
    }
    return out;
  }

  // ── Multilevel planning (base-included subgroups) ──────────────────

  auto plan_multilevel(const std::vector<double> &start,
                       const std::vector<double> &goal,
                       const std::string &planner_name, double time_limit,
                       bool simplify, bool interpolate) -> PlanResult {
    // Build a 2-level hierarchy:
    //   Level 0: SE(2) (base pose, non-holonomic — Dubins or Reeds-Shepp)
    //   Level 1: Compound(SE(2) + R^N) (full active space)
    // OMPL's ProjectionFactory cannot auto-detect this mapping because
    // the base subclass carries STATE_SPACE_DUBINS / STATE_SPACE_REEDS_SHEPP
    // rather than STATE_SPACE_SE2.  We supply a custom projection that
    // works directly on the state layout (all three share SE2 storage).
    std::vector<ob::SpaceInformationPtr> si_vec;
    std::vector<om::ProjectionPtr> proj_vec;

    // Level 0: SE(2) base ────────────────────────────────────────────
    auto level0_se2 = make_se2_base_space(turning_radius_, allow_reverse_);
    {
      ob::RealVectorBounds bnd(2);
      bnd.setLow(0, base_x_lo_);
      bnd.setHigh(0, base_x_hi_);
      bnd.setLow(1, base_y_lo_);
      bnd.setHigh(1, base_y_hi_);
      level0_se2->setBounds(bnd);
      level0_se2->setLongestValidSegmentFraction(0.02);

      auto si = std::make_shared<ob::SpaceInformation>(level0_se2);
      // Base-only check — a proper relaxation of the full-body check,
      // using the dedicated FetchBase VAMP model (3 DOF, 14 spheres).
      // Any pose valid at the top level projects to a base-
      // collision-free pose here, giving the multilevel framework an
      // admissible abstraction.
      si->setStateValidityChecker(
          std::make_shared<BaseOnlyValidityChecker>(si, env_));
      si->setup();
      si_vec.push_back(si);
    }

    // Level 1: Compound(SE(2) + R^N)  (only when arm joints are active) ──
    if (!base_only_) {
      auto si = std::make_shared<ob::SpaceInformation>(space_);
      si->setStateValidityChecker(std::make_shared<SubgroupValidityChecker>(
          si, env_, active_indices_, frozen_config_,
          /*has_base=*/true, /*base_only=*/false));
      si->setup();
      si_vec.push_back(si);

      // OMPL's own Projection_SE2RN_SE2 works here because both
      // ReedsSheppStateSpace and DubinsStateSpace inherit from
      // SE2StateSpace (same state layout).  The ProjectionFactory
      // can't auto-pick it since the concrete space's type tag is
      // STATE_SPACE_REEDS_SHEPP / STATE_SPACE_DUBINS, but constructing
      // it manually bypasses that check.
      auto proj = std::make_shared<om::Projection_SE2RN_SE2>(space_, level0_se2);
      // Eagerly initialise the fiber space (sampler + scratch state),
      // which the framework relies on for lifting.
      proj->makeFiberSpace();
      proj_vec.push_back(proj);
    }

    // Create multilevel planner and problem definition ─────────────────
    auto &top_si = si_vec.back();
    auto planner = create_multilevel_planner(si_vec, proj_vec, planner_name);

    auto pdef = std::make_shared<ob::ProblemDefinition>(top_si);
    ob::ScopedState<> ompl_start(top_si->getStateSpace());
    ob::ScopedState<> ompl_goal(top_si->getStateSpace());
    write_scoped_state(ompl_start, start);
    write_scoped_state(ompl_goal, goal);
    pdef->setStartAndGoalStates(ompl_start, ompl_goal);

    // Soft costs are integrated by QRRTStar/RRTstar/BIT* family on top
    // of the base SE(2) distance.  No reverse penalty lives in the
    // distance metric — forward-only behavior comes from the curve
    // selection (Dubins) rather than from a cost weighting.
    if (!cost_libs_.empty()) {
      pdef->setOptimizationObjective(
          build_objective(top_si, /*active_top=*/true));
    }

    planner->setProblemDefinition(pdef);
    planner->setup();

    // Solve ────────────────────────────────────────────────────────────
    // The multilevel framework only early-exits on non-final levels
    // (BundleSpaceSequenceImpl.h: `foundKLevelSolution_ && k < size - 1`),
    // so the top level runs to the full time budget even after an exact
    // solution exists.  For feasibility planners like QRRT that don't
    // refine, that's pure waste.  Combine the time budget with an
    // exact-solution PTC so we return as soon as the top level has a
    // solution.
    auto ptc_time = ob::timedPlannerTerminationCondition(time_limit);
    auto ptc_soln = ob::exactSolnPlannerTerminationCondition(pdef);
    auto ptc = ob::plannerOrTerminationCondition(ptc_time, ptc_soln);
    auto t0 = std::chrono::steady_clock::now();
    auto status = planner->solve(ptc);
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    PlanResult result;
    result.planning_time_ns = elapsed_ns;
    result.solved = static_cast<bool>(status);

    if (result.solved) {
      auto &path = *pdef->getSolutionPath()->as<og::PathGeometric>();

      if (simplify) {
        og::PathSimplifier simplifier(top_si);
        simplifier.simplifyMax(path);
      }
      if (interpolate) path.interpolate();
      result.path_cost = path.length();

      for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        result.path.push_back(read_state(path.getState(i)));
      }
    } else {
      result.path_cost = std::numeric_limits<double>::infinity();
    }
    return result;
  }

  // ── Decomposed planning (base roadmap + layered arm scheduler) ────
  //
  // Adapter around ``plan_decomposed`` (plan_decomposed.hpp).  The
  // decomposed planner already returns a ready-to-use sequence of
  // active-subgroup configurations sampled at the base-resample
  // density, so ``simplify`` / ``interpolate`` are no-ops here — the
  // path is already dense in the 11-D active space.  (If callers ever
  // want to trim waypoints, shortcut-smoothing should run on the
  // reconstructed PathGeometric, not on the layered DAG output, so we
  // leave that for a follow-up.)

  auto plan_decomposed_wrapper(const std::vector<double> &start,
                               const std::vector<double> &goal,
                               double time_limit, bool /*simplify*/,
                               bool /*interpolate*/) -> PlanResult {
    std::array<double, 4> base_xy_bounds{base_x_lo_, base_x_hi_, base_y_lo_,
                                         base_y_hi_};
    DecomposedConfig cfg{};
    auto r = plan_decomposed(allow_reverse_, turning_radius_, base_xy_bounds,
                             start, goal, active_indices_, frozen_config_,
                             base_dim_, active_dim_, env_, time_limit, cfg);

    PlanResult out;
    out.solved = r.solved;
    out.planning_time_ns = r.planning_time_ns;
    if (r.solved) {
      out.path = std::move(r.path);
      out.path_cost = r.path_cost;
    } else {
      out.path_cost = std::numeric_limits<double>::infinity();
    }
    return out;
  }

  // ── Geometric planning (arm-only subgroups, with constraints) ─────

  auto plan_geometric(const std::vector<double> &start,
                      const std::vector<double> &goal,
                      const std::string &planner_name, double time_limit,
                      bool simplify, bool interpolate) -> PlanResult {
    const bool constrained = !constraints_.empty();
    if (constrained) {
      reject_incompatible_planner(planner_name);
      check_constraint_satisfaction(start, "start");
      check_constraint_satisfaction(goal, "goal");
    }

    ob::StateSpacePtr active_space = space_;
    ob::SpaceInformationPtr si;
    if (constrained) {
      auto intersection = std::make_shared<ob::ConstraintIntersection>(
          static_cast<unsigned int>(active_dim_), constraints_);
      auto css =
          std::make_shared<ob::ProjectedStateSpace>(space_, intersection);
      auto csi = std::make_shared<ob::ConstrainedSpaceInformation>(css);
      css->setup();
      si = csi;
      active_space = css;
    } else {
      si = std::make_shared<ob::SpaceInformation>(space_);
    }

    si->setStateValidityChecker(std::make_shared<SubgroupValidityChecker>(
        si, env_, active_indices_, frozen_config_, has_base_, base_only_));
    if (!constrained) {
      si->setMotionValidator(std::make_shared<SubgroupMotionValidator>(
          si, env_, active_indices_, frozen_config_, has_base_, base_only_));
    }
    si->setup();

    og::SimpleSetup ss(si);
    ss.setPlanner(create_geometric_planner(si, planner_name));

    ob::ScopedState<> ompl_start(active_space);
    ob::ScopedState<> ompl_goal(active_space);
    write_scoped_state(ompl_start, start);
    write_scoped_state(ompl_goal, goal);
    ss.setStartAndGoalStates(ompl_start, ompl_goal);

    // Apply user-supplied soft costs.  Only consumed by asymptotically
    // optimal planners; for planners that don't read it the call is
    // harmless.
    if (!cost_libs_.empty()) {
      ss.setOptimizationObjective(
          build_objective(si, /*active_top=*/false));
    }

    auto t0 = std::chrono::steady_clock::now();
    auto status = ss.solve(time_limit);
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    PlanResult result;
    result.planning_time_ns = elapsed_ns;
    result.solved = static_cast<bool>(status);

    if (result.solved) {
      if (simplify) ss.simplifySolution();

      auto &path = ss.getSolutionPath();
      if (interpolate) path.interpolate();
      result.path_cost = path.length();

      for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        result.path.push_back(read_state(path.getState(i)));
      }
    } else {
      result.path_cost = std::numeric_limits<double>::infinity();
    }
    return result;
  }

  // ── Constraint helpers ────────────────────────────────────────────

  void check_constraint_satisfaction(const std::vector<double> &active_q,
                                     const char *which) const {
    Eigen::VectorXd q(active_dim_);
    for (int i = 0; i < active_dim_; ++i) q[i] = active_q[i];
    for (std::size_t i = 0; i < constraints_.size(); ++i) {
      const auto &c = constraints_[i];
      Eigen::VectorXd r(c->getCoDimension());
      c->function(q, r);
      const double residual = r.norm();
      if (residual > c->getTolerance()) {
        throw std::invalid_argument(
            std::string("Constraint #") + std::to_string(i) +
            " is violated at " + which + " (residual " +
            std::to_string(residual) + " > tolerance " +
            std::to_string(c->getTolerance()) +
            ").  Both start and goal must already lie on the constraint "
            "manifold.");
      }
    }
  }

  static void reject_incompatible_planner(const std::string &name) {
    static const std::vector<std::string> bad = {
        "bitstar",  "abitstar",   "aitstar",  "eitstar",
        "blitstar", "fmt",        "bfmt",     "informed_rrtstar",
        "rrtsharp", "rrtxstatic", "strrtstar"};
    for (const auto &b : bad) {
      if (name == b) {
        throw std::invalid_argument(
            "Planner '" + name +
            "' is incompatible with constrained planning.  Use one of: "
            "rrtc, rrt, rrtstar, prm, prmstar, kpiece, bkpiece, lbkpiece, "
            "est, biest, sbl, stride.");
      }
    }
  }

  // ── Objective builder ────────────────────────────────────────────
  //
  // Combine every CompiledCost into a single OMPL objective.  The
  // ``active_top`` flag selects the right state-extraction layout for
  // the multilevel "top" SpaceInformation (Compound when arm is
  // active, SE2-only when base-only) versus the geometric / arm-only
  // RealVector layout.  The dlopen'd ``CostLibrary`` is reused; only
  // the thin adapter is rebuilt per plan() call.
  //
  // For base-including subgroups we ALWAYS keep a path-length term as
  // the baseline so the SE(2) distance contribution (Dubins / RS) stays
  // inside the optimisation objective QRRTStar / RRT* rewires against.
  // Without this, a CompiledCost would silently shadow the car-like
  // distance in motionCost and the non-holonomic shaping would only
  // affect nearest-neighbour selection, not rewiring decisions.
  auto build_objective(const ob::SpaceInformationPtr &si, bool active_top) const
      -> std::shared_ptr<ob::OptimizationObjective> {
    const CompiledCost::Layout layout =
        active_top ? (base_only_ ? CompiledCost::Layout::kSE2Only
                                 : CompiledCost::Layout::kCompound)
                   : CompiledCost::Layout::kPlain;

    const bool keep_path_length = active_top && has_base_;

    if (cost_libs_.size() == 1 && !keep_path_length) {
      return std::make_shared<CompiledCost>(si, cost_libs_[0],
                                            cost_weights_[0], layout,
                                            active_dim_);
    }

    auto multi = std::make_shared<ob::MultiOptimizationObjective>(si);
    if (keep_path_length) {
      // Path-length objective integrates ob::motionCost via the SE(2)
      // distance — forward-only Dubins length or Reeds-Shepp length,
      // depending on ``allow_reverse_``.
      multi->addObjective(
          std::make_shared<ob::PathLengthOptimizationObjective>(si), 1.0);
    }
    for (std::size_t i = 0; i < cost_libs_.size(); ++i) {
      // MultiOptimizationObjective applies its own weight on top of
      // whatever stateCost() returns; bake the per-cost weight into
      // the adapter (weight 1.0 here) to keep a single source of
      // truth for the scaling factor.
      multi->addObjective(
          std::make_shared<CompiledCost>(si, cost_libs_[i], cost_weights_[i],
                                         layout, active_dim_),
          1.0);
    }
    return multi;
  }

  // ── Planner factories ─────────────────────────────────────────────

  static auto create_multilevel_planner(
      std::vector<ob::SpaceInformationPtr> &si_vec,
      std::vector<om::ProjectionPtr> &proj_vec,
      const std::string &name) -> ob::PlannerPtr {
    // Use the 3-arg constructor whenever we have custom projections
    // (proj_vec.size() == si_vec.size() - 1); otherwise fall back to the
    // 1-arg auto-detection constructor (used when there is a single
    // level with no projection needed).
    const bool use_custom = !proj_vec.empty();

    // Direct multilevel planner names.
    if (name == "qrrt")
      return use_custom ? std::make_shared<om::QRRT>(si_vec, proj_vec)
                        : std::make_shared<om::QRRT>(si_vec);
    if (name == "qmp")
      return use_custom ? std::make_shared<om::QMP>(si_vec, proj_vec)
                        : std::make_shared<om::QMP>(si_vec);
    if (name == "qmpstar")
      return use_custom ? std::make_shared<om::QMPStar>(si_vec, proj_vec)
                        : std::make_shared<om::QMPStar>(si_vec);
    if (name == "qrrtstar")
      return use_custom ? std::make_shared<om::QRRTStar>(si_vec, proj_vec)
                        : std::make_shared<om::QRRTStar>(si_vec);
    // PRM-style names → QMP.
    if (name == "prm" || name == "lazyprm")
      return use_custom ? std::make_shared<om::QMP>(si_vec, proj_vec)
                        : std::make_shared<om::QMP>(si_vec);
    if (name == "prmstar" || name == "lazyprmstar")
      return use_custom ? std::make_shared<om::QMPStar>(si_vec, proj_vec)
                        : std::make_shared<om::QMPStar>(si_vec);
    // Non-optimal tree names → QRRT.
    if (name == "rrtc" || name == "rrtconnect" || name == "rrt")
      return use_custom ? std::make_shared<om::QRRT>(si_vec, proj_vec)
                        : std::make_shared<om::QRRT>(si_vec);
    // Default (including all "star/optimal" geometric names) → QRRTStar.
    // The reverse penalty only bites for asymptotically optimal planners.
    return use_custom ? std::make_shared<om::QRRTStar>(si_vec, proj_vec)
                      : std::make_shared<om::QRRTStar>(si_vec);
  }

  static auto create_geometric_planner(const ob::SpaceInformationPtr &si,
                                       const std::string &name)
      -> ob::PlannerPtr {
    if (name == "rrtc" || name == "rrtconnect")
      return std::make_shared<og::RRTConnect>(si);
    if (name == "rrt") return std::make_shared<og::RRT>(si);
    if (name == "rrtstar") return std::make_shared<og::RRTstar>(si);
    if (name == "informed_rrtstar")
      return std::make_shared<og::InformedRRTstar>(si);
    if (name == "rrtsharp") return std::make_shared<og::RRTsharp>(si);
    if (name == "rrtxstatic") return std::make_shared<og::RRTXstatic>(si);
    if (name == "strrtstar") return std::make_shared<og::STRRTstar>(si);
    if (name == "lbtrrt") return std::make_shared<og::LBTRRT>(si);
    if (name == "trrt") return std::make_shared<og::TRRT>(si);
    if (name == "bitrrt") return std::make_shared<og::BiTRRT>(si);
    if (name == "bitstar") return std::make_shared<og::BITstar>(si);
    if (name == "abitstar") return std::make_shared<og::ABITstar>(si);
    if (name == "aitstar") return std::make_shared<og::AITstar>(si);
    if (name == "eitstar") return std::make_shared<og::EITstar>(si);
    if (name == "blitstar") return std::make_shared<og::BLITstar>(si);
    if (name == "fmt") return std::make_shared<og::FMT>(si);
    if (name == "bfmt") return std::make_shared<og::BFMT>(si);
    if (name == "kpiece") return std::make_shared<og::KPIECE1>(si);
    if (name == "bkpiece") return std::make_shared<og::BKPIECE1>(si);
    if (name == "lbkpiece") return std::make_shared<og::LBKPIECE1>(si);
    if (name == "prm") return std::make_shared<og::PRM>(si);
    if (name == "prmstar") return std::make_shared<og::PRMstar>(si);
    if (name == "lazyprm") return std::make_shared<og::LazyPRM>(si);
    if (name == "lazyprmstar") return std::make_shared<og::LazyPRMstar>(si);
    if (name == "spars") return std::make_shared<og::SPARS>(si);
    if (name == "spars2") return std::make_shared<og::SPARStwo>(si);
    if (name == "est") return std::make_shared<og::EST>(si);
    if (name == "biest") return std::make_shared<og::BiEST>(si);
    if (name == "sbl") return std::make_shared<og::SBL>(si);
    if (name == "stride") return std::make_shared<og::STRIDE>(si);
    if (name == "pdst") return std::make_shared<og::PDST>(si);
    throw std::invalid_argument("Unknown planner: " + name);
  }
};

}  // namespace fetch_planning