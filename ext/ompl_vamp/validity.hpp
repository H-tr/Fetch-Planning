/**
 * OMPL ↔ VAMP state validity and motion validators for whole-body Fetch.
 *
 * The full-body robot is 11-DOF (3 base + 8 arm_with_torso). The first
 * three joints form a planar SE(2) pose for a nonholonomic differential-
 * drive base; the remaining eight joints are the torso_lift + 7-DOF arm.
 *
 * OMPL state-space layouts (chosen by the planner based on which joints
 * are active in the subgroup):
 *
 *  1. ``kPlain``   — a plain RealVectorStateSpace(active_dim_). Used when
 *     the active subgroup contains no base joints (e.g. "fetch_arm",
 *     "fetch_arm_with_torso"). States read directly into a VAMP
 *     configuration after expanding the subgroup via active_indices +
 *     frozen_config.
 *
 *  2. ``kCompound`` — a CompoundStateSpace of a car-like SE(2) subspace
 *     (``DubinsStateSpace`` or ``ReedsSheppStateSpace`` depending on the
 *     ``allow_reverse`` flag) + RealVectorStateSpace(active_dim_ − 3).
 *     Used when the active subgroup includes base joints (e.g.
 *     "fetch_base", "fetch_base_arm", "fetch_whole_body"). The SE(2)
 *     subspace carries (base_x, base_y, base_theta) and its built-in
 *     distance()/interpolate() enforces the nonholonomic constraint.
 *     State extraction reads SE(2) and the RealVector subspace separately,
 *     then concatenates them into a VAMP configuration in URDF order
 *     (base first, then arm).
 *
 * In all modes collision checking is delegated to VAMP's SIMD sphere
 * checks of the whole body.  Motion edges are sampled along the state
 * space's own interpolation (the Dubins / Reeds-Shepp curve for the
 * base) at the robot's resolution.
 */

#pragma once

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/DubinsStateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/WrapperStateSpace.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/fetch_base.hh>
#include <vamp/robots/fetch_whole_body.hh>
#include <vector>

namespace fetch_planning {

namespace ob = ompl::base;

using Robot = vamp::robots::FetchWholeBody;
using BaseRobot = vamp::robots::FetchBase;
inline constexpr std::size_t kRake = vamp::FloatVectorWidth;
using VampEnv = vamp::collision::Environment<vamp::FloatVector<kRake>>;
using FloatEnv = vamp::collision::Environment<float>;

// Robot dimension (used only by the full-body validity checkers below).
inline constexpr int kFullDim = Robot::dimension;

// Unwrap a ConstrainedStateSpace wrapper (if present) to get the
// underlying RealVector state.
inline auto extract_real_state(const ob::State* state)
    -> const ob::RealVectorStateSpace::StateType* {
  if (auto* wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType*>(state)) {
    return wrapper->getState()->as<ob::RealVectorStateSpace::StateType>();
  }
  return state->as<ob::RealVectorStateSpace::StateType>();
}

// ─── Validity / motion checkers ─────────────────────────────────────

/// The active joints of a subgroup, the frozen values of the others, and
/// the state layout: with the base (``has_base``) the leading three
/// active values are the SE(2) pose of a bare SE(2) state (``base_only``)
/// or of the first subspace of Compound(SE2 + RealVector); without it the
/// state is a RealVectorStateSpace (possibly wrapped by a constrained
/// space).
struct Subgroup {
  std::vector<int> active;
  std::vector<float> frozen;
  bool has_base;
  bool base_only;

  auto dim() const -> int { return static_cast<int>(active.size()); }

  auto se2(const ob::State* state) const
      -> const ob::SE2StateSpace::StateType* {
    return base_only ? state->as<ob::SE2StateSpace::StateType>()
                     : state->as<ob::CompoundStateSpace::StateType>()
                           ->as<ob::SE2StateSpace::StateType>(0);
  }

  /// Active-DOF values of an OMPL state.
  void read(const ob::State* state, double* out) const {
    if (const auto* wrapper =
            dynamic_cast<const ob::WrapperStateSpace::StateType*>(state))
      state = wrapper->getState();
    if (!has_base) {
      const auto* rv = state->as<ob::RealVectorStateSpace::StateType>();
      for (int i = 0; i < dim(); ++i) out[i] = rv->values[i];
      return;
    }
    const auto* pose = se2(state);
    out[0] = pose->getX();
    out[1] = pose->getY();
    out[2] = pose->getYaw();
    if (base_only) return;
    const auto* rv = state->as<ob::CompoundStateSpace::StateType>()
                         ->as<ob::RealVectorStateSpace::StateType>(1);
    for (int i = 3; i < dim(); ++i) out[i] = rv->values[i - 3];
  }

  /// Full-body configuration from active-DOF values.
  void expand(const double* values, float* q) const {
    std::copy(frozen.begin(), frozen.end(), q);
    for (int i = 0; i < dim(); ++i)
      q[active[i]] = static_cast<float>(values[i]);
  }

  auto configuration(const ob::State* state) const -> Robot::Configuration {
    std::array<double, Robot::dimension> values{};
    read(state, values.data());
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            q{};
    expand(values.data(), q.data());
    return Robot::Configuration(q.data());
  }
};

/// Collision-check ``n`` full-body configurations in SIMD blocks.
/// ``config(i, q)`` writes configuration ``i`` into ``q``.  Block ``b``
/// holds samples ``{b, b + m, b + 2m, ...}``, so the first blocks already
/// span the whole motion (FLASK Alg. 3; VAMP orders its edge checks the
/// same way).
template <typename Config>
inline auto samples_valid(const VampEnv& env, int n, Config&& config) -> bool {
  const int m = (n + static_cast<int>(kRake) - 1) / static_cast<int>(kRake);
  alignas(vamp::FloatVectorAlignment)
      std::array<float, Robot::dimension * kRake>
          buf{};
  std::array<float, Robot::dimension> q{};
  for (int b = 0; b < m; ++b) {
    for (std::size_t lane = 0; lane < kRake; ++lane) {
      config(std::min(static_cast<int>(lane) * m + b, n - 1), q.data());
      for (std::size_t d = 0; d < Robot::dimension; ++d)
        buf[d * kRake + lane] = q[d];
    }
    Robot::ConfigurationBlock<kRake> block(buf.data());
    const bool ok = env.attachments ? Robot::fkcc_attach<kRake>(env, block)
                                    : Robot::fkcc<kRake>(env, block);
    if (!ok) return false;
  }
  return true;
}

/// Validity checker for the *lower* (SE(2)) level of the multilevel
/// hierarchy when arm joints are active.  Uses the dedicated ``FetchBase``
/// VAMP robot model, which includes only the 14 base-link collision
/// spheres — the torso/arm/gripper links don't exist in its URDF.
///
/// This is a proper relaxation of the full 11-DOF validity check:
/// if the whole robot (with any arm configuration) fits at (x,y,θ),
/// then the base certainly fits, so this checker returns true.
/// Conversely, a pass here does NOT imply the full robot fits — that's
/// what the upper level is for.  The multilevel probabilistic
/// completeness guarantee holds with this abstraction.
///
/// The ``FetchBase`` header is generated by cricket from
/// ``fetch_planning/resources/robot/fetch/fetch_base.urdf`` (a trimmed
/// copy of the whole-body URDF stopping at ``base_link``) by
/// ``scripts/generate_fk.sh``.
class BaseOnlyValidityChecker : public ob::StateValidityChecker {
 public:
  BaseOnlyValidityChecker(const ob::SpaceInformationPtr& si, const VampEnv& env)
      : ob::StateValidityChecker(si), env_(env) {}

  auto isValid(const ob::State* state) const -> bool override {
    const auto* wrapper =
        dynamic_cast<const ob::WrapperStateSpace::StateType*>(state);
    const auto* se2 =
        wrapper ? wrapper->getState()->as<ob::SE2StateSpace::StateType>()
                : state->as<ob::SE2StateSpace::StateType>();

    // FetchBase is a 3-DOF VAMP robot — its configuration *is* the SE2
    // pose.  Broadcast into a rake-wide block and call its fkcc, which
    // does forward kinematics + collision check over the 14 base-link
    // spheres in a single SIMD sweep.
    BaseRobot::template ConfigurationBlock<kRake> block;
    block[0] = vamp::FloatVector<kRake, 1>(static_cast<float>(se2->getX()));
    block[1] = vamp::FloatVector<kRake, 1>(static_cast<float>(se2->getY()));
    block[2] = vamp::FloatVector<kRake, 1>(static_cast<float>(se2->getYaw()));
    return BaseRobot::template fkcc<kRake>(env_, block);
  }

 private:
  const VampEnv& env_;
};

/// Whole-body check of a state, with the inactive joints frozen.
class SubgroupValidityChecker : public ob::StateValidityChecker {
 public:
  SubgroupValidityChecker(const ob::SpaceInformationPtr& si, const VampEnv& env,
                          Subgroup subgroup)
      : ob::StateValidityChecker(si), env_(env), sub_(std::move(subgroup)) {}

  auto isValid(const ob::State* state) const -> bool override {
    const auto q = sub_.configuration(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
  }

 private:
  const VampEnv& env_;
  Subgroup sub_;
};

/// Whole-body check of a motion along the state space's own
/// interpolation, which is also what the planners, path simplification
/// and interpolation produce.  Without the base the motion is VAMP's
/// straight edge check; with it, the base follows its Dubins or
/// Reeds-Shepp curve (straight in the arm joints), sampled no farther
/// apart than VAMP's straight-edge resolution.
class SubgroupMotionValidator : public ob::MotionValidator {
 public:
  SubgroupMotionValidator(const ob::SpaceInformationPtr& si, const VampEnv& env,
                          Subgroup subgroup, double turning_radius)
      : ob::MotionValidator(si),
        env_(env),
        sub_(std::move(subgroup)),
        turning_radius_(turning_radius) {
    if (!sub_.has_base) return;
    const ob::StateSpace* base = si->getStateSpace().get();
    if (!sub_.base_only)
      base = base->as<ob::CompoundStateSpace>()->getSubspace(0).get();
    dubins_ = dynamic_cast<const ob::DubinsStateSpace*>(base);
    rs_ = dynamic_cast<const ob::ReedsSheppStateSpace*>(base);
  }

  auto checkMotion(const ob::State* s1, const ob::State* s2) const
      -> bool override {
    if (!sub_.has_base) {
      return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
          sub_.configuration(s1), sub_.configuration(s2), env_);
    }
    Motion motion(*this, s1, s2);
    return samples_valid(env_, motion.n + 1, [&](int i, float* q) {
      sub_.expand(motion.at(static_cast<double>(i) / motion.n), q);
    });
  }

  /// On failure, walks the samples ``t = k / n`` of the check in order to
  /// the last valid one.
  auto checkMotion(const ob::State* s1, const ob::State* s2,
                   std::pair<ob::State*, double>& last_valid) const
      -> bool override {
    if (checkMotion(s1, s2)) return true;
    Motion motion(*this, s1, s2);
    int k = 1;
    for (; k <= motion.n; ++k) {
      const double* values = motion.at(static_cast<double>(k) / motion.n);
      if (!samples_valid(env_, 1,
                         [&](int, float* q) { sub_.expand(values, q); }))
        break;
    }
    last_valid.second = static_cast<double>(k - 1) / motion.n;
    if (last_valid.first)
      si_->getStateSpace()->interpolate(s1, s2, last_valid.second,
                                        last_valid.first);
    return false;
  }

 private:
  const VampEnv& env_;
  Subgroup sub_;
  double turning_radius_;
  const ob::DubinsStateSpace* dubins_ = nullptr;
  const ob::ReedsSheppStateSpace* rs_ = nullptr;

  /// Active-DOF values along a motion: linear in the arm joints, and the
  /// base's Dubins or Reeds-Shepp curve (computed once).  ``n`` is the
  /// number of samples after the start.
  struct Motion {
    const SubgroupMotionValidator& v;
    std::array<double, Robot::dimension> a{}, b{}, values{};
    const ob::SE2StateSpace::StateType* from = nullptr;
    const ob::SE2StateSpace::StateType* to = nullptr;
    ob::DubinsStateSpace::PathType dubins_path;
    ob::ReedsSheppStateSpace::PathType rs_path;
    ob::State* scratch = nullptr;
    int n;

    Motion(const SubgroupMotionValidator& validator, const ob::State* s1,
           const ob::State* s2)
        : v(validator) {
      v.sub_.read(s1, a.data());
      v.sub_.read(s2, b.data());
      double joints = 0.0;
      for (int i = v.sub_.has_base ? 3 : 0; i < v.sub_.dim(); ++i)
        joints += (b[i] - a[i]) * (b[i] - a[i]);
      double length = std::sqrt(joints);
      if (v.sub_.has_base) {
        from = v.sub_.se2(s1);
        to = v.sub_.se2(s2);
        double curve;
        if (v.dubins_) {
          dubins_path = v.dubins_->getPath(from, to);
          curve = dubins_path.length();
          scratch = v.dubins_->allocState();
        } else {
          rs_path = v.rs_->getPath(from, to);
          curve = rs_path.length();
          scratch = v.rs_->allocState();
        }
        // A unit of normalized curve length moves the base by the
        // turning radius in the plane, plus one radian of yaw on arcs.
        length += curve * std::hypot(v.turning_radius_, 1.0);
        n = std::max(1,
                     static_cast<int>(std::ceil(length * Robot::resolution)));
      } else {
        // The samples of VAMP's straight edge check, in blocks of kRake.
        n = static_cast<int>(kRake) *
            std::max(1, static_cast<int>(
                            std::ceil(length / kRake * Robot::resolution)));
      }
    }

    ~Motion() {
      if (v.dubins_) v.dubins_->freeState(scratch);
      if (v.rs_) v.rs_->freeState(scratch);
    }

    /// Active-DOF values at fraction ``t`` of the motion.
    auto at(double t) -> const double* {
      for (int i = 0; i < v.sub_.dim(); ++i)
        values[i] = a[i] + t * (b[i] - a[i]);
      if (v.sub_.has_base) {
        bool first = false;
        if (v.dubins_) {
          v.dubins_->interpolate(from, to, t, first, dubins_path, scratch);
        } else {
          v.rs_->interpolate(from, to, t, first, rs_path, scratch);
        }
        const auto* pose = scratch->as<ob::SE2StateSpace::StateType>();
        values[0] = pose->getX();
        values[1] = pose->getY();
        values[2] = pose->getYaw();
      }
      return values.data();
    }
  };
};

}  // namespace fetch_planning
