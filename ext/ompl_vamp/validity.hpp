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
 *  2. ``kCompound`` — a CompoundStateSpace of ReedsSheppStateSpace(ρ) +
 *     RealVectorStateSpace(active_dim_ − 3). Used when the active
 *     subgroup includes base joints (e.g. "fetch_base", "fetch_base_arm",
 *     "fetch_whole_body"). The SE(2) subspace carries (base_x, base_y,
 *     base_theta) and its built-in distance()/interpolate() enforces the
 *     nonholonomic constraint. State extraction reads SE(2) and the
 *     RealVector subspace separately, then concatenates them into a
 *     VAMP configuration in URDF order (base first, then arm).
 *
 * In all modes collision checking is delegated to VAMP's SIMD
 * ``validate_motion`` (resolution 1 for single states, full robot
 * resolution for motion edges).
 */

#pragma once

#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/State.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/WrapperStateSpace.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vamp/collision/environment.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/fetch_whole_body.hh>
#include <vector>

namespace fetch_planning {

namespace ob = ompl::base;

using Robot = vamp::robots::FetchWholeBody;
inline constexpr std::size_t kRake = vamp::FloatVectorWidth;
using VampEnv = vamp::collision::Environment<vamp::FloatVector<kRake>>;
using FloatEnv = vamp::collision::Environment<float>;

// Fetch whole-body DOF layout: the first 3 indices are the base, the
// remaining 8 are the arm_with_torso joints in URDF order.
inline constexpr int kBaseDim = 3;
inline constexpr int kArmDim = 8;
static_assert(Robot::dimension == kBaseDim + kArmDim,
              "FetchWholeBody must be 11 DOF (3 base + 8 arm_with_torso).");

// Given an OMPL state from either a plain RealVectorStateSpace or a
// ConstrainedStateSpace wrapper, return the underlying RealVector state.
// ConstrainedStateSpace inherits from WrapperStateSpace, so we unwrap it
// before casting.
inline auto extract_real_state(const ob::State *state)
    -> const ob::RealVectorStateSpace::StateType * {
  if (auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state)) {
    return wrapper->getState()->as<ob::RealVectorStateSpace::StateType>();
  }
  return state->as<ob::RealVectorStateSpace::StateType>();
}

// Read the SE(2) component and the RealVector component out of a
// compound state (base active). Writes into `out_base` (size 3) and
// `out_arm` (size active_dim_ − 3).
inline void extract_compound_state(const ob::State *state,
                                   float *out_base,
                                   float *out_arm,
                                   std::size_t arm_dim) noexcept {
  const auto *wrapper =
      dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
  const auto *compound =
      wrapper
          ? wrapper->getState()->as<ob::CompoundStateSpace::StateType>()
          : state->as<ob::CompoundStateSpace::StateType>();

  const auto *se2 = compound->as<ob::SE2StateSpace::StateType>(0);
  const auto *rv =
      compound->as<ob::RealVectorStateSpace::StateType>(1);

  out_base[0] = static_cast<float>(se2->getX());
  out_base[1] = static_cast<float>(se2->getY());
  out_base[2] = static_cast<float>(se2->getYaw());
  for (std::size_t i = 0; i < arm_dim; ++i) {
    out_arm[i] = static_cast<float>(rv->values[i]);
  }
}

// ─── Full-body checkers (11-DOF compound state) ─────────────────────
//
// Used when every joint is active, i.e. the default "fetch_whole_body"
// planner without any frozen_config. The state space is the compound
// ReedsSheppStateSpace + RealVectorStateSpace.

class FetchFullBodyValidityChecker : public ob::StateValidityChecker {
 public:
  FetchFullBodyValidityChecker(const ob::SpaceInformationPtr &si,
                               const VampEnv &env)
      : ob::StateValidityChecker(si), env_(env) {}

  auto isValid(const ob::State *state) const -> bool override {
    auto config = compound_to_vamp(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(config, config,
                                                            env_);
  }

 private:
  const VampEnv &env_;

  static auto compound_to_vamp(const ob::State *state) -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    extract_compound_state(state, buf.data(), buf.data() + kBaseDim, kArmDim);
    return Robot::Configuration(buf.data());
  }
};

class FetchFullBodyMotionValidator : public ob::MotionValidator {
 public:
  FetchFullBodyMotionValidator(const ob::SpaceInformationPtr &si,
                               const VampEnv &env)
      : ob::MotionValidator(si), env_(env) {}

  auto checkMotion(const ob::State *s1, const ob::State *s2) const
      -> bool override {
    return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
        compound_to_vamp(s1), compound_to_vamp(s2), env_);
  }

  auto checkMotion(const ob::State *s1, const ob::State *s2,
                   std::pair<ob::State *, double> &last_valid) const
      -> bool override {
    last_valid.first = nullptr;
    last_valid.second = 0.0;
    return checkMotion(s1, s2);
  }

 private:
  const VampEnv &env_;

  static auto compound_to_vamp(const ob::State *state) -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    extract_compound_state(state, buf.data(), buf.data() + kBaseDim, kArmDim);
    return Robot::Configuration(buf.data());
  }
};

// ─── Subgroup checkers (reduced dim → full config → VAMP) ───────────
//
// Subgroup planning projects the 11-DOF full-body state onto the
// active_indices subset. The rest of the body is pinned to frozen_config.
// `has_base` selects the state layout: when true, the first three active
// indices are read from a CompoundStateSpace(SE2 + RealVector); when
// false, the whole state is a plain RealVectorStateSpace.

class SubgroupValidityChecker : public ob::StateValidityChecker {
 public:
  SubgroupValidityChecker(const ob::SpaceInformationPtr &si, const VampEnv &env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config, bool has_base)
      : ob::StateValidityChecker(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)),
        has_base_(has_base) {}

  auto isValid(const ob::State *state) const -> bool override {
    auto config = expand(state);
    return vamp::planning::validate_motion<Robot, kRake, 1>(config, config,
                                                            env_);
  }

 private:
  const VampEnv &env_;
  std::vector<int> active_;
  std::vector<float> frozen_;
  bool has_base_;

  auto expand(const ob::State *state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());

    if (has_base_) {
      // First 3 active indices are always (base_x, base_y, base_theta)
      // in that order. The rest are RealVector joints.
      const auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
      const auto *compound =
          wrapper
              ? wrapper->getState()->as<ob::CompoundStateSpace::StateType>()
              : state->as<ob::CompoundStateSpace::StateType>();
      const auto *se2 = compound->as<ob::SE2StateSpace::StateType>(0);
      const auto *rv =
          compound->as<ob::RealVectorStateSpace::StateType>(1);

      buf[active_[0]] = static_cast<float>(se2->getX());
      buf[active_[1]] = static_cast<float>(se2->getY());
      buf[active_[2]] = static_cast<float>(se2->getYaw());
      for (std::size_t i = 3; i < active_.size(); ++i) {
        buf[active_[i]] = static_cast<float>(rv->values[i - 3]);
      }
    } else {
      const auto *rv = extract_real_state(state);
      for (std::size_t i = 0; i < active_.size(); ++i) {
        buf[active_[i]] = static_cast<float>(rv->values[i]);
      }
    }
    return Robot::Configuration(buf.data());
  }
};

class SubgroupMotionValidator : public ob::MotionValidator {
 public:
  SubgroupMotionValidator(const ob::SpaceInformationPtr &si, const VampEnv &env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config, bool has_base)
      : ob::MotionValidator(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)),
        has_base_(has_base) {}

  auto checkMotion(const ob::State *s1, const ob::State *s2) const
      -> bool override {
    return vamp::planning::validate_motion<Robot, kRake, Robot::resolution>(
        expand(s1), expand(s2), env_);
  }

  auto checkMotion(const ob::State *s1, const ob::State *s2,
                   std::pair<ob::State *, double> &last_valid) const
      -> bool override {
    last_valid.first = nullptr;
    last_valid.second = 0.0;
    return checkMotion(s1, s2);
  }

 private:
  const VampEnv &env_;
  std::vector<int> active_;
  std::vector<float> frozen_;
  bool has_base_;

  auto expand(const ob::State *state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());

    if (has_base_) {
      const auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
      const auto *compound =
          wrapper
              ? wrapper->getState()->as<ob::CompoundStateSpace::StateType>()
              : state->as<ob::CompoundStateSpace::StateType>();
      const auto *se2 = compound->as<ob::SE2StateSpace::StateType>(0);
      const auto *rv =
          compound->as<ob::RealVectorStateSpace::StateType>(1);

      buf[active_[0]] = static_cast<float>(se2->getX());
      buf[active_[1]] = static_cast<float>(se2->getY());
      buf[active_[2]] = static_cast<float>(se2->getYaw());
      for (std::size_t i = 3; i < active_.size(); ++i) {
        buf[active_[i]] = static_cast<float>(rv->values[i - 3]);
      }
    } else {
      const auto *rv = extract_real_state(state);
      for (std::size_t i = 0; i < active_.size(); ++i) {
        buf[active_[i]] = static_cast<float>(rv->values[i]);
      }
    }
    return Robot::Configuration(buf.data());
  }
};

}  // namespace fetch_planning
