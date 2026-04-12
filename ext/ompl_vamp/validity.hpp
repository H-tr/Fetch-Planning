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

// Robot dimension (used only by the full-body validity checkers below).
inline constexpr int kFullDim = Robot::dimension;

// Unwrap a ConstrainedStateSpace wrapper (if present) to get the
// underlying RealVector state.
inline auto extract_real_state(const ob::State *state)
    -> const ob::RealVectorStateSpace::StateType * {
  if (auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state)) {
    return wrapper->getState()->as<ob::RealVectorStateSpace::StateType>();
  }
  return state->as<ob::RealVectorStateSpace::StateType>();
}

// ─── Validity / motion checkers ─────────────────────────────────────
//
// ``has_base`` selects the state layout: when true, the leading
// active indices are read from a CompoundStateSpace(SE2 + RealVector)
// or a bare SE2 (base_only); when false, the whole state is a plain
// RealVectorStateSpace.

class SubgroupValidityChecker : public ob::StateValidityChecker {
 public:
  SubgroupValidityChecker(const ob::SpaceInformationPtr &si, const VampEnv &env,
                          std::vector<int> active_indices,
                          std::vector<float> frozen_config, bool has_base,
                          bool base_only = false)
      : ob::StateValidityChecker(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)),
        has_base_(has_base),
        base_only_(base_only) {}

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
  bool base_only_;

  auto expand(const ob::State *state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());

    if (base_only_) {
      // SE2 state only — no compound wrapper, no arm joints.
      const auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
      const auto *se2 =
          wrapper
              ? wrapper->getState()->as<ob::SE2StateSpace::StateType>()
              : state->as<ob::SE2StateSpace::StateType>();
      buf[active_[0]] = static_cast<float>(se2->getX());
      buf[active_[1]] = static_cast<float>(se2->getY());
      buf[active_[2]] = static_cast<float>(se2->getYaw());
    } else if (has_base_) {
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
                          std::vector<float> frozen_config, bool has_base,
                          bool base_only = false)
      : ob::MotionValidator(si),
        env_(env),
        active_(std::move(active_indices)),
        frozen_(std::move(frozen_config)),
        has_base_(has_base),
        base_only_(base_only) {}

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
  bool base_only_;

  auto expand(const ob::State *state) const -> Robot::Configuration {
    alignas(Robot::Configuration::S::Alignment)
        std::array<float, Robot::Configuration::num_scalars_rounded>
            buf{};
    std::copy(frozen_.begin(), frozen_.end(), buf.begin());

    if (base_only_) {
      const auto *wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType *>(state);
      const auto *se2 =
          wrapper
              ? wrapper->getState()->as<ob::SE2StateSpace::StateType>()
              : state->as<ob::SE2StateSpace::StateType>();
      buf[active_[0]] = static_cast<float>(se2->getX());
      buf[active_[1]] = static_cast<float>(se2->getY());
      buf[active_[2]] = static_cast<float>(se2->getYaw());
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
