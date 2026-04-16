/**
 * CompiledCost — an ompl::base::StateCostIntegralObjective that
 * dispatches stateCost() to a CasADi-generated C function loaded
 * at runtime via dlopen.
 *
 * The CasADi function is produced by
 * ``fetch_planning.planning.costs.Cost`` and has the uniform
 * "1 input, 2 outputs" ABI shared with CompiledConstraint:
 *
 *      q  →  [scalar_cost, gradient(q)]
 *
 * OMPL integrates the per-state scalar along each motion
 * (trapezoidal rule, optionally sub-sampled via
 * ``enableMotionCostInterpolation``).  This is the standard soft-cost
 * plug-in point for asymptotically-optimal planners such as RRT*,
 * BIT*, AIT*, and the multilevel QRRT*.
 *
 * For whole-body / base-including subgroups the planner combines this
 * user-supplied objective on top of the ``PenalizedReedsSheppStateSpace``
 * distance penalty, so the non-holonomic reverse shaping is preserved
 * — CompiledCost is additive, not a replacement for the base objective.
 *
 * The dlopen'd handle and scratch buffers are owned by
 * ``CostLibrary`` so that the thin ``CompiledCost`` wrapper can be
 * reconstructed on each plan() call against whatever
 * ``SpaceInformation`` the planner happens to be using at that time
 * (flat vs. constrained vs. multilevel top) — the shared library is
 * only loaded once per user-defined cost.
 */

#pragma once

#include <dlfcn.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/objectives/StateCostIntegralObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/WrapperStateSpace.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "validity.hpp"  // extract_real_state

namespace fetch_planning {

namespace ob = ompl::base;

// CasADi C ABI — casadi_int is long long on 64-bit Linux.
using casadi_cost_fn = int (*)(const double** arg, double** res, long long* iw,
                               double* w, int mem);
using casadi_cost_work_fn = int (*)(long long* sz_arg, long long* sz_res,
                                    long long* sz_iw, long long* sz_w);

/// Owns a dlopen'd CasADi cost library plus its scratch buffers.
/// Thread-unsafe: OMPL calls stateCost() serially from a single
/// planner thread, so we store the scratch inline without locking.
class CostLibrary {
 public:
  CostLibrary(unsigned int ambient_dim, const std::string& so_path,
              const std::string& symbol_name)
      : ambient_dim_(ambient_dim),
        so_path_(so_path),
        symbol_name_(symbol_name) {
    handle_ = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_)
      throw std::runtime_error("CompiledCost: dlopen failed: " +
                               std::string(dlerror()));

    fn_ = reinterpret_cast<casadi_cost_fn>(dlsym(handle_, symbol_name.c_str()));
    if (!fn_) {
      dlclose(handle_);
      throw std::runtime_error("CompiledCost: dlsym('" + symbol_name +
                               "') failed: " + std::string(dlerror()));
    }
    std::string work_name = symbol_name + "_work";
    work_fn_ = reinterpret_cast<casadi_cost_work_fn>(
        dlsym(handle_, work_name.c_str()));
    if (!work_fn_) {
      dlclose(handle_);
      throw std::runtime_error("CompiledCost: dlsym('" + work_name +
                               "') failed: " + std::string(dlerror()));
    }

    long long sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
    work_fn_(&sz_arg, &sz_res, &sz_iw, &sz_w);
    iw_.resize(static_cast<std::size_t>(sz_iw));
    w_.resize(static_cast<std::size_t>(sz_w));
    grad_scratch_.resize(ambient_dim_);

    if (sz_arg != 1 || sz_res != 2) {
      dlclose(handle_);
      throw std::runtime_error(
          "CompiledCost: generated function has wrong signature "
          "(expected 1 input, 2 outputs; got " +
          std::to_string(sz_arg) + " in, " + std::to_string(sz_res) + " out)");
    }
  }

  ~CostLibrary() {
    if (handle_) dlclose(handle_);
  }

  CostLibrary(const CostLibrary&) = delete;
  CostLibrary& operator=(const CostLibrary&) = delete;

  /// Evaluate the scalar cost at ``q``.  The gradient output is
  /// discarded into a scratch buffer so the ABI matches Constraint's
  /// ``[residual, jacobian]``.
  double evaluate(const double* q) const {
    double value = 0.0;
    const double* arg[1] = {q};
    double* res[2] = {&value, grad_scratch_.data()};
    fn_(arg, res, iw_.data(), w_.data(), 0);
    return value;
  }

  unsigned int ambient_dim() const { return ambient_dim_; }
  const std::string& so_path() const { return so_path_; }
  const std::string& symbol_name() const { return symbol_name_; }

 private:
  unsigned int ambient_dim_;
  std::string so_path_;
  std::string symbol_name_;
  void* handle_ = nullptr;
  casadi_cost_fn fn_ = nullptr;
  casadi_cost_work_fn work_fn_ = nullptr;
  mutable std::vector<long long> iw_;
  mutable std::vector<double> w_;
  mutable std::vector<double> grad_scratch_;
};

/// Adapter: OMPL ``StateCostIntegralObjective`` backed by a
/// ``CostLibrary``.  One instance is built per plan() call — it is
/// cheap because the shared library lives in the library object.
///
/// The state layout is one of three:
///   * ``kPlain``    — RealVectorStateSpace(active_dim) (arm-only).
///   * ``kSE2Only``  — SE2/ReedsShepp state (base-only subgroups).
///   * ``kCompound`` — CompoundStateSpace(SE2/RS + RealVector) for
///                     subgroups that include the base AND arm joints,
///                     including the multilevel "top" space.
/// The Cost CasADi expression is always written against the planner's
/// active-DOF vector ``q`` (length ``active_dim``), with base joints in
/// indices [0, base_dim) when present.
class CompiledCost : public ob::StateCostIntegralObjective {
 public:
  enum class Layout { kPlain, kSE2Only, kCompound };

  CompiledCost(const ob::SpaceInformationPtr& si,
               std::shared_ptr<CostLibrary> lib, double weight, Layout layout,
               int active_dim, bool interpolate_motion = true)
      : ob::StateCostIntegralObjective(si, interpolate_motion),
        lib_(std::move(lib)),
        weight_(weight),
        layout_(layout),
        active_dim_(active_dim),
        scratch_(static_cast<std::size_t>(active_dim)) {
    setCostThreshold(ob::Cost(std::numeric_limits<double>::infinity()));
  }

  ob::Cost stateCost(const ob::State* s) const override {
    fill_active_config(s, scratch_.data());
    const double value = lib_->evaluate(scratch_.data());
    return ob::Cost(weight_ * value);
  }

 private:
  std::shared_ptr<CostLibrary> lib_;
  double weight_;
  Layout layout_;
  int active_dim_;
  mutable std::vector<double> scratch_;

  void fill_active_config(const ob::State* state, double* out) const {
    if (layout_ == Layout::kPlain) {
      const auto* rv = extract_real_state(state);
      for (int i = 0; i < active_dim_; ++i) out[i] = rv->values[i];
      return;
    }
    if (layout_ == Layout::kSE2Only) {
      const auto* wrapper =
          dynamic_cast<const ob::WrapperStateSpace::StateType*>(state);
      const auto* se2 =
          wrapper ? wrapper->getState()->as<ob::SE2StateSpace::StateType>()
                  : state->as<ob::SE2StateSpace::StateType>();
      out[0] = se2->getX();
      out[1] = se2->getY();
      out[2] = se2->getYaw();
      return;
    }
    // kCompound — first 3 entries are SE2 base; remainder is RealVector.
    const auto* wrapper =
        dynamic_cast<const ob::WrapperStateSpace::StateType*>(state);
    const auto* compound =
        wrapper ? wrapper->getState()->as<ob::CompoundStateSpace::StateType>()
                : state->as<ob::CompoundStateSpace::StateType>();
    const auto* se2 = compound->as<ob::SE2StateSpace::StateType>(0);
    const auto* rv = compound->as<ob::RealVectorStateSpace::StateType>(1);
    out[0] = se2->getX();
    out[1] = se2->getY();
    out[2] = se2->getYaw();
    for (int i = 3; i < active_dim_; ++i) out[i] = rv->values[i - 3];
  }
};

}  // namespace fetch_planning
