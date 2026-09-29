/**
 * Time-optimal path parameterization Python extension (TOPP-RA).
 *
 * Thin nanobind wrapper around the C++ core of toppra (Pham & Pham,
 * "A New Approach to Time-Optimal Path Parameterization Based on
 * Reachability Analysis", T-RO 2018), vendored under toppra/ from
 * https://github.com/hungpham2511/toppra (commit 8121e9a, MIT license)
 * with only the pieces used here: joint velocity / acceleration
 * constraints, the built-in Seidel LP solver, piecewise-polynomial paths
 * and the constant-acceleration parametrizer.
 *
 * The piecewise-linear waypoint path is resampled at ``knot_spacing`` and
 * joined by a natural cubic spline parameterized by arc length; on the
 * straight segments the spline stays within about a tenth of the spacing
 * of the original path.  TOPP-RA then computes the fastest velocity
 * profile along it that starts and ends at rest.
 */
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <optional>
#include <toppra/algorithm/toppra.hpp>
#include <toppra/constraint/linear_joint_acceleration.hpp>
#include <toppra/constraint/linear_joint_velocity.hpp>
#include <toppra/geometric_path/piecewise_poly_path.hpp>
#include <toppra/parametrizer/const_accel.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

struct ToppraTrajectory {
  std::shared_ptr<toppra::PiecewisePolyPath> path;
  std::shared_ptr<toppra::parametrizer::ConstAccel> timing;

  double duration() const {
    const auto interval = timing->pathInterval();
    return interval[1] - interval[0];
  }

  // Evaluate derivative ``order`` at each time, as a (T, ndof) matrix.
  Eigen::MatrixXd eval(const Eigen::VectorXd& times, int order) const {
    const Eigen::VectorXd t = times.cwiseMax(0.0).cwiseMin(duration());
    const toppra::Vectors rows = timing->eval(t, order);
    Eigen::MatrixXd out(static_cast<Eigen::Index>(rows.size()), path->dof());
    for (std::size_t i = 0; i < rows.size(); ++i)
      out.row(static_cast<Eigen::Index>(i)) = rows[i].transpose();
    return out;
  }

  Eigen::VectorXd at(double t, int order) const {
    return eval(Eigen::VectorXd::Constant(1, t), order).row(0).transpose();
  }

  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> sample(
      const Eigen::VectorXd& times) const {
    return {eval(times, 0), eval(times, 1), eval(times, 2)};
  }

  // Uniform rollout with step dt; the last sample is the endpoint.
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
  sample_uniform(double dt) const {
    const double total = duration();
    const auto n = static_cast<Eigen::Index>(std::floor(total / dt));
    Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(n + 1, 0.0, n * dt);
    if (total - times[n] > 1e-12) {
      times.conservativeResize(n + 2);
      times[n + 1] = total;
    }
    auto [p, v, a] = sample(times);
    return {std::move(times), std::move(p), std::move(v), std::move(a)};
  }
};

std::optional<ToppraTrajectory> compute_trajectory(
    const Eigen::Ref<const Eigen::MatrixXd>& waypoints,
    const Eigen::Ref<const Eigen::VectorXd>& max_velocity,
    const Eigen::Ref<const Eigen::VectorXd>& max_acceleration,
    double knot_spacing) {
  // Knots every <= knot_spacing along each segment (zero-length segments
  // add none), at their arc length.
  toppra::Vectors knots{waypoints.row(0).transpose()};
  std::vector<double> s{0.0};
  for (Eigen::Index i = 1; i < waypoints.rows(); ++i) {
    const Eigen::VectorXd a = waypoints.row(i - 1).transpose();
    const Eigen::VectorXd b = waypoints.row(i).transpose();
    const double length = (b - a).norm();
    const int n = static_cast<int>(std::ceil(length / knot_spacing - 1e-9));
    for (int k = 1; k <= n; ++k) {
      knots.push_back(a + (b - a) * (static_cast<double>(k) / n));
      s.push_back(s.back() + length / n);
    }
  }
  auto path = std::make_shared<toppra::PiecewisePolyPath>(
      toppra::PiecewisePolyPath::CubicSpline(
          knots, Eigen::Map<const toppra::Vector>(s.data(), s.size()),
          {toppra::BoundaryCond("natural"), toppra::BoundaryCond("natural")}));

  auto velocity = std::make_shared<toppra::constraint::LinearJointVelocity>(
      -max_velocity, max_velocity);
  auto acceleration =
      std::make_shared<toppra::constraint::LinearJointAcceleration>(
          -max_acceleration, max_acceleration);
  acceleration->discretizationType(toppra::DiscretizationType::Interpolation);

  toppra::algorithm::TOPPRA algo({velocity, acceleration}, path);
  // Grid on the spline knots, refined until the path is locally
  // quadratic to 1e-4 rad between grid points (toppra's defaults).
  algo.setGridpoints(path->proposeGridpoints(
      1e-4, 100, 0.05, 100,
      Eigen::Map<const toppra::Vector>(s.data(), s.size())));
  if (algo.computePathParametrization(0, 0) != toppra::ReturnCode::OK)
    return std::nullopt;
  const auto& data = algo.getParameterizationData();
  auto timing = std::make_shared<toppra::parametrizer::ConstAccel>(
      path, data.gridpoints, data.parametrization);
  return ToppraTrajectory{std::move(path), std::move(timing)};
}

}  // namespace

NB_MODULE(_time_parameterization, m) {
  m.doc() = "Time-optimal path parameterization (TOPP-RA, Pham & Pham 2018).";

  nb::class_<ToppraTrajectory>(m, "ToppraTrajectory")
      .def_prop_ro("duration", &ToppraTrajectory::duration,
                   "Total duration of the trajectory (seconds).")
      .def(
          "position",
          [](const ToppraTrajectory& self, double t) { return self.at(t, 0); },
          nb::arg("t"), "Configuration at time ``t`` (seconds).")
      .def(
          "velocity",
          [](const ToppraTrajectory& self, double t) { return self.at(t, 1); },
          nb::arg("t"), "Joint velocity at time ``t`` (seconds).")
      .def(
          "acceleration",
          [](const ToppraTrajectory& self, double t) { return self.at(t, 2); },
          nb::arg("t"), "Joint acceleration at time ``t`` (seconds).")
      .def("sample", &ToppraTrajectory::sample, nb::arg("times"),
           "Sample (position, velocity, acceleration) at each time in "
           "``times``.\nReturns a 3-tuple of ``(T, ndof)`` matrices.")
      .def("sample_uniform", &ToppraTrajectory::sample_uniform, nb::arg("dt"),
           "Uniformly-spaced rollout with step ``dt``.\n"
           "Returns ``(times, positions, velocities, accelerations)``; "
           "``times`` always include ``0`` and ``duration``.");

  m.def("compute_trajectory", &compute_trajectory, nb::arg("waypoints"),
        nb::arg("max_velocity"), nb::arg("max_acceleration"),
        nb::arg("knot_spacing") = 0.1,
        "Time-optimal trajectory along the piecewise-linear path "
        "``waypoints`` (N, ndof), starting and ending at rest.\n\n"
        "The path is resampled every ``knot_spacing`` and splined; the "
        "result deviates from it by about ``knot_spacing / 10``.  Returns "
        "``None`` if TOPP-RA finds no feasible parameterization.");
}
