/**
 * OMPL + VAMP Python extension — nanobind bindings.
 *
 * The actual planner, validity checkers, constraint primitives, and
 * pinocchio robot loader live in self-contained internal headers
 * under this directory.  This file is intentionally kept thin: it
 * only imports those headers and exposes the C++ API to Python via
 * nanobind.  If you find yourself adding more than a few lines of
 * non-binding code here, that's a sign it belongs in one of the
 * internal headers instead.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "planner.hpp"

namespace nb = nanobind;
using fetch_planning::OmplVampPlanner;
using fetch_planning::PlanResult;

NB_MODULE(_ompl_vamp, m) {
  m.doc() =
      "OMPL + VAMP C++ planning extension for whole-body Fetch "
      "(11 DOF: 3 nonholonomic base + 8 arm_with_torso).";

  nb::class_<PlanResult>(m, "PlanResult")
      .def_ro("solved", &PlanResult::solved)
      .def_ro("path", &PlanResult::path)
      .def_ro("planning_time_ns", &PlanResult::planning_time_ns)
      .def_ro("path_cost", &PlanResult::path_cost);

  nb::class_<OmplVampPlanner>(m, "OmplVampPlanner")
      .def(nb::init<std::vector<int>, std::vector<double>, int, double, double>(),
           "Create a planner.\n\n"
           "active_indices: joints this planner controls.\n"
           "frozen_config: full-body config for inactive joints.\n"
           "base_dim: how many leading active indices form the "
           "nonholonomic base (0 = arm-only, 3 = SE2 base).\n"
           "turning_radius: Reeds-Shepp turning radius (metres).\n"
           "reverse_penalty: cost multiplier for reverse segments "
           "(1.0 = no penalty, >1.0 = discourage backing up).",
           nb::arg("active_indices"), nb::arg("frozen_config"),
           nb::arg("base_dim") = 0,
           nb::arg("turning_radius") = 0.2,
           nb::arg("reverse_penalty") = 1.0)
      .def("set_base_bounds", &OmplVampPlanner::set_base_bounds,
           nb::arg("x_lo"), nb::arg("x_hi"), nb::arg("y_lo"), nb::arg("y_hi"),
           nb::arg("theta_lo") = -M_PI, nb::arg("theta_hi") = M_PI)
      .def("add_pointcloud", &OmplVampPlanner::add_pointcloud,
           nb::arg("points"), nb::arg("r_min"), nb::arg("r_max"),
           nb::arg("point_radius"))
      .def("add_sphere", &OmplVampPlanner::add_sphere, nb::arg("center"),
           nb::arg("radius"))
      .def("clear_environment", &OmplVampPlanner::clear_environment)
      .def("add_compiled_constraint", &OmplVampPlanner::add_compiled_constraint,
           nb::arg("so_path"), nb::arg("symbol_name"), nb::arg("ambient_dim"),
           nb::arg("co_dim"))
      .def("clear_constraints", &OmplVampPlanner::clear_constraints)
      .def("num_constraints", &OmplVampPlanner::num_constraints)
      .def("plan", &OmplVampPlanner::plan, nb::arg("start"), nb::arg("goal"),
           nb::arg("planner_name") = "rrtc", nb::arg("time_limit") = 10.0,
           nb::arg("simplify") = true, nb::arg("interpolate") = true)
      .def("validate", &OmplVampPlanner::validate, nb::arg("config"))
      .def("dimension", &OmplVampPlanner::dimension)
      .def("lower_bounds", &OmplVampPlanner::lower_bounds)
      .def("upper_bounds", &OmplVampPlanner::upper_bounds)
      .def("min_max_radii", &OmplVampPlanner::min_max_radii);
}