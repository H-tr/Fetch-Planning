/**
 * Closed-form local motions in a differentially flat output space.
 *
 * Building blocks for the FLASK kinodynamic planner (Duong et al.,
 * "Ultrafast Sampling-based Kinodynamic Planning via Differential
 * Flatness", T-RO 2026), specialised to a mobile manipulator whose
 * leading active DOF may be a nonholonomic diff-drive base:
 *
 *   * Flat output ``y``: every arm / torso joint is its own flat output
 *     (fully actuated), and the base contributes its planar position
 *     (x, y).  The base heading is not a flat output — it is recovered
 *     from the direction of travel, ``theta = atan2(y_dot, x_dot) +
 *     gear * pi``, with ``gear`` = 0 (forward) or 1 (reverse).
 *   * Flat state ``z = (y, y_dot)`` with pseudo-control ``w = y_ddot``
 *     (order r = 2), so ``z`` follows a chain of double integrators.
 *   * Local motion between two flat states: the linear-quadratic
 *     minimum-time (LQMT) solution.  For r = 2 this is the cubic Hermite
 *     spline through both endpoint positions and velocities, and its
 *     cost ``J(T) = int |y_ddot|_R^2 dt + rho T`` is a rational function
 *     of the duration whose minimiser is a root of a quartic.
 *
 * A trajectory is a sequence of segments of two kinds:
 *
 *   * ``kDrive`` — cubic in every flat output; the heading follows the
 *     velocity.  At a base-rest endpoint (x_dot = y_dot = 0) the heading
 *     is the departure direction ``y_ddot(0)`` (arrival direction
 *     ``-y_ddot(T)``) and the yaw rate stays finite.
 *   * ``kSpin``  — the base is parked at a fixed (x, y) while its heading
 *     follows its own cubic with zero end rates (the diff-drive's
 *     rotate-in-place); arm joints are cubic as usual.
 *
 * Nothing here is robot-specific: callers say whether the leading three
 * active DOF are a base (``base_dim`` = 3) or not (``base_dim`` = 0).
 * Active-DOF vectors use the planner's layout ``[x, y, theta, arm...]``;
 * flat vectors drop theta: ``[x, y, arm...]``.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace fetch_planning {
namespace flat {

inline auto wrap_angle(double a) -> double {
  return std::remainder(a, 2.0 * M_PI);
}

/// Heading of a moving base: direction of travel, flipped when reversing.
inline auto heading_from_direction(double dx, double dy, int gear) -> double {
  return wrap_angle(std::atan2(dy, dx) + (gear != 0 ? M_PI : 0.0));
}

// ─── Scalar cubic ──────────────────────────────────────────────────────

/// ``p(t) = p0 + v0 t + c2 t^2 + c3 t^3`` on ``[0, T]``.
struct Cubic {
  double p0 = 0.0;
  double v0 = 0.0;
  double c2 = 0.0;
  double c3 = 0.0;

  /// Hermite fit: the unique cubic with p(0)=p0, p'(0)=v0, p(T)=p1,
  /// p'(T)=v1.  This is also the minimum-effort (LQMT, r = 2) motion.
  static auto hermite(double p0, double v0, double p1, double v1, double T)
      -> Cubic {
    const double d = p1 - p0;
    return {p0, v0, (3.0 * d - T * (2.0 * v0 + v1)) / (T * T),
            (-2.0 * d + T * (v0 + v1)) / (T * T * T)};
  }

  static auto constant(double p) -> Cubic { return {p, 0.0, 0.0, 0.0}; }

  auto pos(double t) const -> double {
    return p0 + t * (v0 + t * (c2 + t * c3));
  }
  auto vel(double t) const -> double {
    return v0 + t * (2.0 * c2 + 3.0 * c3 * t);
  }
  auto acc(double t) const -> double { return 2.0 * c2 + 6.0 * c3 * t; }
  auto jerk() const -> double { return 6.0 * c3; }

  /// ``int_0^T p''(t)^2 dt``.
  auto effort(double T) const -> double {
    return 4.0 * c2 * c2 * T + 12.0 * c2 * c3 * T * T +
           12.0 * c3 * c3 * T * T * T;
  }

  /// Exact ``max |p'(t)|`` on ``[0, T]`` (endpoints + the vertex of the
  /// quadratic velocity).
  auto max_abs_vel(double T) const -> double {
    double m = std::max(std::abs(vel(0.0)), std::abs(vel(T)));
    if (c3 != 0.0) {
      const double tv = -c2 / (3.0 * c3);
      if (tv > 0.0 && tv < T) m = std::max(m, std::abs(vel(tv)));
    }
    return m;
  }

  /// Exact ``max |p''(t)|`` on ``[0, T]`` (acceleration is linear).
  auto max_abs_acc(double T) const -> double {
    return std::max(std::abs(acc(0.0)), std::abs(acc(T)));
  }

  /// Exact ``[min p, max p]`` on ``[0, T]`` (endpoints + stationary
  /// points of the cubic).
  auto pos_range(double T) const -> std::pair<double, double> {
    double lo = std::min(pos(0.0), pos(T));
    double hi = std::max(pos(0.0), pos(T));
    auto visit = [&](double t) {
      if (t > 0.0 && t < T) {
        const double p = pos(t);
        lo = std::min(lo, p);
        hi = std::max(hi, p);
      }
    };
    // p'(t) = 3 c3 t^2 + 2 c2 t + v0
    const double A = 3.0 * c3, B = 2.0 * c2, C = v0;
    if (std::abs(A) < 1e-15) {
      if (std::abs(B) > 1e-15) visit(-C / B);
    } else {
      const double disc = B * B - 4.0 * A * C;
      if (disc >= 0.0) {
        const double s = std::sqrt(disc);
        visit((-B + s) / (2.0 * A));
        visit((-B - s) / (2.0 * A));
      }
    }
    return {lo, hi};
  }

  /// The same polynomial restricted to ``[s, s + T']``, re-based at 0.
  auto shifted(double s) const -> Cubic {
    return {pos(s), vel(s), c2 + 3.0 * c3 * s, c3};
  }
};

// ─── LQMT cost and optimal duration ────────────────────────────────────

/// ``J(T) = a / T^3 - b / T^2 + c / T + rho T`` — the minimum effort of
/// the Hermite cubic between two flat states over duration T, summed
/// over flat outputs with weights ``r_i`` (Example 4 of the paper, with
/// ``R = diag(r)``), plus the time penalty.
struct LqmtTerms {
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;

  void add(double r, double p0, double v0, double p1, double v1) {
    const double d = p1 - p0;
    a += r * 12.0 * d * d;
    b += r * 12.0 * (v0 + v1) * d;
    c += r * 4.0 * (v0 * v0 + v0 * v1 + v1 * v1);
  }

  auto cost(double T, double rho) const -> double {
    return a / (T * T * T) - b / (T * T) + c / T + rho * T;
  }
};

/// Real roots of ``x^4 + p x^2 + q x + r = 0`` (Ferrari's method with a
/// safeguarded-Newton resolvent root, then a Newton polish on the
/// quartic).  Returns the number of roots written to ``roots``.
inline auto solve_depressed_quartic(double p, double q, double r,
                                    std::array<double, 4>& roots) -> int {
  int n = 0;
  // Roots of x^2 + B x + C = 0, written in the cancellation-free form.
  auto quadratic = [&](double B, double C) {
    const double disc = B * B - 4.0 * C;
    if (disc < 0.0) return;
    const double qq = -0.5 * (B + std::copysign(std::sqrt(disc), B));
    roots[n++] = qq;
    roots[n++] = C / qq;
  };

  const double scale = 1.0 + std::abs(p) + std::sqrt(std::abs(r));
  if (std::abs(q) <= 1e-13 * scale * std::sqrt(scale)) {
    // Biquadratic: u^2 + p u + r = 0 with u = x^2.
    const double disc = p * p - 4.0 * r;
    if (disc < 0.0) return 0;
    const double s = std::sqrt(disc);
    for (double u : {(-p + s) * 0.5, (-p - s) * 0.5}) {
      if (u >= 0.0) {
        roots[n++] = std::sqrt(u);
        roots[n++] = -std::sqrt(u);
      }
    }
  } else {
    // Resolvent cubic g(m) = 8m^3 + 8p m^2 + (2p^2 - 8r) m - q^2 has
    // g(0) = -q^2 < 0, so it owns a root m > 0.
    auto g = [&](double m) {
      return ((8.0 * m + 8.0 * p) * m + (2.0 * p * p - 8.0 * r)) * m - q * q;
    };
    auto dg = [&](double m) {
      return (24.0 * m + 16.0 * p) * m + (2.0 * p * p - 8.0 * r);
    };
    double lo = 0.0, hi = 1.0;
    while (g(hi) < 0.0) hi *= 2.0;
    double m = hi;
    for (int it = 0; it < 200; ++it) {
      const double gm = g(m);
      if (gm > 0.0) {
        hi = m;
      } else {
        lo = m;
      }
      const double d = dg(m);
      double next = (d != 0.0) ? m - gm / d : 0.5 * (lo + hi);
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (std::abs(next - m) <= 1e-15 * std::max(1.0, m)) {
        m = next;
        break;
      }
      m = next;
    }
    const double s = std::sqrt(2.0 * m);
    const double t = q / (2.0 * s);
    quadratic(-s, 0.5 * p + m + t);
    quadratic(s, 0.5 * p + m - t);
  }

  for (int i = 0; i < n; ++i) {
    double x = roots[i];
    for (int it = 0; it < 3; ++it) {
      const double f = ((x * x + p) * x + q) * x + r;
      x -= f / ((4.0 * x * x + 2.0 * p) * x + q);
    }
    roots[i] = x;
  }
  return n;
}

/// Duration minimising ``J(T)`` over ``T > 0``: the best positive root
/// of ``dJ/dT = 0``, i.e. ``rho T^4 - c T^2 + 2 b T - 3 a = 0``
/// (Eq. 28 of the paper, generalised to per-output weights).  Since the
/// quartic is negative at 0 and positive at infinity, a positive root
/// exists unless the two states coincide.
inline auto optimal_time(const LqmtTerms& k, double rho) -> double {
  std::array<double, 4> roots{};
  const int n = solve_depressed_quartic(-k.c / rho, 2.0 * k.b / rho,
                                        -3.0 * k.a / rho, roots);
  double best_t = 0.0;
  double best_j = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    if (roots[i] > 0.0 && k.cost(roots[i], rho) < best_j) {
      best_j = k.cost(roots[i], rho);
      best_t = roots[i];
    }
  }
  return best_t;
}

// ─── Segments and trajectories ─────────────────────────────────────────

enum class SegmentKind : int { kDrive = 0, kSpin = 1 };

/// One local motion.  ``cubics`` holds one cubic per flat output (for a
/// spin the base x / y cubics are constant); ``theta`` is only used by
/// spins.  ``heading0`` / ``heading1`` record the base heading at the
/// segment ends — for a drive segment they are the limit directions at
/// rest endpoints, which is what ``eval`` falls back to when the base
/// speed vanishes.
struct Segment {
  SegmentKind kind = SegmentKind::kDrive;
  double t0 = 0.0;
  double T = 0.0;
  std::vector<Cubic> cubics;
  Cubic theta;
  int gear = 0;
  double heading0 = 0.0;
  double heading1 = 0.0;
  bool base_stationary = false;
};

/// Speeds below this are "at rest" for heading / yaw-rate purposes.
inline constexpr double kRestSpeed = 1e-9;

/// Planar base quantities of a drive segment at local time ``s``:
/// heading, yaw rate and yaw acceleration.  Uses the cubic's analytic
/// limits at rest endpoints.
struct BaseKinematics {
  double theta = 0.0;
  double omega = 0.0;
  double omega_dot = 0.0;
  double speed = 0.0;
};

inline auto drive_base_kinematics(const Segment& seg, double s)
    -> BaseKinematics {
  BaseKinematics out;
  if (seg.base_stationary) {
    out.theta = seg.heading0;
    return out;
  }
  const Cubic& cx = seg.cubics[0];
  const Cubic& cy = seg.cubics[1];
  const double vx = cx.vel(s), vy = cy.vel(s);
  const double ax = cx.acc(s), ay = cy.acc(s);
  const double jx = cx.jerk(), jy = cy.jerk();
  const double v2 = vx * vx + vy * vy;
  out.speed = std::sqrt(v2);
  // Scale-aware "is it moving" test: compare against the acceleration
  // times a tiny time step so rest endpoints are detected robustly.
  const double a_scale = std::sqrt(ax * ax + ay * ay) + 1e-12;
  if (out.speed > std::max(kRestSpeed, 1e-7 * a_scale * seg.T)) {
    out.theta = heading_from_direction(vx, vy, seg.gear);
    out.omega = (vx * ay - ax * vy) / v2;
    out.omega_dot = ((vx * jy - jx * vy) * v2 -
                     (vx * ay - ax * vy) * 2.0 * (vx * ax + vy * ay)) /
                    (v2 * v2);
    return out;
  }
  // At a rest endpoint: the velocity is ~ a(s) * (time since/until
  // rest), so the heading is the (signed) acceleration direction and
  // omega -> cross(a, j) / (2 |a|^2).
  const bool departure = s < 0.5 * seg.T;
  out.theta = departure ? seg.heading0 : seg.heading1;
  out.omega = (ax * jy - jx * ay) / (2.0 * (ax * ax + ay * ay));
  // The yaw acceleration at the rest instant has no simple closed-form
  // limit; evaluate slightly inside the segment.
  const double s2 = departure ? s + 1e-3 * seg.T : s - 1e-3 * seg.T;
  const double wx = cx.vel(s2), wy = cy.vel(s2);
  const double bx = cx.acc(s2), by = cy.acc(s2);
  const double w2 = wx * wx + wy * wy;
  out.omega_dot = ((wx * jy - jx * wy) * w2 -
                   (wx * by - bx * wy) * 2.0 * (wx * bx + wy * by)) /
                  (w2 * w2);
  return out;
}

/// Evaluate a segment at local time ``s`` into active-DOF position /
/// velocity / acceleration (any output pointer may be null).  Velocity
/// and acceleration for the base are world-frame ``(x_dot, y_dot,
/// theta_dot)`` and ``(x_ddot, y_ddot, theta_ddot)``.
inline void eval_segment(const Segment& seg, double s, int base_dim, double* q,
                         double* qd, double* qdd) {
  const int n_flat = static_cast<int>(seg.cubics.size());
  if (base_dim == 0) {
    for (int i = 0; i < n_flat; ++i) {
      const Cubic& c = seg.cubics[i];
      if (q) q[i] = c.pos(s);
      if (qd) qd[i] = c.vel(s);
      if (qdd) qdd[i] = c.acc(s);
    }
    return;
  }
  // Base (x, y) + arm dims.
  for (int f = 0; f < n_flat; ++f) {
    const int a = f < 2 ? f : f + 1;
    const Cubic& c = seg.cubics[f];
    if (q) q[a] = c.pos(s);
    if (qd) qd[a] = c.vel(s);
    if (qdd) qdd[a] = c.acc(s);
  }
  if (seg.kind == SegmentKind::kSpin) {
    if (q) q[2] = wrap_angle(seg.theta.pos(s));
    if (qd) qd[2] = seg.theta.vel(s);
    if (qdd) qdd[2] = seg.theta.acc(s);
    return;
  }
  const BaseKinematics bk = drive_base_kinematics(seg, s);
  if (q) q[2] = bk.theta;
  if (qd) qd[2] = bk.omega;
  if (qdd) qdd[2] = bk.omega_dot;
}

/// A time-parameterised flat trajectory: consecutive segments with
/// C^1 continuity in every flat output (position and velocity), plus
/// the base heading (continuous by construction).
class FlatTrajectory {
 public:
  FlatTrajectory() = default;
  FlatTrajectory(int active_dim, int base_dim)
      : active_dim_(active_dim), base_dim_(base_dim) {}

  void push(Segment seg) {
    seg.t0 = duration_;
    duration_ += seg.T;
    segments_.push_back(std::move(seg));
  }

  auto duration() const -> double { return duration_; }
  auto active_dim() const -> int { return active_dim_; }
  auto base_dim() const -> int { return base_dim_; }
  auto segments() const -> const std::vector<Segment>& { return segments_; }
  auto empty() const -> bool { return segments_.empty(); }

  /// Locate the segment containing ``t`` (clamped to [0, duration]).
  auto locate(double t) const -> std::pair<const Segment*, double> {
    t = std::clamp(t, 0.0, duration_);
    auto it = std::upper_bound(
        segments_.begin(), segments_.end(), t,
        [](double tt, const Segment& sg) { return tt < sg.t0; });
    const Segment& seg =
        (it == segments_.begin()) ? segments_.front() : *(it - 1);
    return {&seg, std::clamp(t - seg.t0, 0.0, seg.T)};
  }

  void evaluate(double t, double* q, double* qd, double* qdd) const {
    auto [seg, s] = locate(t);
    eval_segment(*seg, s, base_dim_, q, qd, qdd);
  }

  using Matrix = std::vector<std::vector<double>>;

  auto sample(const std::vector<double>& times) const
      -> std::tuple<Matrix, Matrix, Matrix> {
    Matrix P(times.size(), std::vector<double>(active_dim_));
    Matrix V = P, A = P;
    for (std::size_t k = 0; k < times.size(); ++k) {
      evaluate(times[k], P[k].data(), V[k].data(), A[k].data());
    }
    return {std::move(P), std::move(V), std::move(A)};
  }

  auto sample_uniform(double dt) const
      -> std::tuple<std::vector<double>, Matrix, Matrix, Matrix> {
    std::vector<double> times;
    const auto n = static_cast<std::size_t>(std::floor(duration_ / dt));
    times.reserve(n + 2);
    for (std::size_t k = 0; k <= n; ++k) times.push_back(k * dt);
    if (duration_ - times.back() > 1e-12) times.push_back(duration_);
    auto [P, V, A] = sample(times);
    return {std::move(times), std::move(P), std::move(V), std::move(A)};
  }

  /// Body-frame base command at ``t``: signed forward speed ``v``
  /// (negative when reversing) and yaw rate ``omega``.  Zero for
  /// planners without a base.
  auto base_twist(double t) const -> std::array<double, 2> {
    if (base_dim_ == 0) return {0.0, 0.0};
    auto [seg, s] = locate(t);
    if (seg->kind == SegmentKind::kSpin) return {0.0, seg->theta.vel(s)};
    const BaseKinematics bk = drive_base_kinematics(*seg, s);
    return {seg->gear != 0 ? -bk.speed : bk.speed, bk.omega};
  }

  /// Segment boundary times ``[0, t_1, ..., duration]``.
  auto knot_times() const -> std::vector<double> {
    std::vector<double> out;
    out.reserve(segments_.size() + 1);
    for (const auto& sg : segments_) out.push_back(sg.t0);
    out.push_back(duration_);
    return out;
  }

  auto segment_kinds() const -> std::vector<int> {
    std::vector<int> out;
    out.reserve(segments_.size());
    for (const auto& sg : segments_) out.push_back(static_cast<int>(sg.kind));
    return out;
  }

  auto segment_gears() const -> std::vector<int> {
    std::vector<int> out;
    out.reserve(segments_.size());
    for (const auto& sg : segments_) out.push_back(sg.gear);
    return out;
  }

 private:
  int active_dim_ = 0;
  int base_dim_ = 0;
  double duration_ = 0.0;
  std::vector<Segment> segments_;
};

}  // namespace flat
}  // namespace fetch_planning
