"""State-space model of the TurtleBot3 Waffle Pi for Track A.

    Wheels -> unicycle   differential-drive kinematics for the Waffle Pi's wheel radius and track
    Wheel actuators      velocity servo with stiction: a deadband and a first-order lag. Derived from the MuJoCo
                         model's parameters and checked in tests/test_model.py
    Unicycle             x = (x, y, theta), u = (v, omega): continuous model (Week 5 D) and the exact-arc discrete
                         model with its Jacobian (Week 7 C.4), shared with the EKF
    Tracking error       linearised about a reference moving at (v_r, omega_r): the full 3-state model and the
                         2-state cross-track/heading model of Week 5 D and Week 8 B
    Discretisation       zero-order hold at the control period (Week 5 C5)

Sign conventions follow Week 5 - errors are actual minus reference, e = (e_x along track, e_y cross track to the
left, e_theta), all in the reference frame. (Week 8 B writes the heading error the other way round, theta_d - theta.)
"""
import control as ct
import numpy as np

from src import config as cfg

R = cfg.WHEEL_RADIUS
B = cfg.WHEEL_TRACK

# Wheel actuator. Each wheel is a MuJoCo velocity servo, torque = kv (w_cmd - w), on a joint with Coulomb friction
# (frictionloss) and armature. In steady state the servo torque must balance the friction, so the wheel turns
# frictionloss/kv slower than commanded, and not at all for |w_cmd| below that. The response is first order with
# tau = J/kv ; J is the armature plus the wheel's inertia plus its share of the robot's mass (m r^(2)/2).
WHEEL_KV = 0.1 # N m s/rad, actuator gain in the menagerie model
WHEEL_FRICTION = 0.1 # N m, joint frictionloss in the menagerie model
WHEEL_DEADBAND = WHEEL_FRICTION / WHEEL_KV # 1.0 rad/s (measured 1.00-1.02)
WHEEL_TAU = 0.110 # s, (0.01 + 2.1e-5 + 1.857 * 0.033^2 / 2)/kv (measured 0.108)
WHEEL_SPEED_USABLE = cfg.WHEEL_SPEED_MAX - WHEEL_DEADBAND # 6.88 rad/s actually achievable

# Turning resistance. Even with the deadband removed, the robot turns a constant ~0.02rad/s slower than the
# kinematics predict, at any turn rate, spinning or driving - this is a Coulomb-like resisting moment, most likely tyre
# scrub across the width of each wheel's contact. Also seen by the encoders so it is real wheel motion. Identified
# in tests/test_model.py Part 5 rather than derived.
YAW_SCRUB = 0.02 # rad/s
V_MAX = R * WHEEL_SPEED_USABLE # 0.227m/s straight-line top speed
W_MAX = 2 * R * WHEEL_SPEED_USABLE / B # 1.58rad/s top turn rate on the spot


def wrap(a):
    """Fold an angle back into (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# Differential drive <-> unicycle

def wheels_to_unicycle(w_l, w_r):
    """Wheel speeds [rad/s] -> forward speed v [m/s] and turn rate omega [rad/s]."""
    return R * (w_r + w_l) / 2, R * (w_r - w_l) / B


def unicycle_to_wheels(v, w, compensate=True):
    """(v, omega) -> wheel speed commands [rad/s].

    If either wheel would exceed what the actuator can deliver, both are scaled down together so the path
    curvature v/omega is kept and only the speed drops. 
    With compensate=True the turning resistance is added to omega and the deadband to each wheel command, so 
    in steady state the robot moves at the requested (v, omega)."""
    if compensate and abs(w) > 1e-3:
        w = w + np.sign(w) * YAW_SCRUB
    w_l, w_r = (v - w * B / 2) / R, (v + w * B / 2) / R
    scale = max(1.0, abs(w_l) / WHEEL_SPEED_USABLE, abs(w_r) / WHEEL_SPEED_USABLE)
    w_l, w_r = w_l / scale, w_r / scale
    if compensate:
        w_l, w_r = [w + np.sign(w) * WHEEL_DEADBAND if abs(w) > 1e-3 else 0.0 for w in (w_l, w_r)]
    return np.clip([w_l, w_r], -cfg.WHEEL_SPEED_MAX, cfg.WHEEL_SPEED_MAX)


def wheel_response(w_cmd):
    """Steady-state wheel speed for a command  - the actuator deadband."""
    return np.sign(w_cmd) * np.maximum(np.abs(w_cmd) - WHEEL_DEADBAND, 0.0)


def turn_response(w):
    """Turn rate the robot actually achieves when the wheel speeds alone would give w - the turning resistance."""
    return np.sign(w) * np.maximum(np.abs(w) - YAW_SCRUB, 0.0)


# Unicycle Model

def f_cont(x, u):
    """Continuous unicycle, xdot = f(x, u) with x = (x, y, theta), u = (v, omega) (Week 5 D)."""
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


def f_uni(x, u, dt):
    """Exact-arc discrete motion model, x_{k+1} = f(x_k, u_k) for u held for dt (Week 7 C.4)."""
    v, w = u
    px, py, th = x
    if abs(w) < 1e-6:                                  # straight-line limit
        return np.array([px + v * dt * np.cos(th), py + v * dt * np.sin(th), wrap(th)])
    th_n = th + w * dt
    return np.array([px + v / w * (np.sin(th_n) - np.sin(th)),
                     py + v / w * (-np.cos(th_n) + np.cos(th)),
                     wrap(th_n)])


def F_uni(x, u, dt):
    """Jacobian of f_uni with respect to the state (Week 7 C.4)."""
    v, w = u
    th = x[2]
    F = np.eye(3)
    if abs(w) < 1e-6:
        F[0, 2] = -v * dt * np.sin(th)
        F[1, 2] = v * dt * np.cos(th)
    else:
        th_n = th + w * dt
        F[0, 2] = v / w * (np.cos(th_n) - np.cos(th))
        F[1, 2] = v / w * (np.sin(th_n) - np.sin(th))
    return F


# Tracking-error model

def tracking_error(x, x_ref):
    """Pose error of x relative to the reference pose x_ref, in the reference frame: (e_x, e_y, e_theta)."""
    dx, dy = x[0] - x_ref[0], x[1] - x_ref[1]
    c, s = np.cos(x_ref[2]), np.sin(x_ref[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(x[2] - x_ref[2])])


def error_model(v_r, w_r):
    """Linearised tracking-error dynamics about a reference moving at (v_r, omega_r):
        edot = A e + B du,  e = (e_x, e_y, e_theta),  du = (v - v_r, omega - omega_r)
    from e_x' = v cos(e_theta) - v_r + omega_r e_y,  e_y' = v sin(e_theta) - omega_r e_x,  e_theta' = omega - omega_r.
    """
    A = np.array([[0.0, w_r, 0.0],
                  [-w_r, 0.0, v_r],
                  [0.0, 0.0, 0.0]])
    Bm = np.array([[1.0, 0.0],
                   [0.0, 0.0],
                   [0.0, 1.0]])
    return A, Bm


def reduced_error_model(v_r):
    """The 2-state model of Week 5 D and Week 8 B, i.e. cross-track and heading error steered by the turn rate,
    e = (e_y, e_theta), du = omega - omega_r. It is the (e_y, e_theta) block of error_model when the speed is not
    controlled (dv = 0) and the coupling to the along-track error through omega_r is neglected."""
    return np.array([[0.0, v_r], [0.0, 0.0]]), np.array([[0.0], [1.0]])


def discretise(A, Bm, dt=cfg.CONTROL_DT):
    """Zero-order-hold discretisation at the control period (Week 5 C5): returns (A_d, B_d)."""
    d = ct.c2d(ct.ss(A, Bm, np.eye(A.shape[0]), np.zeros((A.shape[0], Bm.shape[1]))), dt, method="zoh")
    return d.A, d.B
