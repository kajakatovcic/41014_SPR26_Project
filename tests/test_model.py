# State-space model of the Waffle Pi (src/control/model.py): derivation checks, controllability, observability with
# the project's sensors, discretisation, and validation against the MuJoCo robot (marking criterion A).
#   Part 1 - derivation: the tracking-error Jacobians derived symbolically (Week 5 B1) and the motion-model Jacobian
#            checked numerically (Week 7 C.4)
#   Part 2 - controllability of the error model against speed (Week 5 C3, D1)
#   Part 3 - observability of the pose with encoders, ArUco range and bearing. Range only and bearing only, still
#            and moving (Week 5 C4, Week 7 C/D) and where in the room the camera sees 0, 1 or 2+ markers
#   Part 4 - discretisation at the control period: zero-order hold against forward Euler (Week 5 C5)
#   Part 5 - wheel actuators: deadband and time constant measured in MuJoCo against the values derived from the
#            model's parameters, and the turning resistance identified
#   Part 6 - does the robot agree? (Week 5 D2) The unicycle model, with and without the actuator model, predicts a
#            drive through a sequence of commands. This is compared with the MuJoCo robot's true path
# Usage: python tests/test_model.py [--no-plot]
import argparse
import sys
from pathlib import Path

import control as ct
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg                            
from src.control import model as mdl                          
from src.perception import perception as per              
from src.simulation import world as sim               

DT = cfg.CONTROL_DT
CAMERA_DT = 1 / cfg.CAMERA_RATE
HFOV = 2 * np.arctan(cfg.CAMERA_WIDTH / 2 / per.CAMERA.K[0, 0]) # horizontal field of view, calibrated camera


# PART 1: derivation checks

def num_jac(fn, x, eps=1e-6):
    """Central-difference Jacobian (Week 7 C.4)."""
    x = np.asarray(x, float)
    f0 = np.atleast_1d(fn(x))
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = eps
        J[:, i] = (np.atleast_1d(fn(x + dx)) - np.atleast_1d(fn(x - dx))) / (2 * eps)
    return J


def part1():
    print("\n=== Part 1: derivation checks")
    # Tracking error in the reference frame differentiated with sympy (Week 5 B1: Jacobians symbolically first)
    ex, ey, eth, v, w, vr, wr = sp.symbols("e_x e_y e_theta v omega v_r omega_r", real=True)
    edot = sp.Matrix([v * sp.cos(eth) - vr + wr * ey,
                      v * sp.sin(eth) - wr * ex,
                      w - wr])
    e, u = sp.Matrix([ex, ey, eth]), sp.Matrix([v, w])
    at = {ex: 0, ey: 0, eth: 0, v: vr, w: wr}
    A_sym = sp.simplify(edot.jacobian(e).subs(at))
    B_sym = sp.simplify(edot.jacobian(u).subs(at))
    print("A =", A_sym.tolist(), "   B =", B_sym.tolist())
    ok = True
    for vr_, wr_ in [(0.15, 0.0), (0.2, 0.4), (0.0, 0.8)]:
        A, Bm = mdl.error_model(vr_, wr_)
        A_n = np.array(A_sym.subs({vr: vr_, wr: wr_}), float)
        B_n = np.array(B_sym, float)
        ok &= np.allclose(A, A_n) and np.allclose(Bm, B_n)

    # The tracking-error kinematics themselves checked by simulating the nonlinear unicycle
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, xr = rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 3)
        u_, ur = rng.uniform([0, -0.5], [0.2, 0.5]), rng.uniform([0, -0.5], [0.2, 0.5])
        e0 = mdl.tracking_error(x, xr)
        h = 1e-6
        de = (mdl.tracking_error(x + h * mdl.f_cont(x, u_), xr + h * mdl.f_cont(xr, ur)) - e0) / h
        pred = np.array(edot.subs({ex: e0[0], ey: e0[1], eth: e0[2], v: u_[0], w: u_[1],
                                   vr: ur[0], wr: ur[1]}), float).ravel()
        ok &= np.allclose(de, pred, atol=1e-4)

    # Motion-model Jacobian (Week 7: "always check a hand-derived Jacobian numerically")
    worst = 0
    for x_, u_ in [((1.2, -0.4, 0.7), (0.15, 0.3)), ((0, 0, -2.0), (0.2, 0.0)), ((0.5, 1, 3.0), (0.0, 1.0))]:
        diff = np.abs(mdl.F_uni(x_, u_, DT) - num_jac(lambda s: mdl.f_uni(s, u_, DT), x_)).max()
        worst = max(worst, diff)
    ok &= worst < 1e-6
    print(f"error-model Jacobians match sympy; error kinematics match the simulated unicycle; "
          f"F_uni max |analytic - numerical| = {worst:.1e}")
    print("PASS" if ok else "FAIL")
    return ok


# PART 2: controllability

def part2():
    print("\n=== Part 2: controllability of the tracking-error model against speed (Week 5 D1)")
    print(f'{"v_r [m/s]":>10} {"omega_r":>8} | {"3-state (dv, dw)":>18} {"min sing. value":>16} | '
          f'{"2-state (dw)":>13} {"det":>8}')
    rows = []
    for vr, wr in [(0.2, 0.0), (0.1, 0.0), (0.02, 0.0), (0.0, 0.0), (0.15, 0.4), (0.0, 0.8)]:
        A, Bm = mdl.error_model(vr, wr)
        Co = ct.ctrb(A, Bm)
        sv = np.linalg.svd(Co, compute_uv=False)
        A2, B2 = mdl.reduced_error_model(vr)
        Co2 = ct.ctrb(A2, B2)
        rows.append((vr, wr, np.linalg.matrix_rank(Co), sv.min()))
        print(f"{vr:>10.2f} {wr:>8.2f} | {f'rank {np.linalg.matrix_rank(Co)}':>18} {sv.min():>16.3f} | "
              f"{f'rank {np.linalg.matrix_rank(Co2)}':>13} {np.linalg.det(Co2):>8.3f}")
    ok = rows[0][2] == 3 and rows[3][2] == 2 and rows[4][2] == 3 and rows[5][2] == 3
    print("rank 3 whenever the reference moves (v_r or omega_r non-zero), rank 2 at rest: "
          + ("PASS" if ok else "FAIL"))
    return ok, rows


# PART 3: observability

def h_rb(x, m):
    """Range and bearing to marker position m from pose x (Week 7 C.4)."""
    dx, dy = m[0] - x[0], m[1] - x[1]
    return np.array([np.hypot(dx, dy), mdl.wrap(np.arctan2(dy, dx) - x[2])])


def H_rb(x, m):
    dx, dy = m[0] - x[0], m[1] - x[1]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]])


def observability(x0, u, schedule, rows):
    """Stack the noise-weighted measurement Jacobians along the motion (local observability of the time-varying
    linearised system): O = [W H_0; W H_1 F_0; W H_2 F_1 F_0; ...], W = R^-1/2 from perception's noise model so
    the singular values are in units of 'standard deviations per unit of state error'. schedule lists the marker
    positions measured at each camera frame."""
    blocks, x, Phi = [], np.array(x0, float), np.eye(3)
    for markers in schedule:
        for m in markers:
            H = H_rb(x, m)[rows]
            R = per.measurement_covariance(np.hypot(m[0] - x[0], m[1] - x[1]))
            W = np.diag(1 / np.sqrt(np.diag(R)[rows]))
            blocks.append(W @ H @ Phi)
        Phi = mdl.F_uni(x, u, CAMERA_DT) @ Phi
        x = mdl.f_uni(x, u, CAMERA_DT)
    if not blocks:
        return 0, np.zeros(3)
    O = np.vstack(blocks)
    sv = np.linalg.svd(O, compute_uv=False)
    return np.linalg.matrix_rank(O, tol=1e-9 * sv.max()), sv


def _segment_hits(p, q, objs):
    """Does the segment p -> q pass through any obstacle? Uses Week 4 ray casting (_box, _sphere) on a finite ray."""
    d = q - p
    for kind, c, size in objs:
        if kind == "box":
            lo, hi = np.array(c) - size, np.array(c) + size
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (lo - p) / d, (hi - p) / d
            tmin, tmax = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
            if tmin <= tmax and tmax > 0 and tmin < 1:
                return True
        else:
            oc = p - np.array(c)
            a, b, cc = d @ d, 2 * oc @ d, oc @ oc - size ** 2
            disc = b * b - 4 * a * cc
            if disc >= 0:
                t = (-b - np.sqrt(disc)) / (2 * a)
                if 0 < t < 1:
                    return True
    return False


def visible_markers(x, max_range=3.0, max_angle=60.0):
    """Markers the camera can use from pose x: inside the field of view, within max_range (detection >= 90% in
    tests/test_perception.py), seen less than max_angle off square-on, and not hidden by an obstacle or the dock."""
    objs = [(k, c, np.array(s) if k == "box" else s) for _, k, c, s, _ in cfg.OBSTACLES]
    objs.append(("box", cfg.DOCK_POS, np.array(cfg.DOCK_SIZE[:2]) - 1e-3))
    cam = (per.pose_to_SE3(x) * cfg.T_RC).t[:2]
    out = []
    for mid, T in cfg.MARKERS.items():
        m = T.t[:2]
        to_cam = cam - m
        r = np.linalg.norm(to_cam)
        bearing = mdl.wrap(np.arctan2(m[1] - cam[1], m[0] - cam[0]) - x[2])
        view = np.degrees(np.arccos(np.clip(T.R[:2, 2] @ to_cam / r, -1, 1)))
        if r <= max_range and abs(bearing) <= HFOV / 2 and view < max_angle and not _segment_hits(cam, m, objs):
            out.append(mid)
    return out


def free(p, margin):
    _in = cfg.ROOM_HALF - cfg.WALL_THICK - margin
    if abs(p[0]) > _in or abs(p[1]) > _in:
        return False
    for _, kind, c, size, _ in cfg.OBSTACLES + [("dock", "box", cfg.DOCK_POS, cfg.DOCK_SIZE[:2], 0)]:
        if kind == "box":
            q = np.abs(np.array(p) - c) - size
            if np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0) < margin:
                return False
        elif np.linalg.norm(np.array(p) - c) - size < margin:
            return False
    return True


def part3():
    print("\n=== Part 3: observability of the pose (x, y, theta) with the project's sensors")
    # A pose where the camera sees two markers at once (found on the same grid as the room map below)
    x0, pair = None, None
    for px in np.arange(-1.9, 1.91, 0.1):
        for py in np.arange(-1.9, 1.91, 0.1):
            for th in np.deg2rad(np.arange(0, 360, 15)):
                if x0 is None and free((px, py), cfg.ROBOT_RADIUS):
                    vis = visible_markers((px, py, th))
                    if len(vis) >= 2:
                        x0, pair = np.array([px, py, th]), vis[:2]
    mA, mB = cfg.MARKERS[pair[0]].t[:2], cfg.MARKERS[pair[1]].t[:2]
    print(f"test pose ({x0[0]:.1f}, {x0[1]:.1f}, {np.degrees(x0[2]):.0f} deg) sees markers {pair[0]} and {pair[1]}; "
          f"10 camera frames (1 s), still or driving at (0.15 m/s, 0.2 rad/s)")
    n = 10
    sets = [("encoders only", [[]] * n, [0, 1]),
            ("1 marker, range + bearing", [[mA]] * n, [0, 1]),
            ("1 marker, range only", [[mA]] * n, [0]),
            ("1 marker, bearing only", [[mA]] * n, [1]),
            ("1 marker, then another", [[mA]] * (n // 2) + [[mB]] * (n // 2), [0, 1]),
            ("2 markers at once", [[mA, mB]] * n, [0, 1])]
    print(f'{"sensor set":>26} | {"still: rank":>11} {"singular values":>24} | {"moving: rank":>12} '
          f'{"singular values":>24}')
    results = {}
    for name, schedule, rows in sets:
        r0, s0 = observability(x0, (0.0, 0.0), schedule, rows)
        r1, s1 = observability(x0, (0.15, 0.2), schedule, rows)
        results[name] = (r0, r1)
        print(f"{name:>26} | {r0:>11d} {str(np.round(s0, 1)):>24} | {r1:>12d} {str(np.round(s1, 1)):>24}")
    ok = (results["encoders only"] == (0, 0) and results["1 marker, range + bearing"] == (2, 2)
          and results["1 marker, then another"][1] == 3 and results["2 markers at once"] == (3, 3))
    print("encoders alone observe nothing. One marker never observes the full pose, moving or not: rotating the "
          "robot and its path about the marker changes no measurement. Two markers, at once or one after the "
          "other, do: " + ("PASS" if ok else "FAIL"))

    # Where in the room does the camera give the EKF what it needs?
    xs = np.arange(-1.9, 1.91, 0.1)
    headings = np.deg2rad(np.arange(0, 360, 15))
    counts, grid = [], np.full((len(xs), len(xs)), np.nan)
    for i, px in enumerate(xs):
        for j, py in enumerate(xs):
            if not free((px, py), cfg.ROBOT_RADIUS):
                continue
            n = [len(visible_markers((px, py, th))) for th in headings]
            counts += n
            grid[j, i] = np.mean(np.array(n) >= 1)
    counts = np.array(counts)
    print(f"over {len(counts)} free poses (0.1 m grid x 24 headings): {100 * np.mean(counts == 0):.0f}% see no "
          f"marker (dead reckoning only), {100 * np.mean(counts == 1):.0f}% see one (corrects 2 of 3 directions), "
          f"{100 * np.mean(counts >= 2):.0f}% see two or more (corrects the full pose at once)")
    return ok, dict(grid=grid, xs=xs)


# PART 4: discretisation

def part4():
    print(f"\n=== Part 4: discretisation at the control period T = {DT} s (Week 5 C5)")
    ok = True
    for name, (A, Bm) in [("2-state, v_r = 0.15", mdl.reduced_error_model(0.15)),
                          ("3-state, v_r = 0.15, omega_r = 0.4", mdl.error_model(0.15, 0.4))]:
        Ad, Bd = mdl.discretise(A, Bm, DT)
        Ad_e, Bd_e = np.eye(A.shape[0]) + A * DT, Bm * DT
        rho_zoh = np.abs(np.linalg.eigvals(Ad)).max()
        rho_eul = np.abs(np.linalg.eigvals(Ad_e)).max()
        print(f"{name}:")
        print(f"   ZOH   B_d = {np.round(Bd, 6).tolist()}")
        print(f"   Euler B_d = {np.round(Bd_e, 6).tolist()}   (misses the T^2/2 terms)")
        print(f"   max |eig(A_d)|: ZOH {rho_zoh:.6f}, Euler {rho_eul:.6f}")
        ok &= rho_zoh <= 1 + 1e-9
    print("zero-order hold keeps the open-loop eigenvalues on the unit circle; Euler pushes rotating references "
          "just outside it: " + ("PASS" if ok else "FAIL"))
    return ok


# PART 5: wheel actuators

def part5(model, data):
    print("\n=== Part 5: wheel actuators, MuJoCo against the derived deadband and time constant")
    dofs = [model.joint("wheel_left").dofadr[0], model.joint("wheel_right").dofadr[0]]
    steps = {}
    print(f'{"command [rad/s]":>16} {"steady speed":>13} {"shortfall":>10} {"tau [s]":>8}')
    shortfall, taus = [], []
    for cmd in [0.8, 1.5, 3.0, 5.0, cfg.WHEEL_SPEED_MAX]:
        mujoco.mj_resetData(model, data)
        sim.set_pose(model, data, 0, 0, 0)
        for _ in range(250):
            mujoco.mj_step(model, data)
        data.ctrl[:] = cmd
        t, w = [], []
        for _ in range(int(1.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            t.append(data.time); w.append(data.qvel[dofs].mean())
        t, w = np.array(t) - t[0], np.array(w)
        ss = w[-100:].mean()
        tau = t[np.argmax(w >= 0.632 * ss)] if ss > 0.05 else np.nan
        steps[cmd] = (t, w)
        if ss > 0.05:
            shortfall.append(cmd - ss); taus.append(tau)
        print(f"{cmd:>16.2f} {ss:>13.3f} {cmd - ss:>10.3f} {tau:>8.3f}")
    print(f"derived: deadband {mdl.WHEEL_DEADBAND:.3f} rad/s, tau {mdl.WHEEL_TAU:.3f} s; "
          f"measured: {np.mean(shortfall):.3f} rad/s, {np.mean(taus):.3f} s")
    print(f"-> usable wheel speed {mdl.WHEEL_SPEED_USABLE:.2f} rad/s: top speed {mdl.V_MAX:.3f} m/s, "
          f"top turn rate {mdl.W_MAX:.2f} rad/s")

    # Turning resistance - deadband-compensated wheel commands, no turn compensation, steady state
    print(f'\n{"v [m/s]":>8} {"omega [rad/s]":>14} {"achieved":>9} {"shortfall":>10}')
    deficits = []
    for v, w in [(0.15, 0.2), (0.15, 0.4), (0.1, 0.8), (0.0, 0.4), (0.0, 0.8), (0.0, 1.4), (0.12, -0.3)]:
        wl, wr = (v - w * mdl.B / 2) / mdl.R, (v + w * mdl.B / 2) / mdl.R
        mujoco.mj_resetData(model, data)
        sim.set_pose(model, data, 0, 0, 0)
        data.ctrl[:] = [x + np.sign(x) * mdl.WHEEL_DEADBAND for x in (wl, wr)]
        for _ in range(1000):
            mujoco.mj_step(model, data)
        th0 = sim.true_pose(model, data)[2]
        for _ in range(1000):
            mujoco.mj_step(model, data)
        w_true = mdl.wrap(sim.true_pose(model, data)[2] - th0) / (1000 * model.opt.timestep)
        deficits.append(abs(w) - abs(w_true))
        print(f"{v:>8.2f} {w:>14.2f} {w_true:>9.3f} {abs(w) - abs(w_true):>10.3f}")
    print(f"turning resistance: measured {np.mean(deficits):.3f} +/- {np.std(deficits):.3f} rad/s, "
          f"model YAW_SCRUB = {mdl.YAW_SCRUB:.3f}")
    ok = (abs(np.mean(shortfall) / mdl.WHEEL_DEADBAND - 1) < 0.05 and abs(np.mean(taus) / mdl.WHEEL_TAU - 1) < 0.1
          and abs(np.mean(deficits) - mdl.YAW_SCRUB) < 0.005)
    print("PASS" if ok else "FAIL")
    return ok, steps


# PART 6: model against the MuJoCo robot

SEQUENCE = [((0.15, 0.0), 3.0), ((0.15, 0.4), 4.0), ((0.0, 0.8), 3.0), ((0.12, -0.3), 4.0), ((0.2, 0.0), 2.0)]


def run_robot(model, data, compensate):
    """Drive the MuJoCo robot through SEQUENCE at the control rate. Returns the true path, the requested (v, omega)
    and the wheel commands sent for each period."""
    mujoco.mj_resetData(model, data)
    sim.set_pose(model, data, 0, 0, 0)
    for _ in range(250):
        mujoco.mj_step(model, data)
    path, reqs, cmds = [sim.true_pose(model, data)], [], []
    n_sub = int(round(DT / model.opt.timestep))
    for u, dur in SEQUENCE:
        for _ in range(int(round(dur / DT))):
            data.ctrl[:] = mdl.unicycle_to_wheels(*u, compensate=compensate)
            reqs.append(u); cmds.append(data.ctrl.copy())
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            path.append(sim.true_pose(model, data))
    return np.array(path), reqs, np.array(cmds)


def predict(x0, reqs, cmds, actuator):
    """Unicycle prediction- either the requested (v, omega) taken literally (ideal unicycle), or the wheel commands
    passed through the actuator model (deadband, first-order lag, turning resistance)."""
    x, w, path = np.array(x0, float), np.zeros(2), [np.array(x0, float)]
    a = 1 - np.exp(-DT / mdl.WHEEL_TAU)
    for u_req, c in zip(reqs, cmds):
        if actuator:
            w_new = w + a * (mdl.wheel_response(c) - w)
            v, om = mdl.wheels_to_unicycle(*(w + w_new) / 2)  # mean speed over the period
            u = (v, mdl.turn_response(om))
            w = w_new
        else:
            u = u_req
        x = mdl.f_uni(x, u, DT)
        path.append(x)
    return np.array(path)


def part6(model, data):
    print("\n=== Part 6: Unicycle model against the MuJoCo robot (Week 5 D2)")
    print(f"command sequence (v, omega) for (s): {SEQUENCE}")
    print(f'{"wheel commands":>22} {"prediction":>26} | {"final position error":>21} {"max":>8} '
          f'{"final heading error":>20}')
    runs, finals = {}, {}
    for compensate in (False, True):
        truth, reqs, cmds = run_robot(model, data, compensate)
        for actuator in (False, True):
            pred = predict(truth[0], reqs, cmds, actuator)
            e_pos = np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1)
            e_th = np.degrees(np.abs(mdl.wrap(pred[-1, 2] - truth[-1, 2])))
            label = ("compensated" if compensate else "raw (v, omega) -> wheels",
                     "unicycle + actuator model" if actuator else "ideal unicycle")
            runs[label] = (truth, pred)
            finals[label] = e_pos[-1]
            print(f"{label[0]:>22} {label[1]:>26} | {1000 * e_pos[-1]:>18.0f} mm {1000 * e_pos.max():>5.0f} mm "
                  f"{e_th:>17.1f} deg")
    ok = (finals[("compensated", "ideal unicycle")] < 0.05
          and finals[("raw (v, omega) -> wheels", "unicycle + actuator model")] < 0.05)
    print("with compensated commands (deadband + turning resistance) the plain unicycle predicts the robot and the "
          "actuator model explains the uncompensated robot: " + ("PASS" if ok else "FAIL"))
    return ok, runs


def plot(rows2, res3, steps, runs):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    vs = np.linspace(0, 0.25, 60)
    a.plot(vs, [np.linalg.svd(ct.ctrb(*mdl.error_model(v, 0.0)), compute_uv=False).min() for v in vs],
           label="3-state, omega_r = 0 (same as 2-state)")
    a.plot(vs, [np.linalg.svd(ct.ctrb(*mdl.error_model(v, 0.4)), compute_uv=False).min() for v in vs],
           label="3-state, omega_r = 0.4")
    a.plot(vs, [abs(np.linalg.det(ct.ctrb(*mdl.reduced_error_model(v)))) for v in vs], "--",
           label="2-state |det|")
    a.axvline(mdl.V_MAX, color="k", ls=":", label="top speed")
    a.set_xlabel("reference speed v_r [m/s]"); a.set_ylabel("smallest singular value of C")
    a.set_title("Part 2: controllability against speed"); a.legend(fontsize=8)

    a = ax[0, 1]
    xs = res3["xs"]
    im = a.imshow(100 * res3["grid"], origin="lower", extent=[xs[0] - 0.05, xs[-1] + 0.05] * 2, cmap="viridis",
                  vmin=0, vmax=100)
    plt.colorbar(im, ax=a, label="% of headings with a marker in view")
    for mid, T in cfg.MARKERS.items():
        a.plot(*T.t[:2], "rs"); a.annotate(str(mid), T.t[:2], color="r", xytext=(4, 4), textcoords="offset points")
    a.set_title("Part 3: where the camera can correct the EKF"); a.set_aspect("equal")

    a = ax[1, 0]
    for cmd, (t, w) in steps.items():
        line, = a.plot(t, w, label=f"command {cmd:.2f}")
        ss = mdl.wheel_response(cmd)
        a.plot(t, ss * (1 - np.exp(-t / mdl.WHEEL_TAU)), "--", color=line.get_color(), lw=1)
    a.set_xlabel("time [s]"); a.set_ylabel("wheel speed [rad/s]"); a.set_xlim(0, 0.8)
    a.set_title("Part 5: wheel step responses (dashed: deadband + lag model)"); a.legend(fontsize=8)

    a = ax[1, 1]
    truth, pred_ideal = runs[("raw (v, omega) -> wheels", "ideal unicycle")]
    truth_c, _ = runs[("compensated", "ideal unicycle")]
    a.plot(truth[:, 0], truth[:, 1], "k", lw=2.5, label="MuJoCo, raw commands")
    a.plot(*runs[("raw (v, omega) -> wheels", "unicycle + actuator model")][1][:, :2].T, "b:",
           label="unicycle + actuator model (raw commands)")
    a.plot(truth_c[:, 0], truth_c[:, 1], color="0.6", lw=2.5, label="MuJoCo, compensated commands")
    a.plot(pred_ideal[:, 0], pred_ideal[:, 1], "r--", label="ideal unicycle (requested v, omega)")
    a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.legend(fontsize=8)
    a.set_title("Part 6: model against the robot")
    plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser(description="Waffle Pi state-space model: analysis and validation")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ok1 = part1()
    ok2, rows2 = part2()
    ok3, res3 = part3()
    ok4 = part4()
    model = sim.build_world(room=False)
    data = mujoco.MjData(model)
    ok5, steps = part5(model, data)
    ok6, runs = part6(model, data)
    print(f'\nmodel test: {"PASS" if all([ok1, ok2, ok3, ok4, ok5, ok6]) else "FAIL"}')
    if not args.no_plot:
        plot(rows2, res3, steps, runs)


if __name__ == "__main__":
    main()
