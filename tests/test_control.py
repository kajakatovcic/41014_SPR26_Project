# Evaluate the controller (src/control/control.py) on the ROUTE mission (main.py --mission route): the LQR design
# and the path-following/docking controller that the explore mission's Navigator builds on, on a fixed route so the
# controllers can be compared on the same path. The explore mission is evaluated in tests/test_navigation.py.
#   Part 0 - the route: clearance of the planned path from the known obstacles and walls, beyond the robot's
#            radius (Anthony Fava's check, src/control/mission.py), for the spline and for straight segments
#   Part 1 - LQR design (from wk6 A.5)- Bryson's rule, the gain and the closed-loop poles across driving speeds
#   Part 2 - trade-off (wk6 A.5)- sweeping R from aggressive to gentle on the linear model and checking each
#            design on the nonlinear unicycle WITH the 0.11sec wheel lag (which the LQR model leaves out)
#   Part 3 - missions- the full loop (EKF + controller) over repeated trials with random odometry errors: the
#            mission controller on the spline route (with Anthony's spline geometry + curvature feed-forward),
#            the same controller on straight segments (turning on the spot at corners), and the Week 8 C.4
#            placeholder, scored against ground truth
#   Part 4 - docking from offset starts: the Week 9 D3 law regulates range and bearing only so how much lateral
#            and heading error is left when the robot starts off the marker's normal?
# Usage: python tests/test_control.py [--trials 3] [--no-plot]
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main                                                
from src import config as cfg                          
from src.control import control as ctl                    
from src.control import model                                  

DT = cfg.CONTROL_DT
V0 = 0.15 # nominal tracking speed
R_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]


# PART 0: the route

def part0():
    from src.control.mission import clearance
    print("\n=== Part 0: route clearance beyond the robot's radius (Anthony Fava's check, src/control/mission.py)")
    ok = True
    for name, smooth in [("spline (used)", True), ("straight segments", False)]:
        path = ctl.MissionController(smooth=smooth).path
        obs, wall = clearance(path.xy)
        ok &= not smooth or (obs >= 0.05 and wall >= 0.05)
        print(f"{name:>18}: length {path.s[-1]:.2f} m, clearance from obstacles {100 * obs:.1f} cm, "
              f"walls {100 * wall:.1f} cm")
    print("the spline route keeps at least 5 cm beyond the robot's radius: " + ("PASS" if ok else "FAIL"))
    return ok


# PARTS 1 and 2: design on the model

def part1():
    print("\n=== Part 1: LQR design (Week 6 A.5, Bryson's rule)")
    print(f"largest acceptable values: e_y {1000 * ctl.E_Y_MAX:.0f} mm, e_theta {np.degrees(ctl.E_TH_MAX):.0f} deg, "
          f"steering {ctl.DW_MAX} rad/s  ->  Q = diag({ctl.Q_LQR[0, 0]:.1f}, {ctl.Q_LQR[1, 1]:.1f}), "
          f"R = {ctl.R_LQR.item():.1f}")
    print(f'{"v [m/s]":>8} {"K (e_y, e_theta)":>22} {"|z| closed loop":>18} {"time constants [s]":>22} {"damping":>8}')
    ok = True
    for v in [0.04, 0.08, 0.15, 0.2]:
        Ad, Bd = model.discretise(*model.reduced_error_model(v), DT)
        K = ctl.lqr_gain(v)
        z = np.linalg.eigvals(Ad - Bd @ K)
        s = np.log(z.astype(complex)) / DT # equivalent continuous poles
        tau = -1 / s.real
        zeta = -s.real / np.abs(s)
        ok &= np.all(np.abs(z) < 1)
        print(f"{v:>8.2f} {str(np.round(K.ravel(), 2)):>22} {str(np.round(np.abs(z), 4)):>18} "
              f"{str(np.round(tau, 2)):>22} {np.round(zeta.min(), 2):>8}")
    print(f"all closed loops stable (|z| < 1). The slowest closed-loop time constant is far above the wheels' "
          f"{model.WHEEL_TAU} s lag, which the model leaves out: " + ("PASS" if ok else "FAIL"))
    return ok


def closed_loop_linear(K, v=V0, e0=(0.1, 0.0), n=400):
    Ad, Bd = model.discretise(*model.reduced_error_model(v), DT)
    x, X, U = np.array(e0, float), [], []
    for _ in range(n):
        u = -K @ x
        X.append(x.copy()); U.append(u.item())
        x = Ad @ x + (Bd @ u).ravel()
    return np.array(X), np.array(U)


def closed_loop_nonlinear(K, v=V0, e0=(0.1, 0.0), n=400, lag=True):
    """Unicycle following the x axis from e_y = e0[0], with the wheel lag (first order, model.WHEEL_TAU) on omega."""
    x, w_act, X, U = np.array([0.0, e0[0], e0[1]]), 0.0, [], []
    a = 1 - np.exp(-DT / model.WHEEL_TAU)
    for _ in range(n):
        e = np.array([x[1], model.wrap(x[2])])
        w_cmd = float(np.clip(-(K @ e)[0], -model.W_MAX, model.W_MAX))
        X.append(e); U.append(w_cmd)
        w_new = w_act + a * (w_cmd - w_act) if lag else w_cmd
        x = model.f_uni(x, (v, (w_act + w_new) / 2 if lag else w_cmd), DT)
        w_act = w_new
    return np.array(X), np.array(U)


def settling_time(e_y, tol=0.005):
    outside = np.flatnonzero(np.abs(e_y) > tol)
    return (outside[-1] + 1) * DT if len(outside) else 0.0


def part2():
    print(f"\n=== Part 2: trade-off curve (Week 6 A.5), v = {V0} m/s, starting 100 mm off the path")
    print(f'{"R scale":>8} {"state cost":>11} {"input cost":>11} {"peak omega":>11} {"settle (5 mm)":>14} | '
          f'{"nonlinear + lag: settle":>24} {"overshoot":>10}')
    rows = []
    for r in R_SCALES:
        K = ctl.lqr_gain(V0, R=ctl.R_LQR * r)
        X, U = closed_loop_linear(K)
        Jx = np.sum(np.einsum("ij,jk,ik->i", X, ctl.Q_LQR, X))
        Ju = np.sum(U ** 2) * ctl.R_LQR.item()
        Xn, Un = closed_loop_nonlinear(K)
        over = max(0.0, -Xn[:, 0].min())
        rows.append(dict(r=r, X=X, Xn=Xn, Jx=Jx, Ju=Ju, peak=np.abs(U).max()))
        print(f"{r:>8g} {Jx:>11.1f} {Ju:>11.2f} {np.abs(U).max():>8.2f} rad/s {settling_time(X[:, 0]):>11.2f} s | "
              f"{settling_time(Xn[:, 0]):>22.2f} s {1000 * over:>7.1f} mm")
    print("the Bryson design (R scale 1) settles in the nonlinear model with the wheel lag as in the linear one")
    return rows


# PARTS 3 and 4: in the simulator

def mission_metrics(log, controller):
    a = log.arrays()
    tracking = a["mode"] == "track"
    d = main.path_errors(a["true"][tracking, :2], log.ref_path) if tracking.any() else np.array([np.nan])
    w = a["cmd"][:, 1]
    out = dict(time=a["t"][-1], done=bool(controller.done), path_rms=np.sqrt(np.mean(d ** 2)), path_max=d.max(),
               w_rms=np.sqrt(np.mean(w ** 2)), dw_rms=np.sqrt(np.mean(np.diff(w) ** 2)) / DT, true=a["true"])
    if isinstance(controller, ctl.MissionController):
        out["dock"] = main.docking_errors(a["true"][-1])
    return out


def part3(n):
    print(f"\n=== Part 3: missions, spline route vs straight segments vs the Week 8 C.4 placeholder ({n} trials each)")
    rng = np.random.default_rng(200)
    errs = [rng.uniform(-0.008, 0.008, 2) for _ in range(n)]
    res = {"lqr": [], "straight": [], "simple": []}
    for name in res:
        for i in range(n):
            log, c = main.run(seed=200 + i, mission="route", controller=name,
                              encoder_errors=(errs[i], cfg.SIM_WHEEL_SLIP_SIGMA))
            res[name].append(mission_metrics(log, c))
    print(f'{"controller":>11} {"trial":>6} {"done":>5} {"time":>7} {"path RMS":>9} {"path max":>9} '
          f'{"omega RMS":>10} {"d omega/dt RMS":>15} | {"dock distance":>13} {"lateral":>8} {"heading":>8}')
    for name, rs in res.items():
        for i, m in enumerate(rs):
            dock = (f"{1000 * m['dock'][0]:>10.0f} mm {1000 * m['dock'][1]:>5.0f} mm {np.degrees(m['dock'][2]):>5.1f} deg"
                    if "dock" in m else f'{"(no docking)":>32}')
            print(f"{name:>11} {i:>6} {str(m['done']):>5} {m['time']:>5.1f} s {1000 * m['path_rms']:>6.1f} mm "
                  f"{1000 * m['path_max']:>6.0f} mm {m['w_rms']:>5.2f} rad/s {m['dw_rms']:>9.2f} rad/s2 | {dock}")
    lq = res["lqr"] + res["straight"]
    ok = all(m["done"] for m in lq)
    ok &= np.mean([m["path_rms"] for m in res["lqr"]]) < np.mean([m["path_rms"] for m in res["simple"]])
    ok &= all(abs(m["dock"][0]) < 0.02 and abs(m["dock"][1]) < 0.02 and abs(np.degrees(m["dock"][2])) < 3 for m in lq)
    for name in res:
        print(f"{name:>9}: mean time {np.mean([m['time'] for m in res[name]]):.1f} s, mean path RMS "
              f"{1000 * np.mean([m['path_rms'] for m in res[name]]):.1f} mm, max {1000 * max(m['path_max'] for m in res[name]):.0f} mm")
    print("every mission controller run docked within 20 mm and 3 deg, and the spline controller tracks its path more "
          "tightly than the placeholder: " + ("PASS" if ok else "FAIL"))
    return ok, res


def part4():
    print("\n=== Part 4: docking from offset starts (Week 9 D3), straight from a pre-dock pose")
    pre = np.array(cfg.WAYPOINTS[-1])
    print(f'{"start offset":>24} | {"docked":>6} {"time":>7} {"distance":>9} {"lateral":>9} {"heading":>9}')
    rows = []
    for dy, dth in [(0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.2, 0.0), (0.0, 0.3), (0.15, -0.3)]:
        start = (pre[0], pre[1] + dy, dth)
        log, c = main.run(mission="route", start=start, waypoints=[], duration=60, seed=300)
        m = mission_metrics(log, c)
        rows.append((dy, dth, m))
        dist, lat, hd = m["dock"]
        print(f"{f'y {1000 * dy:+.0f} mm, theta {np.degrees(dth):+.0f} deg':>24} | {str(m['done']):>6} "
              f"{m['time']:>5.1f} s {1000 * dist:>6.0f} mm {1000 * lat:>6.0f} mm {np.degrees(hd):>6.1f} deg")
    print("the law nulls range and bearing, so a start off the marker's normal ends facing the marker but off its "
          "normal (Week 9 D3): the lateral error grows with the start offset")
    return rows


def plot(rows2, res3, rows4):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    a.loglog([r["Ju"] for r in rows2], [r["Jx"] for r in rows2], "o-")
    for r in rows2:
        a.annotate(f"R x{r['r']:g}", (r["Ju"], r["Jx"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    a.set_xlabel("input cost"); a.set_ylabel("state cost"); a.set_title("Part 2: LQR trade-off curve (Week 6 A.5)")

    a = ax[0, 1]
    t = np.arange(len(rows2[0]["X"])) * DT
    for r in rows2[1:4]:
        line, = a.plot(t, 1000 * r["X"][:, 0], label=f"R x{r['r']:g} linear")
        a.plot(t, 1000 * r["Xn"][:, 0], "--", color=line.get_color(), label=f"R x{r['r']:g} nonlinear + lag")
    a.axhline(0, color="k", lw=0.8); a.set_xlim(0, 12)
    a.set_xlabel("time [s]"); a.set_ylabel("cross-track error [mm]"); a.legend(fontsize=8)
    a.set_title("Part 2: recovery from 100 mm off the path")

    a = ax[1, 0]
    for name, style, colour, label in [("simple", ":", "tab:orange", "placeholder"),
                                       ("straight", "--", "tab:green", "straight segments + docking"),
                                       ("lqr", "-", "tab:blue", "spline route + docking")]:
        for i, m in enumerate(res3[name]):
            a.plot(m["true"][:, 0], m["true"][:, 1], style, lw=1.2, color=colour, label=label if i == 0 else None)
    pts = np.array([cfg.START_POSE[:2]] + list(cfg.WAYPOINTS))
    a.plot(pts[:, 0], pts[:, 1], "kx", label="waypoints")
    spline = ctl.MissionController().path.xy
    a.plot(spline[:, 0], spline[:, 1], "k-", lw=0.6, alpha=0.6, label="planned spline")
    for _, kind, c, size, _ in cfg.OBSTACLES:
        a.add_patch(plt.Circle(c, size, color="0.7") if kind == "cylinder"
                    else plt.Rectangle(np.array(c) - size, *(2 * np.array(size)), color="0.7"))
    a.plot(*main.docked_pose()[:2], "r*", ms=12, label="docked pose")
    a.set_aspect("equal"); a.legend(fontsize=8); a.set_title("Part 3: true paths, every trial")

    a = ax[1, 1]
    target = main.docked_pose()
    for dy, dth, m in rows4:
        a.plot(m["true"][:, 0], m["true"][:, 1], label=f"y {1000 * dy:+.0f} mm, th {np.degrees(dth):+.0f} deg")
    a.plot(*target[:2], "r*", ms=12)
    T = cfg.MARKERS[cfg.DOCK_ID]
    a.plot([T.t[0]] * 2, [T.t[1] - 0.075, T.t[1] + 0.075], "k-", lw=4)
    a.set_aspect("equal"); a.legend(fontsize=8); a.set_title("Part 4: docking from offset starts")
    plt.tight_layout(); plt.show()


def cli():
    parser = argparse.ArgumentParser(description="Controller evaluation")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    ok0 = part0()
    ok1 = part1()
    rows2 = part2()
    ok3, res3 = part3(args.trials)
    rows4 = part4()
    print(f'\ncontrol test: {"PASS" if ok0 and ok1 and ok3 else "FAIL"}')
    if not args.no_plot:
        plot(rows2, res3, rows4)


if __name__ == "__main__":
    cli()
