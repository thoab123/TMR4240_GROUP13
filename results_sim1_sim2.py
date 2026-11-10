"""Report figures and numbers for Part 2, Simulations 1 and 2 (Results section).

    python results_sim1_sim2.py
    python results_sim1_sim2.py --no-show

Runs the two presets exactly as run_case_part_2.py does and condenses the
plots the assignment asks for into one figure per simulation:

  Simulation 1 (free drift, 300 s): position, heading and trajectory, the
  current in BODY, the wind field (total, mean + slow and gust speed,
  direction) and the wind and wave loads.
  Simulation 2 (four-corner test, constrained thrusters, no environment):
  position and heading against reference and command, tracking error,
  velocities against the reference velocities, desired and applied wrench,
  and commanded and actual thrust and azimuth angle of every thruster.

Output (figures/ next to this script):
    fig_sim1.pdf/.png      Simulation 1 report figure
    fig_sim2.pdf/.png      Simulation 2 report figure
    sim1_metrics.csv       all Simulation 1 numbers quoted in the report
    sim2_metrics.csv       all Simulation 2 numbers quoted in the report
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_2.config import Part2SimConfig, default_thrusters_part_2
from run_case_part_2 import (DT, WAVES_HS, WAVES_TP, WIND_FROM, environment,
                             four_corner_commands, run)

OUT_DIR = Path(__file__).resolve().parent / "figures"

T_SIM1 = 300.0                 # Simulation 1 length [s]
HOLD = 300.0                   # Simulation 2 hold time per leg [s]
N_LEGS = 5                     # commanded legs in Simulation 2
LPP = 34.0                     # Gunnerus length between perpendiculars [m]
G = 9.81                       # gravity [m/s^2]
LF_WINDOW = 30.0               # moving average separating slow and wave-frequency motion [s]
SETTLE_POS, SETTLE_PSI = 0.5, np.deg2rad(2.0)   # settling band around the command [m, rad]
COLUMN_STYLE = {"font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,   # text sizes for a
                "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5,  # one-column (3.5 in)
                "lines.linewidth": 0.9}                                             # report figure


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def moving_average(x, dt, window=LF_WINDOW):
    """Centred moving average, edge padded, same length."""
    k = max(int(round(window / dt)), 1)
    xp = np.pad(np.asarray(x, float), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, "valid")


def break_wraps(deg):
    """Angle in degrees with NaN inserted where it wraps between +180 and -180 (no vertical lines)."""
    out = np.asarray(deg, float).copy()
    out[1:][np.abs(np.diff(out)) > 180.0] = np.nan
    return out


def write_csv(path, metrics):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v if isinstance(v, (list, str)) else round(float(v), 4)])


def print_metrics(title, metrics):
    print(f"\n{title}")
    for k, v in metrics.items():
        print(f"  {k:58s} {v if isinstance(v, (list, str)) else round(float(v), 3)}")


# -- SIMULATION 1 -- #

def run_sim1():
    """Free drift, DP off, 300 s in the design environment (as the sim1 preset)."""
    cfg = Part2SimConfig(dt=DT, T=T_SIM1, use_controller=False, use_reference=False)
    return run(cfg, np.zeros(6), environment())


def sim1_metrics(logs):
    dt = float(logs.t[1] - logs.t[0])
    N, E, psi = logs.eta[:, 0], logs.eta[:, 1], np.unwrap(logs.eta[:, 5])
    # Drift velocity in NED over the last 100 s (slow part only)
    late = logs.t >= logs.t[-1] - 100.0
    vN = np.polyfit(logs.t[late], moving_average(N, dt)[late], 1)[0]
    vE = np.polyfit(logs.t[late], moving_average(E, dt)[late], 1)[0]
    # Wave-frequency motion: signal minus its 30 s moving average, edges excluded
    k = int(LF_WINDOW / dt)
    inner = slice(k, len(logs.t) - k)
    wf = [np.std((x - moving_average(x, dt))[inner]) for x in (N, E, psi)]
    # Wind direction: logged "towards" direction converted to "from", deviation from the mean
    beta_from = wrap(logs.beta_w - np.pi)
    d_beta = np.rad2deg(wrap(beta_from - WIND_FROM))
    tw, twa = logs.tau_w6[:, [0, 1, 5]], logs.tau_wave[:, [0, 1, 5]]
    lam_p = G * WAVES_TP ** 2 / (2.0 * np.pi)           # deep-water peak wavelength [m]
    return {
        "final North [m]": N[-1],
        "final East [m]": E[-1],
        "final drift distance [m]": np.hypot(N[-1], E[-1]),
        "drift direction, towards [deg from North]": np.rad2deg(np.arctan2(E[-1], N[-1])) % 360.0,
        "final heading [deg]": np.rad2deg(psi[-1]),
        "max |heading| [deg]": np.rad2deg(np.abs(psi).max()),
        "slow drift velocity North, last 100 s [m/s]": vN,
        "slow drift velocity East, last 100 s [m/s]": vE,
        "wave-frequency motion std N [m]": wf[0],
        "wave-frequency motion std E [m]": wf[1],
        "wave-frequency motion std heading [deg]": np.rad2deg(wf[2]),
        "mean wind speed [m/s]": np.mean(logs.U_w),
        "wind speed std [m/s]": np.std(logs.U_w),
        "gust speed std [m/s]": np.std(logs.U_w_gust),
        "max |direction deviation| [deg]": np.abs(d_beta).max(),
        "mean wind load X [kN]": np.mean(tw[:, 0]) / 1e3,
        "mean wind load Y [kN]": np.mean(tw[:, 1]) / 1e3,
        "mean wind load N [kNm]": np.mean(tw[:, 2]) / 1e3,
        "mean wave load X [kN]": np.mean(twa[:, 0]) / 1e3,
        "mean wave load Y [kN]": np.mean(twa[:, 1]) / 1e3,
        "mean wave load N [kNm]": np.mean(twa[:, 2]) / 1e3,
        "wave load std X [kN]": np.std(twa[:, 0]) / 1e3,
        "wave load std Y [kN]": np.std(twa[:, 1]) / 1e3,
        "wave load std N [kNm]": np.std(twa[:, 2]) / 1e3,
        "max |wave load Y| [kN]": np.abs(twa[:, 1]).max() / 1e3,
        "current in BODY u_c range [m/s]": [round(float(logs.cur_body[:, 0].min()), 3), round(float(logs.cur_body[:, 0].max()), 3)],
        "current in BODY v_c range [m/s]": [round(float(logs.cur_body[:, 1].min()), 3), round(float(logs.cur_body[:, 1].max()), 3)],
        "peak wavelength [m]": lam_p,
        "peak wavelength / Lpp [-]": lam_p / LPP,
        "Hs / Lpp [%]": 100.0 * WAVES_HS / LPP,
    }


def plot_sim1(logs):
    """Report figure in the same style as the course plots: 4 x 2 panels with titles."""
    import matplotlib.pyplot as plt
    t = logs.t
    fig, ax = plt.subplots(4, 2, figsize=(11, 9.5))
    ax = ax.ravel()
    # Trajectory
    ax[0].plot(logs.eta[:, 1], logs.eta[:, 0])
    ax[0].plot(0, 0, "ko", ms=4)
    ax[0].set_xlabel("East [m]"); ax[0].set_ylabel("North [m]"); ax[0].set_title("Trajectory")
    # Position
    ax[1].plot(t, logs.eta[:, 0], label="N [m]"); ax[1].plot(t, logs.eta[:, 1], label="E [m]")
    ax[1].set_title("Position [m]")
    # Heading
    ax[2].plot(t, np.rad2deg(np.unwrap(logs.eta[:, 5])), color="C2")
    ax[2].set_title("Heading [deg]")
    # Current in BODY
    ax[3].plot(t, logs.cur_body[:, 0], label=r"$u_c$ [m/s]"); ax[3].plot(t, logs.cur_body[:, 1], label=r"$v_c$ [m/s]")
    ax[3].set_title("Current in BODY [m/s]")
    # Wind speed components
    ax[4].plot(t, logs.U_w, lw=0.8, label="total")
    ax[4].plot(t, logs.U_w_mean, label="mean + slow")
    ax[4].plot(t, logs.U_w_gust, lw=0.8, label="gust")
    ax[4].set_title("Wind speed [m/s]")
    # Wind direction (from), deviation from the mean
    d_beta = np.rad2deg(wrap(wrap(logs.beta_w - np.pi) - WIND_FROM))
    ax[5].plot(t, d_beta, color="C3")
    ax[5].axhline(5, color="k", ls=":", lw=0.8); ax[5].axhline(-5, color="k", ls=":", lw=0.8)
    ax[5].set_title("Wind direction deviation [deg]")
    # Wind loads
    ax[6].plot(t, logs.tau_w6[:, 0] / 1e3, label="X [kN]"); ax[6].plot(t, logs.tau_w6[:, 1] / 1e3, label="Y [kN]")
    ax[6].plot(t, logs.tau_w6[:, 5] / 1e4, label="N/10 [kNm]")
    ax[6].set_title("Wind load")
    # Wave loads
    ax[7].plot(t, logs.tau_wave[:, 0] / 1e3, lw=0.6, label="X [kN]"); ax[7].plot(t, logs.tau_wave[:, 1] / 1e3, lw=0.6, label="Y [kN]")
    ax[7].plot(t, logs.tau_wave[:, 5] / 1e4, lw=0.6, label="N/10 [kNm]")
    ax[7].set_title("Wave load")
    for i, a in enumerate(ax):
        a.grid(True, alpha=0.3)
        if i >= 6:
            a.set_xlabel("Time [s]")
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -- SIMULATION 2 -- #

def run_sim2():
    """Four-corner test, constrained thrusters, no environment, no observer (as the sim2 preset)."""
    cfg = Part2SimConfig(dt=DT, T=N_LEGS * HOLD, use_reference=True,
                         use_observer=False, thruster_dynamics=True)
    return run(cfg, four_corner_commands(cfg, HOLD),
               environment(current=False, wind=False, waves=False))


def sim2_metrics(logs, thrusters):
    t, dt = logs.t, float(logs.t[1] - logs.t[0])
    u_max = np.array([th.u_max for th in thrusters])
    u_rate = np.array([th.u_rate for th in thrusters])
    rot = np.array([th.rot_speed for th in thrusters])
    azi = rot > 0.0
    e_pos = np.hypot(logs.eta[:, 0] - logs.sp[:, 0], logs.eta[:, 1] - logs.sp[:, 1])
    e_psi = np.abs(wrap(logs.eta[:, 5] - logs.sp[:, 5]))
    d_pos = np.hypot(logs.eta[:, 0] - logs.cmd[:, 0], logs.eta[:, 1] - logs.cmd[:, 1])
    d_psi = np.abs(wrap(logs.eta[:, 5] - logs.cmd[:, 5]))
    settle = []
    for i in range(N_LEGS):
        k0, k1 = int(i * HOLD / dt), int((i + 1) * HOLD / dt)
        outside = np.where((d_pos[k0:k1] > SETTLE_POS) | (d_psi[k0:k1] > SETTLE_PSI))[0]
        settle.append(round(float((outside[-1] + 1) * dt), 1) if outside.size else 0.0)
    du = np.abs(np.diff(logs.u, axis=0)) / dt
    da = np.abs(wrap(np.diff(logs.alpha, axis=0))) / dt
    rot_lim = np.any(da[:, azi] >= 0.99 * rot[azi], axis=1)
    res = logs.tau_d[:, [0, 1, 5]] - logs.tau_thr[:, [0, 1, 5]]
    big = np.any(np.abs(res) > np.array([1e3, 1e3, 10e3]), axis=1)
    speed, speed_ref = np.hypot(logs.nu[:, 0], logs.nu[:, 1]), np.hypot(logs.nu_ref[:, 0], logs.nu_ref[:, 1])
    # Final error at the end of every hold
    end = [int((i + 1) * HOLD / dt) - 1 for i in range(N_LEGS)]
    return {
        "max tracking error position [m]": e_pos.max(),
        "time of max tracking error [s]": t[np.argmax(e_pos)],
        "RMS tracking error position [m]": np.sqrt(np.mean(e_pos ** 2)),
        "max tracking error position per leg [m]": [round(float(e_pos[int(i * HOLD / dt):int((i + 1) * HOLD / dt)].max()), 3) for i in range(N_LEGS)],
        "max tracking error heading [deg]": np.rad2deg(e_psi.max()),
        "settling per leg, 0.5 m / 2 deg [s]": settle,
        "max error at end of holds, position [m]": float(d_pos[end].max()),
        "max error at end of holds, heading [deg]": float(np.rad2deg(d_psi[end].max())),
        "peak vessel speed [m/s]": speed.max(),
        "peak reference speed [m/s]": np.nanmax(speed_ref),
        "peak yaw rate [deg/s]": np.rad2deg(np.abs(logs.nu[:, 5]).max()),
        "peak thrust / max per thruster [-]": [round(float(x), 3) for x in np.abs(logs.u).max(axis=0) / u_max],
        "time with a thrust command at max [%]": 100 * np.mean(np.any(np.abs(logs.u_cmd) >= 0.99 * u_max, axis=1)),
        "time at thrust ramp-rate limit [%]": 100 * np.mean(np.any(du >= 0.99 * u_rate, axis=1)),
        "time at azimuth rate limit [%]": 100 * np.mean(rot_lim),
        "latest azimuth rate limit after a setpoint change [s]": float((t[1:][rot_lim] % HOLD).max()) if rot_lim.any() else 0.0,
        "max wrench residual X, Y [kN], N [kNm]": [round(float(x), 2) for x in np.abs(res).max(axis=0) / 1e3],
        "RMS wrench residual X, Y [kN], N [kNm]": [round(float(x), 3) for x in np.sqrt(np.mean(res ** 2, axis=0)) / 1e3],
        "time with residual > 1 kN or 10 kNm [s]": big.sum() * dt,
        "latest residual > 1 kN / 10 kNm after a setpoint change [s]": float((t[big] % HOLD).max()) if big.any() else 0.0,
    }


def plot_sim2(logs, thrusters):
    import matplotlib.pyplot as plt
    t = logs.t
    fig, ax = plt.subplots(2, 4, figsize=(7.16, 3.3))
    ax = ax.ravel()
    # (a) trajectory with setpoints
    ax[0].plot(logs.sp[:, 1], logs.sp[:, 0], "--", lw=0.9, label="reference")
    ax[0].plot(logs.eta[:, 1], logs.eta[:, 0], lw=0.9, label="vessel")
    ax[0].plot(logs.cmd[:, 1], logs.cmd[:, 0], "kx", ms=4, label="setpoints")
    ax[0].set_xlabel("East [m]"); ax[0].set_ylabel("North [m]")
    ax[0].set_aspect("equal", adjustable="datalim")
    # (b) position against reference and command
    for i, lab in ((0, "N"), (1, "E")):
        ax[1].plot(t, logs.cmd[:, i], "k:", lw=0.7)
        ax[1].plot(t, logs.sp[:, i], "--", lw=0.9, color=f"C{i}")
        ax[1].plot(t, logs.eta[:, i], lw=0.9, color=f"C{i}", label=lab)
    ax[1].set_ylabel("Position [m]")
    # (c) heading
    ax[2].plot(t, np.rad2deg(logs.cmd[:, 5]), "k:", lw=0.7, label="command")
    ax[2].plot(t, np.rad2deg(logs.sp[:, 5]), "--", lw=0.9, color="C2", label="reference")
    ax[2].plot(t, np.rad2deg(logs.eta[:, 5]), lw=0.9, color="C2", label="vessel")
    ax[2].set_ylabel("Heading [deg]")
    # (d) tracking error against the reference
    e_pos = np.hypot(logs.eta[:, 0] - logs.sp[:, 0], logs.eta[:, 1] - logs.sp[:, 1])
    ax[3].plot(t, e_pos, color="C3", lw=0.9, label="position [m]")
    ax[3].plot(t, np.rad2deg(np.abs(wrap(logs.eta[:, 5] - logs.sp[:, 5]))), color="C2", lw=0.9, label="heading [deg]")
    ax[3].set_ylabel("Tracking error")
    # (e) BODY velocities against the reference velocities
    for i, lab, sc in ((0, "u", 1.0), (1, "v", 1.0), (5, "r (deg/s)", 180.0 / np.pi)):
        c = {0: "C0", 1: "C1", 5: "C2"}[i]
        ax[4].plot(t, sc * logs.nu_ref[:, i], "--", lw=0.9, color=c)
        ax[4].plot(t, sc * logs.nu[:, i], lw=0.9, color=c, label=lab)
    ax[4].set_ylabel("Velocity [m/s, deg/s]")
    # (f) desired and applied wrench
    for i, lab, sc in ((0, "X", 1e3), (1, "Y", 1e3), (5, "N/10", 1e4)):
        c = {0: "C0", 1: "C1", 5: "C2"}[i]
        ax[5].plot(t, logs.tau_d[:, i] / sc, "--", lw=0.9, color=c)
        ax[5].plot(t, logs.tau_thr[:, i] / sc, lw=0.7, color=c, label=lab)
    ax[5].set_ylabel("Wrench [kN, kNm]")
    # (g) commanded and actual thrust
    for i, th in enumerate(thrusters):
        ax[6].plot(t, logs.u_cmd[:, i] / 1e3, "--", lw=0.9, color=f"C{i}")
        ax[6].plot(t, logs.u[:, i] / 1e3, lw=0.7, color=f"C{i}", label=th.name.replace("_", " "))
    ax[6].set_ylabel("Thrust [kN]")
    # (h) commanded and actual azimuth angle
    for i, th in enumerate(thrusters):
        if th.rot_speed > 0.0:
            ax[7].plot(t, break_wraps(np.rad2deg(wrap(logs.alpha_cmd[:, i]))), "--", lw=0.9, color=f"C{i}")
            ax[7].plot(t, break_wraps(np.rad2deg(wrap(logs.alpha[:, i]))), lw=0.7, color=f"C{i}", label=th.name.replace("_", " "))
    ax[7].set_ylabel("Azimuth angle [deg]")
    for i, a in enumerate(ax):
        a.grid(True, alpha=0.3)
        a.tick_params(labelsize=6)
        a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
        a.set_title(f"({'abcdefgh'[i]})", fontsize=7, loc="left")
        if i > 0:
            a.set_xlabel("Time [s]")
            a.set_xticks(np.arange(0, N_LEGS * HOLD + 1, 600))
            a.set_xticks(np.arange(0, N_LEGS * HOLD + 1, 300), minor=True)
            a.grid(True, which="minor", alpha=0.15)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=5.5, loc="best")
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    return fig


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-show", action="store_true", help="save figures without showing them")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    thrusters = default_thrusters_part_2()

    logs1 = run_sim1()
    m1 = sim1_metrics(logs1)
    write_csv(OUT_DIR / "sim1_metrics.csv", m1)
    print_metrics("Simulation 1 (free drift)", m1)

    logs2 = run_sim2()
    m2 = sim2_metrics(logs2, thrusters)
    write_csv(OUT_DIR / "sim2_metrics.csv", m2)
    print_metrics("Simulation 2 (four-corner test)", m2)

    for name, fig in (("fig_sim1", plot_sim1(logs1)), ("fig_sim2", plot_sim2(logs2, thrusters))):
        fig.savefig(OUT_DIR / f"{name}.pdf")
        fig.savefig(OUT_DIR / f"{name}.png", dpi=250)
    print(f"\nsaved fig_sim1, fig_sim2, sim1_metrics.csv and sim2_metrics.csv to {OUT_DIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
