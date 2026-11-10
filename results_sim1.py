"""Report figure and numbers for Part 2, Simulation 1 (Results section).

    python results_sim1.py
    python results_sim1.py --no-show

Runs the sim1 preset of run_case_part_2.py (free drift, DP off, 300 s in the
design environment) and collects the plots the assignment asks for in one
figure, in the same style as the course plots: trajectory, position,
heading, current in BODY, wind speed (total, mean + slow, gust), wind
direction and the wind and wave loads.

Output (figures/ next to this script):
    fig_sim1.pdf/.png      Simulation 1 report figure
    sim1_metrics.csv       all Simulation 1 numbers quoted in the report
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_2.config import Part2SimConfig
from run_case_part_2 import DT, WAVES_HS, WAVES_TP, WIND_FROM, environment, run

OUT_DIR = Path(__file__).resolve().parent / "figures"

T_SIM1 = 300.0                 # Simulation 1 length [s]
LPP = 34.0                     # Gunnerus length between perpendiculars [m]
G = 9.81                       # gravity [m/s^2]
LF_WINDOW = 30.0               # moving average separating slow and wave-frequency motion [s]


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def moving_average(x, dt, window=LF_WINDOW):
    """Centred moving average, edge padded, same length."""
    k = max(int(round(window / dt)), 1)
    xp = np.pad(np.asarray(x, float), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(xp, np.ones(k) / k, "valid")


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
    d_beta = np.rad2deg(wrap(wrap(logs.beta_w - np.pi) - WIND_FROM))
    tw, twa = logs.tau_w6[:, [0, 1, 5]], logs.tau_wave[:, [0, 1, 5]]
    lam_p = G * WAVES_TP ** 2 / (2.0 * np.pi)           # deep-water peak wavelength [m]
    return {
        "final North [m]": N[-1],
        "final East [m]": E[-1],
        "final drift distance [m]": np.hypot(N[-1], E[-1]),
        "drift direction, towards [deg from North]": np.rad2deg(np.arctan2(E[-1], N[-1])) % 360.0,
        "final heading [deg]": np.rad2deg(psi[-1]),
        "slow drift velocity North, last 100 s [m/s]": vN,
        "slow drift velocity East, last 100 s [m/s]": vE,
        "wave-frequency motion std N [m]": wf[0],
        "wave-frequency motion std E [m]": wf[1],
        "wave-frequency motion std heading [deg]": np.rad2deg(wf[2]),
        "gust speed std [m/s]": np.std(logs.U_w_gust),
        "max |direction deviation| [deg]": np.abs(d_beta).max(),
        "mean wind load X [kN]": np.mean(tw[:, 0]) / 1e3,
        "mean wave load X [kN]": np.mean(twa[:, 0]) / 1e3,
        "mean wave load Y [kN]": np.mean(twa[:, 1]) / 1e3,
        "wave load std Y [kN]": np.std(twa[:, 1]) / 1e3,
        "max |wave load Y| [kN]": np.abs(twa[:, 1]).max() / 1e3,
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
    # Wind direction (from), deviation from the mean, with the +/-5 deg bound
    d_beta = np.rad2deg(wrap(wrap(logs.beta_w - np.pi) - WIND_FROM))
    ax[5].plot(t, d_beta, color="C3")
    ax[5].axhline(5, color="k", ls=":", lw=0.8); ax[5].axhline(-5, color="k", ls=":", lw=0.8)
    ax[5].set_title("Wind direction deviation [deg]")
    # Wind loads (yaw moment divided by 10 to share the axis)
    ax[6].plot(t, logs.tau_w6[:, 0] / 1e3, label="X [kN]"); ax[6].plot(t, logs.tau_w6[:, 1] / 1e3, label="Y [kN]")
    ax[6].plot(t, logs.tau_w6[:, 5] / 1e4, label="N/10 [kNm]")
    ax[6].set_title("Wind load")
    # Wave loads, first order plus drift
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-show", action="store_true", help="save the figure without showing it")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    logs = run_sim1()
    metrics = sim1_metrics(logs)
    with open(OUT_DIR / "sim1_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        for k, v in metrics.items():
            writer.writerow([k, round(float(v), 4)])
    print("Simulation 1 (free drift)")
    for k, v in metrics.items():
        print(f"  {k:46s} {float(v):.3f}")

    fig = plot_sim1(logs)
    fig.savefig(OUT_DIR / "fig_sim1.pdf")
    fig.savefig(OUT_DIR / "fig_sim1.png", dpi=250)
    print(f"saved fig_sim1 and sim1_metrics.csv to {OUT_DIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
