"""Effect of the wind gust on DP station keeping (environmental loads section).

    python compare_gust_dp.py
    python compare_gust_dp.py --no-show

Holds the origin with the complete DP system (reference model, LQR
controller, thrust allocation, thruster dynamics and the selected observer)
in the Simulation 1 environment, once with the gust and once without.  The
wind uses separate random streams for gust, slow speed and direction, so
switching the gust off leaves the slow speed, the direction, the current and
the waves unchanged: the difference between the runs is the gust alone.

Output (figures/ next to this script):
    gust_dp_comparison.csv          all numbers quoted in the report
    fig_gust_dp.pdf/.png            wind speed, wind sway force and tunnel thrust
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_2.config import Part2SimConfig, default_thrusters_part_2
from run_case_part_2 import DT, environment, run, selected_observer
from simulation.capability import average_utilization, max_deviation

OUT_DIR = Path(__file__).resolve().parent / "figures"

T_RUN = 1800.0          # run length [s]; longer than Simulation 4 for steadier statistics
EVAL_FROM = 300.0       # evaluation starts after the start-up transient [s]
ZOOM = (600.0, 900.0)   # time window shown in the figure [s]


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def run_case(gust):
    """Station keeping at the origin with the selected observer, gust on or off."""
    cfg = Part2SimConfig(dt=DT, T=T_RUN, use_observer=True, observer_type=selected_observer())
    return run(cfg, np.zeros(6), environment(gust=gust))


def metrics(logs, thrusters):
    """Wind, station keeping and actuator activity over the evaluation window."""
    dt = float(logs.t[1] - logs.t[0])
    sel = logs.t >= EVAL_FROM
    u_rate = np.array([th.u_rate for th in thrusters])
    pos = np.hypot(logs.eta[sel, 0] - logs.sp[sel, 0], logs.eta[sel, 1] - logs.sp[sel, 1])
    psi = np.rad2deg(np.abs(wrap(logs.eta[sel, 5] - logs.sp[sel, 5])))
    lf_pos, lf_psi = max_deviation(logs, EVAL_FROM, deviation="lf")
    du = np.abs(np.diff(logs.u, axis=0))[sel[1:]] / dt          # thrust rates [N/s]
    tau_wind = logs.tau_w6[sel][:, [0, 1, 5]]                     # wind load X, Y, N
    return {
        "wind speed std [m/s]": float(np.std(logs.U_w[sel])),
        "wind load std X [kN]": np.std(tau_wind[:, 0]) / 1e3,
        "wind load std Y [kN]": np.std(tau_wind[:, 1]) / 1e3,
        "wind load std N [kNm]": np.std(tau_wind[:, 2]) / 1e3,
        "RMS position error [m]": rms(pos),
        "max position error [m]": float(pos.max()),
        "max LF position error [m]": lf_pos,
        "RMS heading error [deg]": rms(psi),
        "max LF heading error [deg]": float(np.rad2deg(lf_psi)),
        "mean thrust utilisation [%]": average_utilization(logs, thrusters, EVAL_FROM),
        "mean thrust rate, sum of thrusters [kN/s]": du.sum(axis=1).mean() / 1e3,
        "time at ramp-rate limit [%]": 100 * np.mean(np.any(du >= 0.99 * u_rate, axis=1)),
    }


def plot(logs_by_case):
    """Wind speed, wind sway force and tunnel thrust with and without the gust."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
    for label, logs in logs_by_case.items():
        sel = (logs.t >= ZOOM[0]) & (logs.t <= ZOOM[1])
        axes[0].plot(logs.t[sel], logs.U_w[sel], lw=0.9, label=label)
        axes[1].plot(logs.t[sel], logs.tau_w6[sel, 1] / 1e3, lw=0.9, label=label)
        axes[2].plot(logs.t[sel], logs.u[sel, 0] / 1e3, lw=0.9, label=label)
    axes[0].set_ylabel("Wind speed [m/s]")
    axes[1].set_ylabel("Wind sway force [kN]")
    axes[2].set_ylabel("Tunnel thrust [kN]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Station keeping with and without the wind gust (Simulation 1 environment)")
    fig.tight_layout()
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
    logs_by_case, rows = {}, []
    for label, gust in (("gust on", True), ("gust off", False)):
        logs = run_case(gust)
        logs_by_case[label] = logs
        rows.append({"case": label, **metrics(logs, thrusters)})

    with open(OUT_DIR / "gust_dp_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow([round(v, 4) if isinstance(v, float) else v for v in r.values()])
    print(f"{'':44s}" + "".join(f"{r['case']:>12s}" for r in rows))
    for key in rows[0]:
        if key != "case":
            print(f"{key:44s}" + "".join(f"{r[key]:12.3f}" for r in rows))

    fig = plot(logs_by_case)
    fig.savefig(OUT_DIR / "fig_gust_dp.pdf")
    fig.savefig(OUT_DIR / "fig_gust_dp.png", dpi=200)
    print(f"saved gust_dp_comparison.csv and fig_gust_dp to {OUT_DIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
