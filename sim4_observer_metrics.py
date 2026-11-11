"""Observer metrics for Part 2, Simulation 4 (observer selection, Results section).

    python sim4_observer_metrics.py

Runs the Simulation 4 fixed-wrench observer test of run_case_part_2.py
(desired wrench [2, 2, 2] kN (kNm), Simulation 1 environment, 600 s, sensor
noise off) for both observers, with and without waves, and prints the
estimation errors quoted in the Simulation 4 text:

  with waves     errors against the low-frequency truth (centred 30 s moving
                 average of the true pose and velocity), as in Simulation 6
                 and the extra-credit study;
  without waves  errors against the true pose and velocity.  Without waves the
                 truth has no wave-frequency motion, and the moving average
                 would lag behind the drifting vessel.

All errors are RMS over 300-600 s.  Same settings as the sim4 preset
(seed, time step, WAVES_N), so the runs are the ones behind its figures.

Output (figures/ next to this script):
    sim4_observer_metrics.csv     all numbers printed here
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_case_part_2 as rc
from simulation.plotter_part_2 import _moving_average
from simulation.simulation_part_2 import Part2SimConfig

OUT_DIR = Path(__file__).resolve().parent / "figures"
T_FROM, T_TO = 300.0, 600.0


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def observer_errors(logs, against_lf: bool) -> dict:
    """RMS position [m], heading [deg] and surge/sway velocity [m/s] errors."""
    dt = float(logs.t[1] - logs.t[0])
    sel = (logs.t >= T_FROM) & (logs.t <= T_TO)
    truth = logs.eta[:, [0, 1, 5]].copy()
    vel = logs.nu[:, [0, 1, 5]].copy()
    if against_lf:
        truth[:, 2] = np.unwrap(truth[:, 2])
        truth = np.column_stack([_moving_average(truth[:, i], dt) for i in range(3)])
        vel = np.column_stack([_moving_average(vel[:, i], dt) for i in range(3)])
    e_pose = logs.eta_est[:, [0, 1, 5]] - truth
    e_pose[:, 2] = _wrap(e_pose[:, 2])
    e_vel = logs.nu_est[:, [0, 1, 5]] - vel
    return {
        "position RMS [m]": _rms(np.linalg.norm(e_pose[sel, :2], axis=1)),
        "heading RMS [deg]": float(np.rad2deg(_rms(e_pose[sel, 2]))),
        "surge/sway velocity RMS [m/s]": _rms(np.linalg.norm(e_vel[sel, :2], axis=1)),
    }


def main():
    fixed = rc.WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
    rows = []
    for kind in ("nonlinear_passive", "kalman"):
        for waves in (True, False):
            cfg = Part2SimConfig(dt=rc.DT, T=600.0, use_reference=False,
                                 use_observer=True, observer_type=kind)
            logs = rc.run(cfg, np.zeros(6), rc.environment(waves=waves), controller=fixed)
            metrics = observer_errors(logs, against_lf=waves)
            case = "with waves (vs LF truth)" if waves else "without waves (vs truth)"
            rows.append({"observer": kind, "case": case, **metrics})

    print(f"Simulation 4 fixed-wrench observer test, RMS over {T_FROM:.0f}-{T_TO:.0f} s, "
          f"seed {rc.SEED}, dt = {rc.DT} s, {rc.WAVES_N} wave components")
    keys = list(rows[0].keys())
    print(f"{'observer':<19}{'case':<27}" + "".join(f"{k:>32}" for k in keys[2:]))
    for r in rows:
        print(f"{r['observer']:<19}{r['case']:<27}"
              + "".join(f"{r[k]:>32.3f}" for k in keys[2:]))

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "sim4_observer_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"saved sim4_observer_metrics.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
