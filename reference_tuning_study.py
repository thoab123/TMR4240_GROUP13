"""
reference_tuning_study.py
-------------------------
Closed-loop study behind the reference-model tables in the report.

Runs Simulation 3 (setpoint change) and Simulation 4 (four-corner test) with
the SAME controller (DPController defaults) and thrust allocation every time,
so only the reference model changes:

  Part A - filter order and wn:  no reference, 2nd order (project Eq. (2))
           and 3rd order (our choice) for several natural frequencies.
  Part B - robustness: the four-corner test repeated with the Simulation 1
           environments (0.5 m/s current / 15 m/s wind from east), which load
           the tunnel thruster before any setpoint change.

Metrics:
  t_s       time until position error < 0.5 m and heading error < 2 deg
            (Sim 3), or worst leg of the four-corner test (Sim 4)
  util      peak |u_j| / u_max,j over the three thrusters [%]
  tun_sat   total time the tunnel thruster is at >= 99 % of u_max [s]
  lag       peak horizontal distance between vessel and reference [m]
  fuel      sum_j |u_j[kN]|^1.5 * dt / 1000 - proxy for propeller energy
            (power ~ thrust^1.5); only meaningful as a relative number

Run from the repository root (takes ~10 minutes):
    python reference_tuning_study.py

Prints both tables and saves figures/reference_tuning_table.csv and
figures/reference_robustness_table.csv
"""
from pathlib import Path
import csv

import numpy as np

from part_1.config import SimConfig, RefAxisConfig, default_thrusters_gunnerus3
from part_1.controller import DPController
from part_1.current import Current
from part_1.reference import ReferenceModel
from part_1.wind import Wind
from simulation.simulation_part_1 import DPSimulator3DOF

# (order, wn) cases for Part A; None = no reference model
CASES_A = [None, (2, 0.05), (3, 0.03), (3, 0.05), (3, 0.07), (3, 0.08)]
# cases for Part B (old design, chosen design, faster alternative)
CASES_B = [(2, 0.05), (3, 0.05), (3, 0.07)]

U_MAX = np.array([32e3, 80e3, 80e3])          # [N] Tunnel, Azimuth_1, Azimuth_2
SETTLE_POS = 0.5                              # [m]   same criterion as the checks
SETTLE_PSI = np.deg2rad(2.0)                  # [rad]
HOLD = 300.0                                  # [s]   four-corner hold per setpoint
CORNERS = [[50, 0, 0], [50, -50, 0], [50, -50, -np.pi / 4],
           [0, -50, -np.pi / 4], [0, 0, 0]]
OUT_DIR = Path("figures")


def wrap(a):
    """Vectorised angle wrap to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def make_ref(case, dt):
    """Build a ReferenceModel for (order, wn), or None for 'no reference'."""
    if case is None:
        return None
    order, wn = case
    cfg = RefAxisConfig(wn=wn, order=order)
    return ReferenceModel(dt, cfg, cfg)             # same wn for N, E and psi


def run(cfg: SimConfig, eta_cmd, case, **env):
    """One closed-loop run with the given reference case and environment."""
    cfg.use_reference = case is not None
    sim = DPSimulator3DOF(cfg, DPController(), default_thrusters_gunnerus3(),
                          reference=make_ref(case, cfg.dt))
    sim.reset_state()
    return sim.run(np.asarray(eta_cmd, dtype=float), **env)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def utilisation(lg):
    return np.max(np.abs(lg.u) / U_MAX, axis=0)


def tunnel_saturated_time(lg):
    return float(np.sum(np.abs(lg.u[:, 0]) >= 0.99 * U_MAX[0]) * (lg.t[1] - lg.t[0]))


def tracking_lag(lg):
    return float(np.max(np.hypot(lg.eta[:, 0] - lg.sp[:, 0],
                                 lg.eta[:, 1] - lg.sp[:, 1])))


def fuel_proxy(lg):
    return float(np.sum(np.abs(lg.u / 1e3) ** 1.5) * (lg.t[1] - lg.t[0]) / 1e3)


def settle_time(t, eta, target):
    pos = np.hypot(eta[:, 0] - target[0], eta[:, 1] - target[1])
    psi = np.abs(wrap(eta[:, 5] - target[2]))
    outside = np.where((pos > SETTLE_POS) | (psi > SETTLE_PSI))[0]
    if len(outside) == 0:
        return 0.0
    return t[outside[-1] + 1] - t[0] if outside[-1] + 1 < len(t) else np.inf


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------
def sim3(case):
    """Simulation 3: [0,0,0] -> [10, 10, 3pi/2], no environment."""
    target = np.zeros(6)
    target[0], target[1], target[5] = 10.0, 10.0, 3 * np.pi / 2
    lg = run(SimConfig(T=600.0), target, case)
    return dict(t_s=settle_time(lg.t, lg.eta, [10.0, 10.0, wrap(target[5])]),
                util=utilisation(lg), fuel=fuel_proxy(lg))


def four_corner_cmd(cfg):
    n = int(round(cfg.T / cfg.dt)) + 1
    per = int(round(HOLD / cfg.dt))
    cmd = np.zeros((n, 6))
    for i, c in enumerate(CORNERS):
        rows = slice(i * per, n if i == len(CORNERS) - 1 else (i + 1) * per)
        cmd[rows, 0], cmd[rows, 1], cmd[rows, 5] = c
    return cmd


def sim4(case, **env):
    """Simulation 4: four-corner test (optionally with current/wind)."""
    cfg = SimConfig(T=HOLD * len(CORNERS))
    lg = run(cfg, four_corner_cmd(cfg), case, **env)
    legs = []
    for i, c in enumerate(CORNERS):
        sel = (lg.t >= i * HOLD) & (lg.t < (i + 1) * HOLD)
        legs.append(settle_time(lg.t[sel], lg.eta[sel], c))
    return dict(t_s=max(legs), util=utilisation(lg), tun_sat=tunnel_saturated_time(lg),
                lag=tracking_lag(lg), fuel=fuel_proxy(lg))


def label(case):
    return "no ref" if case is None else f"order {case[0]}, wn={case[1]}"


def print_table(title, header, rows):
    print(f"\n{title}")
    print(" | ".join(f"{h:>13}" for h in header))
    for r in rows:
        print(" | ".join(f"{str(v):>13}" for v in r))


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ---------------- Part A: order and wn (no environment) ----------------
    header_a = ["case", "Tr [s]", "S3 t_s [s]", "S3 util [%]", "S3 fuel",
                "S4 t_s [s]", "S4 util [%]", "S4 tun_sat[s]", "S4 lag [m]", "S4 fuel"]
    rows_a = []
    for case in CASES_A:
        print(f"Part A: {label(case)} ...", flush=True)
        s3 = sim3(case)
        if case is None:
            # the four-corner test is only defined with a reference model here
            rows_a.append([label(case), "-", f"{s3['t_s']:.0f}", f"{100*s3['util'].max():.0f}",
                           f"{s3['fuel']:.1f}", "-", "-", "-", "-", "-"])
            continue
        s4 = sim4(case)
        rows_a.append([label(case), f"{1/case[1]:.1f}", f"{s3['t_s']:.0f}",
                       f"{100*s3['util'].max():.0f}", f"{s3['fuel']:.1f}",
                       f"{s4['t_s']:.0f}", f"{100*s4['util'].max():.0f}",
                       f"{s4['tun_sat']:.1f}", f"{s4['lag']:.2f}", f"{s4['fuel']:.1f}"])
    print_table("Part A: filter order and natural frequency (no environment)", header_a, rows_a)

    # ---------------- Part B: four-corner test with environment -------------
    envs = {
        "current 0.5 m/s from E": lambda: dict(current=Current(0.5, np.pi / 2, semantics="from")),
        "wind 15 m/s from E": lambda: dict(wind=Wind(15.0, np.pi / 2, semantics="from",
                                                     sigma_slow=1.0, seed=0)),
    }
    header_b = ["environment", "case", "util [%]", "tun_sat [s]", "lag [m]", "t_s [s]"]
    rows_b = []
    for env_name, make_env in envs.items():
        for case in CASES_B:
            print(f"Part B: {env_name}, {label(case)} ...", flush=True)
            s4 = sim4(case, **make_env())
            rows_b.append([env_name, label(case), f"{100*s4['util'].max():.0f}",
                           f"{s4['tun_sat']:.1f}", f"{s4['lag']:.2f}", f"{s4['t_s']:.0f}"])
    print_table("Part B: four-corner test with environmental loads", header_b, rows_b)

    with open(OUT_DIR / "reference_tuning_table.csv", "w", newline="") as f:
        csv.writer(f).writerows([header_a] + rows_a)
    with open(OUT_DIR / "reference_robustness_table.csv", "w", newline="") as f:
        csv.writer(f).writerows([header_b] + rows_b)
    print(f"\nsaved tables to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
