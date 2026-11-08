"""Verify the Part 2 reference model (report Section 7, Reference Model).

    python verify_reference_model.py
    python verify_reference_model.py --no-show

Two parts:

1) Reference model alone.  Runs ONLY the reference model (no vessel,
   controller or allocation) on the Part 2, Simulation 2 four-corner sequence
   at the Part 2 time step, and computes the inertial feedforward the
   controller adds,

       tau_ff = M_b * acc_ref   (BODY frame, M_b = diag(m11, m22, m66)),

   for the tuned third-order filter and the rejected alternatives.  The
   acceleration is rotated to BODY with the reference heading (the controller
   uses the vessel heading; the two coincide when the vessel tracks the
   reference).  The controller also subtracts a Coriolis term
   S(r_d) J^T eta_dot_d; it is left out because position and heading never
   change at the same time in the four-corner test.  The peaks of tau_ff and
   of its rate of change are compared with the thrust and ramp-rate limits.

2) Closed loop, Part 2 Simulation 2.  Runs the preset (constrained thrusters,
   no environment, no observer) and computes the tracking error against the
   reference, the settling time of every leg, the thruster limits that become
   active and the difference between desired and applied wrench.

Output (figures/ next to this script):
    fig_reference_feedforward.pdf/.png   feedforward force and rate, 2nd vs 3rd order
    reference_verification.csv           part 1: all candidate numbers
    reference_sim2_metrics.csv           part 2: Simulation 2 numbers
"""
import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_2.config import Part2SimConfig, RefAxisConfig, default_thrusters_part_2
from part_2.reference import ReferenceModel
from run_case_part_2 import DT, environment, four_corner_commands, run

OUT_DIR = Path(__file__).resolve().parent / "figures"

HOLD = 300.0                                   # hold time per four-corner leg [s]
N_LEGS = 5                                     # commanded legs in the four-corner test
M_B = np.array([6.007e5, 7.067e5, 5.456e7])    # m11, m22, m66 used by the controller feedforward
SETTLE_POS, SETTLE_PSI = 0.5, np.deg2rad(2.0)  # settling band around the setpoint [m, rad]

TUNED = RefAxisConfig()                        # tuned defaults from part_2/config.py
CASES = [                                      # (label, config) compared in the table
    ("3rd order, wn = 0.05 (used)", TUNED),
    ("2nd order, wn = 0.05", replace(TUNED, order=2)),
    ("3rd order, wn = 0.03", replace(TUNED, wn=0.03)),
    ("3rd order, wn = 0.07", replace(TUNED, wn=0.07)),
    ("3rd order, wn = 0.08", replace(TUNED, wn=0.08)),
]


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# -- PART 1: REFERENCE MODEL ALONE -- #

def run_reference(cfg):
    """Reference model alone on the four-corner sequence; returns t, eta_ref, nu_ref, tau_ff, eta_cmd."""
    sim_cfg = Part2SimConfig(dt=DT, T=N_LEGS * HOLD)
    eta_cmd = four_corner_commands(sim_cfg, HOLD)
    ref = ReferenceModel(dt=DT, cfg_xy=cfg, cfg_psi=cfg)
    ref.reset(np.zeros(6))
    n = eta_cmd.shape[0]
    t = np.arange(n) * DT
    eta_ref, nu_ref, tau_ff = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(n):
        eta, nu, acc = ref.step(t[k], DT, eta_cmd[k])
        eta_ref[k], nu_ref[k] = eta[[0, 1, 5]], nu[[0, 1, 5]]
        # Rotates the NED acceleration reference to BODY with the reference heading
        c, s = np.cos(eta[5]), np.sin(eta[5])
        acc_b = np.array([c * acc[0] + s * acc[1], -s * acc[0] + c * acc[1], acc[5]])
        # Inertial feedforward, as in the controller
        tau_ff[k] = M_B * acc_b
    return t, eta_ref, nu_ref, tau_ff, eta_cmd[:, [0, 1, 5]]


def settling_times(t, eta_ref, eta_cmd):
    """2 % settling time of the reference after each setpoint change [s]."""
    out = []
    for i in range(N_LEGS):
        k0, k1 = int(i * HOLD / DT), int((i + 1) * HOLD / DT)
        prev = np.zeros(3) if i == 0 else eta_cmd[k0 - 1]
        step = np.abs(eta_cmd[k0] - prev)
        # Settled when every changed axis is within 2 % of its step
        err = np.abs(eta_ref[k0:k1] - eta_cmd[k0:k1])
        ok = np.all((err <= 0.02 * step) | (step < 1e-9), axis=1)
        bad = np.where(~ok)[0]
        out.append(round(float((bad[-1] + 1) * DT), 1) if bad.size else 0.0)
    return out


def reference_alone(thrusters):
    """Part 1: feedforward demand of every candidate; returns the logs for the figure."""
    tunnel = thrusters[0]
    rows = [["case", "peak ff X [kN]", "peak ff Y [kN]", "peak ff N [kNm]",
             "peak ff rate X [kN/s]", "peak ff rate Y [kN/s]", "peak ff rate N [kNm/s]",
             "peak speed [m/s]", "peak yaw rate [deg/s]", "settling per leg [s]"]]
    results = {}
    for label, cfg in CASES:
        t, eta_ref, nu_ref, tau_ff, eta_cmd = run_reference(cfg)
        results[label] = (t, tau_ff)
        peak = np.abs(tau_ff).max(axis=0) / 1e3
        rate = np.abs(np.diff(tau_ff, axis=0)).max(axis=0) / DT / 1e3
        speed = np.hypot(nu_ref[:, 0], nu_ref[:, 1]).max()
        yaw_rate = np.degrees(np.abs(nu_ref[:, 2]).max())
        settle = settling_times(t, eta_ref, eta_cmd)
        rows.append([label, *np.round(peak, 1), *np.round(rate, 2),
                     round(float(speed), 2), round(float(yaw_rate), 2), settle])
        print(f"{label:28s} ff peak {peak.round(1)}  ff rate {rate.round(2)}  "
              f"speed {speed:.2f} m/s  yaw rate {yaw_rate:.2f} deg/s  settling {settle}")
    # Extra rows: the limits the numbers are compared with
    rows.append(["limit: tunnel thruster", tunnel.u_max / 1e3, "", "", tunnel.u_rate / 1e3, "", "", "", "", ""])
    rows.append(["limit: sum of azimuth ramp rates", "", "", "",
                 sum(x.u_rate for x in thrusters[1:]) / 1e3, "", "", "", "", ""])
    with open(OUT_DIR / "reference_verification.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return results


# -- PART 2: CLOSED LOOP, SIMULATION 2 -- #

def closed_loop_sim2(thrusters):
    """Part 2: Simulation 2 preset; tracking, settling, active limits and wrench residual."""
    cfg = Part2SimConfig(dt=DT, T=N_LEGS * HOLD, use_reference=True,
                         use_observer=False, thruster_dynamics=True)
    logs = run(cfg, four_corner_commands(cfg, HOLD),
               environment(current=False, wind=False, waves=False))
    t, dt = logs.t, float(logs.t[1] - logs.t[0])
    u_max = np.array([th.u_max for th in thrusters])
    u_rate = np.array([th.u_rate for th in thrusters])
    rot = np.array([th.rot_speed for th in thrusters])

    # Tracking error: vessel against the reference output (logs.sp)
    e_pos = np.hypot(logs.eta[:, 0] - logs.sp[:, 0], logs.eta[:, 1] - logs.sp[:, 1])
    e_psi = np.abs(wrap(logs.eta[:, 5] - logs.sp[:, 5]))
    # Settling: time after each setpoint change until the vessel stays within the band of the command
    d_pos = np.hypot(logs.eta[:, 0] - logs.cmd[:, 0], logs.eta[:, 1] - logs.cmd[:, 1])
    d_psi = np.abs(wrap(logs.eta[:, 5] - logs.cmd[:, 5]))
    settle = []
    for i in range(N_LEGS):
        k0, k1 = int(i * HOLD / dt), int((i + 1) * HOLD / dt)
        outside = np.where((d_pos[k0:k1] > SETTLE_POS) | (d_psi[k0:k1] > SETTLE_PSI))[0]
        settle.append(round(float((outside[-1] + 1) * dt), 1) if outside.size else 0.0)

    # Thruster limits: commanded thrust at maximum, thrust rate and azimuth rate at their limits
    du = np.abs(np.diff(logs.u, axis=0)) / dt
    da = np.abs(wrap(np.diff(logs.alpha, axis=0))) / dt
    azi = rot > 0.0
    rot_lim = np.any(da[:, azi] >= 0.99 * rot[azi], axis=1)
    # Wrench residual: desired minus applied thruster wrench (BODY X, Y, N)
    res = logs.tau_d[:, [0, 1, 5]] - logs.tau_thr[:, [0, 1, 5]]
    big = np.any(np.abs(res) > np.array([1e3, 1e3, 10e3]), axis=1)   # above 1 kN / 10 kNm
    big_times = t[big]
    since_leg = big_times % HOLD                                      # time since the last setpoint change

    m = {
        "max tracking error position [m]": float(e_pos.max()),
        "time of max tracking error [s]": float(t[np.argmax(e_pos)]),
        "RMS tracking error position [m]": float(np.sqrt(np.mean(e_pos ** 2))),
        "max tracking error heading [deg]": float(np.rad2deg(e_psi.max())),
        "settling per leg, 0.5 m / 2 deg [s]": settle,
        "peak thrust / max per thruster [-]": [round(float(x), 3) for x in np.abs(logs.u).max(axis=0) / u_max],
        "time with a thrust command at max [%]": float(100 * np.mean(np.any(np.abs(logs.u_cmd) >= 0.99 * u_max, axis=1))),
        "time at thrust ramp-rate limit [%]": float(100 * np.mean(np.any(du >= 0.99 * u_rate, axis=1))),
        "time at azimuth rate limit [%]": float(100 * np.mean(rot_lim)),
        "latest azimuth rate limit after a setpoint change [s]": float((t[1:][rot_lim] % HOLD).max()) if rot_lim.any() else 0.0,
        "max wrench residual X, Y [kN], N [kNm]": [round(float(x), 2) for x in np.abs(res).max(axis=0) / 1e3],
        "RMS wrench residual X, Y [kN], N [kNm]": [round(float(x), 3) for x in np.sqrt(np.mean(res ** 2, axis=0)) / 1e3],
        "time with residual > 1 kN or 10 kNm [s]": float(big.sum() * dt),
        "latest residual > 1 kN / 10 kNm after a setpoint change [s]": float(since_leg.max()) if big.any() else 0.0,
    }
    with open(OUT_DIR / "reference_sim2_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        for k, v in m.items():
            writer.writerow([k, v if isinstance(v, list) else round(v, 3)])
    print("\nPart 2, Simulation 2 (closed loop)")
    for k, v in m.items():
        print(f"  {k:62s} {v if isinstance(v, list) else round(v, 3)}")


def plot_feedforward(results, tunnel):
    """Sway feedforward and its rate at the start of the first sway leg, 2nd vs 3rd order."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    for label, style in [("2nd order, wn = 0.05", "--"), ("3rd order, wn = 0.05 (used)", "-")]:
        t, tau_ff = results[label]
        axes[0].plot(t, tau_ff[:, 1] / 1e3, style, label=label)
        axes[1].plot(t[1:], np.diff(tau_ff[:, 1]) / DT / 1e3, style, label=label)
    for s in (-1, 1):
        axes[0].axhline(s * tunnel.u_max / 1e3, color="k", ls=":", lw=1)
        axes[1].axhline(s * tunnel.u_rate / 1e3, color="k", ls=":", lw=1)
    axes[0].text(452, -tunnel.u_max / 1e3 + 4, "tunnel max thrust", fontsize=8)
    axes[1].text(452, -tunnel.u_rate / 1e3 - 9, "tunnel ramp rate", fontsize=8)
    axes[0].set_ylabel("sway feedforward [kN]")
    axes[1].set_ylabel("rate of change [kN/s]")
    axes[1].set_ylim(-30, 30)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_xlim(290, 520)
    axes[0].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Inertial feedforward at the start of the first sway leg (four-corner test)")
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
    th = thrusters
    print(f"limits: tunnel {th[0].u_max/1e3:.0f} kN, {th[0].u_rate/1e3:.0f} kN/s; "
          f"azimuths {th[1].u_max/1e3:.0f} kN, {th[1].u_rate/1e3:.0f} kN/s each")

    results = reference_alone(thrusters)
    closed_loop_sim2(thrusters)

    fig = plot_feedforward(results, thrusters[0])
    fig.savefig(OUT_DIR / "fig_reference_feedforward.pdf")
    fig.savefig(OUT_DIR / "fig_reference_feedforward.png", dpi=200)
    print(f"\nsaved fig_reference_feedforward, reference_verification.csv and "
          f"reference_sim2_metrics.csv to {OUT_DIR}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
