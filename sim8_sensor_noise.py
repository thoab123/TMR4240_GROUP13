"""Part 2, extra credit ("Sim 8"): sensor noise, noise off vs noise on.

    python sim8_sensor_noise.py
    python sim8_sensor_noise.py --no-show

Repeats the two observer tests of Part 2, Simulation 4 with the prescribed
sensor noise switched on.  Every case is run twice, noise off and noise on,
with everything else identical (seeds, setpoints, controller, observer tuning,
evaluation window), so the difference is caused by the noise alone:

  A) Observer comparison: fixed desired wrench [2, 2, 2] kN (kN m), controller
     out of the loop, waves on, both observers.
  B) Closed loop: hold the origin with raw feedback and with the selected
     observer (SELECTED_OBSERVER in run_case_part_2.py).

The noise levels are fixed course values (models/sensors.py): 0.20 m North and
East, 0.20 deg heading, and 0.02 m/s, 0.05 deg/s on the simulated velocity
that raw feedback uses.  The observers only receive the noisy pose.

Estimation errors are measured against the low-frequency (LF) truth: the true
state smoothed with a 30 s centred moving average, the same definition as
plot_observer_error (simulation/plotter_part_2.py) and the Simulation 5
deviation measure (simulation/capability.py).  Wave filtering and estimate
noise are measured in frequency bands: the wave band 0.5-2 wp around the
design-sea peak wp = 2 pi / Tp, and the noise band above 3 rad/s, where the
vessel itself barely moves, so what is left there in an estimate is noise.

Output (figures/ next to this script):
    sim8_observer_metrics.csv      part A, one row per observer and noise setting
    sim8_closed_loop_metrics.csv   part B, one row per feedback and noise setting
    fig_sim8_estimates.pdf/.png    measured, true and estimated pose and sway velocity, noise on
    fig_sim8_thrust.pdf/.png       thrust with raw feedback vs observer, noise on
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_2.config import Part2SimConfig, default_thrusters_part_2
from run_case_part_2 import DT, WAVES_TP, WrenchSequence, environment, run, selected_observer
from simulation.capability import average_utilization, max_deviation

OUT_DIR = Path(__file__).resolve().parent / "figures"

T_RUN = 600.0                    # length of every run [s], as in Simulation 4
EVAL_FROM = 300.0                # evaluation window: second half of each run [s]
LF_WINDOW = 30.0                 # moving average that defines the LF truth [s]
MAX_LAG = 20.0                   # largest estimate lag or lead searched for [s]
W_PEAK = 2.0 * np.pi / WAVES_TP  # design-sea peak frequency [rad/s]
WAVE_BAND = (0.5 * W_PEAK, 2.0 * W_PEAK)   # wave band [rad/s]
NOISE_BAND = (3.0, np.inf)       # noise band [rad/s], above the wave band
WELCH_SEGMENT = 1024             # samples per Welch segment (102 s at dt = 0.1 s)
FIXED_WRENCH = [2e3, 2e3, 2e3]   # desired wrench of part A [N, N, N m]
ZOOM = (400.0, 460.0)            # time window shown in the estimate figure [s]
ZOOM_THRUST = (300.0, 400.0)     # time window shown in the thrust figure [s]
NAMES = {"nonlinear_passive": "NPO", "kalman": "EKF"}


# -- SIGNAL HELPERS -- #

def moving_average(x, dt, window=LF_WINDOW):
    """Centred moving average along time (axis 0), edge padded, same length."""
    k = max(int(round(window / dt)), 1)
    pad = [(k // 2, k - 1 - k // 2)] + [(0, 0)] * (np.ndim(x) - 1)
    xp = np.pad(np.asarray(x, float), pad, mode="edge")
    kernel = np.ones(k) / k
    if np.ndim(x) == 1:
        return np.convolve(xp, kernel, "valid")
    return np.column_stack([np.convolve(xp[:, i], kernel, "valid") for i in range(xp.shape[1])])


def pose(eta):
    """North, East and unwrapped heading [m, m, rad] from a logged 6-DOF pose."""
    return np.column_stack([eta[:, 0], eta[:, 1], np.unwrap(eta[:, 5])])


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def lag(est, ref, dt):
    """Estimate lag [s]: the time shift tau in [-MAX_LAG, MAX_LAG] that minimises
    the RMS of est(t) - ref(t - tau).  Positive: the estimate trails the reference."""
    n, k_max = len(est), int(MAX_LAG / dt)
    shifts = range(-k_max, k_max + 1)
    cost = [rms(est[k:] - ref[:n - k]) if k >= 0 else rms(est[:n + k] - ref[-k:]) for k in shifts]
    return float(shifts[int(np.argmin(cost))] * dt)


def band_std(x, dt, band):
    """Standard deviation of the part of x inside a frequency band [rad/s].

    The slow drift is removed first (x minus its 30 s moving average), so it
    cannot leak into the band; the variance is then the Welch spectrum
    integrated over the band."""
    x = np.asarray(x, float)
    fast = x - moving_average(x, dt)
    f, S = welch(fast, fs=1.0 / dt, nperseg=WELCH_SEGMENT)   # f [Hz], S [unit^2/Hz]
    w = 2.0 * np.pi * f                                     # frequency [rad/s]
    inside = (w >= band[0]) & (w <= band[1])
    return float(np.sqrt(trapezoid(S[inside], f[inside])))


# -- RUNS -- #

def run_observer(kind, noise):
    """Part A: fixed wrench, controller out of the loop, waves on."""
    cfg = Part2SimConfig(dt=DT, T=T_RUN, use_reference=False, use_observer=True,
                         observer_type=kind, use_sensor_noise=noise)
    return run(cfg, np.zeros(6), environment(), controller=WrenchSequence([(1.0, FIXED_WRENCH)]))


def run_closed_loop(use_observer, noise):
    """Part B: hold the origin, raw feedback or the selected observer."""
    cfg = Part2SimConfig(dt=DT, T=T_RUN, use_observer=use_observer,
                         observer_type=selected_observer(), use_sensor_noise=noise)
    return run(cfg, np.zeros(6), environment())


# -- METRICS -- #

def observer_metrics(logs):
    """Estimation quality over the evaluation window (part A)."""
    dt = float(logs.t[1] - logs.t[0])
    sel = logs.t >= EVAL_FROM
    true, est = pose(logs.eta), pose(logs.eta_est)
    true_lf = moving_average(true, dt)
    err = est - true_lf                                   # estimate minus LF truth
    nu_est = logs.nu_est[:, [0, 1, 5]]
    nu_err = nu_est - moving_average(logs.nu[:, [0, 1, 5]], dt)
    # Wave motion left: wave-band content of the estimate relative to the true motion
    left = [100 * band_std(est[sel, i], dt, WAVE_BAND) / band_std(true[sel, i], dt, WAVE_BAND)
            for i in range(3)]
    # Estimate noise: content of the estimate in the noise band
    noise = [band_std(est[sel, i], dt, NOISE_BAND) for i in range(3)]
    noise_nu = [band_std(nu_est[sel, i], dt, NOISE_BAND) for i in range(3)]
    m = {
        "LF position error RMS [m]": rms(np.hypot(err[sel, 0], err[sel, 1])),
        "LF heading error RMS [deg]": np.rad2deg(rms(err[sel, 2])),
        "wave motion left N [%]": left[0],
        "wave motion left E [%]": left[1],
        "wave motion left heading [%]": left[2],
        "estimate noise N, E [mm]": 1e3 * np.hypot(noise[0], noise[1]),
        "estimate noise heading [mdeg]": 1e3 * np.rad2deg(noise[2]),
        "u, v error RMS [m/s]": rms(np.hypot(nu_err[sel, 0], nu_err[sel, 1])),
        "r error RMS [deg/s]": np.rad2deg(rms(nu_err[sel, 2])),
        "estimate noise u, v [mm/s]": 1e3 * np.hypot(noise_nu[0], noise_nu[1]),
        "estimate noise r [mdeg/s]": 1e3 * np.rad2deg(noise_nu[2]),
        "position lag [s]": lag(est[sel, :2], true_lf[sel, :2], dt),
        "heading lag [s]": lag(est[sel, 2], true_lf[sel, 2], dt),
    }
    # Mean normalised innovation squared (EKF only): about 3 when R matches the noise
    m["mean NIS [-]"] = float(np.nanmean(logs.nis[sel])) if logs.nis is not None else float("nan")
    return m


def closed_loop_metrics(logs, thrusters):
    """Station keeping and actuator activity over the evaluation window (part B)."""
    dt = float(logs.t[1] - logs.t[0])
    sel = logs.t >= EVAL_FROM
    u_max = np.array([th.u_max for th in thrusters])
    u_rate = np.array([th.u_rate for th in thrusters])
    pos = np.hypot(logs.eta[sel, 0] - logs.sp[sel, 0], logs.eta[sel, 1] - logs.sp[sel, 1])
    psi = np.rad2deg(np.abs(wrap(logs.eta[sel, 5] - logs.sp[sel, 5])))
    lf_pos, lf_psi = max_deviation(logs, EVAL_FROM, deviation="lf")
    # Thrust and azimuth rates between time steps
    du = np.abs(np.diff(logs.u, axis=0))[sel[1:]] / dt
    da = np.rad2deg(np.abs(wrap(np.diff(logs.alpha, axis=0))))[sel[1:]] / dt
    m = {
        "max position error [m]": pos.max(),
        "RMS position error [m]": rms(pos),
        "max LF position error [m]": lf_pos,
        "max heading error [deg]": psi.max(),
        "RMS heading error [deg]": rms(psi),
        "max LF heading error [deg]": np.rad2deg(lf_psi),
        "mean thrust utilisation [%]": average_utilization(logs, thrusters, EVAL_FROM),
        "time with a thrust command at max [%]": 100 * np.mean(np.any(np.abs(logs.u_cmd[sel]) >= 0.99 * u_max, axis=1)),
        "time at ramp-rate limit [%]": 100 * np.mean(np.any(du >= 0.99 * u_rate, axis=1)),
    }
    for i, th in enumerate(thrusters):
        m[f"mean thrust rate {th.name} [kN/s]"] = du[:, i].mean() / 1e3
    for i, th in enumerate(thrusters):
        if th.kind == "azimuth":                         # azimuths only (tunnel is fixed)
            m[f"mean azimuth rate {th.name} [deg/s]"] = da[:, i].mean()
    return m


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in rows:
            writer.writerow([round(v, 4) if isinstance(v, float) else v for v in r.values()])


def print_table(title, rows):
    print(f"\n{title}")
    cases = [r["case"] for r in rows]
    print(f"{'':36s}" + "".join(f"{c:>24s}" for c in cases))
    for key in rows[0]:
        if key == "case":
            continue
        print(f"{key:36s}" + "".join(f"{r.get(key, float('nan')):24.3f}" for r in rows))


# -- FIGURES -- #

def plot_estimates(logs_on):
    """Measured, true, LF true and estimated North, heading and sway velocity, noise on."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(logs_on), figsize=(11, 7.5), sharex=True)
    for col, (kind, logs) in enumerate(logs_on.items()):
        dt = float(logs.t[1] - logs.t[0])
        sel = (logs.t >= ZOOM[0]) & (logs.t <= ZOOM[1])
        meas, true, est = pose(logs.eta_measured), pose(logs.eta), pose(logs.eta_est)
        true_lf = moving_average(true, dt)
        for row, (i, label, scale) in enumerate(((0, "North [m]", 1.0), (2, "Heading [deg]", 180 / np.pi))):
            ax = axes[row, col]
            ax.plot(logs.t[sel], scale * meas[sel, i], ".", ms=2, color="0.6", label="measured (noisy)")
            ax.plot(logs.t[sel], scale * true[sel, i], lw=0.8, label="true")
            ax.plot(logs.t[sel], scale * true_lf[sel, i], "k--", lw=1.0, label="LF true (30 s mean)")
            ax.plot(logs.t[sel], scale * est[sel, i], lw=1.8, label="estimate")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        # Sway velocity: the observers get no velocity measurement, only the estimate exists
        ax = axes[2, col]
        ax.plot(logs.t[sel], logs.nu[sel, 1], lw=0.8, label="true")
        ax.plot(logs.t[sel], moving_average(logs.nu[:, 1], dt)[sel], "k--", lw=1.0, label="LF true (30 s mean)")
        ax.plot(logs.t[sel], logs.nu_est[sel, 1], lw=1.8, label="estimate")
        ax.set_ylabel("Sway velocity v [m/s]")
        ax.grid(True, alpha=0.3)
        axes[0, col].set_title(f"{NAMES[kind]}, waves + sensor noise")
        axes[2, col].set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_thrust(logs_on, thrusters):
    """Actual thrust per thruster with raw feedback and with the observer, noise on."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(logs_on), 1, figsize=(9, 5.5), sharex=True, sharey=True)
    for ax, (label, logs) in zip(axes, logs_on.items()):
        sel = (logs.t >= ZOOM_THRUST[0]) & (logs.t <= ZOOM_THRUST[1])
        for i, th in enumerate(thrusters):
            ax.plot(logs.t[sel], logs.u[sel, i] / 1e3, lw=0.9, label=th.name)
        ax.axhline(thrusters[1].u_max / 1e3, color="k", ls=":", lw=0.8)
        ax.axhline(-thrusters[1].u_max / 1e3, color="k", ls=":", lw=0.8)
        ax.set_title(f"{label}, waves + sensor noise")
        ax.set_ylabel("Thrust [kN]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time [s]")
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

    # Part A: both observers, noise off and on
    rows_a, est_on = [], {}
    for kind in ("nonlinear_passive", "kalman"):
        for noise in (False, True):
            logs = run_observer(kind, noise)
            rows_a.append({"case": f"{NAMES[kind]} noise {'on' if noise else 'off'}",
                           **observer_metrics(logs)})
            if noise:
                est_on[kind] = logs
    write_csv(OUT_DIR / "sim8_observer_metrics.csv", rows_a)
    print_table("A) Observer comparison, fixed wrench, waves on", rows_a)

    # Part B: raw feedback and the selected observer, noise off and on
    rows_b, thrust_on = [], {}
    obs_name = NAMES[selected_observer()]
    for label, use_obs in (("raw feedback", False), (obs_name, True)):
        for noise in (False, True):
            logs = run_closed_loop(use_obs, noise)
            rows_b.append({"case": f"{label} noise {'on' if noise else 'off'}",
                           **closed_loop_metrics(logs, thrusters)})
            if noise:
                thrust_on[label] = logs
    write_csv(OUT_DIR / "sim8_closed_loop_metrics.csv", rows_b)
    print_table("B) Closed-loop station keeping", rows_b)

    for name, fig in (("fig_sim8_estimates", plot_estimates(est_on)),
                      ("fig_sim8_thrust", plot_thrust(thrust_on, thrusters))):
        fig.savefig(OUT_DIR / f"{name}.pdf")
        fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    print(f"\nsaved sim8_observer_metrics.csv, sim8_closed_loop_metrics.csv and figures to {OUT_DIR}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
