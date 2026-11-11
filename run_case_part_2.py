"""
Part 2 scenario runner — the file you edit and run.

    python run_case_part_2.py                 # runs SCENARIO below
    python run_case_part_2.py sim3            # runs one named scenario
    python run_case_part_2.py --list          # lists the scenarios

Every mandatory simulation of the Part 2 description has a preset here
(``sim1`` … ``sim7``), plus ``bonus`` for the optional extra-credit sensor-noise
task; all regular simulations run with sensor noise OFF.  Each preset builds
the configuration, the environment and the set-point sequence, runs
``DPSimulatorPart2`` and plots.  Edit the
presets freely: the environmental numbers are collected at the top, the
per-simulation switches live in each function.  All Part 2 blocks are
imported from ``part_2/`` — the checks (``python check.py --part 2``) build
them with the constructor defaults documented in those templates, so put
your final tuning there, not only in this file.

``SELECTED_OBSERVER`` starts as ``None``: compare both observers in Part 2,
Simulation 4, then set it to the one you chose and justify the choice in the
report.  Part 2, Simulations 5-7 and the raw-vs-observer comparison of
Part 2, Simulation 4 use it explicitly, and stop with a message until you
have set it.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from part_2.config import Part2SimConfig, default_thrusters_part_2
from part_2.controller import DPController
from part_2.current import Current
from part_2.observer import select_observer
from simulation.plotter_part_2 import (plot_nis, plot_observer, plot_observer_error,
                                       plot_thruster_setpoints, plot_waves,
                                       plot_wind_field, plot_wrench_residual, plot_wave_rejection)
from part_2.reference import ReferenceModel
from part_2.thrust_allocation import ThrustAllocator
from models.waves import Waves
from part_2.wind import Wind
from simulation.capability import capability_sweep, plot_capability
from simulation.plotter import (plot_current, plot_dashboard, plot_time_histories,
                                plot_wrench, plot_xy)
from simulation.simulation_part_2 import DPSimulatorPart2

# Some presets deliberately open more figures than matplotlib's default
# 20-figure warning threshold: Part 2, Simulation 4 alone plots both observers
# in four conditions.  Raise the threshold so a harmless warning does not read
# as a fault.
plt.rcParams["figure.max_open_warning"] = 60

# ---------------------------------------------------------------------------
# Environmental conditions prescribed for the supplied Gunnerus vessel model
# and its three-thruster configuration.
# ---------------------------------------------------------------------------
DT = 0.1
CURRENT_SPEED, CURRENT_FROM = 0.2, np.deg2rad(90.0)       # 0.2 m/s from east
WIND_SPEED, WIND_FROM = 10.0, np.deg2rad(0.0)             # 10 m/s from north
WAVES_HS, WAVES_TP, WAVES_FROM = 1.5, 8.0, np.deg2rad(45.0)   # design sea, from north-east
WAVES_N = 100                                             # wave frequency components (range 0.25-3 wp, default), see plot_wave_model.py
ROBUST_HS, ROBUST_TP = 2.5, 10.0                          # Simulation 6 (heavy sea for a vessel of this size, Lpp 34 m)
CAP_WIND, CAP_CURRENT, CAP_HS, CAP_TP = 12.0, 0.2, 1.5, 8.0   # Simulation 5
CAP_POS_LIMIT, CAP_PSI_LIMIT_DEG = 5.0, 5.0               # DP watch circles (max LF excursion)
CAP_TOTAL_POS_LIMIT, CAP_TOTAL_PSI_LIMIT_DEG = 3.0, 3.0   # operability limits (max total motion)
SEED = 123

# Observer used by Part 2, Simulations 5-7 (and by the raw-vs-observer
# comparison in Part 2, Simulation 4); options "nonlinear_passive" or "kalman".
SELECTED_OBSERVER = "kalman"

# Initial observer error used by the convergence demonstration in Part 2,
# Simulation 4.  The engine resets every observer at the TRUE initial pose, so
# an estimate that starts on top of the truth has nothing to converge from.
# The value below is an EXAMPLE (5 m north, 5 m west, 10 deg of heading):
# choose your own, and say in the report what it is and why it is a fair test.
OBSERVER_OFFSET = np.array([5.0, -5.0, 0.0, 0.0, 0.0, np.deg2rad(10.0)])

SCENARIO = "sim2"          # default scenario when run without arguments


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def selected_observer() -> str:
    """The observer you chose in Part 2, Simulation 4 (``SELECTED_OBSERVER``).

    Raises ``NotImplementedError`` while the choice is still unmade, so a
    preset never silently runs with an observer you did not pick.
    """
    if SELECTED_OBSERVER is None:
        raise NotImplementedError(
            "set SELECTED_OBSERVER at the top of run_case_part_2.py to the "
            "observer you selected in Part 2, Simulation 4 — "
            "\"nonlinear_passive\" or \"kalman\" — and justify the choice in "
            "the report; run Part 2, Simulation 4 first if you have not yet")
    return SELECTED_OBSERVER


def environment(*, current=True, wind=True, waves=True, gust=True,
                hs=WAVES_HS, tp=WAVES_TP, seed=SEED):
    """Standard Part 2 environment; switch components off with False."""
    cur = Current(CURRENT_SPEED, CURRENT_FROM, semantics="from") if current else Current()
    wnd = (Wind(WIND_SPEED, WIND_FROM, semantics="from", sigma_slow=0.5, gust=gust,
                sigma_dir=np.deg2rad(2.0), dir_limit=np.deg2rad(5.0), seed=seed)
           if wind else Wind())
    wav = Waves(hs=hs, tp=tp, direction=WAVES_FROM, n_components=WAVES_N, seed=seed) if waves else None
    return cur, wnd, wav


def four_corner_commands(cfg: Part2SimConfig, hold: float):
    """Set-point sequence of the four-corner test, ``hold`` seconds per leg."""
    corners = [[50.0, 0.0, 0.0], [50.0, -50.0, 0.0], [50.0, -50.0, -np.pi / 4],
               [0.0, -50.0, -np.pi / 4], [0.0, 0.0, 0.0]]
    n_steps = int(round(cfg.T / cfg.dt)) + 1
    per_leg = int(round(hold / cfg.dt))
    eta_cmd = np.zeros((n_steps, 6))
    for i, c in enumerate(corners):
        rows = slice(i * per_leg, n_steps if i == len(corners) - 1 else (i + 1) * per_leg)
        eta_cmd[rows, 0], eta_cmd[rows, 1], eta_cmd[rows, 5] = c
    return eta_cmd


def make_sim(cfg: Part2SimConfig, controller=None):
    thrusters = default_thrusters_part_2()
    return DPSimulatorPart2(
        cfg, controller or DPController(), thrusters,
        reference=ReferenceModel(dt=cfg.dt),
        allocator=ThrustAllocator(thrusters),
        observer=select_observer(cfg.observer_type, **cfg.observer_kwargs),
    )


class WrenchSequence:
    """Stand-in controller: a piecewise-constant desired wrench.

    ``segments`` is a list of ``(duration_s, [Fx, Fy, Mz])``; the last segment
    holds to the end of the run. Used by Simulation 4 for its fixed-wrench
    observer test.
    """

    def __init__(self, segments):
        self.segments = [(float(d), np.asarray(w, dtype=float).reshape(3)) for d, w in segments]

    def compute(self, t, dt, eta, nu, eta_ref, nu_ref, acc_ref):
        t_seg = 0.0
        wrench = self.segments[-1][1]
        for duration, w in self.segments:
            if t < t_seg + duration:
                wrench = w
                break
            t_seg += duration
        tau = np.zeros(6)
        tau[0], tau[1], tau[5] = wrench
        return tau


def run(cfg, eta_cmd, env, controller=None):
    current, wind, waves = env
    sim = make_sim(cfg, controller)
    sim.reset_state()
    return sim.run(eta_cmd, current=current, wind=wind, waves=waves)


def standard_plots(logs, *, setpoints=False, observer=False, environment_plots=False):
    plot_dashboard(logs)
    plot_time_histories(logs)
    plot_xy(logs)
    if setpoints:
        plot_wrench(logs)
        plot_wrench_residual(logs)
        plot_thruster_setpoints(logs)
    if observer:
        plot_observer(logs)
        plot_observer_error(logs)
    if environment_plots:
        plot_wind_field(logs)
        plot_current(logs)
        plot_waves(logs)


# ---------------------------------------------------------------------------
# Mandatory simulations
# ---------------------------------------------------------------------------

def sim1():
    """Part 2, Simulation 1 — environmental loads: free drift, DP off, 300 s; plots
    the wind field (total, mean+slow and gust speed, direction), current and
    wave loads you built."""
    cfg = Part2SimConfig(dt=DT, T=300.0, use_controller=False, use_reference=False)
    logs = run(cfg, np.zeros(6), environment())
    plot_time_histories(logs)
    plot_xy(logs)
    plot_wind_field(logs)
    plot_current(logs)
    plot_waves(logs)
    return logs


def sim2():
    """Part 2, Simulation 2 — four-corner DP test with constrained Part 2
    thruster dynamics enabled, and with environment and observer disabled."""
    hold = 300.0
    cfg = Part2SimConfig(dt=DT, T=5 * hold, use_reference=True,
                         use_observer=False, thruster_dynamics=True)
    logs = run(cfg, four_corner_commands(cfg, hold),
               environment(current=False, wind=False, waves=False))
    standard_plots(logs, setpoints=True)
    return logs


def sim3():
    """Part 2, Simulation 3 — the prescribed four-corner DP test in the Part 2,
    Simulation 1 environment, with raw measurements fed back (no observer).

    This deliberately keeps the reference model and constrained thrusters on:
    the purpose is to validate the integrated DP system under current, detailed
    wind and waves before observers are introduced in Simulation 4.
    """
    hold = 300.0
    cfg = Part2SimConfig(dt=DT, T=5 * hold, use_observer=False)
    logs = run(cfg, four_corner_commands(cfg, hold), environment())
    standard_plots(logs, setpoints=True, environment_plots=True)
    return logs


def sim4():
    """Part 2, Simulation 4 — observer selection, in three parts: (a) fixed desired
    wrench [2 2 2]e3 in the Part 2, Simulation 1 environment, with and without waves,
    for both observers; (b) the same run started with OBSERVER_OFFSET so the estimate
    has something to converge from; (c) closed-loop station keeping with raw
    measurements vs. SELECTED_OBSERVER."""
    out = {}
    fixed = WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
    variants = (("with waves", dict(waves=True)), ("without waves", dict(waves=False)))
    for kind in ("nonlinear_passive", "kalman"):
        for label, env_kw in variants:
            cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True,
                                 observer_type=kind) # , bypass_actuators=True)
            logs = run(cfg, np.zeros(6), environment(**env_kw), controller=fixed)
            plot_observer(logs).suptitle(f"Observer {kind}, {label}")
            plot_observer_error(logs).suptitle(f"Observer {kind}, {label} — estimation error")
            # Add wave-rejection plot for "with waves" scenarios to show filtering capability
            if env_kw.get("waves", False):
                plot_wave_rejection(logs).suptitle(f"Observer {kind}, {label} — wave rejection")
            # NIS plot for observer diagnostics (all scenarios)
            if kind == "kalman":
                plot_nis(logs).suptitle(f"Observer {kind}, {label} — NIS diagnostic")
            out[(kind, label)] = logs
    # Convergence: start the estimate away from the truth (the engine resets it
    # ON the truth), then watch the error decay.  Quote the convergence time.
    for kind in ("nonlinear_passive", "kalman"):
        cfg = Part2SimConfig(dt=DT, T=300.0, use_reference=False, use_observer=True,
                             observer_type=kind) # , bypass_actuators=True)
        sim = make_sim(cfg, fixed)
        sim.reset_state()
        sim.observer.reset(OBSERVER_OFFSET)
        current, wind, waves = environment()
        logs = sim.run(np.zeros(6), current=current, wind=wind, waves=waves)
        plot_observer(logs).suptitle(f"Observer {kind}, offset start — convergence")
        plot_observer_error(logs).suptitle(f"Observer {kind}, offset start — error convergence")
        if kind == "kalman":
            plot_nis(logs).suptitle(f"Observer {kind}, offset start — NIS diagnostic")
        out[(kind, "offset start")] = logs
    # Stage (c) is the only part that needs the choice you make from the
    # evidence above, so it is skipped until you have made it.  Stages (a) and
    # (b) are returned either way: they are the measurements you select on.
    if SELECTED_OBSERVER is None:
        print("\nPart 2, Simulation 4: stages (a) and (b) are done and plotted, and "
              "their logs are returned.\nCompare the observers on that evidence, set "
              "SELECTED_OBSERVER at the top of run_case_part_2.py,\nthen run sim4 again "
              "for stage (c), the raw-vs-observer closed loop."
              "\n(In a notebook, restart the kernel or importlib.reload(rc) first, so the "
              "new value is picked up.)")
        return out
    # Closed loop: does the selected observer improve the DP system?
    for label, use_obs in (("raw measurements", False), (f"observer {selected_observer()}", True)):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=use_obs,
                             observer_type=selected_observer())
        logs = run(cfg, np.zeros(6), environment())
        standard_plots(logs, setpoints=True, observer=use_obs)
        plt.gcf().suptitle(f"Closed loop — {label}")
        if use_obs:
            plot_observer_error(logs).suptitle(f"Closed loop — {label} (estimation error)")
        out[("closed loop", label)] = logs
    return out


def sim5(step_deg: float = 10.0):
    """Part 2, Simulation 5 — average thrust-utilisation polar plot: co-linear
    wind/waves/current swept around the vessel, DP at the origin, heading 0,
    SELECTED_OBSERVER in the loop.

    Three polar plots: the utilisation for all directions; the directions
    whose maximum low-frequency excursion stays inside the DP watch circles
    (CAP_POS_LIMIT / CAP_PSI_LIMIT_DEG) — the motion a DP system checks on
    its screen; and the directions whose maximum total motion (with the
    wave-frequency oscillation) stays inside the operability limits
    (CAP_TOTAL_POS_LIMIT / CAP_TOTAL_PSI_LIMIT_DEG) — what a motion-critical
    operation such as a gangway transfer experiences."""
    thrusters = default_thrusters_part_2()

    def run_direction(direction):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=True, observer_type=selected_observer())
        env = (Current(CAP_CURRENT, direction, semantics="from"),
               Wind(CAP_WIND, direction, semantics="from", sigma_slow=0.5, gust=True,
                    sigma_dir=np.deg2rad(2.0), seed=SEED),
               Waves(hs=CAP_HS, tp=CAP_TP, direction=direction, n_components=WAVES_N, seed=SEED))
        return run(cfg, np.zeros(6), env)

    result = capability_sweep(run_direction, thrusters, step_deg=step_deg,
                              pos_limit=CAP_POS_LIMIT, psi_limit_deg=CAP_PSI_LIMIT_DEG,
                              deviation="lf")
    plot_capability(result, total_limits=(CAP_TOTAL_POS_LIMIT, CAP_TOTAL_PSI_LIMIT_DEG))
    return result


def station_keeping_metrics(logs, thrusters, start=300.0, end=600.0):
    """Station-keeping, LF estimation and actuator metrics on a bounded interval."""
    from simulation.plotter_part_2 import _moving_average
    from scipy.integrate import trapezoid
    from scipy.signal import welch

    dt = float(logs.t[1] - logs.t[0])
    selected = (logs.t >= start) & (logs.t <= end)
    rate_window = (logs.t[:-1] >= start) & (logs.t[1:] <= end)
    wrap = lambda angle: np.arctan2(np.sin(angle), np.cos(angle))
    rms = lambda values: float(np.sqrt(np.mean(np.square(values))))
    position = np.linalg.norm(logs.eta[:, :2] - logs.sp[:, :2], axis=1)
    heading = np.rad2deg(wrap(logs.eta[:, 5] - logs.sp[:, 5]))
    truth = logs.eta[:, [0, 1, 5]].copy()
    truth[:, 2] = np.unwrap(truth[:, 2])
    truth_lf = np.column_stack([
        _moving_average(truth[:, index], dt) for index in range(3)
    ])
    estimation_error = logs.eta_est[:, [0, 1, 5]] - truth_lf
    estimation_error[:, 2] = wrap(estimation_error[:, 2])
    lf_position = np.linalg.norm(truth_lf[:, :2] - logs.sp[:, :2], axis=1)
    wave_band = (0.5 * 2.0 * np.pi / WAVES_TP, 2.0 * 2.0 * np.pi / WAVES_TP)

    def horizontal_wave_power(pose):
        horizontal = pose[selected, :2]
        fast = horizontal - np.column_stack([
            _moving_average(horizontal[:, index], dt) for index in range(2)
        ])
        frequency, spectrum = welch(fast, fs=1.0 / dt,
                                    nperseg=min(1024, len(fast)), axis=0)
        angular_frequency = 2.0 * np.pi * frequency
        band = (angular_frequency >= wave_band[0]) & (angular_frequency <= wave_band[1])
        return float(np.sum(trapezoid(spectrum[band], frequency[band], axis=0)))

    true_wave_power = horizontal_wave_power(logs.eta)
    estimated_wave_power = horizontal_wave_power(logs.eta_est)
    wave_ratio_db = (float(10.0 * np.log10(estimated_wave_power / true_wave_power))
                     if true_wave_power > 0 and estimated_wave_power > 0 else float("nan"))
    velocity_lf = np.column_stack([
        _moving_average(logs.nu[:, index], dt) for index in (0, 1, 5)
    ])
    velocity_error = logs.nu_est[:, [0, 1, 5]] - velocity_lf
    thrust_limits = np.array([thruster.u_max for thruster in thrusters])
    ramp_limits = np.array([thruster.u_rate for thruster in thrusters])
    azimuth_indices = [index for index, thruster in enumerate(thrusters)
                       if thruster.kind == "azimuth"]
    slew_limits = np.array([thrusters[index].rot_speed for index in azimuth_indices])
    thrust_rates = np.abs(np.diff(logs.u, axis=0))[rate_window] / dt
    azimuth_rates = np.abs(wrap(np.diff(logs.alpha, axis=0)))[rate_window] / dt
    ramp_active = np.any(thrust_rates >= 0.99 * ramp_limits, axis=1)
    slew_active = np.any(azimuth_rates[:, azimuth_indices] >= 0.99 * slew_limits, axis=1)
    residual = (logs.tau_d - logs.tau_thr)[selected][:, [0, 1, 5]]
    metrics = {
        "Position RMS [m]": rms(position[selected]),
        "True LF position deviation RMS [m]": rms(lf_position[selected]),
        "Maximum horizontal deviation [m]": float(position[selected].max()),
        "Heading RMS [deg]": rms(heading[selected]),
        "Maximum absolute heading error [deg]": float(np.abs(heading[selected]).max()),
        "LF position estimation RMS [m]": rms(np.linalg.norm(estimation_error[selected, :2], axis=1)),
        "LF heading estimation RMS [deg]": rms(np.rad2deg(estimation_error[selected, 2])),
        "LF surge/sway estimation RMS [m/s]": rms(np.linalg.norm(velocity_error[selected, :2], axis=1)),
        "Horizontal WF estimate/true power ratio [dB]": wave_ratio_db,
        "Mean thrust utilisation [%]": float(np.mean(100 * np.sum(np.abs(logs.u[selected]), axis=1) / thrust_limits.sum())),
        "Mean absolute thrust rate [kN/s]": float(thrust_rates.mean() / 1000),
        "Time at thrust-amplitude limit [%]": float(100 * np.mean(np.any(np.abs(logs.u[selected]) >= 0.99 * thrust_limits, axis=1))),
        "Time at thrust-ramp limit [%]": float(100 * ramp_active.mean()),
        "Time at azimuth-slew limit [%]": float(100 * slew_active.mean()),
        "Time at either rate limit [%]": float(100 * np.mean(ramp_active | slew_active)),
        "Peak absolute East deviation [m]": float(np.abs(logs.eta[selected, 1]).max()),
        "Time of peak absolute East deviation [s]": float(logs.t[selected][np.argmax(np.abs(logs.eta[selected, 1]))]),
        "Peak surge wrench residual [kN]": float(np.abs(residual[:, 0]).max() / 1000),
        "RMS surge wrench residual [kN]": rms(residual[:, 0]) / 1000,
    }
    for index, thruster in enumerate(thrusters):
        metrics[f"Peak actual thrust {thruster.name} [kN]"] = float(np.abs(logs.u[selected, index]).max() / 1000)
        metrics[f"Mean absolute thrust rate {thruster.name} [kN/s]"] = float(thrust_rates[:, index].mean() / 1000)
    return metrics


def sim6():
    """Part 2, Simulation 6 — heavy-sea station keeping for 1000 s; plots first 600 s and full run."""
    cfg = Part2SimConfig(dt=DT, T=1000.0, use_observer=True, observer_type=selected_observer())
    logs = run(cfg, np.zeros(6), environment(hs=ROBUST_HS, tp=ROBUST_TP))
    plot_count = int(round(600.0 / cfg.dt)) + 1
    plot_logs = replace(logs, **{
        name: value[:plot_count]
        for name, value in vars(logs).items()
        if isinstance(value, np.ndarray) and value.ndim > 0
    })
    figures_before_comparison = set(plt.get_fignums())
    standard_plots(plot_logs, observer=True)
    plot_wrench_residual(plot_logs)
    thruster_figure = plot_thruster_setpoints(plot_logs)
    thruster_figure.suptitle("First 600 s — Thruster set-points vs. actual")
    thruster_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figures_before_full_run = set(plt.get_fignums())
    standard_plots(logs, observer=True)
    plot_wrench_residual(logs)
    plot_thruster_setpoints(logs)
    for figure_number in sorted(set(plt.get_fignums()) - figures_before_comparison):
        figure = plt.figure(figure_number)
        full_run = figure_number not in figures_before_full_run
        displayed_logs = logs if full_run else plot_logs
        for axis in figure.axes:
            if any(np.array_equal(line.get_xdata(), displayed_logs.t) for line in axis.lines):
                axis.axvline(300.0, color="0.35", linestyle="--", linewidth=0.8)
                axis.axvline(600.0, color="0.35", linestyle="--", linewidth=0.8)
        suptitle = getattr(figure, "_suptitle", None)
        existing_title = suptitle.get_text() if suptitle is not None else ""
        period = "Full 1000 s run" if full_run else "First 600 s"
        figure.suptitle(f"Sim 6: {period} | Evaluation: 300-600 s\n{existing_title}")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    print("Simulation 6: 1000 s run; comparison figures show 0-600 s.")
    print("Matched evaluation window: 300-600 s; full-run figures are shown separately.")
    print("Running the matched 600 s Sim 4 design-sea baseline for the metrics table...")
    baseline_logs = run(replace(cfg, T=600.0), np.zeros(6), environment())
    thrusters = default_thrusters_part_2()
    baseline_metrics = station_keeping_metrics(baseline_logs, thrusters)
    heavy_metrics = station_keeping_metrics(logs, thrusters)
    print(f"Observer: {selected_observer()}; dt={cfg.dt} s; seed={SEED}; wave components={WAVES_N}")
    print(f"Design sea: Hs={WAVES_HS} m, Tp={WAVES_TP} s; heavy sea: Hs={ROBUST_HS} m, Tp={ROBUST_TP} s")
    print("Metrics: 300 <= t <= 600 s; LF truth: centred 30 s mean of each complete run.")
    print(f"WF band: {0.5 * 2 * np.pi / WAVES_TP:.6f}-{2 * 2 * np.pi / WAVES_TP:.6f} rad/s, fixed for both seas.")
    print("WF ratio: 10*log10(sum estimated N/E band power / sum true N/E band power).")
    print("Spectra: subtract 30 s mean within the selected interval, Welch with 1024-sample segments.")
    print("True LF position deviation is measured against the setpoint, not the estimate.")
    print("Limit activity: at least one thruster reaches 99% of its configured limit.")
    print("Mean thrust rate: mean absolute rate across time and all three thrusters.")
    print(f"{'Metric':<58} {'Sim 4':>12} {'Sim 6':>12}")
    for name, value in heavy_metrics.items():
        print(f"{name:<58} {baseline_metrics[name]:12.3f} {value:12.3f}")
    return logs


def sim7():
    """Part 2, Simulation 7 — hold the origin through a 45-degree heading maneuver."""
    cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=True,
                         observer_type=selected_observer(), thruster_dynamics=True)
    n_steps = int(round(cfg.T / cfg.dt)) + 1
    eta_cmd = np.zeros((n_steps, 6))
    turn_start = int(round(120.0 / cfg.dt))
    turn_back = int(round(360.0 / cfg.dt))
    eta_cmd[turn_start:turn_back, 5] = np.deg2rad(45.0)
    logs = run(cfg, eta_cmd, environment())
    baseline_logs = run(cfg, np.zeros_like(eta_cmd), environment())
    standard_plots(logs, setpoints=True, observer=True)

    def position_deviation(logs):
        total = np.hypot(logs.eta[:, 0], logs.eta[:, 1])
        north = np.convolve(
            np.pad(logs.eta[:, 0], pad, mode="edge"), window_weights, mode="valid"
        )
        east = np.convolve(
            np.pad(logs.eta[:, 1], pad, mode="edge"), window_weights, mode="valid"
        )
        low_frequency = np.hypot(north, east)
        return total, low_frequency

    def heading_error_degrees(logs):
        error = np.arctan2(
            np.sin(logs.eta[:, 5] - logs.sp[:, 5]),
            np.cos(logs.eta[:, 5] - logs.sp[:, 5]),
        )
        return np.rad2deg(error)

    heading_error = heading_error_degrees(logs)
    baseline_heading_error = heading_error_degrees(baseline_logs)
    window_samples = int(round(30.0 / cfg.dt))
    if window_samples % 2 == 0:
        window_samples += 1
    window_weights = np.full(window_samples, 1.0 / window_samples)
    pad = window_samples // 2
    position_error, position_error_lf = position_deviation(logs)
    baseline_position_error, baseline_position_error_lf = position_deviation(baseline_logs)

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(logs.t, position_error, color="tab:blue", alpha=0.25,
              label="45 deg turn: total deviation")
    axis.plot(logs.t, position_error_lf, color="tab:blue",
              label="45 deg turn: 30 s low-frequency deviation")
    axis.plot(baseline_logs.t, baseline_position_error, color="tab:orange", alpha=0.25,
              label="No-turn baseline: total deviation")
    axis.plot(baseline_logs.t, baseline_position_error_lf, "--", color="tab:orange",
              label="No-turn baseline: 30 s low-frequency deviation")
    axis.axvspan(120.0, 270.0, color="tab:orange", alpha=0.15, label="0 to 45 deg turn")
    axis.axvspan(360.0, 480.0, color="tab:green", alpha=0.12, label="45 to 0 deg turn")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Position deviation from origin [m]")
    axis.set_title("Simulation 7 — heading maneuver vs. no-turn baseline")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()

    print("Simulation 7: 45-degree heading maneuver vs. no-turn baseline")
    print(f"  max position excursion: {np.max(position_error):.2f} m")
    print(f"  RMS position excursion: {np.sqrt(np.mean(position_error**2)):.2f} m")
    print(
        "  max heading error to filtered reference (turn / baseline): "
        f"{np.max(np.abs(heading_error)):.2f} / "
        f"{np.max(np.abs(baseline_heading_error)):.2f} deg"
    )
    print("  paired interval comparison (turn case vs. no-turn baseline):")
    intervals = (
        ("initial hold", 30.0, 105.0),
        ("0 to 45 deg turn", 135.0, 255.0),
        ("45 deg hold", 285.0, 345.0),
        ("45 to 0 deg turn", 375.0, 465.0),
        ("final hold", 495.0, 585.0),
    )
    for label, start, end in intervals:
        interval = (logs.t >= start) & (logs.t < end)
        turn_rms = np.sqrt(np.mean(position_error_lf[interval]**2))
        baseline_rms = np.sqrt(np.mean(baseline_position_error_lf[interval]**2))
        turn_peak = np.max(position_error[interval])
        baseline_peak = np.max(baseline_position_error[interval])
        turn_heading_rms = np.sqrt(np.mean(heading_error[interval]**2))
        baseline_heading_rms = np.sqrt(np.mean(baseline_heading_error[interval]**2))
        print(
            f"    {label}: LF RMS {turn_rms:.2f} / {baseline_rms:.2f} m "
            f"(delta {turn_rms - baseline_rms:+.2f}); total peak "
            f"{turn_peak:.2f} / {baseline_peak:.2f} m; heading RMS "
            f"{turn_heading_rms:.2f} / {baseline_heading_rms:.2f} deg"
        )
    return logs


def sim9():
    """Part 2, Simulation 9 — Sim 3's four-corner test with the selected observer."""
    hold = 300.0
    cfg = Part2SimConfig(dt=DT, T=5 * hold, use_observer=True,
                         observer_type=selected_observer())
    logs = run(cfg, four_corner_commands(cfg, hold), environment())
    standard_plots(logs, setpoints=True, environment_plots=True, observer=True)
    return logs


def bonus_sensor_noise():
    """Extra credit (3 points) — sensor noise: repeat the Part 2, Simulation 4
    observer comparison and the closed-loop raw-vs-observer run with
    measurement noise ON (fixed course noise levels, models/sensors.py).  All
    regular simulations run with noise off."""
    out = {}
    fixed = WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
    for kind in ("nonlinear_passive", "kalman"):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True,
                             observer_type=kind, use_sensor_noise=True)
        logs = run(cfg, np.zeros(6), environment(), controller=fixed)
        plot_observer(logs).suptitle(f"Bonus: observer {kind}, waves + sensor noise")
        # Keep NIS plot only for Kalman (diagnostic)
        if kind == "kalman":
            plot_nis(logs).suptitle("Bonus: Kalman, waves + sensor noise — NIS diagnostic")
        out[kind] = logs
    for label, use_obs in (("raw noisy measurements", False),
                           (f"observer {selected_observer()}", True)):
        cfg = Part2SimConfig(dt=DT, T=600.0, use_observer=use_obs,
                             observer_type=selected_observer(), use_sensor_noise=True)
        logs = run(cfg, np.zeros(6), environment())
        standard_plots(logs, setpoints=True, observer=use_obs)
        plt.gcf().suptitle(f"Bonus closed loop, sensor noise on — {label}")
        out[("closed loop", label)] = logs
    return out


SCENARIOS = {"sim1": sim1, "sim2": sim2, "sim3": sim3, "sim4": sim4,
             "sim5": sim5, "sim6": sim6, "sim7": sim7, "sim9": sim9,
             "bonus": bonus_sensor_noise}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--list" in argv:
        for k, f in SCENARIOS.items():
            print(f"{k}: {f.__doc__.splitlines()[0]}")
        return
    name = argv[0] if argv else SCENARIO
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario {name!r}; try --list")
    try:
        SCENARIOS[name]()
    except NotImplementedError as exc:
        # A block you have not written yet, or a choice you have not made.
        # Stop with the message rather than a traceback, and still show
        # whatever the scenario managed to plot before it stopped.
        print(f"\n{name}: stopped — {exc}")
        if plt.get_fignums():
            print(f"showing the {len(plt.get_fignums())} figure(s) produced "
                  "before that point.")
            plt.show()
        return
    plt.show()
    print("Simulation finished.")


if __name__ == "__main__":
    main()
