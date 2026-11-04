"""Plot and verify the supplied irregular-wave model (models.waves.Waves).

    python plot_wave_model.py
    python plot_wave_model.py --hs 2.5 --tp 10 --direction 45 --psi 0 --n 100
    python plot_wave_model.py --convergence      # also run the N-convergence study
    python plot_wave_model.py --help

The wave-load generator itself is supplied and is not modified here.  This
script only configures it and produces the evidence asked for in Section 4.3:

1. the ITTC / modified Pierson-Moskowitz spectrum, the chosen frequency range
   and the discretised components (with the share of energy they cover),
2. time series of the first-order wave-frequency (WF) load and the
   second-order slowly varying drift load, plotted separately,
3. a direction check: mean drift force vs wave direction (vessel at psi),
4. optionally, a convergence study of the load statistics vs number of
   components N (several seeds).

The sea state, number of components and seed are imported from
run_case_part_2.py, so the figures and numbers describe exactly the waves used
in the Part 2 simulations.

Output (figures/ next to this script):
    fig_wave_spectrum, fig_wave_loads, fig_wave_direction (.pdf/.png)
    wave_verification.csv          all numbers quoted in the report
    wave_convergence_table.csv     Table A1 (with --convergence)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcsimpy.waves.wave_spectra import ModifiedPiersonMoskowitz
from models.waves import Waves
from run_case_part_2 import SEED as RUNNER_SEED
from run_case_part_2 import WAVES_FROM, WAVES_HS, WAVES_N, WAVES_TP

OUT_DIR = Path(__file__).resolve().parent / "figures"

DT = 0.1
T_END = 600.0

# Design sea and settings taken from run_case_part_2.py
HS = WAVES_HS                          # significant wave height [m]
TP = WAVES_TP                          # peak period [s]
DIRECTION = float(np.degrees(WAVES_FROM))  # waves FROM [deg]
N_COMPONENTS = WAVES_N                 # number of frequency components
SEED = RUNNER_SEED                     # random seed

PSI_SHIP = 0.0                 # vessel heading used in the time series [deg]

CONV_N = [20, 50, 100, 200]    # N values in the convergence study
CONV_SEEDS = [1, 2, 3]         # seeds averaged in the convergence study
CONV_T, CONV_DT = 1800.0, 0.5  # length and sampling of the convergence runs [s]


def make_waves(hs, tp, direction_deg, n, seed):
    """Build the supplied wave model with the default frequency range."""
    return Waves(hs=hs, tp=tp, direction=np.deg2rad(direction_deg),
                 n_components=n, seed=seed)


def split_loads(waves, t, psi):
    """First-order (WF) and second-order (drift) loads for a vessel at rest."""
    load = waves._load
    eta = np.zeros(6)
    eta[5] = psi
    # First-order wave-frequency load from the force RAOs
    tau_wf = np.array([load.first_order_loads(tk, eta) for tk in t])
    # Second-order slowly varying drift load from the QTFs
    tau_sv = np.array([load.second_order_loads(tk, psi) for tk in t])
    return tau_wf, tau_sv


def energy_share(hs, tp, w_lo, w_hi):
    """Fraction of the spectral energy m0 inside [w_lo, w_hi]."""
    w = np.linspace(0.01, 10.0, 40000)
    _, S = ModifiedPiersonMoskowitz(w)(hs, tp)
    band = (w >= w_lo) & (w <= w_hi)
    return np.trapezoid(S[band], w[band]) / np.trapezoid(S, w)


def direction_sweep(waves, psi, t_end=1200.0, dt=1.0):
    """Mean drift load for wave directions 0..350 deg ("from"), vessel at psi."""
    directions = np.arange(0.0, 360.0, 10.0)
    t = np.arange(0.0, t_end, dt)
    mean_drift = np.zeros((directions.size, 6))
    for k, d in enumerate(directions):
        # Turns the whole sea without changing spectrum, amplitudes or phases
        waves.set_direction(np.deg2rad(d))
        mean_drift[k] = np.mean([waves._load.second_order_loads(tk, psi) for tk in t], axis=0)
    return directions, mean_drift


def convergence_study(hs, tp, direction_deg, psi):
    """Load statistics vs number of components, averaged over several seeds."""
    t = np.arange(0.0, CONV_T, CONV_DT)
    rows = []
    for n in CONV_N:
        stats = []
        for seed in CONV_SEEDS:
            tau_wf, tau_sv = split_loads(make_waves(hs, tp, direction_deg, n, seed), t, psi)
            # [WF std Fx, Fy, Mz, drift mean Fx, Fy, Mz, drift std Fx, Fy, Mz]
            stats.append(np.r_[tau_wf[:, [0, 1, 5]].std(0),
                               tau_sv[:, [0, 1, 5]].mean(0),
                               tau_sv[:, [0, 1, 5]].std(0)])
        mean = np.mean(stats, axis=0) * 1e-3
        rows.append([n] + [round(float(v), 2) for v in mean])
        print(f"N = {n:3d}: WF std Fx/Fy/Mz = {mean[0]:.1f} / {mean[1]:.1f} kN / {mean[2]:.0f} kNm, "
              f"drift mean = {mean[3]:.2f} / {mean[4]:.2f} kN / {mean[5]:.2f} kNm, "
              f"drift std = {mean[6]:.2f} / {mean[7]:.2f} kN / {mean[8]:.2f} kNm")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hs", type=float, default=HS, help="significant wave height [m]")
    p.add_argument("--tp", type=float, default=TP, help="peak period [s]")
    p.add_argument("--direction", type=float, default=DIRECTION,
                   help="wave direction, FROM [deg] (0 = North, 90 = East)")
    p.add_argument("--psi", type=float, default=PSI_SHIP, help="vessel heading [deg]")
    p.add_argument("--n", type=int, default=N_COMPONENTS, help="number of frequency components")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--t-end", type=float, default=T_END, help="time series length [s]")
    p.add_argument("--convergence", action="store_true",
                   help="also run the convergence study vs N (takes a few minutes)")
    p.add_argument("--no-show", action="store_true", help="save figures without showing them")
    return p.parse_args(argv)


def main(argv=None):
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    psi = np.deg2rad(args.psi)
    OUT_DIR.mkdir(exist_ok=True)
    waves = make_waves(args.hs, args.tp, args.direction, args.n, args.seed)

    # --- 1. Spectrum, frequency range and discretised components ---
    wp = 2.0 * np.pi / args.tp
    w_lo, w_hi = waves.frequencies.min(), waves.frequencies.max()
    w = np.linspace(0.01, 4.0 * wp, 2000)
    _, S = ModifiedPiersonMoskowitz(w)(args.hs, args.tp)
    # Spectral density represented by each component, a_i^2 / (2 dw)
    dw = (w_hi - w_lo) / (args.n - 1)
    S_comp = waves.amplitudes ** 2 / (2.0 * dw)
    share = energy_share(args.hs, args.tp, 0.25 * wp, 3.0 * wp)

    fig1, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w, S, "k", label="modified PM (ITTC) spectrum")
    ax.plot(waves.frequencies, S_comp, "o", ms=3, label=f"{args.n} components")
    ax.axvspan(0.25 * wp, 3.0 * wp, color="tab:blue", alpha=0.08,
               label=f"range 0.25-3 $\\omega_p$ ({share * 100:.1f} % of energy)")
    ax.axvline(wp, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$S(\omega)$ [m$^2$s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig1.suptitle(f"Wave spectrum — $H_s$ = {args.hs} m, $T_p$ = {args.tp} s")
    fig1.tight_layout()

    # --- 2. First-order (WF) and second-order (drift) loads ---
    t = np.arange(0.0, args.t_end, DT)
    tau_wf, tau_sv = split_loads(waves, t, psi)

    fig2, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)
    for row, (label, i) in enumerate([("$F_x$ [kN]", 0), ("$F_y$ [kN]", 1), ("$M_z$ [kNm]", 5)]):
        axes[row, 0].plot(t, tau_wf[:, i] * 1e-3, lw=0.8)
        axes[row, 1].plot(t, tau_sv[:, i] * 1e-3, color="tab:orange", lw=0.8)
        # Mean drift shown as a dotted line
        axes[row, 1].axhline(tau_sv[:, i].mean() * 1e-3, color="k", ls=":", lw=1)
        axes[row, 0].set_ylabel(label)
    axes[0, 0].set_title("First-order wave-frequency load (filtered by observer)")
    axes[0, 1].set_title("Second-order drift load (rejected by DP controller)")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    fig2.suptitle(f"Wave loads in BODY — $H_s$ = {args.hs} m, $T_p$ = {args.tp} s, "
                  f"from {args.direction:.0f}$\\degree$, $\\psi$ = {args.psi:.0f}$\\degree$, "
                  f"N = {args.n}, seed {args.seed}")
    fig2.tight_layout()

    # --- 3. Direction check: mean drift vs wave direction ---
    sweep_waves = make_waves(args.hs, args.tp, args.direction, args.n, args.seed)
    directions, mean_drift = direction_sweep(sweep_waves, psi)
    # Direction the waves travel towards, relative to the bow
    towards_rel = np.degrees((np.deg2rad(directions) + np.pi - psi) % (2.0 * np.pi))
    drift_dir = np.degrees(np.arctan2(mean_drift[:, 1], mean_drift[:, 0]) % (2.0 * np.pi))

    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes3[0].plot(directions, mean_drift[:, 0] * 1e-3, "o-", ms=3, label="$F_x$ [kN]")
    axes3[0].plot(directions, mean_drift[:, 1] * 1e-3, "o-", ms=3, label="$F_y$ [kN]")
    axes3[0].plot(directions, mean_drift[:, 5] * 1e-3, "o-", ms=3, label="$M_z$ [kNm]")
    axes3[0].set_ylabel("Mean drift load (BODY)")
    axes3[0].legend(loc="best")
    axes3[1].plot(directions, towards_rel, "k--", label="wave propagation (towards)")
    axes3[1].plot(directions, drift_dir, "o", ms=4, label="mean drift force direction")
    axes3[1].set_ylabel("Direction rel. bow [deg]")
    axes3[1].set_xlabel("Wave direction, from [deg]")
    axes3[1].set_xlim(0.0, 350.0)
    axes3[1].legend(loc="best")
    for ax in axes3:
        ax.grid(True, alpha=0.3)
    fig3.suptitle(f"Direction check — vessel at $\\psi$ = {args.psi:.0f}$\\degree$, "
                  f"$H_s$ = {args.hs} m, $T_p$ = {args.tp} s")
    fig3.tight_layout()

    # Saves the figures for the report
    for fig, name in [(fig1, "fig_wave_spectrum"), (fig2, "fig_wave_loads"),
                      (fig3, "fig_wave_direction")]:
        fig.savefig(OUT_DIR / f"{name}.pdf")
        fig.savefig(OUT_DIR / f"{name}.png", dpi=200)

    # Prints the key numbers for the report
    finite = np.all(np.isfinite(tau_wf)) and np.all(np.isfinite(tau_sv))
    print(f"loads finite: {finite}")
    print(f"frequency range {w_lo:.3f}-{w_hi:.3f} rad/s, energy share {share * 100:.1f} %")
    print(f"WF std   Fx/Fy/Mz = {tau_wf[:, 0].std() * 1e-3:.1f} / {tau_wf[:, 1].std() * 1e-3:.1f} kN"
          f" / {tau_wf[:, 5].std() * 1e-3:.0f} kNm")
    print(f"drift mean Fx/Fy/Mz = {tau_sv[:, 0].mean() * 1e-3:.2f} / {tau_sv[:, 1].mean() * 1e-3:.2f} kN"
          f" / {tau_sv[:, 5].mean() * 1e-3:.2f} kNm")

    # Largest angle between mean drift force and wave propagation in the sweep
    dev = (drift_dir - towards_rel + 180.0) % 360.0 - 180.0
    # Period after which the drift load would repeat without frequency jitter
    repeat = 2.0 * np.pi / dw
    print(f"max angle drift force vs propagation = {np.max(np.abs(dev)):.1f} deg")
    print(f"repeat period without frequency jitter = {repeat:.0f} s")

    # Saves every number quoted in the report
    rows = [["quantity", "value", "unit", "note"],
            ["Hs", args.hs, "m", ""], ["Tp", args.tp, "s", ""],
            ["direction (from)", args.direction, "deg", ""],
            ["vessel heading", args.psi, "deg", ""],
            ["components N", args.n, "", ""], ["seed", args.seed, "", ""],
            ["frequency range", f"{w_lo:.3f}-{w_hi:.3f}", "rad/s", "0.25-3 wp, random jitter"],
            ["energy share in range", round(share * 100, 2), "%", ""],
            ["repeat period without jitter", round(repeat), "s", "2 pi / d_omega"],
            ["loads finite", finite, "", f"{args.t_end:g} s"],
            ["WF std Fx", round(tau_wf[:, 0].std() * 1e-3, 1), "kN", f"seed {args.seed}, {args.t_end:g} s"],
            ["WF std Fy", round(tau_wf[:, 1].std() * 1e-3, 1), "kN", ""],
            ["WF std Mz", round(tau_wf[:, 5].std() * 1e-3, 0), "kNm", ""],
            ["WF max |Fy|", round(np.abs(tau_wf[:, 1]).max() * 1e-3, 0), "kN", ""],
            ["drift mean Fx", round(tau_sv[:, 0].mean() * 1e-3, 2), "kN", ""],
            ["drift mean Fy", round(tau_sv[:, 1].mean() * 1e-3, 2), "kN", ""],
            ["drift mean Mz", round(tau_sv[:, 5].mean() * 1e-3, 2), "kNm", ""],
            ["max angle drift vs propagation", round(float(np.max(np.abs(dev))), 1), "deg", "direction sweep"]]
    with open(OUT_DIR / "wave_verification.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # --- 4. Optional convergence study vs N ---
    if args.convergence:
        rows = convergence_study(args.hs, args.tp, args.direction, psi)
        header = ["N", "WF_std_Fx_kN", "WF_std_Fy_kN", "WF_std_Mz_kNm",
                  "drift_mean_Fx_kN", "drift_mean_Fy_kN", "drift_mean_Mz_kNm",
                  "drift_std_Fx_kN", "drift_std_Fy_kN", "drift_std_Mz_kNm"]
        with open(OUT_DIR / "wave_convergence_table.csv", "w", newline="") as f:
            csv.writer(f).writerows([header] + rows)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
