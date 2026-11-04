"""Verify the Part 2 wind model and produce the evidence for the report.

    python verify_wind_model.py
    python verify_wind_model.py --no-show
    python verify_wind_model.py --hours 6

The wind model is built with ``environment()`` from run_case_part_2.py, so it
is exactly the wind used in the Part 2 simulations (same mean wind, slow
processes, gust spectrum and seed).  Every number quoted in the wind section
of the report is printed here and written to figures/wind_verification.csv.

Output (figures/ next to this script):
    fig_wind_coefficients.pdf/.png   Fig. A1: table points + linear interpolation
    fig_wind_gust_spectrum.pdf/.png  Fig. A2: realised gust spectrum vs NORSOK
    fig_wind_direction.pdf/.png      Fig. A3: slowly varying direction, +/-5 deg bound
    wind_verification.csv            all numbers quoted in the report
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_1.wind import Wind as WindPart1
from part_2.wind import Wind, load_wind_coefficients, norsok_spectrum
from run_case_part_2 import DT, environment

OUT_DIR = Path(__file__).resolve().parent / "figures"

N_EQUIV_CASES = 1000           # random cases in the Part 1 / Part 2 comparison
WELCH_SEGMENT = 1800.0         # segment length of the spectrum estimate [s]
SIM1_LENGTH = 300.0            # length of Part 2, Simulation 1 [s]
N_BANDS = 12                   # log bands for the spectrum comparison (0.003-0.3 Hz)


def runner_wind() -> Wind:
    """The wind model exactly as run_case_part_2.py builds it."""
    _, wind, _ = environment(current=False, wind=True, waves=False)
    return wind


def coefficient_quality(alpha_deg, C6):
    """Table step, periodicity and leave-one-out error of linear interpolation."""
    step = np.unique(np.round(np.diff(alpha_deg), 6))
    # The table must close the circle: C(0 deg) = C(360 deg)
    wrap_gap = np.abs(C6[0] - C6[-1])
    # Leave-one-out: interpolate each interior point from its two neighbours.
    # This is the error linear interpolation makes half-way between table points.
    loo = np.zeros(C6.shape[1])
    for i in range(1, len(alpha_deg) - 1):
        a0, a1, a2 = alpha_deg[i - 1], alpha_deg[i], alpha_deg[i + 1]
        guess = C6[i - 1] + (C6[i + 1] - C6[i - 1]) * (a1 - a0) / (a2 - a0)
        loo = np.maximum(loo, np.abs(guess - C6[i]))
    scale = np.max(np.abs(C6), axis=0)
    return step, wrap_gap, 100.0 * loo / np.where(scale > 0, scale, 1.0)


def part1_equivalence(n_cases, seed=0):
    """Max load difference Part 1 vs Part 2 with gust and slow processes off."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for k in range(n_cases):
        semantics = "from" if k % 2 == 0 else "towards"
        speed, beta = rng.uniform(0.0, 20.0), rng.uniform(-np.pi, 2.0 * np.pi)
        eta, nu = np.zeros(6), np.zeros(6)
        eta[5] = rng.uniform(-np.pi, np.pi)
        nu[:2] = rng.uniform(-2.0, 2.0, 2)
        tau1, _ = WindPart1(speed, beta, semantics=semantics).step(0.0, DT, eta, nu)
        tau2, _ = Wind(speed, beta, semantics=semantics).step(0.0, DT, eta, nu)
        worst = max(worst, float(np.max(np.abs(np.asarray(tau1) - np.asarray(tau2)))))
    return worst


def realise(wind, t_end):
    """Step the wind on a vessel at rest; return t, U_mean, U_gust, d_beta."""
    t = np.arange(0.0, t_end, DT)
    U_mean, U_gust, d_beta = (np.zeros(t.size) for _ in range(3))
    eta, nu = np.zeros(6), np.zeros(6)
    for k, tk in enumerate(t):
        _, info = wind.step(tk, DT, eta, nu)
        U_mean[k], U_gust[k], d_beta[k] = info["U_mean"], info["U_gust"], info["dir_offset"]
    return t, U_mean, U_gust, d_beta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=6.0, help="length of the realisation [h]")
    p.add_argument("--no-show", action="store_true", help="save figures without showing them")
    return p.parse_args(argv)


def main(argv=None):
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    OUT_DIR.mkdir(exist_ok=True)
    wind = runner_wind()
    rows = []                                   # (quantity, value, unit, note)

    def record(name, value, unit="", note=""):
        rows.append([name, value, unit, note])
        print(f"{name:45s} {value} {unit}  {note}")

    # --- Parameters actually used by the runner ---
    gp = wind.gust_params
    record("mean wind speed", wind.mean_speed, "m/s", "run_case_part_2.WIND_SPEED")
    record("mean wind direction (from)", round(np.degrees(wind.beta), 2), "deg")
    record("sigma_slow", wind.sigma_slow, "m/s")
    record("tau_slow", wind.tau_slow, "s")
    record("sigma_dir", round(np.degrees(wind.sigma_dir), 3), "deg")
    record("tau_dir", wind.tau_dir, "s")
    record("direction bound", round(np.degrees(wind.dir_limit), 3), "deg")
    record("gust spectrum", gp["spectrum"], "", f"z = {gp['z']} m")
    record("gust harmonics N", gp["n_freq"])
    record("gust frequency range", f"{gp['f_min']}-{gp['f_max']}", "Hz", "log-spaced bins")
    record("seed", wind.seed)
    record("time step", DT, "s")

    # --- 1. Coefficient table and interpolation quality (Fig. A1) ---
    alpha_deg, C6 = load_wind_coefficients()
    step, wrap_gap, loo = coefficient_quality(alpha_deg, C6)
    record("coefficient table step", ",".join(str(s) for s in step), "deg")
    record("coefficient table points", len(alpha_deg))
    for name, i in [("C_X", 0), ("C_Y", 1), ("C_N", 5)]:
        record(f"{name} gap C(0)-C(360)", f"{wrap_gap[i]:.3g}")
        record(f"{name} leave-one-out interp. error", round(float(loo[i]), 2), "% of max|C|")

    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    a_fine = np.linspace(0.0, 360.0, 3601)
    for ax, (name, i, unit) in zip(axes1, [("$C_X$", 0, "N/(m/s)$^2$"),
                                           ("$C_Y$", 1, "N/(m/s)$^2$"),
                                           ("$C_N$", 5, "Nm/(m/s)$^2$")]):
        ax.plot(a_fine, np.interp(a_fine, alpha_deg, C6[:, i]), lw=1, label="linear interpolation")
        ax.plot(alpha_deg, C6[:, i], "o", ms=3, label="table points")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.grid(True, alpha=0.3)
    axes1[0].legend(loc="best", fontsize=8)
    axes1[-1].set_xlabel(r"Relative wind angle $\alpha_{rw}$ [deg]")
    axes1[-1].set_xlim(0.0, 360.0)
    fig1.suptitle("Gunnerus wind coefficients: table and interpolation")
    fig1.tight_layout()

    # --- 2. Base model identical to Part 1 (gust and slow processes off) ---
    record("max |tau| difference Part 1 vs Part 2", f"{part1_equivalence(N_EQUIV_CASES):.3g}", "N or Nm",
           f"{N_EQUIV_CASES} random cases, extensions off")

    # --- 3. Realisation with the runner settings ---
    t, U_mean, U_gust, d_beta = realise(wind, args.hours * 3600.0)
    record("realisation length", args.hours, "h")
    record("slow speed std (realised)", round(float(np.std(U_mean - wind.mean_speed)), 3), "m/s",
           f"target sigma_slow = {wind.sigma_slow}")

    # Gust statistics vs the discretised and the full NORSOK spectrum
    f_full = np.geomspace(1e-5, 10.0, 200001)
    std_full = np.sqrt(np.trapezoid(norsok_spectrum(f_full, wind.mean_speed, gp["z"]), f_full))
    record("gust std, full NORSOK spectrum", round(float(std_full), 3), "m/s")
    record("gust std, discretised (theory)", round(wind.gust_std(), 3), "m/s")
    record("gust std (realised)", round(float(np.std(U_gust)), 3), "m/s")
    record("gust mean (realised)", round(float(np.mean(U_gust)), 4), "m/s")

    # Spectrum estimate (Welch). The gust is a sum of discrete harmonics, so its
    # energy sits in narrow lines: compare the ENERGY in log-spaced bands, not
    # the spectral density point by point.
    f_w, P_w = welch(U_gust, fs=1.0 / DT, nperseg=int(WELCH_SEGMENT / DT))
    edges = np.geomspace(0.003, 0.3, N_BANDS + 1)
    f_band, S_band_real, S_band_target, ratio = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f_w >= lo) & (f_w < hi)
        f_t = np.linspace(lo, hi, 200)
        e_real = np.trapezoid(P_w[m], f_w[m]) if m.sum() > 1 else np.nan
        e_target = np.trapezoid(norsok_spectrum(f_t, wind.mean_speed, gp["z"]), f_t)
        # Band energy shown as the mean density over the band
        f_band.append(np.sqrt(lo * hi))
        S_band_real.append(e_real / (hi - lo))
        S_band_target.append(e_target / (hi - lo))
        ratio.append(e_real / e_target)
    ratio = np.array(ratio)
    record("band energy realised/target, 0.003-0.3 Hz", f"{np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f}", "",
           f"{N_BANDS} log bands, mean {np.nanmean(ratio):.2f}")
    # Above 0.005 Hz the 1800 s Welch segments resolve the spectrum well; the
    # lowest band sits next to f_min, where the estimate is too coarse.
    upper = edges[:-1] >= 0.005
    record("band energy realised/target, 0.005-0.3 Hz", f"{np.nanmin(ratio[upper]):.2f}-{np.nanmax(ratio[upper]):.2f}", "",
           f"mean {np.nanmean(ratio[upper]):.2f}")

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    f_plot = np.geomspace(1e-4, 1.0, 500)
    ax2.loglog(f_w[1:], P_w[1:], lw=0.6, color="0.7", label=f"realised gust (Welch, {WELCH_SEGMENT:.0f} s segments)")
    ax2.loglog(f_band, S_band_real, "o", color="tab:blue", label="realised, band average")
    ax2.loglog(f_plot, norsok_spectrum(f_plot, wind.mean_speed, gp["z"]), "k", label="NORSOK target")
    ax2.axvspan(gp["f_min"], gp["f_max"], color="tab:blue", alpha=0.08, label="discretised band")
    ax2.axvline(1.0 / (2.0 * np.pi * wind.tau_slow), color="tab:red", ls=":",
                label=r"slow-speed corner $1/(2\pi\tau_{slow})$")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel(r"$S(f)$ [(m/s)$^2$/Hz]")
    ax2.set_xlim(1e-4, 1.0)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(1e-3, 1e3)
    ax2.legend(loc="lower left", fontsize=8)
    fig2.suptitle(f"Gust spectrum — $\\bar U$ = {wind.mean_speed} m/s, N = {gp['n_freq']}, "
                  f"seed {wind.seed}, {args.hours:g} h")
    fig2.tight_layout()

    # Slowly varying direction and its bound
    lim = np.degrees(wind.dir_limit)
    d_deg = np.degrees(d_beta)
    at_bound = np.mean(np.abs(d_deg) >= lim - 1e-9) * 100.0
    record("direction std (realised)", round(float(np.std(d_deg)), 3), "deg",
           f"target sigma_dir = {np.degrees(wind.sigma_dir):.1f} deg (before clipping)")
    record("max |direction excursion|", round(float(np.max(np.abs(d_deg))), 3), "deg")
    record("time at direction bound", round(float(at_bound), 3), "%")
    n_sim1 = int(SIM1_LENGTH / DT)
    record("max |direction excursion| in first 300 s", round(float(np.max(np.abs(d_deg[:n_sim1]))), 3),
           "deg", "length of Simulation 1")

    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    ax3.plot(t / 3600.0, d_deg, lw=0.8)
    for s in (-1, 1):
        ax3.axhline(s * lim, color="k", ls="--", lw=1)
    ax3.axvspan(0.0, SIM1_LENGTH / 3600.0, color="tab:orange", alpha=0.15, label="Simulation 1 (300 s)")
    ax3.set_xlabel("Time [h]")
    ax3.set_ylabel(r"$\delta\beta$ [deg]")
    ax3.set_xlim(0.0, args.hours)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)
    fig3.suptitle(f"Slowly varying wind direction — $\\sigma$ = {np.degrees(wind.sigma_dir):.0f}$\\degree$, "
                  f"$\\tau$ = {wind.tau_dir:.0f} s, bound $\\pm${lim:.0f}$\\degree$ ({at_bound:.2f} % of time at bound)")
    fig3.tight_layout()

    # Saves figures and the table of numbers
    for fig, name in [(fig1, "fig_wind_coefficients"), (fig2, "fig_wind_gust_spectrum"),
                      (fig3, "fig_wind_direction")]:
        fig.savefig(OUT_DIR / f"{name}.pdf")
        fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    with open(OUT_DIR / "wind_verification.csv", "w", newline="") as f:
        csv.writer(f).writerows([["quantity", "value", "unit", "note"]] + rows)
    print(f"saved figures and wind_verification.csv to {OUT_DIR}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
