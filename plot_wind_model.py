"""Plot the behaviour of part_1.wind.Wind over time and over relative wind angle.

    python plot_wind_model.py
    python plot_wind_model.py --psi 30 --beta 90 --speed 15 --sigma-slow 1.5 --u 2.0
    python plot_wind_model.py --help

Sign and magnitude verification lives in check.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from part_1.wind import Wind

DT = 0.05
T_END = 900.0

MEAN_SPEED = 15.0
BETA = np.pi / 2          # wind FROM east
SIGMA_SLOW = 1.5
TAU_SLOW = 120.0
SEED = 0

PSI_SHIP = np.deg2rad(30.0)   # vessel heading used in the time simulation


def simulate(wind, psi, nu_body=(0.0, 0.0), t_end=T_END, dt=DT):
    """Step the wind model on a vessel held at a fixed heading and velocity."""
    t = np.arange(0.0, t_end, dt)
    eta = np.zeros(6)
    eta[5] = psi
    nu = np.zeros(6)
    nu[0], nu[1] = nu_body

    U = np.zeros(t.size)
    beta_ned = np.zeros(t.size)
    alpha_body = np.zeros(t.size)
    tau = np.zeros((t.size, 6))

    for k, tk in enumerate(t):
        tau_k, info = wind.step(tk, dt, eta, nu)
        tau[k] = tau_k
        U[k] = info["U"]
        beta_ned[k] = info["beta_ned"]
        alpha_body[k] = info["alpha_body"]

    return t, U, beta_ned, alpha_body, tau


def sweep_relative_angle(wind, psi, n=721):
    """Loads as a function of relative wind angle, at the model's mean speed."""
    alpha = np.linspace(0.0, 2.0 * np.pi, n)
    eta = np.zeros(6)
    eta[5] = psi
    nu = np.zeros(6)

    tau = np.zeros((n, 6))
    for k, a in enumerate(alpha):
        # alpha_rw = 0 means wind blowing towards the bow, i.e. beta_towards = psi + alpha
        probe = Wind(wind.mean_speed, psi + a, semantics="towards")
        tau_k, _ = probe.step(0.0, 0.0, eta, nu)
        tau[k] = tau_k
    return np.degrees(alpha), tau


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psi", type=float, default=np.degrees(PSI_SHIP),
                   help="vessel heading [deg]")
    p.add_argument("--beta", type=float, default=np.degrees(BETA),
                   help="wind direction [deg]")
    p.add_argument("--semantics", choices=("from", "towards"), default="from",
                   help="interpretation of --beta")
    p.add_argument("--speed", type=float, default=MEAN_SPEED,
                   help="mean wind speed [m/s]")
    p.add_argument("--sigma-slow", type=float, default=SIGMA_SLOW,
                   help="std of the slowly-varying speed [m/s]")
    p.add_argument("--tau-slow", type=float, default=TAU_SLOW,
                   help="time constant of the slow variation [s]")
    p.add_argument("--u", type=float, default=0.0, help="vessel surge speed [m/s]")
    p.add_argument("--v", type=float, default=0.0, help="vessel sway speed [m/s]")
    p.add_argument("--t-end", type=float, default=T_END, help="simulation length [s]")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def draw_situation(ax, psi, beta_towards, alpha_rw, u=0.0, v=0.0):
    """Bird's-eye NED sketch of hull heading, wind direction and relative angle."""
    # Plotted in (East, North) so North is up, with the hull drawn in BODY and rotated.
    hull = np.array([[1.0, 0.0], [0.55, 0.32], [-0.75, 0.32],
                     [-0.9, 0.0], [-0.75, -0.32], [0.55, -0.32]])
    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    hull_ned = hull @ R.T
    ax.fill(hull_ned[:, 1], hull_ned[:, 0], color="0.75",
            edgecolor="k", lw=1.5, zorder=3)

    L = 2.2
    bow = np.array([np.sin(psi), np.cos(psi)])
    bow_perp = np.array([bow[1], -bow[0]])
    ax.annotate("", xy=(L * bow[0], L * bow[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=2), zorder=4)
    ax.text(*(L * bow + 0.3 * bow_perp), r"bow $\psi$", fontsize=9,
            ha="center", va="center",
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))

    # Wind arrow flies in from upwind and stops short of the hull.
    upwind = -np.array([np.sin(beta_towards), np.cos(beta_towards)])
    upwind_perp = np.array([upwind[1], -upwind[0]])
    ax.annotate("", xy=(0.8 * upwind[0], 0.8 * upwind[1]),
                xytext=(2.1 * upwind[0], 2.1 * upwind[1]),
                arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2), zorder=4)
    ax.text(*(1.6 * upwind - 0.3 * upwind_perp), "wind", color="tab:blue",
            fontsize=9, ha="center", va="center",
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))

    if abs(u) > 1e-9 or abs(v) > 1e-9:
        vel_ned = R @ np.array([u, v])
        d = vel_ned[::-1] / max(np.hypot(u, v), 1e-9)
        ax.annotate("", xy=(1.2 * d[0], 1.2 * d[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=2), zorder=4)
        ax.text(*(1.2 * d + 0.3 * np.array([d[1], -d[0]])), "vessel vel.",
                color="tab:red", fontsize=9, ha="center", va="center",
                bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))

    alpha = alpha_rw % (2.0 * np.pi)
    if alpha > 2.0 * np.pi - 1e-6:
        alpha = 0.0

    # alpha_rw is measured from the bow, positive to starboard, to the direction
    # the relative wind vector points (i.e. 180 deg = wind hitting the bow).
    ax.annotate("", xy=(1.6 * np.sin(psi + alpha), 1.6 * np.cos(psi + alpha)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2,
                                ls="--"), zorder=4)
    ax.text(1.75 * np.sin(psi + alpha), 1.75 * np.cos(psi + alpha),
            r"$V_{rw}$", color="tab:green", fontsize=9, ha="center", va="center",
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))

    arc = np.linspace(psi, psi + alpha, 120)
    ax.plot(0.9 * np.sin(arc), 0.9 * np.cos(arc), color="tab:green", lw=1.5, zorder=5)
    if alpha > np.deg2rad(5.0):
        ax.annotate("", xy=(0.9 * np.sin(arc[-1]), 0.9 * np.cos(arc[-1])),
                    xytext=(0.9 * np.sin(arc[-6]), 0.9 * np.cos(arc[-6])),
                    arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=1.5),
                    zorder=5)
    mid = psi + alpha / 2.0
    ax.text(1.15 * np.sin(mid), 1.15 * np.cos(mid),
            rf"$\alpha_{{rw}}$ = {np.degrees(alpha):.0f}$\degree$",
            color="tab:green", fontsize=9, ha="center", va="center",
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))

    lim = 2.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    for label, (x, y) in {"N": (0, lim * 0.93), "S": (0, -lim * 0.93),
                          "E": (lim * 0.93, 0), "W": (-lim * 0.93, 0)}.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv=None):
    import matplotlib.pyplot as plt

    args = parse_args(argv)
    psi = np.deg2rad(args.psi)
    beta = np.deg2rad(args.beta)

    wind = Wind(args.speed, beta, semantics=args.semantics,
                sigma_slow=args.sigma_slow, tau_slow=args.tau_slow, seed=args.seed)
    t, U, beta_ned, alpha_body, tau = simulate(
        wind, psi, nu_body=(args.u, args.v), t_end=args.t_end)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, U)
    axes[0].axhline(args.speed, color="k", ls=":", lw=1, label="mean")
    axes[0].set_ylabel("Ambient speed [m/s]")
    axes[0].legend(loc="best")

    axes[1].plot(t, tau[:, 0] * 1e-3, label="$F_x$ [kN]")
    axes[1].plot(t, tau[:, 1] * 1e-3, label="$F_y$ [kN]")
    axes[1].plot(t, tau[:, 5] * 1e-3, label="$M_z$ [kNm]")
    axes[1].set_ylabel("BODY loads")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Wind time history — $\\psi$ = {args.psi:.0f}$\\degree$, "
                 f"wind {args.semantics} {args.beta:.0f}$\\degree$, "
                 f"$\\sigma_{{slow}}$ = {args.sigma_slow} m/s")
    fig.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(5.5, 5.5))
    draw_situation(ax3, psi, beta_ned[0], alpha_body[0], args.u, args.v)
    ax3.set_title(f"Situation at t = 0 — wind {args.semantics} "
                  f"{args.beta:.0f}$\\degree$, $\\psi$ = {args.psi:.0f}$\\degree$")
    fig3.tight_layout()

    alpha_deg, tau_sweep = sweep_relative_angle(wind, psi)
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(alpha_deg, tau_sweep[:, 0] * 1e-3, label="$F_x$ [kN]")
    ax.plot(alpha_deg, tau_sweep[:, 1] * 1e-3, label="$F_y$ [kN]")
    ax.plot(alpha_deg, tau_sweep[:, 5] * 1e-3, label="$M_z$ [kNm]")
    ax.axvline(180.0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"Relative wind angle $\alpha_{rw}$ [deg]")
    ax.set_ylabel("BODY loads")
    ax.set_xlim(0.0, 360.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig2.suptitle(f"Loads vs relative wind angle at U = {args.speed} m/s")
    fig2.tight_layout()

    print(f"mean |Fx| = {np.mean(np.abs(tau[:, 0])) * 1e-3:.2f} kN")
    print(f"mean |Fy| = {np.mean(np.abs(tau[:, 1])) * 1e-3:.2f} kN")
    print(f"speed std = {np.std(U):.2f} m/s")

    plt.show()


if __name__ == "__main__":
    main()
