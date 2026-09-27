"""
plot_reference_model.py
-----------------------
Standalone step response of the reference model (part_1/reference.py) for the
report figure "fig_reference_step.pdf".

Runs ONLY the reference model (no vessel, no controller), fed with the
Simulation 3 setpoint: 10 m step in North and psi_sp = 3*pi/2.

The tuned third-order filter (config default) is plotted together with the
second-order filter of the project's Eq. (2) at the same wn, so the figure
shows why the third-order filter was chosen: its acceleration reference
starts at zero instead of jumping to A*wn^2.

Run from the repository root:
    python plot_reference_model.py

Output: figures/fig_reference_step.pdf and .png
"""
from pathlib import Path
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from part_1.reference import ReferenceModel
from part_1.config import RefAxisConfig

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DT = 0.05                 # [s] same step as the simulator
T_END = 300.0             # [s] plotted time span
STEP_N = 10.0             # [m] North step (Simulation 3)
PSI_SP = 3 * np.pi / 2    # [rad] commanded heading (Simulation 3)
OUT_DIR = Path("figures")

# Tuned defaults from part_1/config.py (wn, zeta, order = 3)
CFG_3 = RefAxisConfig()
CFG_2 = replace(CFG_3, order=2)     # same wn/zeta, second order (Eq. (2))
WN = CFG_3.wn

# IEEE single-column look
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C_3RD = "#1f4e8c"   # third-order filter (used)
C_2ND = "#9a9a9a"   # second-order filter (comparison)
C_AUX = "#666666"   # setpoints / annotations


# ---------------------------------------------------------------------------
# Run the reference model
# ---------------------------------------------------------------------------
def run_reference(cfg: RefAxisConfig):
    """Step the reference model alone; returns t, N_d, Ndot_d, Nddot_d, psi_d."""
    t = np.arange(int(T_END / DT)) * DT
    rm = ReferenceModel(dt=DT, cfg_xy=cfg, cfg_psi=cfg)
    rm.reset(np.zeros(6))                   # start at rest in the origin

    eta_cmd = np.zeros(6)
    eta_cmd[0] = STEP_N
    eta_cmd[5] = PSI_SP

    N_d, Ndot_d, Nddot_d, psi_d = (np.zeros_like(t) for _ in range(4))
    for k, tk in enumerate(t):
        eta_ref, nu_ref, acc_ref = rm.step(tk, DT, eta_cmd)
        N_d[k], Ndot_d[k], Nddot_d[k] = eta_ref[0], nu_ref[0], acc_ref[0]
        psi_d[k] = eta_ref[5]
    return t, N_d, Ndot_d, Nddot_d, psi_d


def settling_time(t, x, target, tol=0.02):
    """Last time |x - target| leaves the +-tol*|target| band."""
    outside = np.where(np.abs(x - target) > tol * abs(target))[0]
    return t[outside[-1] + 1] if len(outside) else 0.0


def main():
    t, N3, V3, A3, P3 = run_reference(CFG_3)
    _, N2, V2, A2, P2 = run_reference(CFG_2)

    ts3 = settling_time(t, N3, STEP_N)
    ts2 = settling_time(t, N2, STEP_N)

    # Key numbers. Analytic values for zeta = 1:
    #   2nd order: v_max = A*wn/e       at t = 1/wn,  a_max = A*wn^2 at t = 0
    #   3rd order: v_max = 2*A*wn/e^2   at t = 2/wn,
    #              a_max = 0.230*A*wn^2 at t = (2 - sqrt(2))/wn
    print(f"wn = {WN} rad/s, Tr = {1/WN:.1f} s, zeta = {CFG_3.zeta}")
    print(f"{'':12s}{'v_max [m/s]':>13s}{'a_max [m/s^2]':>15s}{'t_2% [s]':>10s}")
    print(f"{'2nd order':12s}{V2.max():13.3f}{np.abs(A2).max():15.4f}{ts2:10.1f}")
    print(f"{'3rd order':12s}{V3.max():13.3f}{np.abs(A3).max():15.4f}{ts3:10.1f}")
    print(f"analytic 3rd: v_max = {2*STEP_N*WN/np.e**2:.3f}, "
          f"a_max = {0.2303*STEP_N*WN**2:.4f}")
    print(f"final psi_d = {np.degrees(P3[-1]):.2f} deg")

    fig, ax = plt.subplots(4, 1, figsize=(3.45, 4.8), sharex=True)

    # (1) position
    ax[0].plot(t, np.full_like(t, STEP_N), ":", color=C_AUX, lw=1, label=r"$N_{sp}$")
    ax[0].plot(t, N2, "--", color=C_2ND, lw=1.1, label="2nd order")
    ax[0].plot(t, N3, color=C_3RD, label="3rd order (used)")
    ax[0].set_ylabel(r"$N_d$ [m]")
    ax[0].legend(loc="lower right", frameon=False)

    # (2) velocity
    ax[1].plot(t, V2, "--", color=C_2ND, lw=1.1)
    ax[1].plot(t, V3, color=C_3RD)
    ax[1].set_ylabel(r"$\dot N_d$ [m/s]")

    # (3) acceleration: the reason for the third-order filter
    ax[2].plot(t, A2, "--", color=C_2ND, lw=1.1)
    ax[2].plot(t, A3, color=C_3RD)
    ax[2].set_ylabel(r"$\ddot N_d$ [m/s$^2$]")
    ax[2].annotate(r"jump to $A\omega_r^2$ at $t=0$",
                   xy=(0.5, A2.max()), xytext=(60, 0.85 * A2.max()), fontsize=7,
                   arrowprops=dict(arrowstyle="-", color=C_AUX, lw=0.6))
    ax[2].annotate("3rd order: starts at zero",
                   xy=(2.0, A3[int(2.0 / DT)]), xytext=(60, 0.45 * A2.max()), fontsize=7,
                   arrowprops=dict(arrowstyle="-", color=C_AUX, lw=0.6))

    # (4) heading (third-order filter only)
    ax[3].axhline(np.degrees(PSI_SP), ls=":", color=C_AUX, lw=1,
                  label=r"$\psi_{sp}=3\pi/2$ (as commanded)")
    psi_wrapped = np.degrees(np.arctan2(np.sin(PSI_SP), np.cos(PSI_SP)))
    ax[3].axhline(psi_wrapped, ls="-.", color=C_AUX, lw=1,
                  label=r"$\mathrm{wrap}(\psi_{sp})=-\pi/2$")
    ax[3].plot(t, np.degrees(P3), color=C_3RD, label=r"$\psi_d$ (3rd order)")
    ax[3].set_ylabel(r"$\psi_d$ [deg]")
    ax[3].set_ylim(-120, 300)
    ax[3].set_yticks([-90, 0, 90, 180, 270])
    ax[3].legend(loc="center right", frameon=False, fontsize=6.5)
    ax[3].set_xlabel("Time [s]")
    ax[3].set_xlim(0, T_END)

    # 2 % settling time of the third-order filter
    for a in ax:
        a.axvline(ts3, color="#bbbbbb", lw=0.6, ls="-.")
    ax[0].text(ts3 - 3, 1.0, rf"$t_{{2\%}}\approx{ts3:.0f}$ s", fontsize=7,
               color=C_AUX, ha="right")

    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.4)

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "fig_reference_step.pdf")
    fig.savefig(OUT_DIR / "fig_reference_step.png", dpi=200)
    print(f"saved to {OUT_DIR.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
