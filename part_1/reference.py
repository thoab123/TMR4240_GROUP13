"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.
"""
from typing import Tuple
import numpy as np

from part_1.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


class _AxisFilter:
    """
    Reusable single-axis second-order filter (one instance per axis: N, E,
    psi). State-space form of the project's Eq. (2):

        eta_ddot + 2*zeta*wn*eta_dot + wn^2*eta = wn^2*target

    rearranged for acceleration:

        a = wn^2 * (target - x) - 2*zeta*wn * v

    then advanced with plain forward Euler (matching the vessel's own
    integration method, Sec. 3.4):

        v_new = v + dt * a
        x_new = x + dt * v          (uses the OLD v)
    """

    def __init__(self, dt: float, wn: float, zeta: float,
                 rate_limit: float | None = None):
        self.dt = float(dt)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.rate_limit = rate_limit
        self.x = 0.0
        self.v = 0.0

    def reset(self, x0: float = 0.0, v0: float = 0.0) -> None:
        self.x = float(x0)
        self.v = float(v0)

    def step(self, target: float) -> Tuple[float, float, float]:
        a = self.wn**2 * (target - self.x) - 2.0 * self.zeta * self.wn * self.v
        v_old = self.v
        self.v = self.v + self.dt * a
        if self.rate_limit is not None:
            self.v = float(np.clip(self.v, -self.rate_limit, self.rate_limit))
        self.x = self.x + self.dt * v_old
        return self.x, self.v, a


class ReferenceModel:
    """Second-order filter reference model, one _AxisFilter per DOF."""

    def __init__(
        self,
        dt: float,
        cfg_xy: RefAxisConfig | None = None,
        cfg_psi: RefAxisConfig | None = None,
    ):
        self.dt = float(dt)
        self.cfg_xy = cfg_xy if cfg_xy is not None else RefAxisConfig()
        self.cfg_psi = cfg_psi if cfg_psi is not None else RefAxisConfig()
        self.eta_ref = np.zeros(6)
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

        self.filt_N = _AxisFilter(
            dt=self.dt, wn=self.cfg_xy.wn, zeta=self.cfg_xy.zeta,
            rate_limit=self.cfg_xy.rate_limit,
        )
        self.filt_E = _AxisFilter(
            dt=self.dt, wn=self.cfg_xy.wn, zeta=self.cfg_xy.zeta,
            rate_limit=self.cfg_xy.rate_limit,
        )
        self.filt_psi = _AxisFilter(
            dt=self.dt, wn=self.cfg_psi.wn, zeta=self.cfg_psi.zeta,
            rate_limit=self.cfg_psi.rate_limit,
        )

    def reset(self, eta0: np.ndarray) -> None:
        """Initialize the reference at the vessel's current (6,) state."""
        eta0 = np.asarray(eta0, dtype=float).reshape(6)
        self.eta_ref = eta0.copy()
        self.nu_ref = np.zeros(6)
        self.acc_ref = np.zeros(6)

        self.filt_N.reset(x0=eta0[0], v0=0.0)
        self.filt_E.reset(x0=eta0[1], v0=0.0)
        self.filt_psi.reset(x0=eta0[5], v0=0.0)

    def step(
        self, t: float, dt: float, eta_cmd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        eta_cmd = np.asarray(eta_cmd, dtype=float).reshape(6)

        N_ref, Ndot_ref, Nddot_ref = self.filt_N.step(eta_cmd[0])
        E_ref, Edot_ref, Eddot_ref = self.filt_E.step(eta_cmd[1])

        # psi: shift the commanded heading to the nearest equivalent of
        # itself relative to where the filter currently is, so it always
        # turns the short way (see project Tips: "should not produce an
        # unnecessary 2*pi turn").
        psi_cmd = eta_cmd[5]
        psi_target_eff = self.filt_psi.x + wrap_angle_pi(psi_cmd - self.filt_psi.x)
        psi_ref_raw, psidot_ref, psiddot_ref = self.filt_psi.step(psi_target_eff)
        psi_ref = wrap_angle_pi(psi_ref_raw)

        self.eta_ref = np.zeros(6)
        self.eta_ref[0], self.eta_ref[1], self.eta_ref[5] = N_ref, E_ref, psi_ref

        self.nu_ref = np.zeros(6)
        self.nu_ref[0], self.nu_ref[1], self.nu_ref[5] = Ndot_ref, Edot_ref, psidot_ref

        self.acc_ref = np.zeros(6)
        self.acc_ref[0], self.acc_ref[1], self.acc_ref[5] = Nddot_ref, Eddot_ref, psiddot_ref

        return self.eta_ref, self.nu_ref, self.acc_ref