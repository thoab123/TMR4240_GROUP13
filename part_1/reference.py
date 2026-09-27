"""
Reference template

Students should filter or shape commanded setpoints before they are sent to
the controller. The simulator calls, once per step:

    ref.step(t, dt, eta_cmd) -> (eta_ref, nu_ref, acc_ref)

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; leave the rest zero.

Design (Group 13):
    Third-order low-pass filter per axis (N, E, psi) by default. The
    second-order filter of the project's Eq. (2) is kept as an option
    (RefAxisConfig.order = 2) so the comparison in the report can be
    reproduced. Why third order: with the second-order filter the
    acceleration reference jumps from 0 to A*wn^2 at every setpoint step, and
    the controller's acceleration feedforward (M * acc_ref) passes that jump
    straight to the thrusters. This saturated the 32 kN tunnel thruster at
    the start of the four-corner legs. The third-order filter makes the
    acceleration start from zero and ramp up, which removed the saturation
    (also with current/wind) at the cost of ~35 s slower transitions.
"""
from typing import Tuple
import numpy as np

from part_1.config import RefAxisConfig
from simulation.utils import wrap_angle_pi


class _AxisFilter:
    """
    Reusable single-axis reference filter (one instance per axis: N, E,
    psi). Two versions, selected by `order`:

    order = 3 (default) - third-order filter: a first-order low-pass filter
    in cascade with the second-order filter of the project's Eq. (2),
    see Fossen (2021), Ch. 12 (reference models):

        eta_d''' + (2*zeta+1)*wn*eta_d'' + (2*zeta+1)*wn^2*eta_d'
                 + wn^3*eta_d = wn^3 * target

    rearranged for the jerk (derivative of acceleration):

        j = wn^3 * (target - x) - (2*zeta+1)*wn^2 * v - (2*zeta+1)*wn * a

    With zeta = 1 all three poles sit at s = -wn (no overshoot). The
    acceleration a is now a STATE, so it is continuous and starts at zero
    after a setpoint step.

    order = 2 - second-order filter, the project's Eq. (2):

        eta_ddot + 2*zeta*wn*eta_dot + wn^2*eta = wn^2*target

    rearranged for acceleration:

        a = wn^2 * (target - x) - 2*zeta*wn * v

    Here a is computed algebraically, so it jumps when the target jumps.

    Both are advanced with plain forward Euler (matching the vessel's own
    integration method, Sec. 3.4). Every state is updated from the OLD
    values of the others:

        x_new = x + dt * v
        v_new = v + dt * a
        a_new = a + dt * j          (order 3 only)

    Stability: wn*dt = 0.0025 for wn = 0.05, dt = 0.05, far inside the
    forward-Euler stability region, so the discrete response matches the
    continuous one.
    """

    def __init__(self, dt: float, wn: float, zeta: float,
                 rate_limit: float | None = None, order: int = 3):
        if order not in (2, 3):
            raise ValueError(f"order must be 2 or 3, got {order}")
        self.dt = float(dt)
        self.wn = float(wn)
        self.zeta = float(zeta)
        self.rate_limit = rate_limit
        self.order = int(order)
        self.x = 0.0   # desired position / heading
        self.v = 0.0   # desired velocity
        self.a = 0.0   # desired acceleration (a state only for order 3)

    def reset(self, x0: float = 0.0, v0: float = 0.0) -> None:
        # Start at rest: zero velocity AND zero acceleration, so the first
        # step after reset produces no force spike from the feedforward.
        self.x = float(x0)
        self.v = float(v0)
        self.a = 0.0

    def step(self, target: float) -> Tuple[float, float, float]:
        wn, z, dt = self.wn, self.zeta, self.dt
        x_old, v_old, a_old = self.x, self.v, self.a

        if self.order == 3:
            # jerk from the third-order filter (see class docstring)
            j = (wn**3 * (target - x_old)
                 - (2.0 * z + 1.0) * wn**2 * v_old
                 - (2.0 * z + 1.0) * wn * a_old)
            self.x = x_old + dt * v_old
            self.v = v_old + dt * a_old
            self.a = a_old + dt * j
        else:
            # acceleration from the second-order filter, Eq. (2)
            self.a = wn**2 * (target - x_old) - 2.0 * z * wn * v_old
            self.x = x_old + dt * v_old
            self.v = v_old + dt * self.a

        # Optional velocity limit (off by default, rate_limit = None).
        # Not needed in Part 1: peak reference speed for the 50 m four-corner
        # legs is below 1 m/s.
        if self.rate_limit is not None:
            self.v = float(np.clip(self.v, -self.rate_limit, self.rate_limit))

        return self.x, self.v, self.a


class ReferenceModel:
    """Reference model: one _AxisFilter per DOF (N, E, psi)."""

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
            rate_limit=self.cfg_xy.rate_limit, order=self.cfg_xy.order,
        )
        self.filt_E = _AxisFilter(
            dt=self.dt, wn=self.cfg_xy.wn, zeta=self.cfg_xy.zeta,
            rate_limit=self.cfg_xy.rate_limit, order=self.cfg_xy.order,
        )
        self.filt_psi = _AxisFilter(
            dt=self.dt, wn=self.cfg_psi.wn, zeta=self.cfg_psi.zeta,
            rate_limit=self.cfg_psi.rate_limit, order=self.cfg_psi.order,
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
        # unnecessary 2*pi turn"). Works the same for order 2 and 3.
        psi_cmd = eta_cmd[5]
        psi_target_eff = self.filt_psi.x + wrap_angle_pi(psi_cmd - self.filt_psi.x)
        psi_ref_raw, psidot_ref, psiddot_ref = self.filt_psi.step(psi_target_eff)
        psi_ref = wrap_angle_pi(psi_ref_raw)

        # Outputs are NED quantities; the controller rotates nu_ref and
        # acc_ref to BODY with R(psi)^T before using them.
        self.eta_ref = np.zeros(6)
        self.eta_ref[0], self.eta_ref[1], self.eta_ref[5] = N_ref, E_ref, psi_ref

        self.nu_ref = np.zeros(6)
        self.nu_ref[0], self.nu_ref[1], self.nu_ref[5] = Ndot_ref, Edot_ref, psidot_ref

        self.acc_ref = np.zeros(6)
        self.acc_ref[0], self.acc_ref[1], self.acc_ref[5] = Nddot_ref, Eddot_ref, psiddot_ref

        return self.eta_ref, self.nu_ref, self.acc_ref
