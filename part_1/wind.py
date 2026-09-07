"""
Wind template

Students should compute generalized BODY-frame wind loads:
    tau_w6 = [Fx, Fy, Fz, Mx, My, Mz]

The simulator uses the 3-DOF subset [Fx, Fy, Mz] = tau_w6 indices [0, 1, 5]
and calls, once per step:

    wind.step(t, dt, eta, nu) -> (tau_w6, info)

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (gust spectra, time variation)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5])
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (RELATIVE wind: compute the loads from V_rw = V_wind - V_vessel,
            using the horizontal components nu[0], nu[1])

Outputs:
    tau_w6 : (6,) BODY loads
    info   : optional dict for logging, e.g.
             {"U": ambient speed, "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad)}
             Return {} (or None) if you do not need it.
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from" —
             convert before logging, do not log the raw constructor value.

Wind coefficient data
---------------------
The vessel wind coefficients C(alpha) = [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] are
provided in `data/wind_coeff.csv` (repository root), tabulated against the relative
wind angle alpha in degrees (0..360). Load them with:

    alpha_deg, C6 = load_wind_coefficients()

The wind loads are then computed as F_wind = U_rw^2 * C(alpha_rw), where U_rw
and alpha_rw are the relative wind speed and angle in the BODY frame.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

_WIND_COEFF_FILE = Path(__file__).resolve().parent.parent / "data" / "wind_coeff.csv"


def load_wind_coefficients() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the vessel wind coefficient table.

    Returns
    -------
    alpha_deg : (M,) ndarray
        Relative wind angle grid [deg], from 0 to 360.
    C6 : (M, 6) ndarray
        Coefficients [Cx, Cy, Cz, Cphi, Ctheta, Cpsi] at each angle.
    """
    table = np.loadtxt(_WIND_COEFF_FILE, delimiter=",", skiprows=1)
    return table[:, 0], table[:, 1:]


class Wind:
    """Template for student wind model.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention —
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly-varying wind speed
        component [m/s] (required in Part 1; 0 disables it).
    tau_slow : time constant of the slow variation [s].
    seed : random seed for the slow component, so runs are reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, seed: int | None = None):
        # TODO: Store and use the parameters above in step().
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.seed = seed

        if semantics not in {"from", "towards"}:
            raise ValueError("semantics must be 'from' or 'towards'")

        self.rng = np.random.default_rng(seed)
        self.slow_speed = 0.0
        self.alpha_deg, self.C6 = load_wind_coefficients()

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        # The slow component is an OU process with stationary standard
        # deviation sigma_slow.
        if self.sigma_slow > 0.0 and dt > 0.0:
            decay = np.exp(-dt / self.tau_slow)
            noise_std = self.sigma_slow * np.sqrt(1.0 - decay**2)
            self.slow_speed = (decay * self.slow_speed
                               + noise_std * self.rng.normal())

        # Finds the wind in ned when it goes towards beta
        ambient_speed = max(0.0, self.mean_speed + self.slow_speed)
        beta_towards = self.beta + (np.pi if self.semantics == "from" else 0.0)
        wind_ned = ambient_speed * np.array([
            np.cos(beta_towards), np.sin(beta_towards)
        ])

        # Wind in body frame
        psi = float(eta[5])
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        wind_body = np.array([
            c_psi * wind_ned[0] + s_psi * wind_ned[1],
            -s_psi * wind_ned[0] + c_psi * wind_ned[1],
        ])

        # Relative speed in body and the relaative wind angle
        relative_wind_x = wind_body[0] - nu[0]
        relative_wind_y = wind_body[1] - nu[1]
        relative_speed = np.sqrt(relative_wind_x**2 + relative_wind_y**2)
        alpha_body = float(np.arctan2(relative_wind_y, relative_wind_x))
        alpha_deg = np.degrees(alpha_body) % 360.0

        # Does a linear interpolation to find the correct wind coefficients
        coefficients = np.array([
            np.interp(alpha_deg, self.alpha_deg, coefficient_values)
            for coefficient_values in self.C6.T
        ])

        # Calculating the output
        tau_w6 = relative_speed**2 * coefficients
        info = {
            "U": ambient_speed,
            "beta_ned": float(beta_towards % (2.0 * np.pi)),
            "alpha_body": alpha_body,
        }
        return tau_w6, info
