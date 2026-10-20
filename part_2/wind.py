"""
Wind model (Project Part 2)

Part 2 extends the Part 1 wind model.  In addition to the mean and slowly
varying wind speed, the wind field now contains

* a fluctuating **gust** component of the wind speed, generated from the
  NORSOK (Froya) wind spectrum (Harris can be selected in ``gust_params``),
  and
* a **slowly varying wind direction**, without a gust component, bounded to
  at most +/- 5 degrees around the mean direction.

The coefficient lookup (relative wind -> loads) is unchanged from Part 1;
only the ambient wind field (speed and direction as functions of time)
becomes richer:

    U_w(t)    = U_mean + U_slow(t) + U_gust(t)
    beta_w(t) = wrap(beta_mean + d_beta(t)),   |d_beta| <= dir_limit

Model summary
-------------
* U_slow : first-order Gauss-Markov (Ornstein-Uhlenbeck) process with
  stationary standard deviation ``sigma_slow`` and time constant ``tau_slow``
  (same as Part 1).
* d_beta : first-order Gauss-Markov process with stationary standard
  deviation ``sigma_dir`` and time constant ``tau_dir``, clipped to
  +/- ``dir_limit``.
* U_gust : zero-mean sum of N harmonics drawn from the gust spectrum S(f),

      U_gust(t) = sum_j sqrt(2 S(f_j) df_j) cos(2 pi f_j t + eps_j),

  with log-spaced frequency bins between ``f_min`` and ``f_max``, one random
  frequency inside each bin (avoids a periodic signal) and random phases
  eps_j.  ``f_min`` is placed above the band already covered by U_slow so
  the low-frequency energy is not counted twice.

All random components use separate random streams spawned from ``seed``, so
switching the gust on/off does not change the slow speed or direction
realisations (needed for paired comparisons).

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
    info   : dict for logging
             {"U": ambient speed (total), "beta_ned": direction (towards, rad),
              "alpha_body": relative wind angle in BODY (rad),
              "U_mean": mean + slowly varying speed, "U_gust": gust speed,
              "dir_offset": slow direction excursion d_beta (rad)}
             NOTE: "beta_ned" is always the direction the wind blows
             TOWARDS, even when the constructor semantics is "from".

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

# Default gust parameters, overridden key by key through ``gust_params``
DEFAULT_GUST_PARAMS = {
    "spectrum": "norsok",   # "norsok" (Froya, NORSOK N-003) or "harris"
    "z": 10.0,              # height of the wind speed U_mean [m] (NORSOK only)
    "n_freq": 200,          # number of harmonic components [-]
    "f_min": 0.002,         # lowest gust frequency [Hz] (period 500 s, above the U_slow band)
    "f_max": 0.5,           # highest gust frequency [Hz] (period 2 s)
    "kappa": 0.0026,        # surface drag coefficient [-] (Harris only)
    "L": 1800.0,            # length scale [m] (Harris only)
}


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


def norsok_spectrum(f: np.ndarray, U0: float, z: float = 10.0) -> np.ndarray:
    """
    NORSOK (Froya) wind spectrum, NORSOK N-003 / DNV-RP-C205.

    Parameters
    ----------
    f : frequency [Hz].
    U0 : 1-hour mean wind speed at 10 m height [m/s] (must be > 0).
    z : height above sea level [m].

    Returns
    -------
    S : one-sided spectral density [(m/s)^2 / Hz].
    """
    f = np.asarray(f, dtype=float)
    n = 0.468
    # Non-dimensional frequency
    f_tilde = 172.0 * f * (z / 10.0) ** (2.0 / 3.0) * (U0 / 10.0) ** (-0.75)
    return (320.0 * (U0 / 10.0) ** 2 * (z / 10.0) ** 0.45
            / (1.0 + f_tilde ** n) ** (5.0 / (3.0 * n)))


def harris_spectrum(f: np.ndarray, U10: float, kappa: float = 0.0026,
                    L: float = 1800.0) -> np.ndarray:
    """
    Harris wind spectrum (Harris 1971, see Fossen 2021).

    Parameters
    ----------
    f : frequency [Hz].
    U10 : mean wind speed at 10 m height [m/s] (must be > 0).
    kappa : surface drag coefficient [-].
    L : length scale [m].

    Returns
    -------
    S : one-sided spectral density [(m/s)^2 / Hz].
    """
    f = np.asarray(f, dtype=float)
    # Non-dimensional frequency
    x = L * f / U10
    return 4.0 * kappa * L * U10 / (2.0 + x ** 2) ** (5.0 / 6.0)


class Wind:
    """Part 2 wind model: Part 1 loads + gust and slowly varying direction.

    Constructor contract -- the automated checks (``python check.py``,
    ``pytest``) and the Part 2 runner construct your model with this
    signature, so keep it working.  It is a superset of the Part 1 contract.

        Wind(mean_speed, beta, semantics=..., sigma_slow=..., tau_slow=...,
             gust=..., gust_params=..., sigma_dir=..., tau_dir=...,
             dir_limit=..., seed=...)

    Parameters
    ----------
    mean_speed : mean wind speed [m/s].
    beta : mean direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"from"`` (default, the usual meteorological convention --
        "wind from south" blows northward) or ``"towards"``.
    sigma_slow : standard deviation of the slowly varying wind speed
        component [m/s] (Part 1; 0 disables it).
    tau_slow : time constant of the slow speed variation [s].
    gust : switch the fluctuating (gust) speed component on.  ``False``
        gives the Part 1 wind field.
    gust_params : dict overriding entries of ``DEFAULT_GUST_PARAMS``
        (spectrum, z, n_freq, f_min, f_max, kappa, L).  ``None`` uses the
        defaults (NORSOK, 200 harmonics, 0.002-0.5 Hz).
    sigma_dir : standard deviation of the slowly varying wind direction
        [rad] (0 disables it).  The direction has NO gust component.
    tau_dir : time constant of the slow direction variation [s].
    dir_limit : hard bound on the direction excursion |beta(t) - beta| [rad];
        the project requires at most 5 degrees.  The excursion is clipped.
    seed : random seed for all stochastic components, so runs are
        reproducible.
    """

    def __init__(self, mean_speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "from", sigma_slow: float = 0.0,
                 tau_slow: float = 120.0, gust: bool = False,
                 gust_params: Dict | None = None, sigma_dir: float = 0.0,
                 tau_dir: float = 300.0, dir_limit: float = np.deg2rad(5.0),
                 seed: int | None = None):
        self.mean_speed = float(mean_speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.sigma_slow = float(sigma_slow)
        self.tau_slow = float(tau_slow)
        self.gust = bool(gust)
        self.sigma_dir = float(sigma_dir)
        self.tau_dir = float(tau_dir)
        self.dir_limit = float(dir_limit)
        self.seed = seed

        if semantics not in {"from", "towards"}:
            raise ValueError("semantics must be 'from' or 'towards'")
        if self.tau_slow <= 0.0 or self.tau_dir <= 0.0:
            raise ValueError("tau_slow and tau_dir must be positive")
        if self.dir_limit < 0.0:
            raise ValueError("dir_limit must be non-negative")

        # Merges the user gust parameters into the defaults (catches typos)
        unknown = set(gust_params or {}) - set(DEFAULT_GUST_PARAMS)
        if unknown:
            raise ValueError(f"unknown gust_params keys: {sorted(unknown)}")
        self.gust_params = {**DEFAULT_GUST_PARAMS, **(gust_params or {})}

        # Separate random streams for slow speed, slow direction and gust
        seeds = np.random.SeedSequence(seed).spawn(3)
        self.rng_speed, self.rng_dir, rng_gust = (np.random.default_rng(s) for s in seeds)

        # States of the slowly varying processes
        self.slow_speed = 0.0
        self.dir_offset = 0.0
        self.alpha_deg, self.C6 = load_wind_coefficients()

        # Gust harmonics are drawn once, so the gust is a fixed function of t
        self.gust_omega, self.gust_amp, self.gust_phase = self._build_gust(rng_gust)

    def _build_gust(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Discretise the gust spectrum into harmonics (omega, amplitude, phase)."""
        p = self.gust_params
        n = int(p["n_freq"])
        f_min, f_max = float(p["f_min"]), float(p["f_max"])
        if n < 1 or not 0.0 < f_min < f_max:
            raise ValueError("gust_params need n_freq >= 1 and 0 < f_min < f_max")

        # Log-spaced bins give fine resolution near the spectral peak (~0.005 Hz)
        edges = np.geomspace(f_min, f_max, n + 1)
        df = np.diff(edges)
        # One random frequency inside each bin, avoids a periodic gust signal
        f = edges[:-1] + rng.uniform(0.0, 1.0, n) * df
        # Random phases, uniformly distributed
        phase = rng.uniform(0.0, 2.0 * np.pi, n)

        # No gust energy when the gust is off or there is no mean wind
        if not self.gust or self.mean_speed <= 0.0:
            return 2.0 * np.pi * f, np.zeros(n), phase

        # Evaluates the chosen spectrum at the component frequencies
        spectrum = str(p["spectrum"]).lower()
        if spectrum == "norsok":
            S = norsok_spectrum(f, self.mean_speed, float(p["z"]))
        elif spectrum == "harris":
            S = harris_spectrum(f, self.mean_speed, float(p["kappa"]), float(p["L"]))
        else:
            raise ValueError("gust_params['spectrum'] must be 'norsok' or 'harris'")

        # Amplitude of each harmonic from the energy in its bin
        amp = np.sqrt(2.0 * S * df)
        return 2.0 * np.pi * f, amp, phase

    def gust_std(self) -> float:
        """Theoretical standard deviation of the discretised gust [m/s]."""
        return float(np.sqrt(0.5 * np.sum(self.gust_amp ** 2)))

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        # Slow speed: OU process with stationary standard deviation sigma_slow
        if self.sigma_slow > 0.0 and dt > 0.0:
            decay = np.exp(-dt / self.tau_slow)
            noise_std = self.sigma_slow * np.sqrt(1.0 - decay**2)
            self.slow_speed = (decay * self.slow_speed
                               + noise_std * self.rng_speed.normal())

        # Slow direction: OU process, clipped to +/- dir_limit
        if self.sigma_dir > 0.0 and dt > 0.0:
            decay = np.exp(-dt / self.tau_dir)
            noise_std = self.sigma_dir * np.sqrt(1.0 - decay**2)
            self.dir_offset = (decay * self.dir_offset
                               + noise_std * self.rng_dir.normal())
            self.dir_offset = float(np.clip(self.dir_offset,
                                            -self.dir_limit, self.dir_limit))

        # Gust: sum of the harmonics evaluated at time t
        gust_speed = (float(np.sum(self.gust_amp * np.cos(self.gust_omega * t + self.gust_phase)))
                      if self.gust else 0.0)

        # Mean + slow speed, and total speed (both kept non-negative)
        mean_slow_speed = max(0.0, self.mean_speed + self.slow_speed)
        ambient_speed = max(0.0, mean_slow_speed + gust_speed)
        # Logged gust, so that U = U_mean + U_gust holds exactly
        gust_logged = ambient_speed - mean_slow_speed

        # Finds the wind in ned when it goes towards beta (mean + slow direction)
        beta_towards = (self.beta + self.dir_offset
                        + (np.pi if self.semantics == "from" else 0.0))
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

        # Relative speed in body and the relative wind angle
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
            "U_mean": mean_slow_speed,
            "U_gust": gust_logged,
            "dir_offset": self.dir_offset,
        }
        return tau_w6, info
