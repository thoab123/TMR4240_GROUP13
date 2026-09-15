"""
Current template

Students should provide the ambient current as a generalized NED velocity
vector. The simulator calls, once per step:

    current.step(t, dt, eta, nu) -> nu_c_ned

Inputs (full 6-DOF state — use what your model needs):
    t    : current simulation time [s]        (time-varying currents)
    dt   : time step [s]                      (slowly-varying components)
    eta  : (6,) vessel state [N, E, z, phi, theta, psi] in NED
           (heading is eta[5]; position for spatially varying fields)
    nu   : (6,) vessel BODY velocities [u, v, w, p, q, r]
           (only indices [0, 1, 5] are nonzero in the 3-DOF model)

Output:
    nu_c_ned : (6,) generalized NED current velocity [m/s]
               [V_N, V_E, V_D, 0, 0, 0]
               Only the horizontal components are used by the 3-DOF model.
               Direction convention is 'towards' (the direction the current
               flows to): a current with V_N > 0, V_E = 0 pushes the vessel
               North.
"""
import numpy as np


class Current:
    """Student current model: constant speed, direction optionally
    ramping linearly from `beta` to `beta_end` over `duration` seconds
    (used for Simulation 2), then holding steady at `beta_end`.

    Constructor contract — the automated checks (``python check.py``,
    ``pytest``, ``notebooks/part_1_demo.ipynb``) construct your model with
    this signature, so keep it working:

        Current(speed, beta, semantics=..., beta_end=..., duration=...)

    Parameters
    ----------
    speed : current speed [m/s].
    beta : direction [rad] in NED (0 = North, pi/2 = East).
    semantics : ``"towards"`` (default) — ``beta`` is the direction the
        current flows to — or ``"from"`` — the direction it comes from.
    beta_end, duration : if given, the direction varies linearly from
        ``beta`` to ``beta_end`` over ``duration`` seconds (Simulation 2),
        then stays at ``beta_end``.  Constant direction if ``beta_end`` is
        ``None``.
    """

    def __init__(self, speed: float = 0.0, beta: float = 0.0, *,
                 semantics: str = "towards",
                 beta_end: float | None = None, duration: float = 0.0):
        # Store the constructor arguments as-is; step() below is where
        # they actually get turned into a current vector at each call.
        self.speed = float(speed)
        self.beta = float(beta)
        self.semantics = semantics
        self.beta_end = beta_end
        self.duration = float(duration)

    def step(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        # --- 1. Direction at time t -----------------------------------
        # Constant direction unless beta_end was given, in which case we
        # linearly ramp from `beta` at t=0 to `beta_end` at t=duration,
        # then hold at `beta_end` for all later times (Simulation 2:
        # current rotates from "north" to "east" over 300 s).
        if self.beta_end is None:
            beta_t = self.beta
        else:
            # fraction goes from 0 -> 1 over [0, duration], then clamps
            # at 1 so we don't keep rotating past beta_end.
            fraction = min(t / self.duration, 1.0) if self.duration > 0 else 1.0
            beta_t = self.beta + fraction * (self.beta_end - self.beta)

        # --- 2. Resolve to the "towards" convention --------------------
        # nu_c_ned must describe the direction the water FLOWS TOWARDS.
        # If the user instead specified where it comes FROM (e.g. "0.5
        # m/s from east"), flip by 180 degrees to get the actual flow
        # direction. (Report Task 5: state this convention explicitly.)
        if self.semantics == "towards":
            beta_towards = beta_t
        elif self.semantics == "from":
            beta_towards = beta_t + np.pi
        else:
            raise ValueError(f"Unknown semantics: {self.semantics!r}")

        # --- 3. NED components ------------------------------------------
        # Standard compass-style decomposition: beta measured clockwise
        # from North (0 = North, pi/2 = East), so North component uses
        # cos and East component uses sin.
        V_N = self.speed * np.cos(beta_towards)
        V_E = self.speed * np.sin(beta_towards)

        # --- 4. Pack into the 6-DOF interface ----------------------------
        # V_D (vertical current) and the three rotational slots are
        # always zero: we only model a horizontal translational current,
        # never a vertical component or a "current moment".
        return np.array([V_N, V_E, 0.0, 0.0, 0.0, 0.0])