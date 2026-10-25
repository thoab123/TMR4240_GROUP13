"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    # Minimal 6-DOF Nonlinear Passive Observer

    name = "nonlinear_passive"

    def __init__(
        self,
        m: tuple[float, ...] = (
            5.0e5,  # surge [kg]
            7.5e5,  # sway  [kg]
            1.0e6,  # heave [kg]
            1.0e7,  # roll  [kg*m^2]
            1.0e8,  # pitch [kg*m^2]
            6.0e7,  # yaw   [kg*m^2]
        ),

        d: tuple[float, ...] = (
            5.0e4,  # surge [N*s/m]
            1.0e5,  # sway  [N*s/m]
            2.0e5,  # heave [N*s/m]
            5.0e6,  # roll  [N*m*s/rad]
            1.0e7,  # pitch [N*m*s/rad]
            1.0e7,  # yaw   [N*m*s/rad]
        ),
        
        w0: float = 0.8,    # Peak wave frequency [rad/s] (typically 0.6 - 1.0 rad/s in JONSWAP)
        wc: float = 1.1,    # Cutoff frequency (> w0)
        zeta: float = 0.1,  # Wave filter relative damping
        k3: float = 5.0e4,  # Scale k3 with vessel bias force scale
        k4: float = 5.0e5,  # k4 ≈ 0.5 * m * wc
        T_b: float = 100.0,
        **kwargs,
    ):
        
        self.M = np.asarray(m, dtype=float)
        self.D = np.asarray(d, dtype=float)
        self.w0, self.wc, self.zeta = w0, wc, zeta
        self.T_b = T_b
        self.k4 = 0.5 * self.wc * self.M
        self.k3 = 0.05 * self.k4
        self.reset()

    def reset(self, eta0: np.ndarray | None = None):
        self.xi1 = np.zeros(6)  # Wave filter state 1
        self.xi2 = np.zeros(6)  # Wave filter state 2 (wave pose estimate eta_w)
        self.eta = np.zeros(6) if eta0 is None else np.asarray(eta0, dtype=float).reshape(6).copy()
        self.nu = np.zeros(6)
        self.bias = np.zeros(6)
        self._init = eta0 is not None

    def step(self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray) -> ObserverEstimate:
        y = np.asarray(eta_measured, dtype=float).reshape(6)
        tau = np.asarray(tau_est, dtype=float).reshape(6)
        if not self._init:
            self.reset(y)

        # 1. Estimation error (wrap roll, pitch, yaw to [-pi, pi])
        y_tilde = y - (self.eta + self.xi2)
        y_tilde[3:6] = (y_tilde[3:6] + np.pi) % (2 * np.pi) - np.pi

        # 2. Simple yaw-only rotation matrix J(psi) (assumes small roll & pitch)
        c, s = np.cos(y[5]), np.sin(y[5])
        J = np.eye(6)
        J[:2, :2] = [[c, -s], [s, c]]

        # 3. Wave filter gains (Fossen standard formulas)
        k1_1 = -2.0 * (1.0 - self.zeta) * (self.wc / self.w0)
        k1_2 = 2.0 * self.w0 * (1.0 - self.zeta)
        k2 = self.wc

        # 4. Forward Euler step (x_next = x + dt * x_dot)
        d_xi1 = self.xi2 + k1_1 * y_tilde
        d_xi2 = -(self.w0**2) * self.xi1 - 2.0 * self.zeta * self.w0 * self.xi2 + k1_2 * y_tilde
        d_eta = J @ self.nu + k2 * y_tilde
        d_bias = -(self.bias / self.T_b) + self.k3 * y_tilde
        d_nu = (-self.D * self.nu + J.T @ (self.bias + self.k4 * y_tilde) + tau) / self.M

        self.xi1 += dt * d_xi1
        self.xi2 += dt * d_xi2
        self.eta += dt * d_eta
        self.bias += dt * d_bias
        self.nu += dt * d_nu
        self.eta[3:6] = (self.eta[3:6] + np.pi) % (2 * np.pi) - np.pi

        return ObserverEstimate(self.eta.copy(), self.nu.copy(), self.bias.copy())


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the Kalman filter or extended Kalman filter.
        return self._placeholder(eta_measured)


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc
