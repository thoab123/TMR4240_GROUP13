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

    name = "nonlinear_passive"

    def __init__(self, *args, **kwargs):
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the nonlinear passive observer.
        return self._placeholder(eta_measured)


class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        self.n_states = 15
        self.x = np.zeros(self.n_states)
        self.P = np.eye(self.n_states) * 1.0 # Initial covariance, need to tune

        # Tuning parameters (starting guesses)
        self.w0 = 2 * np.pi / 8.0   # dominant wave frequency [rad/s], Tp = 8 s design sea
        self.lam = 0.1              # wave relative damping ratio [-]
        self.K_w = 0.1              # wave noise gain (tune with Q)
        self.Tb = 1000.0            # bias time constant [s]

        # Noise covariances (starting guesses, need tuning); Q is a continuous-time intensity
        self.Q = np.diag([1e-6] * 3 + [1e-2] * 3 + [1e-6] * 3 + [1e-4] * 3 + [1e2] * 3)
        self.R = np.diag([0.1**2, 0.1**2, np.deg2rad(0.5)**2])

        # Writing the M and D matrices, maybe it should be somewhere else? 
        # These are given in the assignment
        self.M = np.array([
            [6.007e5, 0.0, 0.0],
            [0.0, 7.067e5, -4.733e5],
            [0.0, -5.712e5, 5.456e7]
        ])
        self.Minv = np.linalg.inv(self.M)
        self.D = np.array([
            [1.118e3, 0.0, 0.0],
            [0.0, 2.229e4, 0.0],
            [0.0, 0.0, 1.950e6]
        ])

    #reset the kalman filter when done
    def reset(self, eta0: np.ndarray | None = None):
        self.x = np.zeros(self.n_states)
        if eta0 is not None:
            # Initialize LF pose part of state vector (indices for eta: 6 to 8)
            self.x[6] = eta0[0]  # N
            self.x[7] = eta0[1]  # E
            self.x[8] = eta0[5]  # psi
        self.P = np.eye(self.n_states) * 0.1

    def wave_matrices(self):
        # Calculate the wave model, have 3 DOF and 6 states
        I3, Z3 = np.eye(3), np.zeros((3, 3))

        Om = self.w0 * I3
        Lam = self.lam * I3
        Aw = np.block([[Z3, I3], [-Om @ Om, -2.0 * Lam @ Om]]) 
        Ew = np.vstack([Z3, self.K_w * I3])  # We have white noise in the WF
        Cw = np.hstack([Z3, I3]) # Wave frequency is the three last states in xi
        return Aw, Ew, Cw

    def f(self, x, tau):
        # x_dot = f*x, Nonlinear trensformation
        Aw, _, _ = self.wave_matrices()
        xi, eta, nu, b = x[0:6], x[6:9], x[9:12], x[12:15]
        c, s = np.cos(eta[2]), np.sin(eta[2])
        J = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        # Equation 11 to 13 in the project description
        xi_dot = Aw @ xi
        eta_dot = J @ nu
        nu_dot = self.Minv @ (-self.D @ nu + J.T @ b + tau)
        b_dot = -b / self.Tb

        #Merge all the arrays into one large 15 state array
        return np.concatenate([xi_dot, eta_dot, nu_dot, b_dot])

    def F(self, x, tau):
        # Calculate the jacobian of f, We want to linearize by first order taylor
        Aw, _, _ = self.wave_matrices()
        eta, nu, b = x[6:9], x[9:12], x[12:15]
        psi = eta[2]
        c, s = np.cos(psi), np.sin(psi)
        J = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        dJ = np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])  # dJ/dpsi

        # The derivative of eq 11 - 13
        F = np.zeros((15, 15)) # Starting with a zero matrix and filling in the non zero values
        F[0:6, 0:6] = Aw                                 # (d/dxi)*xi_dot = Aw
        F[6:9, 9:12] = J                                 # (d/dnu)*eta_dot = J
        F[6:9, 8] = dJ @ nu                              # (d/dpsi)*eta_dot = dJ*nu
        F[9:12, 8] = self.Minv @ (dJ.T @ b)              # (d/dpsi)*nu_dot = ...
        F[9:12, 9:12] = -self.Minv @ self.D              # (d/dnu)*nu_dot = ...
        F[9:12, 12:15] = self.Minv @ J.T                 # (d/db)*nu_dot = ...
        F[12:15, 12:15] = -np.eye(3) / self.Tb           # (d/db)*b_dot
        return F

    def h(self):
        # h is already linear
        _, _, Cw = self.wave_matrices()
        H = np.zeros((3, 15))
        H[:, 0:6] = Cw
        H[:, 6:9] = np.eye(3)
        return H

    def step(self, t, dt, eta_measured, tau_est):
        # Extracting the 3 DOF
        z_meas = np.array([eta_measured[0], eta_measured[1], eta_measured[5]])
        tau_3dof = np.array([tau_est[0], tau_est[1], tau_est[5]])

        # Prediction (Euler)
        F = np.eye(self.n_states) + dt * self.F(self.x, tau_3dof) # Linearized f by taylor
        x_pred = self.x + dt * self.f(self.x, tau_3dof)
        P_pred = F @ self.P @ F.T + self.Q * dt

        # Predicted measurement
        H = self.h()
        z_pred = H @ x_pred

        # Innovation
        innovation = z_meas - z_pred
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2])) # Need to wrap the heading angle

        # Correction
        S = H @ P_pred @ H.T + self.R
        K = np.linalg.solve(S, H @ P_pred).T            # P H^T S^-1
        self.x = x_pred + K @ innovation
        self.x[8] = np.arctan2(np.sin(self.x[8]), np.cos(self.x[8]))
        IKH = np.eye(self.n_states) - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T   # Joseph form

        # Assemble 6-DOF outputs
        eta_hat = np.asarray(eta_measured, dtype=float).reshape(6).copy()
        eta_hat[0], eta_hat[1], eta_hat[5] = self.x[6], self.x[7], self.x[8]
        nu_hat = np.zeros(6)
        nu_hat[0], nu_hat[1], nu_hat[5] = self.x[9], self.x[10], self.x[11]
        bias_hat = np.zeros(6)
        bias_hat[0], bias_hat[1], bias_hat[5] = self.x[12], self.x[13], self.x[14]
        return ObserverEstimate(eta=eta_hat, nu=nu_hat, bias=bias_hat)



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

