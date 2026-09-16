"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np
import scipy.linalg # for the LQR (solves Algebraic Riccati Equation)

class DPController:
    """
    Template for student DP controller.
    [x] PID
    [x] LQR
    [ ] backstepping

    Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs):

            # PID initialization (Pole Placement)
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            # Main diagonal elements from the mass matrix M3
            m_u = 6.007e5  # Surge mass
            m_v = 7.067e5  # Sway mass
            I_z = 5.456e7  # Yaw inertia
            M = np.array([m_u, m_v, I_z])

            # Define desired physical response characteristics
            w_n = np.array([0.1, 0.1, 0.2])         # w_n: natural frequency [rad/s] (determines speed of response)
            zeta = np.array([1.0, 1.0, 1.0])        # zeta: damping ratio (1.0 = critically damped, no overshoot)
            T_i = np.array([100.0, 100.0, 100.0])   # T_i: integral time constant [s]

            # Calculate gains based on second-order system dynamics
            self.Kp = M * (w_n**2)
            self.Kd = 2.0 * M * zeta * w_n
            self.Ki = self.Kp / T_i
            
            # integrator initialization
            self.int_e = np.zeros(3)
            #****************************************************************************

            # LQR initialization
            # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
            A = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, -1.86e-3, 0, 0],
                [0, 0, 0, 0, -0.03176, -0.0241],
                [0, 0, 0, 0, -3.325e-4, -0.035989]
            ])
            
            B = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1.665e-6, 0, 0],
                [0, 1.425e-6, 1.236e-8],
                [0, 1.492e-8, 1.846e-8]
            ])

            # Q Matrix: State limits (Pos: 20m, Heading: 0.1 rad, Vel: 2 m/s)
            Q = np.diag([
                1.0 / (20.0**2),   # Surge error
                1.0 / (20.0**2),   # Sway error
                1.0 / (0.1**2),    # Yaw error
                1.0 / (2.0**2),    # Surge velocity
                1.0 / (2.0**2),    # Sway velocity
                1.0 / (2.0**2)     # Yaw velocity
            ])

            # R Matrix: Thruster limits
            tunnleThruster = 32000.0 # [N]
            backThruster = 2*80000.0 # [N]

            max_tau_x = backThruster
            max_tau_y = backThruster + tunnleThruster
            max_tau_n = tunnleThruster * 12 +  backThruster * 13

            R = np.diag([
                1.0 / (max_tau_x**2),
                1.0 / (max_tau_y**2),
                1.0 / (max_tau_n**2)
            ])

            # Normalize to improve numerical stability in the ARE solver
            scale_factor = max_tau_x**2
            Q_scaled = Q * scale_factor
            R_scaled = R * scale_factor

            # Compute LQR gain
            P = scipy.linalg.solve_continuous_are(A, B, Q_scaled, R_scaled)
            self.K_lqr = np.linalg.inv(R_scaled) @ B.T @ P
            #****************************************************************************

    def reset(self) -> None:
        """Resetta gli stati interni prima di ogni run."""
        self.int_e = np.zeros(3)

    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        
        # extracting variables [surge, sway, yaw]
        psi         = eta[5]
        eta_3       = np.array([    eta[0],     eta[1],     eta[5]])
        eta_ref_3   = np.array([eta_ref[0], eta_ref[1], eta_ref[5]])
        nu_3        = np.array([     nu[0],      nu[1],      nu[5]])

        # computing the error {n}
        e_n = eta_3 - eta_ref_3
        
        # normalizing psi -> [-pi, pi]
        e_n[2] = (e_n[2] + np.pi) % (2 * np.pi) - np.pi

        # converting error form {n} -> {b} ---
        R_yaw = _yaw_matrix(psi)
        e_b = R_yaw.T @ e_n

        nu_ref_3 = np.array([nu_ref[0], nu_ref[1], nu_ref[5]])
        nu_ref_b = R_yaw.T @ nu_ref_3

        # computing velocity error
        e_nu_b = nu_3 - nu_ref_b
        
        # PID
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        # updating integral
        self.int_e += e_b * dt
        
        # computing tau = - Kp*e - Kd*e_dot - Ki*int(e)
        tau_3 = - (self.Kp * e_b) - (self.Kd * e_nu_b) - (self.Ki * self.int_e)
        #****************************************************************************

        # # LQR
        # # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        # # state vector x = [e_b, e_nu_b]^T
        # x_err = np.concatenate([e_b, e_nu_b])
        
        # tau_3 = - self.K_lqr @ x_err

        # # Uncomment the line below to use the LQR output instead of the PID output
        # # tau_3 = tau_3_lqr
        # #****************************************************************************

        # assembling output (6 GdL)
        tau_d = np.zeros(6)
        tau_d[0] = tau_3[0]  # Fx
        tau_d[1] = tau_3[1]  # Fy
        tau_d[5] = tau_3[2]  # Mz

        return tau_d

def _yaw_matrix(psi: float) -> np.ndarray:
    """
    Computes the 3x3 rotation matrix around the Z-axis (yaw).
    """
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])