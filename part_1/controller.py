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
        self.int_n = np.zeros(3)
        self.last_tau_3 = np.zeros(3)
        self.last_e_n = np.zeros(3)

        # LQR initialization (9-State Augmented)
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        A_base = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, -1.86e-3, 0, 0],
            [0, 0, 0, 0, -0.03176, -0.0241],
            [0, 0, 0, 0, -3.325e-4, -0.035989]
        ])
        
        B_base = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1.665e-6, 0, 0],
            [0, 1.425e-6, 1.236e-8],
            [0, 1.492e-8, 1.846e-8]
        ])

        # Augment to 9x9 (Adding int_e)
        A_aug = np.zeros((9, 9))
        A_aug[0:6, 0:6] = A_base
        A_aug[6:9, 0:3] = np.eye(3)  # int_e_dot = e_b

        B_aug = np.zeros((9, 3))
        B_aug[0:6, :] = B_base

        # Q Matrix: 9 States
        Q = np.diag([
            1.0 / (10.0**2),    # Surge error (tolleranza rigida: 1 metro massimo)
            1.0 / (10.0**2),    # Sway error (tolleranza rigida: 1 metro massimo)
            1.0 / (0.1**2),     # Yaw error (tolleranza rigida: ~5.7 gradi)

            1.0 / (0.2**2),     # Surge velocity (forte freno dinamico)
            1.0 / (0.2**2),     # Sway velocity (forte freno dinamico)
            1.0 / (0.1**2),    # Yaw velocity (forte freno dinamico)

            1.0 / (100.0**2),   # Integral Surge error 
            1.0 / (100.0**2),   # Integral Sway error
            1.0 / (1**2)        # Integral Yaw error
        ])
                    
        tunnleThruster = 32000.0
        backThruster   = 80000.0

        max_tau_x = 1.5 * backThruster
        max_tau_y = backThruster + tunnleThruster
        max_tau_n = tunnleThruster * 12 + backThruster * 13

        R = np.diag([
            1.0 / (max_tau_x**2),
            1.0 / (max_tau_y**2),
            1.0 / (max_tau_n**2)
        ])

        scale_factor = max_tau_x**2
        Q_scaled = Q * scale_factor
        R_scaled = R * scale_factor

        # Compute 3x9 LQR gain
        P = scipy.linalg.solve_continuous_are(A_aug, B_aug, Q_scaled, R_scaled)
        self.K_lqr = np.linalg.inv(R_scaled) @ B_aug.T @ P

        #****************************************************************************

    def reset(self) -> None:
        """Resetta gli stati interni prima di ogni run."""
        self.int_n = np.zeros(3)
        self.last_tau_3 = np.zeros(3)
        self.last_e_n = np.zeros(3)

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

        acc_ref_3 = np.array([acc_ref[0], acc_ref[1], acc_ref[5]])
        acc_ref_b = R_yaw.T @ acc_ref_3 

        # computing velocity error
        e_nu_b = nu_3 - nu_ref_b

        # LQR + feedforward (9-State)
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
        # 1. Update unbounded integral in the GLOBAL (NED) frame
        self.int_n += e_n * dt
        self.last_e_n = e_n.copy()

        # 2. Rotate the accumulated global integral into the current BODY frame
        int_b = R_yaw.T @ self.int_n

        # 3. Compute feedforward force
        M = np.array([6.007e5, 7.067e5, 5.456e7])
        tau_ff = M * acc_ref_b 
        
        # 4. Assemble 9-DOF error vector using the rotated integral (int_b)
        x_err = np.concatenate([e_b, e_nu_b, int_b])
        
        # 5. Final control law
        tau_3 = -self.K_lqr @ x_err + tau_ff

        # 6. Save for anti-windup back-calculation
        self.last_tau_3 = tau_3
        #****************************************************************************

        # assembling output (6 GdL)
        tau_d = np.zeros(6)
        tau_d[0] = tau_3[0]  # Fx
        tau_d[1] = tau_3[1]  # Fy
        tau_d[5] = tau_3[2]  # Mz

        return tau_d

    def apply_external_aw(self, tau_applied: np.ndarray, psi: float, dt: float) -> None:
        """
        Anti-windup via Clamping (Conditional Integration). 
        Congela l'integratore se i propulsori fisici saturano per evitare falsi accumuli.
        """
        tau_app_3 = np.array([tau_applied[0], tau_applied[1], tau_applied[5]])
        tau_error = tau_app_3 - self.last_tau_3
        
        # Se i propulsori fisici non riescono a seguire la richiesta (saturazione > 100 N)
        if np.max(np.abs(tau_error)) > 100.0:
            # Clamping: annulla l'incremento integrale avvenuto in questo esatto step
            self.int_n -= self.last_e_n * dt

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