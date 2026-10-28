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
    eta_n   : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta_n[0], E = eta_n[1], psi = eta_n[5])
    nu_b    : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu_b[0], v = nu_b[1], r = nu_b[5])
    eta_ref_n : (6,) NED reference state
              (use N_d = eta_ref_n[0], E_d = eta_ref_n[1], psi_d = eta_ref_n[5])
    nu_ref_n  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref_n[0], Edot_d = nu_ref_n[1], psidot_d = nu_ref_n[5])
    acc_ref_n : (6,) NED-frame reference accelerations, same layout as nu_ref_n
              (use for model-based / inertia feedforward)

Output:
    tau_d_b   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d_b[0], Fy = tau_d_b[1], Mz = tau_d_b[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied_b, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
"""
import numpy as np
import scipy.linalg

class DPController:
    """
    DP controller using LQR and full kinematic feedforward.
    """

    def __init__(self, *args, **kwargs):
        self.int_n = np.zeros(3)
        self.last_tau_3_b = np.zeros(3)
        self.last_e_n = np.zeros(3)
        self.last_psi = 0.0

        # LQR initialization (9-State Augmented)
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
        A_aug[6:9, 0:3] = np.eye(3)

        B_aug = np.zeros((9, 3))
        B_aug[0:6, :] = B_base

        # Q Matrix: 9 States
        Q = np.diag([
            1.0 / (10.0**2),    
            1.0 / (10.0**2),    
            1.0 / (0.2**2),     

            1.0 / (0.2**2),     
            1.0 / (0.2**2),     
            1.0 / (0.1**2),    

            1.0 / (100.0**2),   
            1.0 / (100.0**2),   
            1.0 / (1**2)        
        ])
                    
        tunnelThruster = 32000.0
        backThruster   = 80000.0

        max_tau_x = 1.5 * backThruster
        max_tau_y = backThruster + tunnelThruster
        max_tau_n = tunnelThruster * 12 + backThruster * 13

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

    def reset(self) -> None:
        self.int_n = np.zeros(3)
        self.last_tau_3_b = np.zeros(3)
        self.last_e_n = np.zeros(3)
        self.last_psi = 0.0

    def compute(
        self,
        t: float,
        dt: float,
        eta_n: np.ndarray,
        nu_b: np.ndarray,
        eta_ref_n: np.ndarray,
        nu_ref_n: np.ndarray | None = None,
        acc_ref_n: np.ndarray | None = None,
    ) -> np.ndarray:
        
        # Handle None references to prevent runtime subscripting errors
        if nu_ref_n is None:
            nu_ref_n = np.zeros(6)
        if acc_ref_n is None:
            acc_ref_n = np.zeros(6)

        # Extract variables [surge, sway, yaw]
        psi = eta_n[5]
        eta_3_n = np.array([eta_n[0], eta_n[1], eta_n[5]])
        eta_ref_3_n = np.array([eta_ref_n[0], eta_ref_n[1], eta_ref_n[5]])
        nu_3_b = np.array([nu_b[0], nu_b[1], nu_b[5]])

        # Compute NED error
        e_n = eta_3_n - eta_ref_3_n
        
        # Normalize yaw error -> [-pi, pi]
        e_n[2] = (e_n[2] + np.pi) % (2 * np.pi) - np.pi

        # Convert error {n} -> {b}
        R_yaw = _yaw_matrix(psi)
        e_b = R_yaw.T @ e_n

        # Convert reference velocities {n} -> {b}
        nu_ref_3_n = np.array([nu_ref_n[0], nu_ref_n[1], nu_ref_n[5]])
        nu_ref_3_b = R_yaw.T @ nu_ref_3_n

        # Convert reference accelerations {n} -> {b} with Coriolis term correction
        acc_ref_3_n = np.array([acc_ref_n[0], acc_ref_n[1], acc_ref_n[5]])
        
        r_ref_b = nu_ref_3_b[2]
        S_ref_b = np.array([
            [0, -r_ref_b, 0],
            [r_ref_b, 0, 0],
            [0, 0, 0]
        ])
        
        acc_ref_3_b = R_yaw.T @ acc_ref_3_n - S_ref_b @ nu_ref_3_b 

        # Compute velocity error {b}
        e_nu_b = nu_3_b - nu_ref_3_b

        # 1. Update unbounded integral in the GLOBAL (NED) frame
        self.int_n += e_n * dt
        self.last_e_n = e_n.copy()
        self.last_psi = psi

        # 2. Rotate the accumulated global integral into the current BODY frame
        int_b = R_yaw.T @ self.int_n

        # 3. Compute feedforward force
        M_b = np.array([6.007e5, 7.067e5, 5.456e7])
        tau_ff_b = M_b * acc_ref_3_b 
        
        # 4. Assemble 9-DOF error vector
        x_err_b = np.concatenate([e_b, e_nu_b, int_b])
        
        # 5. Final control law
        tau_3_b = -self.K_lqr @ x_err_b + tau_ff_b

        # 6. Save for anti-windup back-calculation
        self.last_tau_3_b = tau_3_b

        # Assemble output (6 GdL)
        tau_d_b = np.zeros(6)
        tau_d_b[0] = tau_3_b[0]  
        tau_d_b[1] = tau_3_b[1]  
        tau_d_b[5] = tau_3_b[2]  

        return tau_d_b

    def apply_external_aw(self, tau_applied_b: np.ndarray, psi: float, dt: float) -> None:
        """
        Anti-windup via Clamping with decoupled axes.
        Prevents scalar coupling where saturating surge would freeze sway/yaw integration.
        """
        tau_app_3_b = np.array([tau_applied_b[0], tau_applied_b[1], tau_applied_b[5]])
        tau_error_b = tau_app_3_b - self.last_tau_3_b
        
        # Check saturation independently per body axis (Threshold widened from 100 to 1000 N)
        is_sat_b = np.abs(tau_error_b) > 1000.0
        
        if np.any(is_sat_b):
            R_yaw = _yaw_matrix(self.last_psi)
            e_b = R_yaw.T @ self.last_e_n
            
            # Zero out the integral increment purely on the saturated body axes
            e_b[is_sat_b] = 0.0
            
            # Map the corrected increment back to NED
            corrected_e_n = R_yaw @ e_b
            
            # Replace the unbounded integral step with the clamped one
            self.int_n -= self.last_e_n * dt
            self.int_n += corrected_e_n * dt

def _yaw_matrix(psi: float) -> np.ndarray:
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])