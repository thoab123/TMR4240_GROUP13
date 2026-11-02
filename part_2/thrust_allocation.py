from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


# Part 2: Constrained Weighted Least Squares (WLS) allocation.

class ThrustAllocator:

    def __init__(self, thrusters: List[ThrusterConfig]):
        
        ## --  SECTION 1: THRUSTER CONFIGURATION -- ##
        
        # Define the thrusters:
        self.thrusters = thrusters
        
        # Check if every thruster is established:
        if len(thrusters) != 3:
            raise ValueError("Expected; one tunnel and two azimuth thrusters -> three thrusters")
        
        # Positions from Table 1 (extended configuration matrix):
        self.B_e = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [12, -3, -13, 3, -13]
        ], dtype = float)
        
        # WLS weights (diagonal weighting):
        self.W = np.diag([1., 1., 1., 1., 1.])   # These weights can be further optimized later.
        
        # Max thrust capacity [N]:
        self.u_max = np.array([32e3, 80e3, 80e3])
        
        # Check if full row rank is satisfied (IMPORTANT!!!):
        if np.linalg.matrix_rank(self.B_e) != 3:   
            raise ValueError(
                "Configuration matrix has insufficient rank"
            )
    
    def _original_wls(self, tau):
        
        ## -- SECTION 2: ORIGINAL WLS SOLUTION -- ##
        
        # Define the inverse diagonal weights:
        w_inv = 1.0 / self.W
        
        # H represents B_e * W^-1 * B_e^T:
        H = (self.B_e * w_inv) @ self.B_e.T
        
        # Define the five allocated thrust components:
        z = w_inv * (
            self.B_e @ np.linalg.solve(H, tau)   # "np.linalg.solve" solves the linear system without needing to calculate the inverse of H
            )
        
        return z

    def _max_limit_restriction(self, z, tol = 1e-5):
        
        ## -- SECTION 3: CHECK MAXIMUM THRUST LIMITS -- ##
        
        # Check whether the allocation method is physically compatible and achievable:
        return (
            abs(z[0]) <= self.u_max[0] + tol
            and np.hypot(z[1], z[2] <= self.u_max[1] + tol)   # "np.hypot" estimates the magnitude of the combined surge and sway forces
            and np.hypot(z[3], z[4] <= self.u_max[2] + tol)
        )
    
    def _new_constrained_wls(self, tau):
        
        ## -- SECTION 4: CONSTRAINED WLS METHOD; OPTIMIZED FOR PART 2 -- ##
        
        # Define initial unconstrained WLS solution: 
        z_0 = self._original_wls(tau)
        
        # Checks if the original WLS method is feasible regarding the limits. If it is, keep it:
        if self._check_max_limit_restriction(z_0):
            return z_0
        
        # Numerical scaling - yaw divided by 10: (in [kN])
        A = self.B_e.copy()
        A[2, :] /= 10.0
        
        target = tau.copy() / 1000
        target[2] /= 10.0
        
        # Define initial guess for constrained optimization: (in [kN])
        v_0 = np.array([
            np.clip(z_0[0] / 1000.0, -32.0, 32.0),
            *np.clip(z_0[1:] / 1000.0, -80.0, 80.0)
        ])
        
        # Define objective function:
        def objective(v):
            return 0.5 * np.dot(self.W * v, v)
        
        # Define gradient of objective function:
        def gradient(v):
            return self.W * v
        
        # Equality constraint, along with inequality constraint for error checking:
        constraints = [
            {
                "type": "eq",
                "fun": lambda v: A @ v - target,
                "jac": lambda v: A
            },
            {
                "type": "ineq",
                "fun": lambda v: 80.0**2 - v[1]**2 - v[2]**2
            },
            {
                "type": "ineq",
                "fun": lambda v: 80.0**2 - v[3]**2 - v[4]**2
            }
        ]
        
        result = minimize(
            objective,
            v_0,
            jac = gradient,
            bounds = [(-32.0, 32.0)] + [(None, None)] * 4,
            constraints = constraints,
            method = "SLSQP",    # Sequential Least Squares Programming.
            options = {
                "ftol": 1e-9,
                "maxiter": 300
            })
        
        # Convertion from [kN] to [N]:
        z = result.x * 1000.0
        
        # Check whether optimization method results are correct or not:
        if (
            not result.success
            or not self._check_max_limit_restriction(z, 0.1)
            or np.linalg.norm(
                A @ result.x - target,
                ord = np.inf
            ) > 1e-4   
        ): 
            return None
        
        return z


    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        
        ## -- SECTION 5: DESIRED 3-DOF WRENCH -- ##
        
        # Define the wrench:
        tau_6 = np.asarray(tau_d, dtype = float)
        
        # Check if this specified wrench contains six limited values:
        if tau_6.shape != (6,) or not  np.all(np.isfinite(tau_6)):
            raise ValueError("tau_d must contain all six restricted/finite values")
        
        # Define the tau vector:
        tau = tau_6[[0, 1, 5]]    # [F_x - Surge force, F_y - Sway force, M_z - Yaw moment]
        
        
        ## -- SECTION 6: CONSTRAINED ALLOCATION SOLVING -- ##
        
        z = self._new_constrained_wls(tau)

        # Infeasible if z is None!
        # Feasible if z is returned - thus, move on to Section 7!
        
        if z is None:
            
            # Scale the requested wrench correctly (using a scaling factor):
            low, high = 0.0, 1.0
            z = np.zeros(5)
            
            # Binary search to find the scaling factor (bisection method):
            for _ in range(28):
                middle = low + (high) / 2.0
                
                # Check scaled wrench:
                check_wrench = self._new_constrained_wls(middle * tau)
                
                if check_wrench is None:
                    high = middle
                else:
                    low = middle
                    z = check_wrench
                
        
        ## -- SECTION 7: CARTESIAN FORCE INTO THRUST & AZIMUTH -- ##
        
        # Define the commanded thrust, u_cmd: 
        u_cmd = np.array([
            z[0],
            np.hypot(z[1], z[2]),
            np.hypot(z[3], z[4])
        ])
        
        # Define the commanded azimuth angles, alpha_cmd:
        alpha_cmd = np.array([
            np.pi / 2,
            np.arctan2(z[2], z[1]),
            np.arctan2(z[4], z[3])
        ])
        
        
        ## -- SECTION 8: MINIMIZE THE AZIMUTH ROTATION -- ##
        
        if alpha_now is not None:
            
            current = np.asarray(alpha_now, dtype = float)
            
            # Check if the current azimuth angles contain three finite values:
            if current.shape != (3, ) or not np.all(np.isfinite(current)):
                raise ValueError("alpha_now must contain all three restricted/finite values")
            
            # Keep the rotation of the azimuth thrusters:
            for i in (1, 2):
                
                # If thrust is zero -> keep the current azimuth angle:
                if abs(u_cmd[i]) < 1e-8:   # Check if the thrust is negligible (close to zero)
                    alpha_cmd[i] = current[i]
                    continue
                
                # Calculate direct rotation:
                direct_rotation = np.arctan2(
                    np.sin(alpha_cmd[i] - current[i]),
                    np.cos(alpha_cmd[i] - current[i])
                )
                
                # Calculate reverse rotation (alternative):
                reverse_rotation = np.arctan2(
                    np.sin(alpha_cmd[i] - current[i] + np.pi),
                    np.cos(alpha_cmd[i] - current[i] + np.pi)
                )
                
                # Select the rotation with the minimum absolute value:
                if abs(reverse_rotation) < abs(direct_rotation):
                    
                    # Reverse the sign of the thrust:
                    u_cmd[i] = -u_cmd[i]
                    
                    # Rotate the azimuth angle by 180 degrees:
                    alpha_cmd[i] += np.pi
                    
        ## -- SECTION 9: WRAP & RETURN THE COMMANDS -- ##        
        
        # Wrap the azimuth angles to the range [-pi, pi]:
        alpha_cmd[:] = np.arctan2(
            np.sin(alpha_cmd[:]), 
            np.cos(alpha_cmd[:])
            )
        
        # Return the thrust and azimuth commands:
        return u_cmd, alpha_cmd

