""" Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design. """

from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig
from scipy.optimize import minimize


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
        w_inv = 1.0 / np.diag(self.W)
        
        # H represents B_e * W^-1 * B_e^T:
        H = (self.B_e * w_inv) @ self.B_e.T
        
        # Define the five allocated thrust components:
        z = w_inv * (
            self.B_e.T @ np.linalg.solve(H, tau)   # "np.linalg.solve" solves the linear system without needing to calculate the inverse of H
            )
        
        return z

    def _max_limit_restriction(self, z, tol=1e-5):
        
        ## -- SECTION 3: CHECK MAXIMUM THRUST LIMITS -- ##
        
        # Check whether the allocation method is physically compatible and achievable:
        return (
            abs(z[0]) <= self.u_max[0] + tol
            and np.hypot(z[1], z[2]) <= self.u_max[1] + tol
            and np.hypot(z[3], z[4]) <= self.u_max[2] + tol
        )
    
    def _feasible_scale_unconstrained(self, tau):
        
        ## -- SECTION 3b: ANALYTIC LOWER BOUND FOR THE SCALING FACTOR (speed-up) -- ##
        
        # The unconstrained WLS solution is linear in tau, so the largest factor s <= 1
        # for which s * z_0 respects every thrust limit follows directly:
        z_0 = self._original_wls(tau)
        magnitudes = np.array([
            abs(z_0[0]),
            np.hypot(z_0[1], z_0[2]),
            np.hypot(z_0[3], z_0[4])
        ])
        with np.errstate(divide = "ignore"):
            factors = np.where(magnitudes > 0, self.u_max / magnitudes, np.inf)
        return min(1.0, float(factors.min())), z_0

    def _new_constrained_wls(self, tau):
        
        ## -- SECTION 4: CONSTRAINED WLS METHOD; OPTIMIZED FOR PART 2 -- ##
        
        # Define initial unconstrained WLS solution: 
        z_0 = self._original_wls(tau)
        
        # Checks if the original WLS method is feasible regarding the limits. If it is, keep it:
        if self._max_limit_restriction(z_0):
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
            return 0.5 * v @ self.W @ v
        
        # Define gradient of objective function:
        def gradient(v):
            return self.W @ v
        
        # Equality constraint, along with inequality constraint for error checking:
        constraints = [
            {
                "type": "eq",
                "fun": lambda v: A @ v - target,
                "jac": lambda v: A
            },
            {
                "type": "ineq",
                "fun": lambda v: 80.0**2 - v[1]**2 - v[2]**2,
                # Analytic Jacobian (speed-up: avoids numerical differencing at every iteration)
                "jac": lambda v: np.array([0.0, -2.0 * v[1], -2.0 * v[2], 0.0, 0.0])
            },
            {
                "type": "ineq",
                "fun": lambda v: 80.0**2 - v[3]**2 - v[4]**2,
                # Analytic Jacobian (speed-up)
                "jac": lambda v: np.array([0.0, 0.0, 0.0, -2.0 * v[3], -2.0 * v[4]])
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
                "ftol": 1e-6,     # was 1e-9; 1e-6 kN is far below what the thrusters can resolve
                "maxiter": 30     # was 300; feasible solves need at most 21 iterations (measured,
                                  # mean 2.8), so 30 keeps a margin while infeasible solves give
                                  # up sooner (they were 92 % of the runtime)
            })
        
        # Convertion from [kN] to [N]:
        z = result.x * 1000.0
        
        # Check whether optimization method results are correct or not:
        if (
            not result.success
            or not self._max_limit_restriction(z, 0.1)
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
            
            # Scale the requested wrench correctly (using a scaling factor).
            # Speed-up: start from the analytic lower bound instead of 0, so every factor
            # below it needs no optimisation, and use 7 steps (precision < 1 %) instead of 28:
            low, z_0 = self._feasible_scale_unconstrained(tau)
            high = 1.0
            z = low * z_0
            
            # Binary search to find the scaling factor (bisection method):
            for _ in range(7):
                middle = (low + high) / 2.0
                
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
