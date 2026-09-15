"""
# Thrust Allocation template

# Students should implement an algorithm that maps the desired body-frame
# wrench to individual thruster commands. The simulator calls, once per step:

#    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

# Inputs (full actuator state — use what your algorithm needs):
#    t         : current simulation time [s]
#    dt        : time step [s]              (rate-aware/dynamic allocation)
#    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
#                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
#                 = [Fx, Fy, Mz]; the other components are zero)
#    u_now     : current actual thrusts [N]     (rate-aware allocation)
#    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

# Outputs:
#    u_cmd     : signed thrust command for each thruster [N]
#    alpha_cmd : thruster angle command for each thruster [rad]

# Students may implement, for example:
#    - pseudo-inverse allocation,
#    - weighted least-squares allocation,
#    - optimization-based allocation,
#    - power-minimizing allocation.
# """

from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig


class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

def allocate(
    self,
    t: float,
    dt: float,
    tau_d: np.ndarray,
    u_now: Optional[np.ndarray] = None,
    alpha_now: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    # Desired 3-DOF wrench
    tau = np.array([
        tau_d[0],  # surge force
        tau_d[1],  # sway force
        tau_d[5],  # yaw moment
    ])

    # Extended configuration matrix
    B_e = np.array([
        [0,  1,   0,  1,   0],
        [1,  0,   1,  0,   1],
        [12, -3, -13, 3, -13],
    ], dtype=float)

    # Weight matrix
    # Larger tunnel weight -> prefer stern azimuths
    W = np.diag([
        10.0,  # tunnel
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    W_inv = np.linalg.inv(W)

    H = B_e @ W_inv @ B_e.T

    z = W_inv @ B_e.T @ np.linalg.solve(H, tau)

    # Extract solution
    u_t = z[0]

    Fx1 = z[1]
    Fy1 = z[2]

    Fx2 = z[3]
    Fy2 = z[4]

    # Convert Cartesian forces -> magnitude + azimuth
    u1 = np.hypot(Fx1, Fy1)
    u2 = np.hypot(Fx2, Fy2)

    alpha1 = np.arctan2(Fy1, Fx1)
    alpha2 = np.arctan2(Fy2, Fx2)

    # Tunnel direction fixed at +90 deg
    alpha_t = np.pi / 2

    # Static limits from project
    MAX_TUNNEL = 32e3   # [N]
    MAX_AZI = 80e3      # [N]

    u_t = np.clip(u_t, -MAX_TUNNEL, MAX_TUNNEL)
    u1 = np.clip(u1, 0, MAX_AZI)
    u2 = np.clip(u2, 0, MAX_AZI)

    u_cmd = np.array([
        u_t,
        u1,
        u2,
    ])

    alpha_cmd = np.array([
        alpha_t,
        alpha1,
        alpha2,
    ])

    return u_cmd, alpha_cmd
