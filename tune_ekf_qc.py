"""Q_c tuning study: run observer comparison across different process noise configurations.

Usage:
    python tune_ekf_qc.py

This script tests multiple Q_c wave-component intensities and tabulates key performance
metrics to justify tuning choices. Metrics include RMS position/velocity errors,
an error-spectrum ratio, and observer convergence rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from part_2.config import Part2SimConfig, default_thrusters_part_2
from part_2.observer import select_observer
from part_2.reference import ReferenceModel
from part_2.thrust_allocation import ThrustAllocator
from part_2.controller import DPController
from part_2.current import Current
from part_2.wind import Wind
from models.waves import Waves
from simulation.simulation_part_2 import DPSimulatorPart2

# Separate tuning scenario, not the prescribed Simulation 4 environment.
DT = 0.1
CURRENT_SPEED, CURRENT_FROM = 0.3, np.deg2rad(180.0)
WIND_SPEED, WIND_FROM = 12.0, np.deg2rad(180.0)
WAVES_HS, WAVES_TP, WAVES_FROM, WAVES_N = 2.0, 8.0, np.deg2rad(180.0), 100
SEED = 42


def make_kalman_observer_with_qc(qc_wave: float, qc_bias: tuple[float, float, float]):
    """Create Kalman observer with custom Q (process noise covariance).
    
    The Kalman observer has 15 states: 6 wave states, 3 LF pose, 3 velocity, 3 bias.
    Q is 6×6: the first 3 are wave inputs (excitation of wave oscillators),
    the last 3 are bias inputs (environmental force noise).
    
    Args:
        qc_wave: Process noise intensity for wave oscillators (scalar, used for N, E, psi)
        qc_bias: Tuple of (q_N, q_E, q_psi) for environmental bias states
    
    Returns:
        KalmanFilterObserver with modified Q
    """
    from part_2.observer import KalmanFilterObserver
    
    obs = KalmanFilterObserver()
    # Q is 6×6: [wave_N, wave_E, wave_psi, bias_N, bias_E, bias_psi]
    q_wave_psi = (np.pi / 180) ** 2  # Convert 1° to rad²
    obs.Q = np.diag([
        qc_wave, qc_wave, qc_wave * q_wave_psi,     # Wave-filter noise (excitation of oscillators)
        *qc_bias,                                    # Environmental bias noise
    ])
    return obs


def compute_metrics(logs, dominant_freq: float = 0.125) -> dict:
    """Extract performance metrics from simulation logs.
    
    Args:
        logs: Simulation log object
        dominant_freq: Expected wave frequency [Hz]
    
    Returns:
        Dictionary of metrics
    """
    dt = logs.t[1] - logs.t[0] if len(logs.t) > 1 else 0.1
    
    # RMS position and velocity errors
    pos_err = np.linalg.norm(logs.eta_est[:, :2] - logs.eta[:, :2], axis=1)
    vel_err = np.linalg.norm(logs.nu_est[:, :2] - logs.nu[:, :2], axis=1)
    heading_err = np.abs(np.unwrap(logs.eta_est[:, 5] - logs.eta[:, 5]))
    
    rms_pos = np.sqrt(np.mean(pos_err ** 2))
    rms_vel = np.sqrt(np.mean(vel_err ** 2))
    rms_heading = np.sqrt(np.mean(heading_err ** 2))
    
    # Error-spectrum ratio, not estimate/true wave attenuation.
    # Compute position error PSD
    freq, pxx = signal.welch(pos_err, fs=1/dt, nperseg=min(2048, len(pos_err)//2))
    
    # Find attenuation at dominant wave frequency
    idx_wave = np.argmin(np.abs(freq - dominant_freq))
    idx_lf = np.argmin(np.abs(freq - 0.02))  # Low-frequency reference
    
    attenuation_db = 10 * np.log10(pxx[idx_wave] / pxx[idx_lf]) if pxx[idx_lf] > 0 else np.nan
    
    # Convergence: time until error drops below 0.5m (steady state)
    threshold = 0.5
    converged_idx = np.where(pos_err < threshold)[0]
    convergence_time = logs.t[converged_idx[0]] if len(converged_idx) > 0 else logs.t[-1]
    
    # NIS consistency (if Kalman, track mean and bounds)
    nis_mean = np.nan
    nis_in_bounds = np.nan
    if hasattr(logs, 'nis') and logs.nis is not None:
        nis_mean = np.mean(logs.nis)
        # 3-sigma bound: mean should be ~6 for 3 dof
        nis_in_bounds = 3.0 < nis_mean < 9.0
    
    return {
        'rms_pos_m': rms_pos,
        'rms_vel_ms': rms_vel,
        'rms_heading_deg': np.rad2deg(rms_heading),
        'wave_atten_db': attenuation_db,
        'conv_time_s': convergence_time,
        'nis_mean': nis_mean,
        'nis_consistent': nis_in_bounds,
    }


def run_qc_tuning_study(qc_configs: list[tuple[float, tuple]]) -> pd.DataFrame:
    """Run observer comparison across multiple Q_c configurations.
    
    Args:
        qc_configs: List of (qc_wave, (qc_N, qc_E, qc_psi)) tuples
    
    Returns:
        DataFrame with metrics for each configuration
    """
    from run_case_part_2 import WrenchSequence, make_sim
    
    results = []
    
    for qc_wave, qc_bias in qc_configs:
        print(f"\nTesting Q_c wave={qc_wave}, bias={qc_bias}...")
        
        # Setup simulation with waves
        cfg = Part2SimConfig(dt=DT, T=600.0, use_reference=False, use_observer=True,
                             observer_type="kalman")
        
        # Create fixed controller (constant wrench)
        fixed_controller = WrenchSequence([(1.0, [2e3, 2e3, 2e3])])
        
        # Create simulator with custom observer Q and fixed controller
        obs = make_kalman_observer_with_qc(qc_wave, qc_bias)
        sim = make_sim(cfg, fixed_controller)
        sim.observer = obs  # Replace observer with tuned version
        
        # Run with waves
        current = Current(CURRENT_SPEED, CURRENT_FROM, semantics="from")
        wind = Wind(WIND_SPEED, WIND_FROM, semantics="from", sigma_slow=0.5, gust=True,
                    sigma_dir=np.deg2rad(2.0), seed=SEED)
        waves = Waves(hs=WAVES_HS, tp=WAVES_TP, direction=WAVES_FROM, 
                      n_components=WAVES_N, seed=SEED)
        
        sim.reset_state()
        logs = sim.run(np.zeros(6), current=current, wind=wind, waves=waves)
        
        # Compute metrics
        metrics = compute_metrics(logs)
        
        # Store result
        result = {
            'Q_c_wave': qc_wave,
            'Q_c_N': qc_bias[0],
            'Q_c_E': qc_bias[1],
            'Q_c_psi': qc_bias[2],
            **metrics,
        }
        results.append(result)
        print(f"  RMS pos: {metrics['rms_pos_m']:.3f}m, "
              f"Error-spectrum ratio: {metrics['wave_atten_db']:.1f}dB, "
              f"Conv: {metrics['conv_time_s']:.1f}s")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    
    study_type = sys.argv[1] if len(sys.argv) > 1 else "wave"
    
    if study_type == "wave":
        # Tune ONLY wave component (bias fixed at 1.0)
        qc_tunings = [
            (0.1, (1000, 1000, 100000)),
            (0.5, (1000, 1000, 100000)),
            (1.0, (1000, 1000, 100000)),
            (2.0, (1000, 1000, 100000)),
            (5.0, (1000, 1000, 100000)),
        ]
        config_name = "Wave Component (Q_c_wave only)"
        
    elif study_type == "bias":
        # Tune ONLY bias component (wave fixed at 1.0)
        qc_tunings = [
            (1.0, (100, 100, 10000)),
            (1.0, (500, 500, 50000)),
            (1.0, (1000, 1000, 100000)),
            (1.0, (2000, 2000, 200000)),
            (1.0, (5000, 5000, 500000)),
        ]
        config_name = "Bias Component (Q_c_N, Q_c_E, Q_c_psi only)"
        
    else:  # coupled (default)
        # Original coupled scaling
        qc_tunings = [
            (0.1, (100, 100, 10000)),
            (0.5, (500, 500, 50000)),
            (1.0, (1000, 1000, 100000)),
            (2.0, (2000, 2000, 200000)),
            (5.0, (5000, 5000, 500000)),
        ]
        config_name = "Coupled Scaling (wave and bias proportional)"
    
    print("=" * 80)
    print(f"Q_c Tuning Study: {config_name}")
    print("=" * 80)
    print("\nConfiguration: Fixed desired wrench [2000 N, 2000 N, 2000 Nm]; run/evaluation 0-600 s")
    print(f"Current={CURRENT_SPEED} m/s; wind={WIND_SPEED} m/s; waves Hs={WAVES_HS} m, Tp={WAVES_TP} s")
    print(f"All environmental directions: from south (180 deg); seed={SEED}; wave components={WAVES_N}; dt={DT} s")
    print("Error-spectrum ratio R_e: 10*log10(S_e(0.125 Hz)/S_e(0.02 Hz)), nearest Welch bins.")
    print("e is the horizontal estimate-minus-true error norm; this is not wave attenuation.")
    print(f"\nRunning {len(qc_tunings)} configurations...\n")
    
    # Run study
    df = run_qc_tuning_study(qc_tunings)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    
    # Format for readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # Show key metrics
    summary_cols = ['Q_c_wave', 'rms_pos_m', 'rms_vel_ms', 'rms_heading_deg', 
                    'wave_atten_db']
    if study_type == "bias":
        summary_cols = ['Q_c_N', 'rms_pos_m', 'rms_vel_ms', 'rms_heading_deg', 
                        'wave_atten_db']
    
    # Rename columns for display (RMS -> RMSE for clarity)
    display_df = df[summary_cols].rename(columns={
        'rms_pos_m': 'RMSE_pos_m',
        'rms_vel_ms': 'RMSE_vel_ms',
        'rms_heading_deg': 'RMSE_heading_deg',
        'wave_atten_db': 'Error_spectrum_ratio_dB'
    })
    print("\n" + display_df.to_string(index=False))
    
    print(f"\n{'='*80}")
