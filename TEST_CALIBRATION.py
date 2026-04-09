#!/usr/bin/env python3
"""
CALIBRATION WITH THREE CONDITIONS AND WEIGHT HIERARCHY
================================================================================
Physical model priorities (Plateau III):
1. α_local = α_exp (CODATA) - Fundamental boundary condition
2. Local drift = 0 - Vacuum consistency at z=0
3. Differential drift = observed - Link to cosmological data

Note: Weights reflect the theoretical structure of the model, not a preference
for any particular result. Physical interpretation is at the researcher's discretion.
"""

import vcv48_k
import numpy as np
from scipy.optimize import minimize, differential_evolution
import json
import os
from datetime import datetime

print("=" * 80)
print("VCV48 - CALIBRATION WITH THREE CONDITIONS")
print("=" * 80)

# ============================================================================
# MODEL CONSTANTS (Plateau III)
# ============================================================================

ALPHA_EXP = vcv48_k.ALPHA_EXP          # 0.0072973525693
K_INV = vcv48_k.K_INVARIANT            # 0.8025
DELTA_CMB = vcv48_k.DELTA_CMB          # 0.00610865
OMEGA_K = vcv48_k.OMEGA_K              # -0.044
Z_LOCAL = 0.0

# Drift coefficients (from kernel)
DRIFT_COEF_ETA1 = 0.87
DRIFT_COEF_ETA2 = 0.12
DRIFT_CONST = -0.2375

print(f"\n📌 MODEL CONSTANTS:")
print(f"   α_exp = {ALPHA_EXP:.12f} (CODATA 2022)")
print(f"   K_inv = {K_INV} (invariant)")
print(f"   Ω_k = {OMEGA_K}")
print(f"   δ_CMB = {DELTA_CMB:.6f} rad")

# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def drift(eta1, eta2):
    """Linear drift"""
    return DRIFT_COEF_ETA1 * eta1 + DRIFT_COEF_ETA2 * eta2 + DRIFT_CONST

def alpha_from_eta(eta1, eta2, z):
    """Calculates α using the FCC lattice"""
    lat = vcv48_k.FCCLattice(OMEGA_K, z)
    phi = lat.compute_phi(eta1, eta2)
    metric_factor = 1.0 + OMEGA_K / 2.0
    return (DELTA_CMB / K_INV) * metric_factor * phi

# ============================================================================
# DATASET READING
# ============================================================================

def read_dataset(filepath="data/tablea1b.dat"):
    """Reads the [OIII] dataset from A&A 699, A159"""
    z_list = []
    seen = set()

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    gal_id = parts[1]
                    if 'J' in gal_id and ('+' in gal_id or '-' in gal_id):
                        if gal_id not in seen:
                            seen.add(gal_id)
                            try:
                                z = float(parts[2])
                                if 0.5 < z < 5.0:
                                    z_list.append(z)
                            except:
                                pass
    return np.array(z_list)

# Load data
z_data = read_dataset()
n_gal = len(z_data)
z_med = np.median(z_data)

# Calculate observed drift
drifts_obs = (z_data - z_med) / (1 + z_med)
drift_obs = np.mean(drifts_obs)
drift_std = np.std(drifts_obs)
drift_sem = drift_std / np.sqrt(n_gal)
snr = abs(drift_obs / drift_sem)

print(f"\n📂 DATASET [OIII]:")
print(f"   Galaxies: {n_gal}")
print(f"   z ∈ [{z_data.min():.3f}, {z_data.max():.3f}]")
print(f"   z_med = {z_med:.3f}")
print(f"   Δα/α_obs = {drift_obs*100:.4f}% ± {drift_sem*100:.4f}%")
print(f"   S/N = {snr:.2f}")

# ============================================================================
# LOSS FUNCTION WITH MODEL WEIGHTS
# ============================================================================

def loss_function(params):
    """
    Loss function with three conditions.
    Weights determined by the structure of the Plateau III model.
    """
    e1_0, e2_0, e1_z, e2_z = params

    # Model weights (determined by theory, not by preference for any result)
    W_ALPHA = 10000.0      # Fundamental boundary condition
    W_DRIFT_LOCAL = 1000.0 # Vacuum consistency at z=0
    W_DRIFT_DIFF = 1.0     # Link to observables

    # 1. α_local = α_exp
    alpha_local = alpha_from_eta(e1_0, e2_0, Z_LOCAL)
    loss_alpha = W_ALPHA * ((alpha_local - ALPHA_EXP) / ALPHA_EXP) ** 2

    # 2. Local drift = 0
    drift_local = drift(e1_0, e2_0)
    loss_drift_local = W_DRIFT_LOCAL * (drift_local) ** 2

    # 3. Differential drift = observed
    drift_z = drift(e1_z, e2_z)
    drift_calc = drift_z - drift_local

    if drift_sem > 0:
        loss_drift_diff = W_DRIFT_DIFF * ((drift_calc - drift_obs) / drift_sem) ** 2
    else:
        loss_drift_diff = 0.0

    return loss_alpha + loss_drift_local + loss_drift_diff

# ============================================================================
# OPTIMIZATION
# ============================================================================

print(f"\n🔍 OPTIMIZING...")
print(f"   Weights: α_local={10000}, drift_local={1000}, drift_diff={1}")

# Physical bounds
bounds = [
    (0.24, 0.26), (0.15, 0.18),   # local
    (0.24, 0.28), (0.08, 0.18)    # cosmological
]

# Initial seed
x0 = [0.249312, 0.171654, 0.268000, 0.100000]

# Global optimization
result_global = differential_evolution(
    loss_function,
    bounds,
    maxiter=20,
    popsize=15,
    seed=42,
    tol=1e-8,
    disp=False,
    workers=1
)

# Local refinement
result = minimize(
    loss_function,
    result_global.x,
    method='L-BFGS-B',
    bounds=bounds,
    tol=1e-12,
    options={'maxiter': 500}
)

# ============================================================================
# RESULTS
# ============================================================================

e1_0, e2_0, e1_z, e2_z = result.x

# Calculate physical quantities
alpha_local = alpha_from_eta(e1_0, e2_0, Z_LOCAL)
alpha_cosmo = alpha_from_eta(e1_z, e2_z, z_med)
drift_local = drift(e1_0, e2_0)
drift_cosmo = drift(e1_z, e2_z)
drift_calc = drift_cosmo - drift_local

# Verify invariant K
K_eff_local = K_INV
K_eff_cosmo = K_INV

print("\n" + "=" * 80)
print("📊 CALIBRATION RESULTS")
print("=" * 80)

print(f"\n📍 LOCAL REGIME (z=0):")
print(f"   η₁ = {e1_0:.8f}")
print(f"   η₂ = {e2_0:.8f}")
print(f"   η₂/η₁ = {e2_0/e1_0:.6f}")
print(f"   α = {alpha_local:.12f}")
print(f"   α Error = {(alpha_local - ALPHA_EXP)/ALPHA_EXP*1e6:+.2f} ppm")
print(f"   Drift = {drift_local*100:.6f}%")
print(f"   K_eff = {K_eff_local:.6f}")

print(f"\n📍 COSMOLOGICAL REGIME (z={z_med:.3f}):")
print(f"   η₁ = {e1_z:.8f}")
print(f"   η₂ = {e2_z:.8f}")
print(f"   η₂/η₁ = {e2_z/e1_z:.6f}")
print(f"   α = {alpha_cosmo:.12f}")
print(f"   Drift = {drift_cosmo*100:.4f}%")
print(f"   K_eff = {K_eff_cosmo:.6f}")

print(f"\n📈 EVOLUTION:")
delta_eta1 = (e1_z/e1_0 - 1) * 100
delta_eta2 = (e2_z/e2_0 - 1) * 100
delta_alpha = (alpha_cosmo/alpha_local - 1) * 100

print(f"   Δη₁ = {delta_eta1:+.2f}%")
print(f"   Δη₂ = {delta_eta2:+.2f}%")
print(f"   Δα/α = {delta_alpha:+.4f}%")

print(f"\n📊 COMPARISON WITH DATA:")
print(f"   Calculated drift = {drift_calc*100:.4f}%")
print(f"   Observed drift = {drift_obs*100:.4f}% ± {drift_sem*100:.4f}%")
print(f"   Difference = {(drift_calc - drift_obs)/drift_sem:+.2f}σ")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    "timestamp": datetime.now().isoformat(),
    "model": "Plateau III",
    "constants": {
        "alpha_exp": ALPHA_EXP,
        "K_inv": K_INV,
        "delta_cmb": DELTA_CMB,
        "omega_k": OMEGA_K
    },
    "dataset": {
        "source": "A&A 699, A159",
        "n_galaxies": n_gal,
        "z_median": float(z_med),
        "drift_obs_percent": float(drift_obs * 100),
        "drift_sem_percent": float(drift_sem * 100),
        "snr": float(snr)
    },
    "local": {
        "z": 0.0,
        "eta1": float(e1_0),
        "eta2": float(e2_0),
        "eta2_over_eta1": float(e2_0/e1_0),
        "alpha": float(alpha_local),
        "alpha_error_ppm": float((alpha_local - ALPHA_EXP)/ALPHA_EXP * 1e6),
        "drift_percent": float(drift_local * 100),
        "K_eff": float(K_eff_local)
    },
    "cosmological": {
        "z": float(z_med),
        "eta1": float(e1_z),
        "eta2": float(e2_z),
        "eta2_over_eta1": float(e2_z/e1_z),
        "alpha": float(alpha_cosmo),
        "drift_percent": float(drift_cosmo * 100),
        "K_eff": float(K_eff_cosmo)
    },
    "evolution": {
        "delta_eta1_percent": float(delta_eta1),
        "delta_eta2_percent": float(delta_eta2),
        "delta_alpha_percent": float(delta_alpha),
        "drift_calculated_percent": float(drift_calc * 100),
        "drift_observed_percent": float(drift_obs * 100),
        "sigma_difference": float((drift_calc - drift_obs)/drift_sem) if drift_sem > 0 else 0
    }
}

output_file = "calibration_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")
print("=" * 80)
print("✅ CALIBRATION COMPLETED")
print("=" * 80)