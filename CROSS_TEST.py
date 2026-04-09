import vcv48_k
import numpy as np
from scipy.optimize import minimize

ALPHA_EXP = vcv48_k.ALPHA_EXP                         K_INV = vcv48_k.K_INVARIANT
DELTA_CMB = vcv48_k.DELTA_CMB
OMEGA_K = vcv48_k.OMEGA_K

metric_factor = 1.0 + OMEGA_K / 2.0
target_phi = ALPHA_EXP * K_INV / (DELTA_CMB * metric_factor)

print(f"Target Φ_α = {target_phi:.10f}")

def loss(params):
    eta1, eta2 = params
    phi = vcv48_k.compute_phi_from_eta(eta1, eta2, OMEGA_K, 0.0)
    return (phi - target_phi) ** 2

# Global search
from scipy.optimize import differential_evolution
bounds = [(0.20, 0.32), (0.05, 0.20)]
result = differential_evolution(loss, bounds, maxiter=100, tol=1e-8)

eta1, eta2 = result.x
phi = vcv48_k.compute_phi_from_eta(eta1, eta2, OMEGA_K, 0.0)
alpha = (DELTA_CMB / K_INV) * metric_factor * phi

print(f"\n✅ CALIBRATION SUCCESSFUL:")
print(f"   η₁ = {eta1:.10f}")
print(f"   η₂ = {eta2:.10f}")
print(f"   η₂/η₁ = {eta2/eta1:.8f}")
print(f"   Φ_α = {phi:.10f}")
print(f"   α_calc = {alpha:.12f}")
print(f"   α_exp  = {ALPHA_EXP:.12f}")
print(f"   Error = {(alpha - ALPHA_EXP)/ALPHA_EXP*1e6:.2f} ppm")