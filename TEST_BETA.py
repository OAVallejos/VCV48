#!/usr/bin/env python3
"""                        
CALCULATION OF SENSITIVITY COEFFICIENTS              
================================================================================
Explicitly loads calibration_results.json (structure: 'local'/'cosmological')
"""

import vcv48_k
import numpy as np
import json

# ============================================================================
# LOAD PARAMETERS FROM calibration_results.json
# ============================================================================

with open('calibration_results.json', 'r') as f:
    data = json.load(f)

print(f"📂 Loaded: calibration_results.json")
print(f"   Timestamp: {data['timestamp']}")
print(f"   Number of galaxies: {data['dataset']['n_galaxies']}")

params = {
    'eta1_local': data['local']['eta1'],
    'eta2_local': data['local']['eta2'],
    'eta1_cosmo': data['cosmological']['eta1'],
    'eta2_cosmo': data['cosmological']['eta2'],
    'z_cosmo': data['cosmological']['z'],
    'omega_k': data['constants']['omega_k'],
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def phi(eta1, eta2, z, omega_k):
    return vcv48_k.compute_phi_from_eta(eta1, eta2, omega_k, z)

def gradient(f, x, y, z, om, h=1e-5):
    """4th order finite differences"""
    fx = (-f(x+2*h, y, z, om) + 8*f(x+h, y, z, om)
          - 8*f(x-h, y, z, om) + f(x-2*h, y, z, om)) / (12*h)
    fy = (-f(x, y+2*h, z, om) + 8*f(x, y+h, z, om)
          - 8*f(x, y-h, z, om) + f(x, y-2*h, z, om)) / (12*h)
    return fx, fy

# ============================================================================
# CALCULATIONS
# ============================================================================

print("\n" + "=" * 70)
print("SENSITIVITY COEFFICIENTS")
print("=" * 70)

# Local regime (z=0)
beta1_local, beta2_local = gradient(phi, params['eta1_local'], params['eta2_local'],
                                    0.0, params['omega_k'])
phi_local = phi(params['eta1_local'], params['eta2_local'], 0.0, params['omega_k'])

# Cosmological regime
beta1_cosmo, beta2_cosmo = gradient(phi, params['eta1_cosmo'], params['eta2_cosmo'],
                                    params['z_cosmo'], params['omega_k'])
phi_cosmo = phi(params['eta1_cosmo'], params['eta2_cosmo'], params['z_cosmo'], params['omega_k'])

# Validation
delta_eta1 = params['eta1_cosmo'] - params['eta1_local']
delta_eta2 = params['eta2_cosmo'] - params['eta2_local']
beta1_avg = (beta1_local + beta1_cosmo) / 2
beta2_avg = (beta2_local + beta2_cosmo) / 2

delta_phi_linear = beta1_avg * delta_eta1 + beta2_avg * delta_eta2
delta_phi_real = phi_cosmo - phi_local
precision = (1 - abs(delta_phi_linear - delta_phi_real) / abs(delta_phi_real)) * 100

# Contributions
contrib1 = abs(beta1_avg * delta_eta1)
contrib2 = abs(beta2_avg * delta_eta2)
total = contrib1 + contrib2

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n📍 LOCAL REGIME (z=0):")
print(f"   η₁ = {params['eta1_local']:.12f}")
print(f"   η₂ = {params['eta2_local']:.12f}")
print(f"   Φ_α = {phi_local:.10f}")
print(f"   β₁ = {beta1_local:+.8f}")
print(f"   β₂ = {beta2_local:+.8f}")
print(f"   |β₂/β₁| = {abs(beta2_local/beta1_local):.2f}")

print(f"\n📍 COSMOLOGICAL REGIME (z={params['z_cosmo']:.3f}):")
print(f"   η₁ = {params['eta1_cosmo']:.12f}")
print(f"   η₂ = {params['eta2_cosmo']:.12f}")
print(f"   Φ_α = {phi_cosmo:.10f}")
print(f"   β₁ = {beta1_cosmo:+.8f}")
print(f"   β₂ = {beta2_cosmo:+.8f}")
print(f"   |β₂/β₁| = {abs(beta2_cosmo/beta1_cosmo):.2f}")

print(f"\n📈 LINEAR VALIDATION:")
print(f"   Δη₁ = {delta_eta1:+.6f} ({delta_eta1/params['eta1_local']*100:+.2f}%)")
print(f"   Δη₂ = {delta_eta2:+.6f} ({delta_eta2/params['eta2_local']*100:+.2f}%)")
print(f"   ΔΦ_α (linear) = {delta_phi_linear:+.8f}")
print(f"   ΔΦ_α (actual) = {delta_phi_real:+.8f}")
print(f"   Precision = {precision:.2f}%")
print(f"   Δη₁ contribution: {contrib1/total*100:.1f}%")
print(f"   Δη₂ contribution: {contrib2/total*100:.1f}%")

print("\n" + "=" * 70)