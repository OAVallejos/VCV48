#!/usr/bin/env python3
"""
🌀 AXION SEARCH AS RESIDUAL PHASE
Residue 24, Fractional Nf
"""                                                     
import vcv48 as vpm
import numpy as np

M_AXION_EV = 1.8e-22
M_REF = 8.75e6  # eV
K_III = 0.8025  # Plateau III

# Predicted effective Nf
Nf_predicted = M_AXION_EV / (M_REF * K_III)
print(f"\n{'='*70}")
print(f"🌀 ULA AXION - RESIDUAL PHASE ANALYSIS")
print(f"{'='*70}")
print(f"Axion mass: {M_AXION_EV:.2e} eV")
print(f"Predicted effective Nf: {Nf_predicted:.2e}")
print(f"Residue: 24 (dark matter sector)")
print("-" * 70)

# Search in multiples of 48 with residue 24
print(f"\n🔍 Scanning configurations with residue 24:")
print(f"{'Nf':>8} | {'Mass (eV)':>15} | {'Mass/Nf':>15} | {'Type'}")
print("-" * 70)

for nf in range(48, 10000, 48):
    if nf % 48 != 24:
        continue

    res = vpm.analizar_por_nf(nf, 3)
    mass_ev = res['energia_ev']

    # Calculate mass/Nf ratio (should be constant)
    ratio = mass_ev / nf if nf > 0 else 0

    print(f"{nf:8d} | {mass_ev:15.6e} | {ratio:15.6e} | {res['tipo']}")

# Expected proportionality constant
expected_constant = M_REF * K_III  # ≈ 7.02e6 eV
print(f"\n📊 Expected constant (m_ref·K_III): {expected_constant:.2e} eV")
print(f"Effective Nf for axion = {M_AXION_EV / expected_constant:.2e}")