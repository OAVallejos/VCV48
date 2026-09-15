# calculate_masses.py
#!/usr/bin/env python3
"""
VCV48 - COMPLETE MASS CALCULATION
Corrected formula: M = N_f · m_e (no double K_I, no phantom QCD)

Dependencies:
    - vpm48_engine_masas (Rust extension module, build with: maturin develop --release)
"""

import sys
import json
from datetime import datetime

try:
    import vpm48_engine_masas as vpm
    print("✅ VPM-48 engine masas loaded")
except ImportError:
    print("❌ Error: maturin develop --release")
    sys.exit(1)

# Experimental masses (PDG 2024)
M_EXP = {
    'electron': 0.5109989461,      # MeV
    'muon': 105.6583755,           # MeV
    'tau': 1776.86,                # MeV
    'up': 2.16,                    # MeV
    'down': 4.70,                  # MeV
    'strange': 93.5,               # MeV
    'charm': 1270.0,               # MeV
    'bottom': 4180.0,              # MeV
    'top': 172760.0,               # MeV (172.76 GeV)
    'proton': 938.2720813,         # MeV
    'neutron': 939.5654205,        # MeV
    'w': 80379.0,                  # MeV (80.379 GeV)
    'z': 91187.6,                  # MeV (91.1876 GeV)
    'higgs': 125180.0,             # MeV (125.18 GeV)
}

def main():
    print("\n" + "="*80)
    print("📊 COMPLETE MASS CALCULATION - VCV48 (CORRECTED FORMULA)")
    print("   M = N_f · m_e  (Plateau I)")
    print("="*80)

    # Constants
    c = vpm.get_constantes()
    print(f"\n📌 CONSTANTS:")
    print(f"   m_e = {c['m_e_mev']:.6f} MeV")
    print(f"   K_I = {c['k_i']:.6f}")
    print(f"   K_III = {c['k_iii']:.6f}")
    print(f"   m_ref = {c['m_ref_mev']:.6f} MeV")

    # Compute all
    masses = vpm.calcular_todas_masas()

    print(f"\n{'Particle':<12} {'Nf':>10} {'M calc (MeV)':>14} {'M exp (MeV)':>14} {'Error':>10}")
    print("-"*80)

    errors = []
    for name, r in masses.items():
        mass_calc = r['mass_mev']
        mass_exp = M_EXP.get(name, 0)
        error = abs(mass_calc - mass_exp) / mass_exp * 100.0

        nf = r.get('nf_real', r.get('nf', 0))
        print(f"{name:<12} {nf:>10,.0f} {mass_calc:>14.6f} {mass_exp:>14.6f} {error:>9.4f}%")
        errors.append(error)

    print("-"*80)
    print(f"\n📊 Average error: {sum(errors)/len(errors):.4f}%")
    print(f"📊 Maximum error: {max(errors):.4f}%")
    print(f"📊 Minimum error: {min(errors):.4f}%")

    # Save
    filename = f"masses_vcv48_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(filename, 'w') as f:
        json.dump(masses, f, indent=2, default=str)
    print(f"\n💾 Saved to {filename}")

if __name__ == "__main__":
    main()
    