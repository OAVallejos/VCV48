#!/usr/bin/env python3
"""
TOP_VPM48.py 
TOP QUARK - MASS CALCULATION

Dependencies:
    - vpm48_engine_top (Rust extension module, build with: maturin develop --release)
"""

import sys
import time
from datetime import datetime
import json

try:
    import vpm48_engine_top as vpm
    print("✅ VPM-48 engine loaded")
except ImportError:
    print("❌ Error: maturin develop --release")
    sys.exit(1)

constants = vpm.get_constantes()

TOP_MASS = constants['masa_top_exp_gev']
TOP_MASS_ERR = constants['masa_top_err_gev']
CENTER = constants['nf_top']
RESIDUE = constants['residuo_top']
K_III = constants['k_iii']
M_REF = constants['m_ref_mev']

RADIUS = 500

def main():
    print("\n" + "="*70)
    print("🚀 TOP QUARK - MASS CALCULATION")
    print("="*70 + "\n")

    print(f"📌 Parameters:")
    print(f"   • m_ref = {M_REF:.6f} MeV")
    print(f"   • K_III = {K_III:.6f}")
    print(f"   • Product = {M_REF * K_III:.6f} MeV")
    print(f"   • Nf = {CENTER:,}")
    print(f"   • Residue = {CENTER % 48}")
    print(f"   • Exp mass = {TOP_MASS:.6f} ± {TOP_MASS_ERR:.6f} GeV\n")

    start = time.time()
    res = vpm.escaneo_ultra_fino(CENTER, RADIUS, True)
    elapsed = res.get('tiempo_segundos', 0)

    print(f"\n✅ Completed in {elapsed:.3f} s")

    best = res.get('mejores', [])
    if best:
        nf, mass, error = best[0]

        print("\n" + "="*70)
        print("📊 RESULT")
        print("="*70)
        print(f"   • Nf = {nf:,}")
        print(f"   • Mass = {mass:.6f} GeV")
        print(f"   • Exp = {TOP_MASS:.6f} ± {TOP_MASS_ERR:.6f} GeV")
        print(f"   • Error = {error:.4f}%")
        print(f"   • |Δ| = {abs(mass - TOP_MASS):.6f} GeV")

        if abs(mass - TOP_MASS) <= TOP_MASS_ERR:
            print(f"   • ✅ WITHIN UNCERTAINTY")
        else:
            print(f"   • ⚠️ OUTSIDE UNCERTAINTY")
        print("="*70)

        filename = f"top_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(filename, 'w') as f:
            json.dump({
                'nf': nf,
                'mass_gev': mass,
                'error_pct': error,
                'within_uncertainty': abs(mass - TOP_MASS) <= TOP_MASS_ERR
            }, f, indent=2)
        print(f"\n💾 {filename}")

if __name__ == "__main__":
    main()