#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCV48 v10.2 - Topological Capture of the Ultralight Axion (ULA)
Python interface for the Rust kernel (PyO3).

Dependencies:
    - axion_ula (Rust extension module, build with: maturin develop --release)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def print_section(title, char="=", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def clean_for_json(obj):
    if isinstance(obj, dict): return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)): return obj
    elif obj is None: return None
    else: return str(obj)

def main():
    try:
        import axion_ula as kernel
    except ImportError as e:
        print(f"[ERROR] Could not import the 'axion_ula' module: {e}")
        print("Make sure you have run: maturin develop --release")
        sys.exit(1)

    print("=== DIAGNOSTICS ===")
    available = [a for a in dir(kernel) if not a.startswith('_')]
    print(f"Available functions: {available}")

    # ✅ FIXED: Now we look for 'search_ula'
    if not hasattr(kernel, 'search_ula'):
        print("[ERROR] The function 'search_ula' is not available")
        sys.exit(1)

    print("=" * 70)
    print("  VCV48 v10.2 - TOPOLOGICAL CAPTURE OF THE ULA AXION (VPM)")
    print("  Vacuum Crystallography O_h(48)")
    print("=" * 70)

    print_section("1. 3-SMOOTH SWEEP IN N_f SPACE (u128)")
    try:
        # ✅ FIXED: Call to the renamed function
        results = kernel.search_ula()
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"    q range explored     : [{results['q_lo']}, {results['q_hi']}]")
    print(f"    States scanned       : {results['total_states_scanned']:,}")
    print(f"    Candidates found     : {len(results['candidates'])}")

    print_section("2. CANDIDATES FOUND (O_h Bosonic Signature)", "-")
    q_qcd = 12230590464
    ratio_teorico = (2**64) / (3**4)

    for i, cand in enumerate(results['candidates']):
        print(f"\n  --- State {i+1} ---")
        print(f"  Mass          : {cand['mass_ev']:.6e} eV")
        print(f"  VPM deviation : {cand['deviation_percent']:+.4f} %")
        print(f"  Factorization : {cand['factorization']}")
        print(f"  Stability     : {cand['stability']:.4f}")
        print(f"  Symmetry      : {cand['symmetry']:.4f}")
        print(f"  Top. Charge   : {cand['charge']:+d}")
        print(f"  (a={cand['a']}, b={cand['b']})")

        q_val = int(cand['q'])
        ratio = q_val / q_qcd
        print(f"  Ratio q_ULA/q_QCD : {ratio:.5e}")
        if abs(ratio - ratio_teorico) / ratio_teorico < 1e-9:
            print(f"  🌟 EXACT MATCH with 2^64 / 3^4 ({ratio_teorico:.5e})")

    best = results.get('best_candidate')
    if best:
        print_section("3. 🏆 BEST CANDIDATE (ULA-α)")
        print(f"  N_f = 1 / {best['q']}")
        print(f"  Refined mass : {best['mass_ev']:.6e} eV (original VPM: 1.80e-22 eV)")
        print(f"  Deviation    : {best['deviation_percent']:+.4f} %")
        print(f"  Factorization: {best['factorization']}")
        print(f"  Stability    : {best['stability']:.4f}")

        print_section("4. DETAILED TOPOLOGICAL ANALYSIS", "-")
        try:
            # ✅ FIXED: Call to the renamed analysis function
            detail = kernel.analyze_ula(1, int(best['q']))
            print(f"  Is 3-smooth (pure boson)? : {'✅ YES' if detail['is_3_smooth'] else '❌ NO'}")
            print(f"  O_h Topological Charge    : {detail['charge']}")
            print(f"  Crystal Stability         : {detail['stability']:.4f}")
            print(f"  Factorization             : {detail['factorization']}")
            print(f"  Ratio q/q_QCD             : {detail['ratio_to_qcd_str']}")
        except Exception as e:
            print(f"  [ERROR] Detailed analysis failed: {e}")

    print_section("5. JSON EXPORT")
    export_data = {
        "metadata": {
            "model": "VCV48 v10.2", "target": "ULA Axion (VPM)",
            "target_mass_ev": 1.8e-22, "band": 0.03,
            "timestamp": datetime.now().isoformat(), "q_qcd": q_qcd, "ratio_teorico": ratio_teorico,
        },
        "search": {
            "q_lo": int(results['q_lo']), "q_hi": int(results['q_hi']),
            "total_states_scanned": results['total_states_scanned'],
            "candidates_found": len(results['candidates']),
        },
        "candidates": clean_for_json(results['candidates']),
        "best_candidate": clean_for_json(results.get('best_candidate')),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"ula_capture_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Results saved to: {json_path}")
    print_section("✅ CAPTURE COMPLETED", "=")
    print("  The ULA is a citizen of the O_h crystal.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()