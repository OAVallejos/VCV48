#!/usr/bin/env python3
"""
oh_48_run.py
VCV48 — Audit of the 48^k structure (O_h group signature).
Lightweight driver for the oh_48 kernel (Rust/PyO3). Android/Termux friendly.
"""
import json
import gc
from datetime import datetime

def print_section(title, char="=", width=70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def clean_for_json(obj):
    if isinstance(obj, dict):  return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [clean_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None: return obj
    return str(obj)

def main():
    try:
        import oh_48 as kernel
    except ImportError as e:
        print(f"[ERROR] Could not import 'oh_48': {e}")
        print("Run first: maturin develop --release")
        raise SystemExit(1)

    print("=" * 70)
    print("  VCV48 — AUDIT OF THE 48^k STRUCTURE (O_h signature)")
    print("  Kernel: oh_48.rs  |  O(1) memory  |  u128 safe")
    print("=" * 70)

    report = {"metadata": {
        "model": "VCV48 — O_h(48) audit",
        "timestamp": datetime.now().isoformat(),
    }}

    # ---------------- 1. THE 48^k LADDER ----------------
    print_section("1. MASS LADDER  m = m_e / 48^k")
    lad = kernel.ladder_48()
    print(f"  Rungs computed: {lad['count']}")
    print(f"  {'k':>3} {'q = 48^k':>22} {'mass (eV)':>14}  window")
    for r in lad['rungs']:
        q = int(r['q'])
        tag = ""
        # mark the QCD axion rung and the ULA window
        if r['window'].startswith('ueV'):
            tag = "  <-- QCD AXION"
        print(f"  {r['k']:>3} {q:>22,d} {r['mass_ev']:>14.6e}  {r['window']}{tag}")
    report['ladder'] = clean_for_json(lad)
    gc.collect()

    # ---------------- 2. CATALOG AUDIT ----------------
    print_section("2. CATALOG AUDIT (factorization from scratch)")
    cat = kernel.analyze_catalog()
    print(f"  Entries analyzed          : {cat['total']}")
    print(f"  EXACT powers of 48        : {cat['exact_power_count']}")
    exact_names = list(cat['exact_power_names'])
    if exact_names:
        print(f"  Exact names               : {', '.join(exact_names)}")

    print("\n  Detail per particle:")
    hdr = f"  {'particle':<11} {'v48':>3} {'exact48':>8} {'3smooth':>8} {'remainder_after_48':>16}"
    print(hdr); print("  " + "-" * 62)
    discrepancies = []
    for e in cat['entries']:
        print(f"  {e['name']:<11} {e['v48']:>3} "
              f"{'YES' if e['is_exact_power_of_48'] else 'no':>8} "
              f"{'YES' if e['is_3smooth'] else 'no':>8} "
              f"{e['remainder_after_48']:>16}")
        # Audit against the Annex: if the Annex claims a residual prime but it is 3-smooth
        if e['is_3smooth'] and ('fermion' in e['class'].lower()):
            discrepancies.append(
                f"{e['name']}: Annex says '{e['anexo_note']}' but it is pure 3-smooth "
                f"(q=2^{e['a']}·3^{e['b']})")
    if discrepancies:
        print("\n  ⚠️  DISCREPANCIES with Annex V:")
        for d in discrepancies:
            print(f"     - {d}")
    report['catalog'] = clean_for_json(cat)
    gc.collect()

    # ---------------- 3. 3-SMOOTH SCAN IN THE QCD AXION BAND ----------------
    print_section("3. 3-SMOOTH DENSITY AROUND q = 48^6 (QCD axion)")
    q_qcd = 12_230_590_464
    for delta in (0.01, 0.03, 0.05):
        q_lo = int(q_qcd / (1.0 + delta))
        q_hi = int(q_qcd * (1.0 + delta))
        band = kernel.scan_3smooth_band(q_lo, q_hi)
        n_exact = band['exact_count']
        print(f"  δ={delta*100:>4.1f}%  3-smooth in band: {band['total_3smooth']:>3}  "
              f"| exact 48^k powers: {n_exact}")
        if n_exact:
            for p in band['exact_48_powers_in_band']:
                print(f"            -> 48^{p['k']} = {int(p['q']):,}")
    # save only the δ=3% band
    band3 = kernel.scan_3smooth_band(int(q_qcd/1.03), int(q_qcd*1.03))
    report['qcd_band_scan'] = clean_for_json(band3)
    gc.collect()

    # ---------------- 4. VERDICT ----------------
    print_section("4. VERDICT")
    if cat['exact_power_count'] == 1 and exact_names == ["QCD Axion"]:
        print("  🎯 The QCD AXION (q=48^6) is the ONLY exact 48^k state in the catalog.")
        print("     -> Elevates q_QCD=48^6 to a distinguished invariant of the O_h crystal.")
    elif cat['exact_power_count'] == 0:
        print("  ❌ No returned candidate is an exact power of 48.")
    else:
        print(f"  ℹ️  {cat['exact_power_count']} exact powers: {', '.join(exact_names)}")
    if discrepancies:
        print("  ⚠️  Factorizations in the Annex needing correction were detected (above).")

    # ---------------- 5. EXPORT ----------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"oh48_audit_{ts}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print_section("✅ AUDIT COMPLETED")
    print(f"  Results in: {path}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by the user.")