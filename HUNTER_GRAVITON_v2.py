#!/usr/bin/env python3
"""
🔬 BOSON CALIBRATION VERIFICATION
Testing with Z boson around Nf=20,176,000           """
import vcv48 as vpm
import time

def verify_boson(name, target_nf, expected_mass, expected_residue, radius=50000, step=1000):
    print(f"\n{'='*70}")
    print(f"🔍 VERIFYING: {name}")
    print(f"{'='*70}")
    print(f"Target Nf: {target_nf:,}")
    print(f"Expected mass: {expected_mass} GeV")
    print(f"Expected residue: {expected_residue}")
    print(f"Range: ±{radius:,} (step={step:,})")
    print("-" * 70)

    nf_min = target_nf - radius
    nf_max = target_nf + radius

    best_candidates = []
    start = time.time()

    for nf in range(nf_min, nf_max + 1, step):
        # Only consider correct residues
        if nf % 48 != expected_residue:
            continue

        try:
            res = vpm.analizar_por_nf(nf, 3)
            calc_mass = res['energia_ev'] / 1e9
            error = abs(calc_mass - expected_mass) / expected_mass * 100
            stability = res['estabilidad_oh']

            # Save if error is small
            if error < 10:  # Less than 10% error
                best_candidates.append({
                    'nf': nf,
                    'mass': calc_mass,
                    'error': error,
                    'stability': stability,
                    'type': res['tipo']
                })

                print(f"    📍 Nf={nf:,} | Mass={calc_mass:.3f} GeV | Error={error:.3f}% | Stab={stability} | {res['tipo']}")

        except Exception as e:
            print(f"   Error at Nf={nf}: {e}")

    elapsed = time.time() - start

    print(f"\n⏱️  Time: {elapsed:.1f} seconds")
    print(f"📊 Candidates found: {len(best_candidates)}")

    if best_candidates:
        best = min(best_candidates, key=lambda x: abs(x['error']))
        print(f"\n🏆 BEST CANDIDATE:")
        print(f"   Nf={best['nf']:,}")
        print(f"   Mass={best['mass']:.4f} GeV (expected={expected_mass})")
        print(f"   Error={best['error']:.4f}%")
        print(f"   Stability={best['stability']}")
        print(f"   Type={best['type']}")

        # Calculate offset
        offset = best['nf'] - target_nf
        print(f"\n📐 DETECTED OFFSET: {offset:+,}")
        return best, offset
    else:
        print("❌ No candidates found")
        return None, None

def verify_all():
    """Verifies all three bosons"""
    bosons = [
        ("W Boson", 15743000, 80.379, 8),
        ("Z Boson", 20176000, 91.1876, 16),
        ("Higgs Boson", 37625000, 125.18, 8)
    ]

    results = {}
    offsets = []

    for name, nf, mass, residue in bosons:
        best, offset = verify_boson(name, nf, mass, residue, radius=200000, step=5000)
        if best:
            results[name] = best
            offsets.append(offset)
        print("\n" + "="*70)

    # Offset analysis
    if offsets:
        avg_offset = sum(offsets) / len(offsets)
        print(f"\n📊 OFFSET ANALYSIS:")
        print(f"   Offsets: {offsets}")
        print(f"   Average offset: {avg_offset:+.1f}")
        print(f"   Consistent? {'✅ YES' if max(offsets)-min(offsets) < 50000 else '❌ NO'}")

    return results

if __name__ == "__main__":
    print("="*70)
    print("🔬 BOSON CALIBRATION VERIFICATION")
    print("="*70)
    print(f"Engine: vpm48_engine_optimized")
    print(f"Available constants:")
    print(f"  BOSON_SCALE: {getattr(vpm, 'BOSON_SCALE', 'Not defined')}")
    print(f"  BOSON_CORRECTION: {getattr(vpm, 'BOSON_CORRECTION', 'Not defined')}")
    print(f"  BOSON_NF_OFFSET: {getattr(vpm, 'BOSON_NF_OFFSET', 'Not defined')}")

    # Verify all bosons
    results = verify_all()

    # Quick test with manual offset
    if getattr(vpm, 'BOSON_NF_OFFSET', None) is None:
        print("\n⚠️  BOSON_NF_OFFSET is not defined in the engine")
        print("   Using manual offset for testing...")

        manual_offset = 220000  # The one we observed before
        print(f"\n🔧 Testing with manual offset: {manual_offset:+}")

        corrected_nf = 20176000 + manual_offset
        res = vpm.analizar_por_nf(corrected_nf, 3)
        mass = res['energia_ev'] / 1e9
        error = abs(mass - 91.1876) / 91.1876 * 100
        print(f"\n📌 Z boson with offset:")
        print(f"   Nf={corrected_nf:,} (target+{manual_offset})")
        print(f"   Mass={mass:.4f} GeV")
        print(f"   Error={error:.4f}%")
        print(f"   Type={res['tipo']}")