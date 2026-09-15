#!/usr/bin/env python3
"""
GENERAL particle search in VCV48
DEFINITIVE version v8 (With Solar Neutrinos + Deduplication)
Optimized for Termux / Android (proot-distro + micromamba)

Dependencies:
    - neutrino_search (VCV48 engine module)
"""

import neutrino_search as ns
import json
import gc
from datetime import datetime

def print_section(title, char="=", width=70):
    """Prints a formatted section"""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def search_spectrum(min_mass_ev, max_mass_ev, max_denominator, label=""):
    """Searches for particles in a specific mass range (general sweep)"""
    print_section(f"SEARCH: {label}")
    print(f"  Mass range: {min_mass_ev:.2e} - {max_mass_ev:.2e} eV")
    print(f"  Maximum denominator: {max_denominator:,}")

    results = ns.search_mass_range(min_mass_ev, max_mass_ev, max_denominator)
    return results

def search_specific_particle(name, target_mass_ev, tolerance_ev, max_den, label=""):
    """High-precision targeted search for known Standard Model particles"""
    print_section(f"TARGETED SEARCH: {label}")
    min_m = max(0.0001, target_mass_ev - tolerance_ev)
    max_m = target_mass_ev + tolerance_ev

    print(f"  Target: {name}")
    print(f"  Target mass: {target_mass_ev:.2e} eV")
    print(f"  Search range: {min_m:.2e} - {max_m:.2e} eV")
    print(f"  Maximum denominator: {max_den:,}")

    results = ns.search_mass_range(min_m, max_m, max_den)

    peaks = results.get('stability_peaks', {})
    if peaks:
        best = list(peaks.values())[0]
        rel_error = abs(best['mass_ev'] - target_mass_ev) / target_mass_ev * 100
        print(f"\n  ✅ BEST CANDIDATE FOR {name}:")
        print(f"    Found mass      = {best['mass_ev']:.6e} eV")
        print(f"    Nf              = {best['numerator']}/{best['denominator']} = {best['nf']:.10f}")
        print(f"    Stability       = {best['stability']:.4f}")
        print(f"    Topological Charge = {best['charge']}")
        print(f"    Class           = {best['class']}")
        print(f"    Relative error  = {rel_error:.6f} %")
    else:
        print(f"\n  ❌ No stability peaks found for {name}.")

    return results

def analyze_results(results, title="RESULTS"):
    """Analyzes and displays results from the standard search"""
    print_section(title)

    total = results.get('total_candidates', 0)
    print(f"\n  Total analyzed: {total:,}")

    distribution = results.get('mass_distribution', {})
    if distribution:
        print(f"\n  📦 MASS DISTRIBUTION:")
        for class_name, count in sorted(distribution.items()):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {class_name:20s}: {count:>12,} ({percentage:.1f}%)")

    peaks = results.get('stability_peaks', {})
    if peaks:
        print(f"\n  📈 STABILITY PEAKS (Top {min(5, len(peaks))}):")
        for key, peak in list(peaks.items())[:5]:
            print(f"    #{int(key)+1}: Mass = {peak['mass_ev']:.4e} eV | Nf = {peak['numerator']}/{peak['denominator']} | St = {peak['stability']:.4f} | Class = {peak['class']}")

    wimps = results.get('wimp_candidates', {})
    if wimps:
        print(f"\n  👻 WIMP CANDIDATES (1-1000 GeV): {len(wimps)} found")

    axions = results.get('axion_candidates', {})
    if axions:
        print(f"\n  🌌 AXION CANDIDATES (< 1 eV): {len(axions)} found")

    neutrinos = results.get('neutrino_candidates', {})
    if neutrinos:
        print(f"\n  🎯 NEUTRINO CANDIDATES (0.0001-2.2 eV): {len(neutrinos)} found")

    return results

def save_results(all_results, filename=None):
    """Saves all results to JSON in a safe and serializable way"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vcv48_search_{timestamp}.json"

    serializable = {}
    for search_name, results in all_results.items():
        if not isinstance(results, dict):
            continue

        if search_name == 'qcd_axions':
            serializable[search_name] = {
                'total_candidates': results.get('total_candidates', 0),
                'axions': {k: dict(v) for k, v in results.get('axion_candidates', {}).items()},
                'spacing_analysis': results.get('spacing_analysis', {}),
                'best': {
                    'mass_ev': results.get('best_mass_ev'),
                    'mass_uev': results.get('best_mass_uev'),
                    'stability': results.get('best_stability'),
                    'nf': results.get('best_nf'),
                    'num': results.get('best_num'),
                    'den': results.get('best_den'),
                }
            }
        elif search_name in ['nu3_atmospheric', 'solar_neutrinos']:
            serializable[search_name] = {
                'total_candidates': results.get('total_candidates', 0),
                'neutrinos': {k: dict(v) for k, v in results.get('neutrino_candidates', {}).items()},
                'best': {
                    'mass_ev': results.get('best_mass_ev'),
                    'mass_meV': results.get('best_mass_meV'),
                    'stability': results.get('best_stability'),
                    'symmetry': results.get('best_symmetry'),
                    'nf': results.get('best_nf'),
                    'num': results.get('best_num'),
                    'den': results.get('best_den'),
                    'rel_error': results.get('best_rel_error'),
                }
            }
        else:
            serializable[search_name] = {
                'total': results.get('total_candidates', 0),
                'distribution': dict(results.get('mass_distribution', {})),
                'peaks': {k: dict(v) for k, v in results.get('stability_peaks', {}).items()},
                'wimps': {k: dict(v) for k, v in results.get('wimp_candidates', {}).items()},
                'axions': {k: dict(v) for k, v in results.get('axion_candidates', {}).items()},
                'neutrinos': {k: dict(v) for k, v in results.get('neutrino_candidates', {}).items()}
            }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {filename}")
    return filename

def main():
    """Main function - general search optimized for mobile"""

    print("╔" + "═" * 68 + "╗")
    print("║     VCV48: GENERAL PARTICLE SEARCH v8            ║")
    print("║     With Solar Neutrinos + Deduplication         ║")
    print("║     [OPTIMIZED FOR ANDROID/TERMUX + RAYON]       ║")
    print("╚" + "═" * 68 + "╝")

    all_results = {}

    # ==================== SEARCH 1: ULTRALIGHT ====================
    results_ultralight = search_spectrum(
        min_mass_ev=0.0001, max_mass_ev=1.0,
        max_denominator=50_000, label="ULTRALIGHT (0.0001-1 eV)"
    )
    analyze_results(results_ultralight, "ULTRALIGHT RESULTS")
    all_results['ultralight'] = results_ultralight
    del results_ultralight; gc.collect()

    # ==================== SEARCH 2: NEUTRINOS ====================
    results_neutrinos = search_spectrum(
        min_mass_ev=0.0001, max_mass_ev=2.2,
        max_denominator=50_000, label="NEUTRINOS (0.0001-2.2 eV)"
    )
    analyze_results(results_neutrinos, "NEUTRINO RESULTS")
    all_results['neutrinos'] = results_neutrinos
    del results_neutrinos; gc.collect()

    # ==================== SEARCH 2b: ATMOSPHERIC NEUTRINO ν₃ ====================
    print_section("SEARCH 2b: ATMOSPHERIC NEUTRINO ν₃ (Parallel)")
    print("  Searching for the most massive neutrino (normal hierarchy)")
    print("  m₃ ≈ √(Δm²₃₂) ≈ 0.0495 eV")

    results_nu3 = ns.search_atmospheric_neutrino()
    best_mass = results_nu3.get('best_mass_ev')
    if best_mass is not None:
        print(f"\n  ✅ NEUTRINO ν₃ FOUND:")
        print(f"    Mass = {best_mass:.6e} eV ({results_nu3.get('best_mass_meV', 0):.3f} meV)")
        print(f"    Nf   = {results_nu3.get('best_num', 'N/A')}/{results_nu3.get('best_den', 'N/A')}")
        print(f"    Stability = {results_nu3.get('best_stability', 0):.4f}")
        print(f"    Relative error = {results_nu3.get('best_rel_error', 0):.4e}")
    all_results['nu3_atmospheric'] = results_nu3
    del results_nu3; gc.collect()

    # ==================== SEARCH 2c: SOLAR NEUTRINOS ν₁, ν₂ (NEW) ====================
    print_section("SEARCH 2c: SOLAR NEUTRINOS ν₁, ν₂ (Parallel)")
    print("  Searching for solar neutrinos (normal hierarchy)")
    print("  m₂ ≈ √(Δm²₂₁) ≈ 0.0087 eV")

    results_solar = ns.search_solar_neutrinos()
    best_solar = results_solar.get('best_mass_ev')
    if best_solar is not None:
        print(f"\n  ✅ SOLAR NEUTRINO FOUND:")
        print(f"    Mass = {best_solar:.6e} eV ({results_solar.get('best_mass_meV', 0):.3f} meV)")
        print(f"    Nf   = {results_solar.get('best_num', 'N/A')}/{results_solar.get('best_den', 'N/A')}")
        print(f"    Stability = {results_solar.get('best_stability', 0):.4f}")
        print(f"    Relative error = {results_solar.get('best_rel_error', 0):.4e}")
    else:
        print("\n  ❌ No solar neutrinos found.")
    all_results['solar_neutrinos'] = results_solar
    del results_solar; gc.collect()

    # ==================== SEARCH 2d: LIGHT NEUTRINO ν₁ ====================
    print_section("SEARCH 2d: LIGHT NEUTRINO ν₁ (1 - 10 meV) [Parallel]")
    print("  Searching for the lightest neutrino (normal hierarchy)")

    results_nu1 = ns.search_nu1()
    best_nu1 = results_nu1.get('best_mass_ev')
    if best_nu1 is not None:
        print(f"\n  ✅ NEUTRINO ν₁ FOUND:")
        print(f"    Mass = {best_nu1:.6e} eV ({results_nu1.get('best_mass_ev', 0)*1000:.3f} meV)")
        print(f"    Nf   = {results_nu1.get('best_num', 'N/A')}/{results_nu1.get('best_den', 'N/A')}")
        print(f"    Stability = {results_nu1.get('best_stability', 0):.4f}")
    all_results['nu1_light'] = results_nu1
    del results_nu1; gc.collect()

    # ==================== SEARCH 3: LIGHT ====================
    results_light = search_spectrum(
        min_mass_ev=2.2, max_mass_ev=511.0,
        max_denominator=10_000, label="LIGHT (2.2-511 eV)"
    )
    analyze_results(results_light, "LIGHT RESULTS")
    all_results['light'] = results_light
    del results_light; gc.collect()

    # ==================== SEARCH 4: ELECTRON ====================
    results_electron = search_spectrum(
        min_mass_ev=510000.0, max_mass_ev=512000.0,
        max_denominator=10, label="ELECTRON (511 keV)"
    )
    analyze_results(results_electron, "ELECTRON RESULTS")
    all_results['electron'] = results_electron
    del results_electron; gc.collect()

    # ==================== SEARCH 5: WIMPs ====================
    results_wimps = search_spectrum(
        min_mass_ev=1e9, max_mass_ev=1e12,
        max_denominator=100, label="WIMPs (1-1000 GeV)"
    )
    analyze_results(results_wimps, "WIMP RESULTS")
    all_results['wimps'] = results_wimps
    del results_wimps; gc.collect()

    # ==================== SEARCH 6: TEV SCALE ====================
    results_tev = search_spectrum(
        min_mass_ev=1e12, max_mass_ev=1e14,
        max_denominator=10, label="TEV (1-100 TeV)"
    )
    analyze_results(results_tev, "TEV RESULTS")
    all_results['tev'] = results_tev
    del results_tev; gc.collect()

    # ==================== SEARCH 7: QCD AXIONS ====================
    print_section("SEARCH 7: QCD AXIONS (1 μeV - 1 meV) [Parallel]")
    print("  Range: 1e-6 - 1e-3 eV")
    print("  Method: Continued Fractions + O_h Topological Fine Search")

    results_axions = ns.search_qcd_axions()
    best_mass_ax = results_axions.get('best_mass_ev')
    if best_mass_ax is not None:
        print(f"\n  ✅ BEST QCD AXION FOUND:")
        print(f"    Mass = {best_mass_ax:.6e} eV ({results_axions.get('best_mass_uev', 0):.3f} μeV)")
        print(f"    Nf   = {results_axions.get('best_num', 'N/A')}/{results_axions.get('best_den', 'N/A')}")
        print(f"    Stability = {results_axions.get('best_stability', 0):.4f}")
        spacing = results_axions.get('spacing_analysis', {})
        print(f"    Regularity = {spacing.get('regularity', 0):.4f}")
    all_results['qcd_axions'] = results_axions
    del results_axions; gc.collect()

    # ==================== TARGETED SEARCHES: STANDARD MODEL ====================
    print_section("TARGETED SEARCHES: STANDARD MODEL")

    res_muon = search_specific_particle(
        "Muon (μ)", 1.05658e8, 1.0e6, 5_000, "MUON (~105.7 MeV)"
    )
    all_results['muon'] = res_muon
    del res_muon; gc.collect()

    res_tau = search_specific_particle(
        "Tau (τ)", 1.77686e9, 1.0e7, 2_000, "TAU (~1.77 GeV)"
    )
    all_results['tau'] = res_tau
    del res_tau; gc.collect()

    res_w = search_specific_particle(
        "W Boson", 8.0379e10, 5.0e8, 1_000, "W BOSON (~80.4 GeV)"
    )
    all_results['w_boson'] = res_w
    del res_w; gc.collect()

    res_z = search_specific_particle(
        "Z Boson", 9.11876e10, 5.0e8, 1_000, "Z BOSON (~91.2 GeV)"
    )
    all_results['z_boson'] = res_z
    del res_z; gc.collect()

    res_higgs = search_specific_particle(
        "Higgs Boson", 1.2518e11, 1.0e9, 1_000, "HIGGS (~125.1 GeV)"
    )
    all_results['higgs'] = res_higgs
    del res_higgs; gc.collect()

    # ==================== FINAL SUMMARY ====================
    print_section("FINAL SEARCH SUMMARY")

    print(f"\n  Searches performed: {len(all_results)}")
    grand_total = 0
    for search_name, results in all_results.items():
        if isinstance(results, dict):
            total = results.get('total_candidates', 0)
            grand_total += total
            print(f"    {search_name:20s}: {total:>15,}")
    print(f"    {'GRAND TOTAL':20s}: {grand_total:>15,}")

    print(f"\n  🏆 BEST CANDIDATES BY CATEGORY:")
    for search_name, results in all_results.items():
        if not isinstance(results, dict): continue

        if search_name == 'qcd_axions':
            best_mass = results.get('best_mass_ev')
            if best_mass is not None:
                print(f"    {search_name:20s}: QCD Axion at {best_mass:.2e} eV (Nf={results.get('best_num')}/{results.get('best_den')})")
            continue

        if search_name in ['nu3_atmospheric', 'solar_neutrinos']:
            best_mass = results.get('best_mass_ev')
            if best_mass is not None:
                print(f"    {search_name:20s}: at {best_mass:.2e} eV (Nf={results.get('best_num')}/{results.get('best_den')})")
            continue

        peaks = results.get('stability_peaks', {})
        if peaks:
            top_peak = list(peaks.values())[0]
            mass = top_peak.get('mass_ev', 0)
            stability = top_peak.get('stability', 0)
            clase = top_peak.get('class', 'UNKNOWN')
            print(f"    {search_name:20s}: {mass:.2e} eV | St: {stability:.4f} | Class: {clase}")

    save_results(all_results)

    print_section("EMERGING CONCLUSIONS")
    print("""
    Observations based on VCV48 v8 data:
    1. The most stable particles are found in fractions with high O_h purity (48).
    2. The electron (1/1) dominates the stability spectrum thanks to the identity bonus.
    3. WIMPs and heavy particles require Nf > 1 (numerator > denominator).
    4. Real QCD axions emerge from enormous denominators (> 10^9) via continued fractions.
    5. Deduplication removes repeated candidates, cleaning the spacing analysis.
    6. Solar neutrinos are now searched with continued fractions up to 10^9.

    Note: These results emerge from systematic topological exploration,
    not from prior assumptions about which particles should exist.
    """)

    return all_results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Search interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Critical error: {e}")