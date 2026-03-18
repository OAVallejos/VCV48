#!/usr/bin/env python3
"""                         🔬 VPM-48 FINE HUNTER - HIGH PRECISION SEARCH      Fine scanning around candidates with specific residues                       """

import vcv48 as vpm
import json
import time
from datetime import datetime
from collections import defaultdict

class FineHunter:
    def __init__(self):
        """Initializes the hunter with engine constants"""
        self.constants = {
            'OH_ORDER': vpm.OH_ORDER,
            'ALPHA_E': vpm.ALPHA_E,
            'ALPHA_EM': vpm.ALPHA_EM,
            'BOSON_SCALE': getattr(vpm, 'BOSON_SCALE', 0.01871),
            'BOSON_CORRECTION': getattr(vpm, 'BOSON_CORRECTION', 1.0827),
            'BOSON_NF_OFFSET': getattr(vpm, 'BOSON_NF_OFFSET', 220000),
            'M_W_EXP': getattr(vpm, 'M_W_EXP', 80.379),
            'M_Z_EXP': getattr(vpm, 'M_Z_EXP', 91.1876),
            'M_HIGGS_EXP': getattr(vpm, 'M_HIGGS_EXP', 125.18)
        }

        # Search configuration by particle
        self.particles = {
            'W': {
                'nf_center': 15743000,
                'expected_mass': self.constants['M_W_EXP'],
                'residue': 8,
                'radius': 500000,  # Increased to capture offset
                'coarse_step': 10000,
                'fine_step': 2000,
                'name': 'W Boson'
            },
            'Z': {
                'nf_center': 20176000,
                'expected_mass': self.constants['M_Z_EXP'],
                'residue': 16,
                'radius': 500000,
                'coarse_step': 10000,
                'fine_step': 2000,
                'name': 'Z Boson'
            },
            'Higgs': {
                'nf_center': 37625000,
                'expected_mass': self.constants['M_HIGGS_EXP'],
                'residue': 8,
                'radius': 500000,
                'coarse_step': 10000,
                'fine_step': 2000,
                'name': 'Higgs Boson'
            }
        }

        self.results = defaultdict(list)
        self.best_candidates = {}

    def scan_by_residue(self, residue, nf_min, nf_max, step=5000, verbose=True):
        """Scans an Nf range filtering by residue"""
        if verbose:
            print(f"\n🔍 Scanning residue {residue}: Nf={nf_min:,} - {nf_max:,} (step={step:,})")
            print("-" * 70)

        results = []
        current_nf = nf_min
        counter = 0

        while current_nf <= nf_max:
            # Adjust to next multiple with correct residue
            current_residue = current_nf % 48
            if current_residue != residue:
                current_nf += (residue - current_residue) % 48
                continue

            try:
                res = vpm.analizar_por_nf(current_nf, 3)

                # Calculate mass in GeV
                mass_gev = res['energia_ev'] / 1e9
                stability = res['estabilidad_oh']

                # Apply offset correction if available
                corrected_nf = current_nf + self.constants['BOSON_NF_OFFSET']

                results.append({
                    'nf': current_nf,
                    'corrected_nf': corrected_nf,
                    'mass_gev': mass_gev,
                    'stability': stability,
                    'residue': residue,
                    'type': res['tipo']
                })

                counter += 1

                # Show progress
                if verbose and counter % 10 == 0:
                    print(f"   Nf={current_nf:,} (corr={corrected_nf:,}) | Mass={mass_gev:.3f} GeV | {res['tipo']}")

            except Exception as e:
                if verbose:
                    print(f"   Error at Nf={current_nf}: {e}")

            current_nf += step * 48  # Keep residue constant

        if verbose:
            print(f"\n✅ Total found: {len(results)}")

        return results

    def full_range_scan(self, nf_min, nf_max, step=48, residues=None, verbose=True):
        """Scans a complete Nf range"""
        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 FULL SCAN: Nf={nf_min:,} - {nf_max:,}")
            print(f"{'='*70}")

        results_by_residue = defaultdict(list)
        total = 0
        start = time.time()

        for nf in range(nf_min, nf_max + 1, step):
            try:
                res = vpm.analizar_por_nf(nf, 3)
                residue = res['residuo']

                # Filter by residues if specified
                if residues is not None and residue not in residues:
                    continue

                mass_gev = res['energia_ev'] / 1e9
                stability = res['estabilidad_oh']
                corrected_nf = nf + self.constants['BOSON_NF_OFFSET']

                results_by_residue[residue].append({
                    'nf': nf,
                    'corrected_nf': corrected_nf,
                    'mass_gev': mass_gev,
                    'stability': stability,
                    'type': res['tipo']
                })

                total += 1

                # Show progress
                if verbose and total % 100 == 0:
                    print(f"   Progress: Nf={nf:,} | Found: {total}")

            except Exception as e:
                if verbose:
                    print(f"   Error at Nf={nf}: {e}")

        elapsed = time.time() - start

        if verbose:
            print(f"\n✅ Scan completed in {elapsed:.1f} seconds")
            print(f"   Total configurations found: {total}")

            # Show statistics by residue
            print(f"\n📊 Distribution by residue:")
            for residue in sorted(results_by_residue.keys()):
                print(f"   Residue {residue:2d}: {len(results_by_residue[residue])} configurations")

        return results_by_residue

    def save_results(self, filename=None):
        """Saves results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fine_hunt_{timestamp}.json'

        # Prepare data for saving
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'constants': self.constants,
            'best_bosons': self.best_candidates,
            'summary': {}
        }

        # Add best bosons
        for name, best in self.best_candidates.items():
            error = abs(best['mass_gev'] - self.particles[name]['expected_mass']) / self.particles[name]['expected_mass'] * 100
            results_json['summary'][name] = {
                'nf': best['nf'],
                'corrected_nf': best.get('corrected_nf', best['nf'] + self.constants['BOSON_NF_OFFSET']),
                'mass_gev': best['mass_gev'],
                'error_percentage': error,
                'stability': best['stability'],
                'type': best['type']
            }

        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\n💾 Results saved in '{filename}'")
        return filename

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fine Hunter VPM-48')
    parser.add_argument('--mode', choices=['bosons', 'graviton', 'full'],
                       default='bosons', help='Search mode')
    parser.add_argument('--nf-min', type=int, default=15000000,
                       help='Minimum Nf for scanning')
    parser.add_argument('--nf-max', type=int, default=38000000,
                       help='Maximum Nf for scanning')
    parser.add_argument('--step', type=int, default=48,
                       help='Scan step')
    parser.add_argument('--residues', type=int, nargs='+', default=None,
                       help='Residues to filter (e.g., --residues 8 16)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Show detailed results')

    args = parser.parse_args()

    hunter = FineHunter()

    print("\n" + "="*70)
    print("🔬 VPM-48 FINE HUNTER - STARTING")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Nf range: {args.nf_min:,} - {args.nf_max:,}")
    print(f"Step: {args.step}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.residues:
        print(f"Filtering residues: {args.residues}")

    # Execute according to mode
    if args.mode == 'bosons':
        # Full range scan with boson residue filter
        boson_residues = args.residues if args.residues else [8, 16]
        results = hunter.full_range_scan(
            args.nf_min,
            args.nf_max,
            step=args.step,
            residues=boson_residues,
            verbose=args.verbose
        )

        # Search for best candidates
        print(f"\n{'='*70}")
        print("🏆 SEARCHING FOR BEST CANDIDATES")
        print(f"{'='*70}")

        for residue in [8, 16]:
            if residue in results:
                candidates = results[residue]
                if candidates:
                    # For residue 8, search near W and Higgs
                    if residue == 8:
                        # Near W (~15.7M)
                        near_w = [c for c in candidates if abs(c['nf'] - 15743000) < 500000]
                        if near_w:
                            best_w = min(near_w, key=lambda x: abs(x['mass_gev'] - 80.379))
                            print(f"\n📌 Best candidate for W (residue 8):")
                            print(f"   Nf={best_w['nf']:,} | Mass={best_w['mass_gev']:.3f} GeV | Error={abs(best_w['mass_gev']-80.379)/80.379*100:.3f}%")
                            hunter.best_candidates['W'] = best_w

                        # Near Higgs (~37.6M)
                        near_higgs = [c for c in candidates if abs(c['nf'] - 37625000) < 500000]
                        if near_higgs:
                            best_higgs = min(near_higgs, key=lambda x: abs(x['mass_gev'] - 125.18))
                            print(f"\n📌 Best candidate for Higgs (residue 8):")
                            print(f"   Nf={best_higgs['nf']:,} | Mass={best_higgs['mass_gev']:.3f} GeV | Error={abs(best_higgs['mass_gev']-125.18)/125.18*100:.3f}%")
                            hunter.best_candidates['Higgs'] = best_higgs

                    # For residue 16, search near Z (~20.2M)
                    if residue == 16:
                        near_z = [c for c in candidates if abs(c['nf'] - 20176000) < 500000]
                        if near_z:
                            best_z = min(near_z, key=lambda x: abs(x['mass_gev'] - 91.1876))
                            print(f"\n📌 Best candidate for Z (residue 16):")
                            print(f"   Nf={best_z['nf']:,} | Mass={best_z['mass_gev']:.3f} GeV | Error={abs(best_z['mass_gev']-91.1876)/91.1876*100:.3f}%")
                            hunter.best_candidates['Z'] = best_z

    elif args.mode == 'graviton':
        # Graviton search (residue 0)
        results = hunter.full_range_scan(
            max(0, args.nf_min),
            args.nf_max,
            step=args.step,
            residues=[0],
            verbose=args.verbose
        )

    elif args.mode == 'full':
        # Full scan without filtering
        results = hunter.full_range_scan(
            args.nf_min,
            args.nf_max,
            step=args.step,
            residues=args.residues,
            verbose=args.verbose
        )

    # Save results
    hunter.save_results()

    print("\n" + "="*70)
    print("✅ FINE HUNTER COMPLETED")
    print("="*70)