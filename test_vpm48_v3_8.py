#!/usr/bin/env python3
"""                         VPM-48 SMART HUNTER - INTEGRATED VERSION           
Search for W, Z and HIGGS with residue filter                         """
import argparse
import numpy as np
from datetime import datetime
import time
import sys
import json

try:
    import vpm48_engine_optimized as vpm
    print("✅ Optimized engine loaded")
except ImportError:
    print("❌ Error: Engine not found")
    sys.exit(1)

ALPHA_E = 22.67

class SmartHunter:
    def __init__(self, name, target, nf_center, range_size, initial_step=50000, tolerance=0.1, residue_filter=None):
        self.name = name
        self.target = target
        self.nf_center = nf_center
        self.range_size = range_size
        self.initial_step = initial_step
        self.tolerance = tolerance
        self.residue_filter = residue_filter
        self.coarse_results = []
        self.fine_results = []
        self.best = None

    def calculate_mass(self, nf):
        """Calculates effective mass for a given Nf"""
        try:
            res = vpm.analizar_por_nf(nf, 3)
            raw_mass = res['energia_ev'] / 1e9
            stability = res['estabilidad_oh']
            alpha_mass = raw_mass * ALPHA_E
            effective_mass = alpha_mass * (1 + (stability - 0.1042) * 1000)
            return effective_mass, stability
        except Exception as e:
            return None, None

    def intelligent_coarse_scan(self):
        """Coarse scan with automatic stop"""
        start = self.nf_center - self.range_size
        end = self.nf_center + self.range_size

        print(f"\n🎯 Searching for {self.name} (~{self.target} GeV)")
        if self.residue_filter is not None:
            print(f"🎯 Filtering by residue: {self.residue_filter}")
        print(f"📊 Coarse range: {start:,} - {end:,} (step {self.initial_step:,})")
        print(f"⏱️  Automatic stop when error < {self.tolerance}% and starts increasing")
        print("-" * 70)

        errors = []
        nfs = []

        for nf in range(start, end + 1, self.initial_step):
            # Apply residue filter if defined
            if self.residue_filter is not None and nf % 48 != self.residue_filter:
                continue

            mass, stability = self.calculate_mass(nf)
            if mass is None:
                continue

            error = abs(mass - self.target) / self.target * 100
            errors.append(error)
            nfs.append(nf)

            self.coarse_results.append({
                'nf': nf,
                'mass': mass,
                'error': error,
                'stability': stability,
                'residue': nf % 48
            })

            print(f"   Nf={nf:,} (residue {nf%48:2d}) | Mass={mass:.2f} GeV | Error={error:.3f}%")

            # MINIMUM DETECTION AND SMART STOP
            if len(errors) > 3:
                if errors[-1] > errors[-2] and min(errors[:-1]) < self.tolerance:
                    idx_min = np.argmin(errors[:-1])
                    nf_min = nfs[idx_min]
                    error_min = errors[idx_min]

                    print(f"\n🛑 MINIMUM DETECTED at Nf={nf_min:,} (error={error_min:.3f}%)")
                    print(f"   Stopping coarse scan...")
                    break

        # Find the best candidate
        if self.coarse_results:
            self.best = min(self.coarse_results, key=lambda x: x['error'])
            print(f"\n✨ BEST CANDIDATE (coarse):")
            print(f"   Nf={self.best['nf']:,} (residue {self.best['residue']})")
            print(f"   Mass={self.best['mass']:.3f} GeV")
            print(f"   Error={self.best['error']:.3f}%")
            print(f"   Stability={self.best['stability']:.6f}")

    def ultrafine_scan(self, radius=50000, fine_step=5000, ultra_step=1000):
        """Fine and ultrafine scan around the best candidate"""
        if not self.best:
            print("❌ No candidate for fine scan")
            return

        print(f"\n🔬 FINE SCAN around Nf={self.best['nf']:,}")
        print("-" * 70)

        # PHASE 1: Fine scan (step 5000)
        start_fine = max(1000000, self.best['nf'] - radius)
        end_fine = self.best['nf'] + radius

        best_fine_candidates = []

        for nf in range(start_fine, end_fine + 1, fine_step):
            mass, stability = self.calculate_mass(nf)
            if mass is None:
                continue

            error = abs(mass - self.target) / self.target * 100

            self.fine_results.append({
                'nf': nf,
                'mass': mass,
                'error': error,
                'stability': stability,
                'residue': nf % 48
            })

            # Show only if relevant
            if error < 0.5 or nf % (fine_step * 5) == 0:
                print(f"    📍 Nf={nf:,} (residue {nf%48:2d}) | Mass={mass:.3f} | Error={error:.3f}%")

            if error < self.best['error'] * 0.5:
                best_fine_candidates.append((nf, error))

        # PHASE 2: ULTRAFINE scan
        if best_fine_candidates:
            print(f"\n⚡ ULTRAFINE SCAN (step {ultra_step})")

            for nf_base, _ in best_fine_candidates[:3]:
                start_ultra = nf_base - 20000
                end_ultra = nf_base + 20000

                for nf in range(start_ultra, end_ultra + 1, ultra_step):
                    mass, stability = self.calculate_mass(nf)
                    if mass is None:
                        continue

                    error = abs(mass - self.target) / self.target * 100

                    self.fine_results.append({
                        'nf': nf,
                        'mass': mass,
                        'error': error,
                        'stability': stability,
                        'residue': nf % 48,
                        'ultrafine': True
                    })

                    if error < 0.01:  # BINGO!
                        print(f"\n🎯🎯🎯 BINGO! Nf={nf:,} (residue {nf%48})")
                        print(f"   Mass={mass:.3f} GeV (target={self.target})")
                        print(f"   Error={error:.4f}%")
                        print(f"   Stability={stability:.6f}")
                    elif error < 0.1 and nf % 10000 == 0:
                        print(f"   🔍 Nf={nf:,} (residue {nf%48:2d}) | Error={error:.3f}%")

        # Best final candidate
        if self.fine_results:
            best_final = min(self.fine_results, key=lambda x: x['error'])
            print(f"\n🏆 BEST FINAL CANDIDATE:")
            print(f"   Nf={best_final['nf']:,} (residue {best_final['residue']})")
            print(f"   Mass={best_final['mass']:.4f} GeV")
            print(f"   Error={best_final['error']:.4f}%")
            print(f"   Stability={best_final['stability']:.6f}")
            return best_final

    def save_results(self):
        """Saves all results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        residue_str = f"_res{self.residue_filter}" if self.residue_filter is not None else ""
        filename = f'{self.name.lower()}{residue_str}_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump({
                'particle': self.name,
                'target_gev': self.target,
                'residue_filter': self.residue_filter,
                'coarse_results': self.coarse_results,
                'fine_results': self.fine_results,
                'best': self.best
            }, f, indent=2)

        print(f"\n💾 Results saved in '{filename}'")

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smart VPM-48 Hunter')
    parser.add_argument('--particle', choices=['w', 'z', 'higgs'], required=True)
    parser.add_argument('--residue', type=int, choices=[0, 16, 24, 48],
                       help='Filter by specific residue (0, 16, 24)')

    args = parser.parse_args()

    # Particle-specific configuration
    if args.particle == 'w':
        hunter = SmartHunter(
            name="W_Boson",
            target=80.379,
            nf_center=15750000,
            range_size=1000000,
            initial_step=50000,
            residue_filter=args.residue
        )
    elif args.particle == 'z':
        hunter = SmartHunter(
            name="Z_Boson",
            target=91.1876,
            nf_center=20000000,
            range_size=2500000,
            initial_step=50000,
            residue_filter=args.residue
        )
    elif args.particle == 'higgs':
        hunter = SmartHunter(
            name="Higgs",
            target=125.18,
            nf_center=37650000,  # Adjusted according to W/Z pattern
            range_size=500000,    # Tighter range to save memory
            initial_step=50000,
            residue_filter=args.residue
        )

    print(f"\n🔍 CONFIGURATION:")
    print(f"   Particle: {args.particle.upper()}")
    print(f"   Target mass: {hunter.target} GeV")
    print(f"   Nf center: {hunter.nf_center:,}")
    print(f"   Range: ±{hunter.range_size:,}")
    if args.residue:
        print(f"   Filtered residue: {args.residue}")

    hunter.intelligent_coarse_scan()
    hunter.ultrafine_scan()
    hunter.save_results()