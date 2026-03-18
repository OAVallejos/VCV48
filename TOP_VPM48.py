#!/usr/bin/env python3     
"""
ULTRA-FINE SCAN FOR THE TOP QUARK
Step 1 around the best candidate                    """                         
import argparse
import sys
import time
from datetime import datetime
import json

try:
    import vpm48_engine_top as vpm
    print("✅ VPM-48 Ultra engine loaded")
except ImportError:
    print("❌ Error: compile with: maturin develop --release")
    sys.exit(1)

TOP_MASS = 172.76
CENTER = 51940366
RADIUS = 500  # Search ±500 around the center

def main():
    print("\n" + "🚀"*40)
    print("🚀 TOP QUARK ULTRA-FINE SCAN - Residue 46")
    print("🚀"*40 + "\n")

    print(f"🎯 Target mass: {TOP_MASS} GeV")
    print(f"🎯 Center: {CENTER:,}")
    print(f"🎯 Radius: ±{RADIUS}")
    print(f"🎯 Step: 1\n")

    start = time.time()
    res = vpm.escaneo_ultra_fino(CENTER, RADIUS, True)

    best = res.get('mejores', [])
    elapsed = res.get('tiempo_segundos', 0)

    print(f"\n✅ Scan completed in {elapsed:.1f} seconds")

    if best and len(best) > 0:
        top_result = best[0]
        nf, mass, error = top_result

        print("\n" + "🏆"*40)
        print(f"🏆 ABSOLUTE CHAMPION:")
        print(f"🏆 Nf = {nf:,}")
        print(f"🏆 Mass = {mass:.6f} GeV")
        print(f"🏆 Error = {error:.6f}%")
        print("🏆"*40)

        # Final verification
        if error < 0.001:
            print("\n✨✨✨ LABORATORY PRECISION! ✨✨✨")

        # Save results
        filename = f"top_ultra_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(filename, 'w') as f:
            json.dump({
                'best': {'nf': nf, 'mass': mass, 'error': error},
                'top10': best[:10]
            }, f, indent=2)
        print(f"\n💾 Results saved in {filename}")

if __name__ == "__main__":
    main()