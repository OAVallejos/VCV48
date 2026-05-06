#!/usr/bin/env python3     
"""
TEST_PHASE.py — Phase Stacking with geometric prediction of φ_offset
======================================================================
Pre-registered prediction:
    φ_offset = arctan(1/√2) + 5×(δ_CMB/K_inv)×√(2/3)
             = 0.646555 rad (37.04°)

Derivation: O_h geometry + vacuum elastic compliance

v3 — Hemisphere test + VDISP purity cuts
"""

import numpy as np
from scipy import stats
from datetime import datetime
import json
import sys

# ============================================================================
# IMPORT RUST MODULE
# ============================================================================
try:
    import phase_core
    print("✅ Rust module phase_core loaded.")
except ImportError:
    print("❌ Rust module not available. Compile with: maturin develop --release")
    sys.exit()

# ============================================================================
# CONFIGURATION
# ============================================================================

PHASE_OFFSET_PREDICTED = 0.646555  # rad — geometrically derived
A0 = 14.075                        # Mpc

# Critical directions for hemisphere test (RA_deg, Dec_deg)
DIRECTIONS = [
    ('CMB_Dipole',         168.0,   -7.0),
    ('Axis_of_Evil',       260.0,   60.0),
    ('SDSS_Alignment',       0.0,   90.0),
    ('CMB_Antipode',       348.0,    7.0),
    ('Galactic_Plane_N',     0.0,   30.0),
    ('Galactic_Plane_S',     0.0,  -30.0),
]

# VDISP cuts for purity test
VDISP_CUTS = [262, 300, 350, 400]

print("=" * 70)
print("🔬 PHASE STACKING — VCV48 (v3 — Hemispheres + Purity)")
print(f"   φ_offset = {PHASE_OFFSET_PREDICTED:.6f} rad ({PHASE_OFFSET_PREDICTED*180/np.pi:.2f}°)")
print(f"   A0 = {A0:.3f} Mpc")
print("=" * 70)

# ============================================================================
# FUNCTIONS
# ============================================================================

def ra_dec_to_xyz(ra_deg, dec_deg):
    """Converts RA, Dec (degrees) to Cartesian unit vector."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def hemisphere_test(engine, ra_arr, dec_arr, z_arr, vd_arr, vdisp_cut, dir_name, ra_dir, dec_dir):
    """Phase Stacking in hemispheres relative to a direction."""
    n_hat = ra_dec_to_xyz(ra_dir, dec_dir)

    ra_rad = np.radians(ra_arr)
    dec_rad = np.radians(dec_arr)
    gx = np.cos(dec_rad) * np.cos(ra_rad)
    gy = np.cos(dec_rad) * np.sin(ra_rad)
    gz = np.sin(dec_rad)
    proj = gx*n_hat[0] + gy*n_hat[1] + gz*n_hat[2]

    results = {}

    for hemi, mask_hemi in [('+', proj > 0.01), ('-', proj < -0.01)]:
        mask_vd = vd_arr > vdisp_cut
        z_hemi = z_arr[mask_hemi & mask_vd]

        if len(z_hemi) < 100:
            continue

        r, phi_obs, dphi, dphi_deg, z_stat, p_val, n_gal, sigma = engine.phase_stacking(
            z_hemi.tolist()
        )

        results[hemi] = {
            'n': int(n_gal), 'R': float(r),
            'phi_rad': float(phi_obs), 'phi_deg': float(phi_obs*180/np.pi),
            'dphi_rad': float(dphi), 'dphi_deg': float(dphi_deg),
            'Z': float(z_stat), 'p': float(p_val), 'sigma': float(sigma),
        }

    return results


# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n📥 [1/4] Loading data...")

# SDSS
try:
    data = np.load('data/sdss_vdisp_calidad.npz')
    m = (data['VDISP'] > 262) & (data['Z'] > 0.03) & (data['Z'] < 0.15)
    z_sdss = data['Z'][m]; ra_sdss = data['RA'][m]; dec_sdss = data['DEC'][m]; vd_sdss = data['VDISP'][m]
    print(f"   SDSS: {len(z_sdss):,} galaxies")
except FileNotFoundError:
    print("   ⚠️  SDSS not found")
    z_sdss = ra_sdss = dec_sdss = vd_sdss = np.array([])

# DESI
try:
    from astropy.table import Table
    t = Table.read('data/DATASET_LRG_VDISP_FLUXR_FINAL.fits')
    vd = np.array(t['VDISP']); zz = np.array(t['Z']); ra = np.array(t['RA']); dec = np.array(t['DEC'])
    m = (vd > 262) & (zz > 0.4) & (zz < 1.0)
    z_desi = zz[m]; ra_desi = ra[m]; dec_desi = dec[m]; vd_desi = vd[m]
    print(f"   DESI: {len(z_desi):,} galaxies")
except FileNotFoundError:
    print("   ⚠️  DESI not found")
    z_desi = ra_desi = dec_desi = vd_desi = np.array([])

# ============================================================================
# 2. DIAGNOSTIC
# ============================================================================

print("\n📐 [2/4] φ_offset diagnostic...")
engine = phase_core.PhaseStackingEngine(70.0, 0.315, 0.685)

for name, val in engine.diagnostic():
    if name in ('phi_base', 'phi_base_deg', 'delta_cmb', 'K_inv', 'compliance',
                 'delta_phi', 'phi_offset_calc', 'phi_offset_deg', 'sdss_measured', 'difference_abs'):
        print(f"   {name:<20} = {val:>12.6f}")

# ============================================================================
# 3. GLOBAL STACKING
# ============================================================================

print("\n🎯 [3/4] GLOBAL Phase Stacking...")

for name, zz in [('SDSS', z_sdss), ('DESI', z_desi)]:
    if not len(zz): continue
    r, ph, dp, dpd, zs, pv, ng, sg = engine.phase_stacking(zz.tolist())
    print(f"\n   {name}: N={ng:,}  R={r:.4f}  φ_obs={ph*180/np.pi:.1f}°  "
          f"Δφ={dpd:.1f}°  Z={zs:.2f}  p={pv:.4f}  σ={sg:.2f}")

# ============================================================================
# 4. HEMISPHERE TEST
# ============================================================================

print(f"\n🌍 [4/4] Hemisphere test + VDISP cuts...")

all_tests = []
n_tests = 0

for name, zz, ra_a, dec_a, vd_a in [
    ('DESI', z_desi, ra_desi, dec_desi, vd_desi),
    ('SDSS', z_sdss, ra_sdss, dec_sdss, vd_sdss),
]:
    if not len(zz): continue

    cuts = VDISP_CUTS if name == 'DESI' else [262]

    for vdc in cuts:
        n_ok = (vd_a > vdc).sum()
        if n_ok < 200: continue

        for dn, dr, dd in DIRECTIONS[:3] if name == 'SDSS' else DIRECTIONS:
            res = hemisphere_test(engine, ra_a, dec_a, zz, vd_a, vdc, dn, dr, dd)
            for hemi, d in res.items():
                n_tests += 1
                key = f"{name}_VDISP>{vdc}_{dn}_{hemi}"
                all_tests.append((key, d))

                if d['p'] < 0.3 or d['sigma'] > 1.0:
                    flag = "⭐" if d['p'] < 0.05 else "  "
                    print(f"   {flag} {key:<45s} R={d['R']:.4f}  φ={d['phi_deg']:.1f}°  "
                          f"Δφ={d['dphi_deg']:.1f}°  p={d['p']:.4f}  σ={d['sigma']:.2f}  N={d['n']:,}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("📊 SUMMARY")

if all_tests:
    all_tests.sort(key=lambda x: x[1]['p'])

    print(f"\n   Top 10:")
    print(f"   {'Rank':<5} {'Configuration':<48} {'p':<10} {'σ':<7} {'N':<10}")
    print(f"   {'-'*80}")
    for i, (k, d) in enumerate(all_tests[:10], 1):
        print(f"   {i:<5} {k:<48} {d['p']:<10.6f} {d['sigma']:<7.2f} {d['n']:<10,}")

    best_p = all_tests[0][1]['p']
    bonf = min(best_p * n_tests, 1.0)
    print(f"\n   Best p: {best_p:.6f}  |  Tests: {n_tests}  |  Bonferroni: {bonf:.6f}")

    if bonf < 0.05:       print("   ✅ SIGNIFICANT")
    elif best_p < 0.05:   print("   ⚠️  Locally significant, does not survive correction")
    else:                 print("   ❌ No signal")
else:
    print("\n   No results.")

# Save
out = {
    'date': datetime.now().isoformat(),
    'phi_offset': PHASE_OFFSET_PREDICTED,
    'n_tests': n_tests,
    'results': [(k, d) for k, d in all_tests],
}
fn = f'phase_stacking_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(fn, 'w') as f: json.dump(out, f, indent=2)
print(f"\n✅ {fn}")
print("=" * 70)