#!/usr/bin/env python3
"""                           WAVE_SWITCH.py — VPM Model with Density Switch          ======================================================================                    Implements environmental phase transition:
  - ρ < ρ_crit: elastic phase (intact O_h lattice) → f_comp = VPM sigmoid
  - ρ ≥ ρ_crit: saturated phase (standard GR) → f_comp = 1.0

Finds the optimal ρ_crit that maximizes global correlation reduction.
"""

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.append('target/release')
from vpm_wave import VPMWaveEngine
engine = VPMWaveEngine()

GAMMA = 1.0/3.0
N_SCAN = 50  # Points to scan ρ_crit

# VPM parameters calibrated on Q1 SDSS (field)
VPM_PARAMS = np.array([3.5801, 0.9739, 156.6, 38.4])

# ================================================================
# 1. LOAD DATA AND DENSITY
# ================================================================
print("=" * 70)
print("  WAVE_SWITCH.py — VPM MODEL WITH DENSITY SWITCH")
print("=" * 70)

print("\n[1/3] Loading DESI LRG...")
lrg = Table.read('data/DATASET_LRG_VDISP_FLUXR_FINAL.fits')
mask = (lrg['VDISP'] > 50) & (lrg['VDISP'] < 500) & (lrg['Z'] > 0.05) & (lrg['Z'] < 0.5)
lrg = lrg[mask]

z_all = np.array(lrg['Z'], dtype=np.float64)
vd_all = np.array(lrg['VDISP'], dtype=np.float64)
ra_all = np.radians(np.array(lrg['RA'], dtype=np.float64))
dec_all = np.radians(np.array(lrg['DEC'], dtype=np.float64))

dc_all = np.array([engine.distancia_comovil(float(z)) for z in z_all], dtype=np.float64)
xyz_all = np.column_stack([dc_all*np.cos(dec_all)*np.cos(ra_all),
                            dc_all*np.cos(dec_all)*np.sin(ra_all),
                            dc_all*np.sin(dec_all)])

tree = cKDTree(xyz_all)
dists, _ = tree.query(xyz_all, k=51)
dens = 50.0 / (4.0/3.0 * np.pi * dists[:,-1]**3)

# ================================================================
# 2. FUNCTIONS
# ================================================================

def xi_vec(z_arr):
    return np.array([engine.xi_vpm(float(zi)) for zi in z_arr], dtype=np.float64)

xi_all = xi_vec(z_all)
ratio_base = 1.0 + xi_all

def f_comp_vpm(vd, z, p):
    sm, fc, s0, sh = p
    su = s0 * (1.0 + z)**GAMMA
    sb = 1.5
    eff_sh = sh * sb
    x = (vd - su) / eff_sh
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    f_comp_val = 1.0 - (1.0 - fc) * sigmoid
    return np.clip(f_comp_val, fc, 1.0)

def evaluar_switch(rho_crit):
    """
    Applies the density switch:
      - ρ < ρ_crit → f_comp = VPM sigmoid
      - ρ ≥ ρ_crit → f_comp = 1.0
    Returns global residual ρ².
    """
    mask_elastico = dens < rho_crit
    mask_saturado = ~mask_elastico

    if mask_elastico.sum() < 100:
        return 1.0, 1.0, 0, 0

    ratio_corr = np.zeros_like(ratio_base)

    # Elastic phase
    ratio_corr[mask_elastico] = ratio_base[mask_elastico] / f_comp_vpm(
        vd_all[mask_elastico], z_all[mask_elastico], VPM_PARAMS
    )

    # Saturated phase (GR)
    ratio_corr[mask_saturado] = ratio_base[mask_saturado]

    # Global correlation
    r, _ = pearsonr(ratio_corr, vd_all)

    # Also measure correlation in each phase separately
    r_elastico, _ = pearsonr(ratio_corr[mask_elastico], vd_all[mask_elastico])
    r_saturado, _ = pearsonr(ratio_corr[mask_saturado], vd_all[mask_saturado])

    return r**2, r_elastico**2, r_saturado**2, mask_elastico.sum()

# ================================================================
# 3. SCAN ρ_crit
# ================================================================
print(f"\n[2/3] Scanning ρ_crit ({N_SCAN} points)...")

# Scan range: percentiles 1 to 30
rho_min = np.percentile(dens, 1)
rho_max = np.percentile(dens, 30)
rho_scan = np.logspace(np.log10(rho_min), np.log10(rho_max), N_SCAN)

resultados_scan = []

for rho_c in rho_scan:
    rho2_global, rho2_elastico, rho2_saturado, n_elastico = evaluar_switch(rho_c)
    resultados_scan.append({
        'rho_crit': rho_c,
        'rho2_global': rho2_global,
        'rho2_elastico': rho2_elastico,
        'rho2_saturado': rho2_saturado,
        'n_elastico': n_elastico,
        'frac_elastico': n_elastico / len(dens) * 100
    })

# Find the optimum
mejor = min(resultados_scan, key=lambda x: x['rho2_global'])
rho_null_global = pearsonr(ratio_base, vd_all)[0]**2

print(f"\n  Global ρ²_null = {rho_null_global:.6f}")
print(f"  Best ρ² with switch = {mejor['rho2_global']:.6f}")
print(f"  Reduction = {(1 - mejor['rho2_global']/rho_null_global)*100:.1f}%")
print(f"  Optimal ρ_crit = {mejor['rho_crit']:.2e}")
print(f"  Galaxies in elastic phase: {mejor['n_elastico']:,} ({mejor['frac_elastico']:.1f}%)")

# ================================================================
# 4. DETAILED ANALYSIS AT OPTIMAL ρ_crit
# ================================================================
print(f"\n[3/3] Detailed analysis at optimal ρ_crit...")

mask_elastico = dens < mejor['rho_crit']
mask_saturado = ~mask_elastico

# Metrics per phase
for nombre, mask_fase in [('ELASTIC (VPM)', mask_elastico), ('SATURATED (GR)', mask_saturado)]:
    z_fase = z_all[mask_fase]
    vd_fase = vd_all[mask_fase]
    n_fase = mask_fase.sum()

    r_null_fase, _ = pearsonr(1.0 + xi_vec(z_fase), vd_fase)

    if nombre.startswith('ELASTIC'):
        ratio_fase = ratio_base[mask_fase] / f_comp_vpm(vd_fase, z_fase, VPM_PARAMS)
    else:
        ratio_fase = ratio_base[mask_fase]

    r_fase, _ = pearsonr(ratio_fase, vd_fase)

    print(f"\n  {nombre} phase:")
    print(f"    Galaxies: {n_fase:,} ({100*n_fase/len(dens):.1f}%)")
    print(f"    z: [{z_fase.min():.3f}, {z_fase.max():.3f}], median: {np.median(z_fase):.3f}")
    print(f"    σ: [{vd_fase.min():.0f}, {vd_fase.max():.0f}], median: {np.median(vd_fase):.0f}")
    print(f"    Density: [{dens[mask_fase].min():.2e}, {dens[mask_fase].max():.2e}]")
    print(f"    ρ²_null = {r_null_fase**2:.6f}")
    print(f"    ρ²_model = {r_fase**2:.6f}")
    print(f"    Reduction = {(1 - r_fase**2/max(r_null_fase**2, 1e-30))*100:.1f}%")

# Comparison with the model without switch
r_sin_switch, _ = pearsonr(ratio_base / f_comp_vpm(vd_all, z_all, VPM_PARAMS), vd_all)
rho2_sin_switch = r_sin_switch**2

print(f"\n  FINAL COMPARISON:")
print(f"    Without switch (global VPM):   ρ² = {rho2_sin_switch:.6f}")
print(f"    With switch (ρ_crit={mejor['rho_crit']:.2e}): ρ² = {mejor['rho2_global']:.6f}")
print(f"    Null (global GR):              ρ² = {rho_null_global:.6f}")

if mejor['rho2_global'] < rho_null_global and mejor['rho2_global'] < rho2_sin_switch:
    print(f"\n  ✅ The density switch IMPROVES both models.")
elif mejor['rho2_global'] < rho_null_global:
    print(f"\n  ✅ The switch improves over GR, but not over global VPM.")
else:
    print(f"\n  ⚠️  The switch does not improve over GR.")

print(f"\n{'='*70}")
print(f"  WAVE_SWITCH.py completed.")
print(f"{'='*70}")