#!/usr/bin/env python3        
"""
WAVE_DENSIDAD.py — Correlation Reduction vs Local Density
======================================================================
Hypothesis:
  The VPM's ability to reduce spurious correlation depends on
  the local density of the cosmic environment.

  - Low density (field): intact O_h lattice → VPM reduces ρ².
  - High density (nodes): saturated O_h lattice → VPM adds nothing (standard GR).

Strategy:
  1. Calculate 3D local density for all DESI galaxies.
  2. Divide into N density bins (not just Q1/Q4).
  3. In each bin, measure ρ²_null and ρ²_VPM.
  4. Plot Δρ² = ρ²_null - ρ²_VPM vs density.
  5. If Δρ² decays with density, the hypothesis is confirmed.
"""

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import sys
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append('target/release')
from vpm_wave import VPMWaveEngine
engine = VPMWaveEngine()

GAMMA = 1.0/3.0
N_BINS_DENSIDAD = 10
N_MIN_POR_BIN = 200

# VPM parameters calibrated on Q1 SDSS (field)
VPM_PARAMS = np.array([3.5801, 0.9739, 156.6, 38.4])

# ================================================================
# 1. LOAD DESI AND CALCULATE LOCAL DENSITY
# ================================================================
print("=" * 70)
print("  WAVE_DENSIDAD.py — CORRELATION REDUCTION VS DENSITY")
print("=" * 70)

print("\n[1/4] Loading DESI LRG and calculating local density...")
lrg = Table.read('data/DATASET_LRG_VDISP_FLUXR_FINAL.fits')
mask = (lrg['VDISP'] > 50) & (lrg['VDISP'] < 500) & (lrg['Z'] > 0.05) & (lrg['Z'] < 0.5)
lrg = lrg[mask]
n_gal = len(lrg)
print(f"  Galaxies: {n_gal:,}")

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

print(f"  Density: [{dens.min():.2e}, {dens.max():.2e}]")
print(f"  Median: {np.median(dens):.2e}")

# ================================================================
# 2. VPM FUNCTIONS
# ================================================================

def xi_vec(z_arr):
    return np.array([engine.xi_vpm(float(zi)) for zi in z_arr], dtype=np.float64)

def f_comp(vd, z, p):
    sm, fc, s0, sh = p
    su = s0 * (1.0 + z)**GAMMA
    sb = 1.5
    eff_sh = sh * sb
    x = (vd - su) / eff_sh
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    f_comp_val = 1.0 - (1.0 - fc) * sigmoid
    return np.clip(f_comp_val, fc, 1.0)

# ================================================================
# 3. ANALYSIS BY DENSITY BINS
# ================================================================
print(f"\n[2/4] Dividing into {N_BINS_DENSIDAD} density bins...")

# Use percentiles to ensure enough galaxies per bin
percentiles = np.linspace(0, 100, N_BINS_DENSIDAD + 1)
dens_edges = np.percentile(dens, percentiles)
dens_centers = 0.5 * (dens_edges[:-1] + dens_edges[1:])

resultados = []

for i in range(N_BINS_DENSIDAD):
    mask_bin = (dens >= dens_edges[i]) & (dens < dens_edges[i+1])
    z_bin = z_all[mask_bin]
    vd_bin = vd_all[mask_bin]
    n_bin = np.sum(mask_bin)

    if n_bin < N_MIN_POR_BIN:
        continue

    xi_bin = xi_vec(z_bin)
    ratio_base = 1.0 + xi_bin

    # Null correlation
    r_null, p_null = pearsonr(ratio_base, vd_bin)
    rho2_null = r_null**2

    # Correlation with VPM
    f_comp_bin = f_comp(vd_bin, z_bin, VPM_PARAMS)
    ratio_vpm = ratio_base / f_comp_bin
    r_vpm, p_vpm = pearsonr(ratio_vpm, vd_bin)
    rho2_vpm = r_vpm**2

    # Metrics
    delta_rho2 = rho2_null - rho2_vpm  # Positive = VPM reduces correlation
    reduccion_pct = 100 * delta_rho2 / rho2_null if rho2_null > 0 else 0

    resultados.append({
        'densidad_media': dens_centers[i],
        'densidad_min': dens_edges[i],
        'densidad_max': dens_edges[i+1],
        'n_galaxias': n_bin,
        'z_mediana': np.median(z_bin),
        'sigma_mediana': np.median(vd_bin),
        'rho2_null': rho2_null,
        'rho2_vpm': rho2_vpm,
        'delta_rho2': delta_rho2,
        'reduccion_pct': reduccion_pct,
        'p_null': p_null,
        'p_vpm': p_vpm
    })

# ================================================================
# 4. RESULTS
# ================================================================
print(f"\n[3/4] Results per density bin:\n")

print(f"  {'Density':>12s} {'N':>7s} {'z_med':>7s} {'σ_med':>7s} {'ρ²_null':>10s} {'ρ²_VPM':>10s} {'Δρ²':>10s} {'Reduction':>10s} {'Verdict'}")
print(f"  {'-'*90}")

for r in resultados:
    if r['delta_rho2'] > 0:
        veredicto = "✅ VPM IMPROVES"
    elif r['delta_rho2'] < 0:
        veredicto = "❌ VPM WORSENS"
    else:
        veredicto = "≈ NO CHANGE"

    print(f"  {r['densidad_media']:12.2e} {r['n_galaxias']:7,} {r['z_mediana']:7.3f} {r['sigma_mediana']:7.0f} {r['rho2_null']:10.6f} {r['rho2_vpm']:10.6f} {r['delta_rho2']:+10.6f} {r['reduccion_pct']:+9.1f}% {veredicto}")

# ================================================================
# 5. TREND ANALYSIS
# ================================================================
print(f"\n[4/4] Trend analysis:")

densidades = np.array([r['densidad_media'] for r in resultados])
deltas = np.array([r['delta_rho2'] for r in resultados])
n_gal_bins = np.array([r['n_galaxias'] for r in resultados])

# Correlation between density and Δρ²
r_dens_delta, p_dens_delta = pearsonr(densidades, deltas)
print(f"  Density vs Δρ² correlation: ρ = {r_dens_delta:+.4f} (p = {p_dens_delta:.4f})")

# Count bins where VPM improves
bines_mejora = sum(1 for r in resultados if r['delta_rho2'] > 0)
bines_empeora = sum(1 for r in resultados if r['delta_rho2'] < 0)
print(f"  Bins where VPM improves:  {bines_mejora}/{len(resultados)}")
print(f"  Bins where VPM worsens:   {bines_empeora}/{len(resultados)}")

# Separate into low and high density
mediana_dens = np.median(densidades)
baja = [r for r in resultados if r['densidad_media'] < mediana_dens]
alta = [r for r in resultados if r['densidad_media'] >= mediana_dens]

delta_baja = np.mean([r['delta_rho2'] for r in baja])
delta_alta = np.mean([r['delta_rho2'] for r in alta])

print(f"\n  Mean Δρ² in low density (< median): {delta_baja:+.6f}")
print(f"  Mean Δρ² in high density (≥ median): {delta_alta:+.6f}")

if delta_baja > 0 and delta_alta <= 0:
    print(f"\n  ✅ HYPOTHESIS CONFIRMED:")
    print(f"     VPM reduces correlation in low density (field)")
    print(f"     and adds nothing in high density (nodes).")
    print(f"     This is consistent with O_h lattice saturation.")
elif delta_baja > delta_alta:
    print(f"\n  ⚠️  HYPOTHESIS PARTIALLY CONFIRMED:")
    print(f"     VPM works better in low density than in high density,")
    print(f"     but the difference is not conclusive.")
else:
    print(f"\n  ❌ HYPOTHESIS NOT CONFIRMED:")
    print(f"     No dependence on local density is observed.")

# ================================================================
# EXPORT
# ================================================================
output = {
    'parametros_vpm': {
        'strength_max': float(VPM_PARAMS[0]),
        'f_comp_min': float(VPM_PARAMS[1]),
        'sigma_0': float(VPM_PARAMS[2]),
        'sharpness': float(VPM_PARAMS[3])
    },
    'n_bins_densidad': N_BINS_DENSIDAD,
    'n_galaxias_total': int(n_gal),
    'correlacion_densidad_vs_delta_rho2': {
        'rho': float(r_dens_delta),
        'p_valor': float(p_dens_delta)
    },
    'delta_rho2_baja_densidad': float(delta_baja),
    'delta_rho2_alta_densidad': float(delta_alta),
    'bines_mejora': bines_mejora,
    'bines_empeora': bines_empeora,
    'resultados_por_bin': [{
        'densidad_media': float(r['densidad_media']),
        'n_galaxias': int(r['n_galaxias']),
        'rho2_null': float(r['rho2_null']),
        'rho2_vpm': float(r['rho2_vpm']),
        'delta_rho2': float(r['delta_rho2']),
        'reduccion_pct': float(r['reduccion_pct'])
    } for r in resultados]
}

with open('resultados_densidad.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  💾 resultados_densidad.json saved")
print(f"{'='*70}")