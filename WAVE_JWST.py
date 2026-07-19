#!/usr/bin/env python3
"""
WAVE_JWST.py — VPM VALIDATION IN 7 JWST FIELDS
======================================================
Independent analysis for the 7 JWST datasets.
Uses the Rust kernel vpm_wave (base VPM, no compression: no σ_ap).
Unified parameters with WAVE_LENSING_c.py.

Datasets: 7 JWST fields (*o.dat.gz)
Test: M_lens/M_dyn = 1 + ξ(z) from Rust kernel
======================================================
"""

import numpy as np
import gzip
import sys
import os
import glob
import pandas as pd

sys.path.append('target/release')
from vpm_wave import VPMWaveEngine
engine = VPMWaveEngine()

DATA_DIR = 'data'

# ================================================================
# PARAMETERS (reference only, the kernel already has them)
# ================================================================
Q1_params = {
    'sigma_0':      175.5,
    'strength_max': 3.0455,
    'f_comp_min':   0.9779,
    'sharpness':    40.2,
    'sat_boost':    1.5
}

# ================================================================
# JWST CATALOG COLUMNS
# ================================================================
COLUMN_NAMES_49 = [
    'id', 'ra', 'dec', 'x_image', 'y_image', 'morph_param',
    'photoz_confidence', 'flag', 'z_phot', 'flux_ref', 'flux_err_ref',
    'chi2_min', 'chi2_peak',
    'fit_param_0', 'fit_param_1', 'fit_param_2', 'fit_param_3',
    'fit_param_4', 'fit_param_5', 'fit_param_6', 'fit_param_7',
    'fit_param_8', 'fit_param_9', 'fit_param_10', 'fit_param_11',
    'fit_param_12', 'fit_param_13', 'fit_param_14', 'fit_param_15',
    'fit_param_16',
    'flux_0', 'flux_1', 'flux_2', 'flux_3', 'flux_4',
    'flux_5', 'flux_6', 'flux_7', 'flux_8', 'flux_9',
    'flux_10', 'flux_11', 'flux_12', 'flux_13', 'flux_14',
    'flux_15', 'flux_16', 'flux_17',
    'template_index'
]

print("=" * 70)
print("VPM VALIDATION — 7 JWST FIELDS")
print("BASE VPM (no compression): M_lens/M_dyn = 1 + ξ(z)")
print("=" * 70)
print(f"  Kernel: VPMWaveEngine (Rust)")
print(f"  A0 = {engine.get_a0():.3f} Mpc")
print(f"  Xi_0 = {engine.get_xi_0():.3f}")
print(f"  Crossing point ξ(z)=0: z ≈ 1.19")
print()

# ================================================================
# FUNCTIONS
# ================================================================

def load_jwst_catalog(filepath):
    """Loads a JWST catalog"""
    try:
        with gzip.open(filepath, 'rt') as f:
            df = pd.read_csv(f, sep=r'\s+', header=None, low_memory=False)
        
        n_cols = min(len(COLUMN_NAMES_49), df.shape[1])
        df.columns = COLUMN_NAMES_49[:n_cols]
        
        df['z_phot'] = pd.to_numeric(df['z_phot'], errors='coerce')
        df['photoz_confidence'] = pd.to_numeric(df['photoz_confidence'], errors='coerce')
        
        # Quality cut
        df = df[(df['z_phot'].notna()) & 
                (df['z_phot'] > 0.01) & 
                (df['z_phot'] < 15.0) &
                (df['photoz_confidence'] > 0.5)]
        
        return df
    except Exception as e:
        print(f"  ❌ Error loading {filepath}: {e}")
        return None


def clasificar_regimen_z(z):
    """Classifies according to ξ(z)"""
    xi = engine.xi_vpm(float(z))
    if xi > 0.01:
        return 'BOOST'
    elif xi < -0.01:
        return 'DEFICIT'
    else:
        return 'TRANSITION'


# ================================================================
# LOAD DATA
# ================================================================
jwst_files = sorted(glob.glob(os.path.join(DATA_DIR, '*o.dat.gz')))

if not jwst_files:
    print("❌ No JWST files (*o.dat.gz) found in data/")
    sys.exit(1)

print(f"Files found: {len(jwst_files)}\n")

all_data = {}
all_zs_combined = []

for idx, filepath in enumerate(jwst_files):
    field_name = os.path.basename(filepath).replace('.dat.gz', '').upper()
    
    df = load_jwst_catalog(filepath)
    if df is None or len(df) == 0:
        continue
    
    zs = df['z_phot'].values.astype(np.float64)
    all_zs_combined.extend(zs)
    all_data[field_name] = zs

if not all_data:
    print("❌ Could not load data from any JWST field")
    sys.exit(1)

all_zs = np.array(all_zs_combined)
print(f"Combined total: {len(all_zs):,} galaxies")
print(f"z range: [{np.min(all_zs):.4f}, {np.max(all_zs):.4f}]")
print(f"Median z: {np.median(all_zs):.4f}")

# ================================================================
# ANALYSIS BY FIELD
# ================================================================
print(f"\n{'='*70}")
print("STATISTICS BY FIELD")
print(f"{'='*70}")
print(f"  {'Field':<22s} {'N':>10s} {'z_med':>8s} {'ξ_med':>10s} {'Ratio_med':>12s} {'Boost%':>8s} {'Deficit%':>10s}")
print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*10} {'─'*12} {'─'*8} {'─'*10}")

field_stats = {}

for field_name, zs in sorted(all_data.items()):
    ratios = np.array([engine.mass_ratio_vpm(float(z)) for z in zs])
    xi_vals = ratios - 1.0
    
    mask_boost = xi_vals > 0.01
    mask_deficit = xi_vals < -0.01
    
    stats = {
        'N': len(zs),
        'z_med': float(np.median(zs)),
        'z_min': float(np.min(zs)),
        'z_max': float(np.max(zs)),
        'xi_med': float(np.median(xi_vals)),
        'xi_mean': float(np.mean(xi_vals)),
        'ratio_med': float(np.median(ratios)),
        'pct_boost': 100 * np.sum(mask_boost) / len(zs),
        'pct_deficit': 100 * np.sum(mask_deficit) / len(zs),
        'pct_cruce': 100 * np.sum(np.abs(xi_vals) <= 0.01) / len(zs),
    }
    field_stats[field_name] = stats
    
    print(f"  {field_name:<22s} {stats['N']:>10,} {stats['z_med']:>8.3f} {stats['xi_med']:>+10.4f} {stats['ratio_med']:>12.4f} {stats['pct_boost']:>7.1f}% {stats['pct_deficit']:>9.1f}%")

# ================================================================
# GLOBAL STATISTICS
# ================================================================
print(f"\n{'='*70}")
print("JWST GLOBAL STATISTICS")
print(f"{'='*70}")

ratios_all = np.array([engine.mass_ratio_vpm(float(z)) for z in all_zs])
xi_all = ratios_all - 1.0

n_boost = np.sum(xi_all > 0.01)
n_deficit = np.sum(xi_all < -0.01)
n_cruce = np.sum(np.abs(xi_all) <= 0.01)

print(f"  Total: {len(all_zs):,}")
print(f"  Median ξ(z): {np.median(xi_all):+.4f}")
print(f"  Median Ratio: {np.median(ratios_all):.4f}")
print(f"")
print(f"  BOOST (ξ > +0.01):    {n_boost:>10,} ({100*n_boost/len(all_zs):.1f}%)")
print(f"  DEFICIT (ξ < -0.01):  {n_deficit:>10,} ({100*n_deficit/len(all_zs):.1f}%)")
print(f"  TRANSITION (|ξ|≤0.01): {n_cruce:>10,} ({100*n_cruce/len(all_zs):.1f}%)")

# ================================================================
# REDSHIFT BINS (WAVE.tex style)
# ================================================================
bins_wave = [
    (0.01, 0.50, "Low local BOOST"),
    (0.50, 0.78, "Low intermediate BOOST"),
    (0.78, 1.00, "High intermediate BOOST"),
    (1.00, 1.22, "TRANSITION"),
    (1.22, 1.50, "Low early DEFICIT"),
    (1.50, 2.00, "High early DEFICIT"),
    (2.00, 3.00, "Moderate DEFICIT"),
    (3.00, 5.00, "Low deep DEFICIT"),
    (5.00, 7.00, "Deep DEFICIT"),
    (7.00, 10.00, "Extreme DEFICIT"),
    (10.00, 20.00, "Ultra-deep DEFICIT"),
]

print(f"\n{'='*90}")
print("JWST: VPM RATIOS BY REDSHIFT BIN (WAVE.tex)")
print(f"{'='*90}")
print(f"  {'z range':<22s} {'N':>10s} {'%':>8s} {'z_med':>8s} {'mean ξ':>12s} {'mean Ratio':>14s} {'Regime'}")
print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*8} {'─'*12} {'─'*14} {'─'*25}")

for z_lo, z_hi, label in bins_wave:
    mask = (all_zs >= z_lo) & (all_zs < z_hi)
    n_bin = np.sum(mask)
    pct = 100 * n_bin / len(all_zs)
    
    if n_bin > 0:
        zs_bin = all_zs[mask]
        xi_bin = xi_all[mask]
        ratio_bin = ratios_all[mask]
        z_med = np.median(zs_bin)
        xi_mean = np.mean(xi_bin)
        ratio_mean = np.mean(ratio_bin)
        vacio = ""
    else:
        z_med = np.nan
        xi_mean = np.nan
        ratio_mean = np.nan
        vacio = " ⚠️ EMPTY"
    
    print(f"  {z_lo:.2f} ≤ z < {z_hi:<6.2f} {n_bin:>10,} {pct:>7.2f}% {z_med:>8.3f} {xi_mean:>+12.6f} {ratio_mean:>14.6f}  {label}{vacio}")

# ================================================================
# CROSSING POINT
# ================================================================
print(f"\n{'='*70}")
print("CROSSING POINT ξ(z) = 0")
print(f"{'='*70}")

z_lo, z_hi = 0.5, 2.0
for _ in range(60):
    z_mid = (z_lo + z_hi) / 2.0
    xi_mid = engine.xi_vpm(z_mid)
    if abs(xi_mid) < 1e-14:
        break
    if xi_mid * engine.xi_vpm(z_lo) > 0:
        z_lo = z_mid
    else:
        z_hi = z_mid

z_cruce = (z_lo + z_hi) / 2.0
print(f"  z_cross (Rust kernel) = {z_cruce:.8f}")
print(f"  ξ(z_cross) = {engine.xi_vpm(z_cruce):.4e}")
print(f"  Ratio(z_cross) = {engine.mass_ratio_vpm(z_cruce):.10f}")

# Galaxy closest to crossing
idx_closest = np.argmin(np.abs(all_zs - z_cruce))
z_closest = all_zs[idx_closest]
print(f"\n  Galaxy closest to crossing:")
print(f"    z = {z_closest:.6f}")
print(f"    ξ = {engine.xi_vpm(float(z_closest)):+.10f}")
print(f"    Ratio = {engine.mass_ratio_vpm(float(z_closest)):.10f}")

# ================================================================
# ξ vs z DISTRIBUTION (fine bins around crossing)
# ================================================================
print(f"\n{'='*70}")
print("DISTRIBUTION AROUND CROSSING (bins of Δz=0.05)")
print(f"{'='*70}")

bins_finos = np.arange(0.80, 1.70, 0.05)
print(f"  {'Bin z':>15s}  {'N':>8s}  {'ξ_mean':>14s}  {'Sign'}")
print(f"  {'─'*15}  {'─'*8}  {'─'*14}  {'─'*15}")

for i in range(len(bins_finos)-1):
    mask = (all_zs >= bins_finos[i]) & (all_zs < bins_finos[i+1])
    n_bin = np.sum(mask)
    if n_bin > 0:
        zs_bin = all_zs[mask]
        xi_bin = np.array([engine.xi_vpm(float(z)) for z in zs_bin])
        xi_mean = np.mean(xi_bin)
        signo = "BOOST (+)" if xi_mean > 0.001 else "DEFICIT (-)" if xi_mean < -0.001 else "≈ ZERO"
        marker = " ⬅" if abs(xi_mean) < 0.005 else ""
        print(f"  {bins_finos[i]:.2f} ≤ z < {bins_finos[i+1]:.2f}  {n_bin:>8,}  {xi_mean:>+14.10f}  {signo}{marker}")
    else:
        print(f"  {bins_finos[i]:.2f} ≤ z < {bins_finos[i+1]:.2f}  {0:>8}  {'---':>14}  ---")

# ================================================================
# FINAL SUMMARY
# ================================================================
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

print(f"""
  DATASETS: 7 JWST fields
  TOTAL GALAXIES: {len(all_zs):,}
  z RANGE: [{np.min(all_zs):.4f}, {np.max(all_zs):.4f}]
  
  BASE VPM (no compression, no σ_ap):
    Median Ratio: {np.median(ratios_all):.4f}
    Median ξ(z):  {np.median(xi_all):+.4f}
  
  CROSSING ξ(z)=0:
    Theoretical (kernel): z = {z_cruce:.6f}
  
  PREDICTIONS:
    z < 1.19 → BOOST (M_lens/M_dyn > 1)
    z > 1.19 → DEFICIT (M_lens/M_dyn < 1)
  
  NOTE: JWST lacks σ_ap → profile compression does not apply.
        For compression, see DESI LRG and eROSITA datasets.
""")

# ================================================================
# LaTeX TABLE FOR WAVE.tex
# ================================================================
print(f"{'='*70}")
print("TABLE FOR WAVE.tex")
print(f"{'='*70}")
print()
print("\\begin{table}[h!]")
print("\\centering")
print("\\begin{tabular}{c c c c c}")
print("\\toprule")
print("Regime & $z_{\\text{med}}$ & N & $\\xi(z)$ & VPM Ratio \\\\")
print("\\midrule")

for z_lo, z_hi, label in bins_wave:
    mask = (all_zs >= z_lo) & (all_zs < z_hi)
    n_bin = np.sum(mask)
    if n_bin > 0:
        z_med = np.median(all_zs[mask])
        xi_mean = np.mean(xi_all[mask])
        ratio_mean = np.mean(ratios_all[mask])
        print(f"  {label} & {z_med:.2f} & {n_bin:,} & {xi_mean:+.3f} & {ratio_mean:.3f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print(f"\n{'='*70}")
print("JWST ANALYSIS COMPLETED")
print(f"{'='*70}")