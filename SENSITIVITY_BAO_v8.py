#!/usr/bin/env python3     
"""
  DESI                      
"""

import numpy as np
import pandas as pd
from astropy.table import Table
import warnings
warnings.filterwarnings('ignore')

try:
    import vpm_core
    engine = vpm_core.VPMEngine()
    print("✅ VCV48 Kernel v7.0 loaded")
except ImportError:
    print("❌ Kernel not available")
    exit(1)

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n📂 Loading DESI...")
tabla = Table.read('data/DATASET_LRG_VDISP_FLUXR_FINAL.fits')
df = tabla.to_pandas()
df = df[df['VDISP'] > 262.2]
print(f"✅ High mass: {len(df):,} galaxies")

# ============================================================================
# STRATIFIED SAMPLING BY VORTICITY
# ============================================================================

# Take 20,000 galaxies per velocity bin
N_SAMPLE = 20000

bins_vdisp = [
    (262.2, 300.0, "Low vorticity"),
    (300.0, 350.0, "Medium-low"),
    (350.0, 400.0, "Medium-high"),
    (400.0, 500.0, "High vorticity"),
    (500.0, 1000.0, "Very high")
]

r_teo = 14 * vpm_core.A0
r_min = r_teo - 1.5
r_max = r_teo + 1.5
n_bins = 15

print(f"\n⚡ Stratified sampling: {N_SAMPLE:,} galaxies per bin")
print(f"   Estimated time: ~{len(bins_vdisp) * 0.5:.1f} minutes\n")

results = []

for vmin, vmax, label in bins_vdisp:
    df_bin = df[(df['VDISP'] >= vmin) & (df['VDISP'] < vmax)]
    n_total = len(df_bin)

    # Sampling
    if n_total > N_SAMPLE:
        df_bin = df_bin.sample(n=N_SAMPLE, random_state=42)
    n_sample = len(df_bin)

    print(f"\n🔬 {label}: [{vmin:.0f}, {vmax:.0f}) km/s")
    print(f"   N = {n_sample:,} (out of {n_total:,} total)")
    print(f"   ⟨z⟩ = {df_bin['Z'].mean():.3f}")
    print(f"   ⟨vdisp⟩ = {df_bin['VDISP'].mean():.1f} km/s")

    # Prepare data
    ra_data = df_bin['RA'].values.astype(np.float64)
    dec_data = df_bin['DEC'].values.astype(np.float64)
    z_data = df_bin['Z'].values.astype(np.float64)
    vdisp_data = df_bin['VDISP'].values.astype(np.float64)

    # Randoms (2x)
    from scipy.stats import gaussian_kde
    n_rand = n_sample * 2
    kde = gaussian_kde(z_data)
    z_rand = kde.resample(n_rand)[0]
    z_rand = np.clip(z_rand, z_data.min(), z_data.max())
    ra_rand = np.random.uniform(ra_data.min(), ra_data.max(), n_rand)
    dec_rand = np.random.uniform(dec_data.min(), dec_data.max(), n_rand)

    # Calculate correlation
    centers, xi, mean_kappa, pred_delta_ns = engine.weighted_correlation(
        ra_data.tolist(), dec_data.tolist(), z_data.tolist(),
        vdisp_data.tolist(), ra_rand.tolist(), dec_rand.tolist(),
        z_rand.tolist(), r_min, r_max, n_bins
    )

    centers = np.array(centers)
    xi = np.array(xi)

    # Analysis
    bin_teo = np.argmin(np.abs(centers - r_teo))
    xi_peak = xi[bin_teo]

    exclude = [bin_teo + i for i in range(-2, 3) if 0 <= bin_teo + i < len(centers)]
    bg_bins = [i for i in range(len(centers)) if i not in exclude]

    bg_mean = np.mean(xi[bg_bins]) if bg_bins else 0
    bg_std = np.std(xi[bg_bins]) if bg_bins else 1
    significance = (xi_peak - bg_mean) / bg_std if bg_std > 0 else 0

    error_abs = abs(centers[bin_teo] - r_teo)
    error_rel = error_abs / r_teo * 100

    print(f"   📍 Peak: {centers[bin_teo]:.2f} Mpc (error: {error_rel:.2f}%)")
    print(f"   📊 Excess: {significance:+.2f}σ")

    results.append({
        'label': label,
        'vmin': vmin,
        'vmax': vmax,
        'n_total': n_total,
        'n_sample': n_sample,
        'z_mean': df_bin['Z'].mean(),
        'vdisp_mean': df_bin['VDISP'].mean(),
        'peak_r': centers[bin_teo],
        'xi_peak': xi_peak,
        'significance': significance,
        'error_rel': error_rel,
        'mean_kappa': mean_kappa,
        'pred_delta_ns': pred_delta_ns
    })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: PHASE COHERENCE vs VORTICITY")
print("="*70)
print(f"{'Bin':<18} {'N':<8} {'⟨vdisp⟩':<10} {'Excess':<10} {'Error':<8} {'κ':<10}")
print("-"*70)

for r in results:
    print(f"{r['label']:<18} {r['n_sample']:<8,} {r['vdisp_mean']:<10.1f} "
          f"{r['significance']:<+10.2f}σ {r['error_rel']:<8.2f}% {r['mean_kappa']:<10.6f}")

# Identify best bin
if results:
    best = max(results, key=lambda x: x['significance'])
    print("\n" + "="*70)
    print("🏆 MAXIMUM PHASE COHERENCE:")
    print(f"   Bin: {best['label']} ({best['vmin']:.0f}-{best['vmax']:.0f} km/s)")
    print(f"   ⟨vdisp⟩ = {best['vdisp_mean']:.1f} km/s")
    print(f"   Excess = {best['significance']:.2f}σ")
    print(f"   Geometric error = {best['error_rel']:.2f}%")
    print("="*70)