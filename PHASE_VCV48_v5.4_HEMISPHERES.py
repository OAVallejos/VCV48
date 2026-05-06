#!/usr/bin/env python3
"""
Final Version                          Includes fix for JSON serialization
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import gaussian_kde
import warnings
import time
import json
warnings.filterwarnings('ignore')

# Fix for JSON serialization
def convert_numpy(obj):
    """Converts NumPy types to native Python types for JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

try:
    import vpm_core
    VPM_AVAILABLE = True
    print("✅ VCV48 Kernel v7.0 loaded")
    A0 = vpm_core.A0
    R_14 = 14 * A0
    OMEGA_OBS = vpm_core.OMEGA_OBS
    print(f"   ω_obs = {OMEGA_OBS:.4f} Gyr⁻¹")
    print(f"   R_14 = {R_14:.2f} Mpc")
except ImportError:
    print("❌ Kernel not available")
    exit(1)

# CMB Dipole
RA_CMB = 168.0
DEC_CMB = -7.0

# ============================================================================
# FUNCTIONS (same as before)
# ============================================================================

def compute_dipole_vector(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

def compute_galaxy_vectors(ra_arr, dec_arr):
    ra = np.radians(ra_arr)
    dec = np.radians(dec_arr)
    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

def split_by_hemisphere(df):
    v_dipole = compute_dipole_vector(RA_CMB, DEC_CMB)
    v_galaxies = compute_galaxy_vectors(df['RA'].values, df['DEC'].values)
    dot_products = v_galaxies @ v_dipole
    return df[dot_products > 0].copy(), df[dot_products < 0].copy()

def compute_correlation_hemisphere(df, label, r_teo, n_sample=25000):
    print(f"\n🔬 {label}: N={len(df):,}, ⟨z⟩={df['Z'].mean():.3f}")

    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=42)

    ra_data = df['RA'].values.astype(np.float64)
    dec_data = df['DEC'].values.astype(np.float64)
    z_data = df['Z'].values.astype(np.float64)
    vdisp_data = df['VDISP'].values.astype(np.float64)

    n_rand = len(df) * 2
    z_rand = np.random.uniform(z_data.min(), z_data.max(), n_rand)
    ra_rand = np.random.uniform(ra_data.min(), ra_data.max(), n_rand)
    dec_rand = np.random.uniform(dec_data.min(), dec_data.max(), n_rand)

    engine = vpm_core.VPMEngine()

    r_min = r_teo - 2.0
    r_max = r_teo + 2.0
    n_bins = 20

    centers, xi, mean_kappa, pred_delta_ns = engine.weighted_correlation(
        ra_data.tolist(), dec_data.tolist(), z_data.tolist(),
        vdisp_data.tolist(), ra_rand.tolist(), dec_rand.tolist(),
        z_rand.tolist(), r_min, r_max, n_bins
    )

    centers = np.array(centers)
    xi = np.array(xi)

    bin_teo = np.argmin(np.abs(centers - r_teo))
    exclude = [bin_teo + i for i in range(-2, 3) if 0 <= bin_teo + i < len(centers)]
    bg_bins = [i for i in range(len(centers)) if i not in exclude]

    bg_mean = np.mean(xi[bg_bins]) if bg_bins else 0
    bg_std = np.std(xi[bg_bins]) if bg_bins else 1
    excess = (xi[bin_teo] - bg_mean) / bg_std if bg_std > 0 else 0

    half_max = (xi[bin_teo] + bg_mean) / 2
    above_half = xi > half_max
    fwhm = np.sum(above_half) * (r_max - r_min) / n_bins

    return {
        'label': label, 'n_gal': len(df), 'z_mean': float(df['Z'].mean()),
        'vdisp_mean': float(df['VDISP'].mean()), 'r_teo': r_teo,
        'r_obs': float(centers[bin_teo]), 'xi_peak': float(xi[bin_teo]),
        'bg_mean': float(bg_mean), 'bg_std': float(bg_std), 'excess': float(excess),
        'error_abs': float(abs(centers[bin_teo] - r_teo)),
        'error_rel': float(abs(centers[bin_teo] - r_teo)/r_teo*100),
        'fwhm': float(fwhm), 'mean_kappa': float(mean_kappa),
        'pred_delta_ns': float(pred_delta_ns)
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("PHASE VCV48 v5.4.1 - FINAL HEMISPHERE ANALYSIS")
    print("="*70)

    tabla = Table.read('data/DATASET_LRG_VDISP_FLUXR_FINAL.fits')
    df = tabla.to_pandas()
    df = df[df['VDISP'] > 262.2]
    print(f"✅ High mass: {len(df):,} galaxies")

    df_toward, df_away = split_by_hemisphere(df)
    print(f"\n🌍 TOWARD: {len(df_toward):,} | AWAY: {len(df_away):,}")

    results = {}

    # Standard
    print("\n📐 STANDARD MODEL (ω=0.1914)")
    results['standard'] = {
        'toward': compute_correlation_hemisphere(df_toward, "TOWARD", R_14),
        'away': compute_correlation_hemisphere(df_away, "AWAY", R_14)
    }

    # Fine (0.1915)
    R_14_FINE = 14 * 14.0679
    print("\n📐 FINE-TUNED MODEL (ω=0.1915)")
    results['fine'] = {
        'toward': compute_correlation_hemisphere(df_toward, "TOWARD", R_14_FINE),
        'away': compute_correlation_hemisphere(df_away, "AWAY", R_14_FINE)
    }

    # Print table
    print("\n" + "="*90)
    print("📊 COMPARATIVE TABLE - HEMISPHERES")
    print("="*90)

    for model, name in [('standard', 'STANDARD (ω=0.1914)'), ('fine', 'FINE-TUNED (ω=0.1915)')]:
        print(f"\n🔷 {name}")
        print("-"*90)
        print(f"{'Hemisphere':<15} {'N':<10} {'⟨z⟩':<8} {'R_theo':<10} {'R_obs':<10} "
              f"{'Error':<10} {'Excess':<10} {'FWHM':<8} {'⟨κ⟩':<10}")
        print("-"*90)
        for hemi in ['toward', 'away']:
            r = results[model][hemi]
            name = "TOWARD (→)" if hemi == 'toward' else "AWAY (←)"
            print(f"{name:<15} {r['n_gal']:<10,} {r['z_mean']:<8.3f} "
                  f"{r['r_teo']:<10.2f} {r['r_obs']:<10.2f} "
                  f"{r['error_abs']:<10.2f} {r['excess']:<+10.2f}σ "
                  f"{r['fwhm']:<8.2f} {r['mean_kappa']:<10.6f}")

    # Differential
    print("\n" + "="*90)
    print("📈 TOWARD - AWAY DIFFERENTIAL")
    print("="*90)
    for model, name in [('standard', 'STANDARD'), ('fine', 'FINE-TUNED')]:
        t = results[model]['toward']
        a = results[model]['away']
        print(f"\n🔷 {name}:")
        print(f"   Δ Excess = {t['excess'] - a['excess']:+.2f}σ")
        print(f"   Δ FWHM   = {t['fwhm'] - a['fwhm']:+.2f} Mpc")

    # Save JSON
    output = convert_numpy({
        'dipole_cmb': {'ra': RA_CMB, 'dec': DEC_CMB},
        'omega_standard': OMEGA_OBS,
        'r14_standard': R_14,
        'r14_fine': R_14_FINE,
        'results': results
    })

    with open('hemisphere_results_final.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to hemisphere_results_final.json")
    print("\n✅ ANALYSIS COMPLETED")

if __name__ == "__main__":
    main()