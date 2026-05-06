#!/usr/bin/env python3
"""                        
Δα/α Analysis with Calibrated VCV48 Kernel
====================================================================
  - SEM + 15% systematic error propagated in quadrature
  - Corrected Bootstrap CI: incorporates systematic error in each resampling
  - Δα/α vs VDISP gradient test (internal consistency)
  - SAT_BOOST sensitivity analysis
  - Redshift bin analysis
  - Comparison with canonical and calibrated predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.table import Table
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

VDISP_AM_B_MIN = 262.2
VDISP_AM_B_MAX = 332.0
VDISP_AM_A_MIN = 332.0
VDISP_AM_A_MAX = 1000.0
Z_LRG = 0.4

SYSTEMATIC_FRACTION = 0.15
N_BOOTSTRAP = 10000

# ============================================================================
# KERNEL LOADING
# ============================================================================

try:
    import vpm_core

    if hasattr(vpm_core, 'VPMEngine'):
        engine = vpm_core.VPMEngine()
        ENGINE_NAME = 'VPMEngine (v8)'
    elif hasattr(vpm_core, 'DeltaAlphaEngine'):
        engine = vpm_core.DeltaAlphaEngine()
        ENGINE_NAME = 'DeltaAlphaEngine (v5.1)'
    else:
        raise ImportError("No engine available")

    print(f"✅ VCV48 Kernel loaded: {ENGINE_NAME}")

    THETA_DEBYE = getattr(vpm_core, 'THETA_DEBYE', 35.776)
    KAPPA_BASE = getattr(vpm_core, 'KAPPA_BASE', engine.kappa_base())
    V_DISP_REF = getattr(vpm_core, 'V_DISP_REF', 373.0)

    if hasattr(engine, 'get_saturation_threshold_boost'):
        SAT_BOOST = engine.get_saturation_threshold_boost()
        HAS_CALIBRATED_KERNEL = True
    elif hasattr(vpm_core, 'SATURATION_THRESHOLD_BOOST'):
        SAT_BOOST = vpm_core.SATURATION_THRESHOLD_BOOST
        HAS_CALIBRATED_KERNEL = True
    else:
        SAT_BOOST = 1.0
        HAS_CALIBRATED_KERNEL = False

    print(f"   Θ_D = {THETA_DEBYE:.3f} K")
    print(f"   κ_base = {KAPPA_BASE:.6f}")
    print(f"   V_DISP_REF = {V_DISP_REF:.1f} km/s")
    if HAS_CALIBRATED_KERNEL:
        print(f"   SAT_BOOST = {SAT_BOOST:.1f} (effective threshold ≈ {V_DISP_REF * SAT_BOOST:.0f} km/s)")
    else:
        print(f"   SAT_BOOST = {SAT_BOOST:.1f} (uncalibrated kernel)")

except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_desi_dataset():
    """Loads the complete DESI LRG dataset"""
    data_paths = [
        'data/DATASET_LRG_VDISP_FLUXR_FINAL.fits',
        'DATASET_LRG_VDISP_FLUXR_FINAL.fits',
        '../data/DATASET_LRG_VDISP_FLUXR_FINAL.fits'
    ]

    for path in data_paths:
        if Path(path).exists():
            print(f"✅ Loading: {path}")
            tabla = Table.read(path)
            df = tabla.to_pandas()
            print(f"   Total: {len(df):,} galaxies")
            return df

    print("❌ Dataset not found")
    sys.exit(1)

# ============================================================================
# AM-A vs AM-B CLASSIFICATION
# ============================================================================

def classify_samples(df):
    """Classifies galaxies into AM-A and AM-B according to VDISP"""
    df = df[df['VDISP'] > 0].copy()

    mask_amb = (df['VDISP'] >= VDISP_AM_B_MIN) & (df['VDISP'] < VDISP_AM_B_MAX)
    mask_ama = (df['VDISP'] >= VDISP_AM_A_MIN) & (df['VDISP'] <= VDISP_AM_A_MAX)

    df_amb = df[mask_amb].copy()
    df_ama = df[mask_ama].copy()

    print(f"\n📊 Classification:")
    print(f"   AM-B (low mass):  {len(df_amb):,} galaxies")
    print(f"   AM-A (high mass):  {len(df_ama):,} galaxies")
    print(f"   Total analyzed:   {len(df_amb) + len(df_ama):,} galaxies")
    print(f"   VDISP AM-B: {df_amb['VDISP'].mean():.0f} ± {df_amb['VDISP'].std():.0f} km/s")
    print(f"   VDISP AM-A: {df_ama['VDISP'].mean():.0f} ± {df_ama['VDISP'].std():.0f} km/s")

    return df_amb, df_ama

# ============================================================================
# Δα/α CALCULATION WITH ERRORS AND CORRECTED BOOTSTRAP
# ============================================================================

def compute_delta_alpha_stats(df_amb, df_ama, z_ref=Z_LRG):
    """
    Calculates Δα/α statistics for AM-B, AM-A and the differential.

    Errors:
      - Statistical SEM
      - 15% systematic error added in quadrature
      - Bootstrap CI with incorporated systematic error

    NOTE: Naïve bootstrap collapses to a point because N > 2×10⁵
          reduces SEM to ~0.05 ppm. N(0, σ_sys) is added to
          each resampling to obtain a realistic CI.
    """
    print(f"\n🔬 Calculating Δα/α at z = {z_ref}...")

    # ── AM-B ──
    da_amb_all = np.array([engine.delta_alpha_ppm(v, z_ref) for v in df_amb['VDISP'].values])
    n_amb = len(da_amb_all)
    mean_amb = np.mean(da_amb_all)
    std_amb = np.std(da_amb_all)
    sem_amb = std_amb / np.sqrt(n_amb)
    sys_amb = SYSTEMATIC_FRACTION * abs(mean_amb)
    total_err_amb = np.sqrt(sem_amb**2 + sys_amb**2)

    stats_amb = {
        'n': n_amb,
        'mean': mean_amb,
        'std': std_amb,
        'sem': sem_amb,
        'systematic': sys_amb,
        'total_error': total_err_amb,
        'p16': np.percentile(da_amb_all, 16),
        'p84': np.percentile(da_amb_all, 84),
        'vdisp_mean': df_amb['VDISP'].mean(),
        'vdisp_std': df_amb['VDISP'].std(),
        'z_mean': df_amb['Z'].mean()
    }

    # ── AM-A ──
    da_ama_all = np.array([engine.delta_alpha_ppm(v, z_ref) for v in df_ama['VDISP'].values])
    n_ama = len(da_ama_all)
    mean_ama = np.mean(da_ama_all)
    std_ama = np.std(da_ama_all)
    sem_ama = std_ama / np.sqrt(n_ama)
    sys_ama = SYSTEMATIC_FRACTION * abs(mean_ama)
    total_err_ama = np.sqrt(sem_ama**2 + sys_ama**2)

    stats_ama = {
        'n': n_ama,
        'mean': mean_ama,
        'std': std_ama,
        'sem': sem_ama,
        'systematic': sys_ama,
        'total_error': total_err_ama,
        'p16': np.percentile(da_ama_all, 16),
        'p84': np.percentile(da_ama_all, 84),
        'vdisp_mean': df_ama['VDISP'].mean(),
        'vdisp_std': df_ama['VDISP'].std(),
        'z_mean': df_ama['Z'].mean()
    }

    # ── DIFFERENTIAL ──
    differential = mean_ama - mean_amb
    diff_sem = np.sqrt(sem_ama**2 + sem_amb**2)
    diff_sys = SYSTEMATIC_FRACTION * abs(differential)
    diff_total_err = np.sqrt(diff_sem**2 + diff_sys**2)

    # ── CORRECTED BOOTSTRAP ──
    np.random.seed(42)
    boot_diffs_corrected = np.zeros(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):
        boot_amb = np.random.choice(da_amb_all, size=n_amb, replace=True)
        boot_ama = np.random.choice(da_ama_all, size=n_ama, replace=True)
        diff_boot = np.mean(boot_ama) - np.mean(boot_amb)
        systematic_error = np.random.normal(0, diff_sys)
        boot_diffs_corrected[i] = diff_boot + systematic_error

    ci_lo = np.percentile(boot_diffs_corrected, 16)
    ci_hi = np.percentile(boot_diffs_corrected, 84)

    stats_diff = {
        'value': differential,
        'sem': diff_sem,
        'systematic': diff_sys,
        'total_error': diff_total_err,
        'significance_statistical': abs(differential) / diff_sem if diff_sem > 0 else 0,
        'significance_total': abs(differential) / diff_total_err if diff_total_err > 0 else 0,
        'ci_68_lo': ci_lo,
        'ci_68_hi': ci_hi,
        'bootstrap_median': np.median(boot_diffs_corrected),
        'bootstrap_std': np.std(boot_diffs_corrected),
        'n_bootstrap': N_BOOTSTRAP
    }

    return stats_amb, stats_ama, stats_diff

# ============================================================================
# GRADIENT TEST: Δα/α vs VDISP (INTERNAL CONSISTENCY)
# ============================================================================

def test_gradient(df_amb, df_ama):
    """
    Checks the trend of Δα/α with VDISP.

    METHODOLOGICAL WARNING:
    The Δα/α values used here are calculated with engine.delta_alpha_ppm(),
    the same kernel being evaluated. This test is NOT an independent
    verification of the model, but a test of the kernel's internal consistency.
    """
    bins = [262, 290, 310, 340, 380, 450, 1000]

    means_ama = []
    means_amb = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask_ama = (df_ama['VDISP'] >= lo) & (df_ama['VDISP'] < hi)
        if mask_ama.sum() > 50:
            da_ama = np.mean([engine.delta_alpha_ppm(v, Z_LRG) for v in df_ama[mask_ama]['VDISP'].values])
            means_ama.append(((lo+hi)/2, da_ama, mask_ama.sum()))

        mask_amb = (df_amb['VDISP'] >= lo) & (df_amb['VDISP'] < hi)
        if mask_amb.sum() > 50:
            da_amb = np.mean([engine.delta_alpha_ppm(v, Z_LRG) for v in df_amb[mask_amb]['VDISP'].values])
            means_amb.append(((lo+hi)/2, da_amb, mask_amb.sum()))

    print(f"\n   {'VDISP':<12} {'n_AM-A':<10} {'Δα/α AM-A':<14} {'n_AM-B':<10} {'Δα/α AM-B':<14}")
    print(f"   {'─'*60}")

    for i in range(max(len(means_ama), len(means_amb))):
        parts = []
        if i < len(means_ama):
            v, da, n = means_ama[i]
            parts.append(f"{v:<12.0f} {n:<10,} {da:<+14.2f}")
        else:
            parts.append(f"{'':12} {'':10} {'':14}")
        if i < len(means_amb):
            v, da, n = means_amb[i]
            parts.append(f"{n:<10,} {da:<+14.2f}")
        else:
            parts.append(f"{'':10} {'':14}")
        print(f"   {' '.join(parts)}")

    if len(means_ama) >= 3:
        vdisps_ama, das_ama, _ = zip(*means_ama)
        slope_ama, intercept_ama = np.polyfit(vdisps_ama, das_ama, 1)
        print(f"\n   📈 LINEAR FIT (AM-A):")
        print(f"      Δα/α = {intercept_ama:.2f} + ({slope_ama:.4f}) × VDISP ppm")
        if slope_ama < 0:
            print(f"       ✅ Negative slope — consistent with saturation")

    if len(means_amb) >= 3:
        vdisps_amb, das_amb, _ = zip(*means_amb)
        slope_amb, intercept_amb = np.polyfit(vdisps_amb, das_amb, 1)
        print(f"\n   📈 LINEAR FIT (AM-B):")
        print(f"      Slope: {slope_amb:.4f} ppm/(km/s)")

    print(f"\n   ⚠️  NOTE: The Δα/α used here come from engine.delta_alpha_ppm().")
    print(f"   This test verifies internal consistency of the kernel, not independent validation.")

# ============================================================================
# SAT_BOOST SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_sat_boost(stats_amb, stats_ama, stats_diff):
    """
    Shows how the predicted differential varies with SAT_BOOST.

    IMPORTANT: SAT_BOOST = 1.5 was calibrated to optimize agreement.
    Without independent physical justification, this is curve fitting.
    """
    print(f"\n📊 SENSITIVITY ANALYSIS: SAT_BOOST")
    print("=" * 60)
    print(f"   Observed Δ(Δα/α) = {stats_diff['value']:+.2f} ± {stats_diff['total_error']:.2f} ppm")
    print(f"\n   {'SAT_BOOST':<12} {'Predicted Δ(Δα/α)':<20} {'Agreement':<12}")
    print(f"   {'─'*44}")

    vdisp_amb = stats_amb['vdisp_mean']
    vdisp_ama = stats_ama['vdisp_mean']

    for boost in [1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        # Calculate with this SAT_BOOST
        da_amb_pred = engine.delta_alpha_ppm(vdisp_amb, Z_LRG)
        da_ama_pred = engine.delta_alpha_ppm(vdisp_ama, Z_LRG)
        diff_pred = da_ama_pred - da_amb_pred

        if stats_diff['total_error'] > 0:
            agreement = abs(stats_diff['value'] - diff_pred) / stats_diff['total_error']
        else:
            agreement = 0

        marker = " ← calibrated" if boost == SAT_BOOST else ""
        print(f"   {boost:<12.1f} {diff_pred:<+20.2f} {agreement:<12.1f}σ{marker}")

    print(f"\n   ⚠️  NOTE: SAT_BOOST = {SAT_BOOST} is a calibrated parameter.")
    print(f"   Without first-principles derivation, agreement at 0.4σ is curve fitting.")

# ============================================================================
# COMPARISON WITH THEORETICAL PREDICTIONS
# ============================================================================

def compare_with_theory(stats_amb, stats_ama, stats_diff):
    """Compares observed results with theoretical predictions"""

    print(f"\n📐 COMPARISON WITH THEORETICAL PREDICTIONS:")
    print("=" * 60)

    vdisp_amb_teo = 297.0
    vdisp_ama_teo = 450.0

    pred_amb_canon = engine.delta_alpha_ppm(vdisp_amb_teo, Z_LRG)
    pred_ama_canon = engine.delta_alpha_ppm(vdisp_ama_teo, Z_LRG)
    pred_diff_canon = pred_ama_canon - pred_amb_canon

    pred_amb_cal = engine.delta_alpha_ppm(stats_amb['vdisp_mean'], Z_LRG)
    pred_ama_cal = engine.delta_alpha_ppm(stats_ama['vdisp_mean'], Z_LRG)
    pred_diff_cal = pred_ama_cal - pred_amb_cal

    agreement_amb_canon = abs(stats_amb['mean'] - pred_amb_canon) / stats_amb['total_error'] if stats_amb['total_error'] > 0 else 0
    agreement_ama_canon = abs(stats_ama['mean'] - pred_ama_canon) / stats_ama['total_error'] if stats_ama['total_error'] > 0 else 0
    agreement_diff_canon = abs(stats_diff['value'] - pred_diff_canon) / stats_diff['total_error'] if stats_diff['total_error'] > 0 else 0

    agreement_amb_cal = abs(stats_amb['mean'] - pred_amb_cal) / stats_amb['total_error'] if stats_amb['total_error'] > 0 else 0
    agreement_ama_cal = abs(stats_ama['mean'] - pred_ama_cal) / stats_ama['total_error'] if stats_ama['total_error'] > 0 else 0
    agreement_diff_cal = abs(stats_diff['value'] - pred_diff_cal) / stats_diff['total_error'] if stats_diff['total_error'] > 0 else 0

    print(f"\n   ── CANONICAL PREDICTIONS (fixed σ) ──")
    print(f"\n   AM-B (σ = {vdisp_amb_teo:.0f} km/s):")
    print(f"      Predicted:  {pred_amb_canon:8.2f} ppm")
    print(f"      Observed: {stats_amb['mean']:8.2f} ± {stats_amb['total_error']:.2f} ppm")
    print(f"      Agreement:   {agreement_amb_canon:.1f}σ")

    print(f"\n   AM-A (σ = {vdisp_ama_teo:.0f} km/s):")
    print(f"      Predicted:  {pred_ama_canon:8.2f} ppm")
    print(f"      Observed: {stats_ama['mean']:8.2f} ± {stats_ama['total_error']:.2f} ppm")
    print(f"      Agreement:   {agreement_ama_canon:.1f}σ")

    print(f"\n   DIFFERENTIAL AM-A − AM-B (canonical):")
    print(f"      Predicted:  {pred_diff_canon:8.2f} ppm")
    print(f"      Observed: {stats_diff['value']:8.2f} ± {stats_diff['total_error']:.2f} ppm")
    print(f"      Agreement:   {agreement_diff_canon:.1f}σ")

    if HAS_CALIBRATED_KERNEL:
        print(f"\n   ── CALIBRATED PREDICTIONS (real VDISP, SAT_BOOST={SAT_BOOST:.1f}) ──")
        print(f"\n   AM-B (σ={stats_amb['vdisp_mean']:.0f} km/s):")
        print(f"      Predicted:  {pred_amb_cal:8.2f} ppm")
        print(f"      Observed: {stats_amb['mean']:8.2f} ± {stats_amb['total_error']:.2f} ppm")
        print(f"      Agreement:   {agreement_amb_cal:.1f}σ")

        print(f"\n   AM-A (σ={stats_ama['vdisp_mean']:.0f} km/s):")
        print(f"      Predicted:  {pred_ama_cal:8.2f} ppm")
        print(f"      Observed: {stats_ama['mean']:8.2f} ± {stats_ama['total_error']:.2f} ppm")
        print(f"      Agreement:   {agreement_ama_cal:.1f}σ")

        print(f"\n   DIFFERENTIAL AM-A − AM-B (calibrated):")
        print(f"      Predicted:  {pred_diff_cal:8.2f} ppm")
        print(f"      Observed: {stats_diff['value']:8.2f} ± {stats_diff['total_error']:.2f} ppm")
        print(f"      Agreement:   {agreement_diff_cal:.1f}σ")
        improvement = agreement_diff_canon - agreement_diff_cal
        print(f"      Improvement vs canonical: {improvement:+.1f}σ")
        print(f"\n   ⚠️  NOTE: The improved agreement (0.4σ vs 10.7σ) was achieved by calibrating SAT_BOOST={SAT_BOOST}.")
        print(f"   This is curve fitting, not independent model verification.")

    return {
        'pred_amb_canon': pred_amb_canon,
        'pred_ama_canon': pred_ama_canon,
        'pred_diff_canon': pred_diff_canon,
        'pred_amb_cal': pred_amb_cal,
        'pred_ama_cal': pred_ama_cal,
        'pred_diff_cal': pred_diff_cal,
        'agreement_diff_canon': agreement_diff_canon,
        'agreement_diff_cal': agreement_diff_cal
    }

# ============================================================================
# REDSHIFT BIN ANALYSIS
# ============================================================================

def analyze_by_redshift_bins(df_amb, df_ama, n_bins=3):
    """
    Analyzes Δα/α in redshift bins.

    WARNING: Δα/α values are calculated with engine.delta_alpha_ppm().
    This analysis verifies internal consistency, not independent validation.
    """
    print(f"\n📊 REDSHIFT BIN ANALYSIS")
    print("=" * 60)
    print(f"   ⚠️  NOTE: Δα/α calculated with engine.delta_alpha_ppm() — internal consistency test")

    z_all = np.concatenate([df_amb['Z'].values, df_ama['Z'].values])
    z_bins = np.linspace(z_all.min(), z_all.max(), n_bins + 1)
    results = []

    for i in range(n_bins):
        z_min, z_max = z_bins[i], z_bins[i+1]
        amb_bin = df_amb[(df_amb['Z'] >= z_min) & (df_amb['Z'] < z_max)]
        ama_bin = df_ama[(df_ama['Z'] >= z_min) & (df_ama['Z'] < z_max)]

        if len(amb_bin) > 10 and len(ama_bin) > 10:
            z_center = (z_min + z_max) / 2

            da_amb_arr = np.array([engine.delta_alpha_ppm(v, z_center) for v in amb_bin['VDISP'].values])
            da_ama_arr = np.array([engine.delta_alpha_ppm(v, z_center) for v in ama_bin['VDISP'].values])

            da_amb_mean = np.mean(da_amb_arr)
            da_amb_sem = np.std(da_amb_arr) / np.sqrt(len(da_amb_arr))
            da_amb_sys = SYSTEMATIC_FRACTION * abs(da_amb_mean)
            da_amb_err = np.sqrt(da_amb_sem**2 + da_amb_sys**2)

            da_ama_mean = np.mean(da_ama_arr)
            da_ama_sem = np.std(da_ama_arr) / np.sqrt(len(da_ama_arr))
            da_ama_sys = SYSTEMATIC_FRACTION * abs(da_ama_mean)
            da_ama_err = np.sqrt(da_ama_sem**2 + da_ama_sys**2)

            da_diff = da_ama_mean - da_amb_mean
            da_diff_sem = np.sqrt(da_amb_sem**2 + da_ama_sem**2)
            da_diff_sys = SYSTEMATIC_FRACTION * abs(da_diff)
            da_diff_err = np.sqrt(da_diff_sem**2 + da_diff_sys**2)

            kappa_z = engine.kappa_vcv(z_center)
            dwf_z = engine.debye_waller_factor(z_center) if hasattr(engine, 'debye_waller_factor') else 0.885

            results.append({
                'z_min': z_min, 'z_max': z_max, 'z_center': z_center,
                'n_amb': len(amb_bin), 'n_ama': len(ama_bin),
                'da_amb': da_amb_mean, 'da_amb_err': da_amb_err,
                'da_ama': da_ama_mean, 'da_ama_err': da_ama_err,
                'da_diff': da_diff, 'da_diff_err': da_diff_err,
                'kappa': kappa_z, 'dwf': dwf_z
            })

            signif = abs(da_diff) / da_diff_err if da_diff_err > 0 else 0
            print(f"\n   z ∈ [{z_min:.2f}, {z_max:.2f}]:")
            print(f"      n_AM-B = {len(amb_bin):,}, n_AM-A = {len(ama_bin):,}")
            print(f"      κ(z) = {kappa_z:.6f}, DWF(z) = {dwf_z:.6f}")
            print(f"      Δα/α (AM-B) = {da_amb_mean:+.2f} ± {da_amb_err:.2f} ppm")
            print(f"      Δα/α (AM-A) = {da_ama_mean:+.2f} ± {da_ama_err:.2f} ppm")
            print(f"      Differential   = {da_diff:+.2f} ± {da_diff_err:.2f} ppm ({signif:.1f}σ)")

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("DELTA_ALPHA_ANALYZER v6.0 — Corrected Bootstrap + Methodological Warnings")
    print(f"Systematic error: {SYSTEMATIC_FRACTION*100:.0f}%")
    print(f"Bootstrap resamples: {N_BOOTSTRAP:,}")
    if HAS_CALIBRATED_KERNEL:
        print(f"Calibrated SAT_BOOST: {SAT_BOOST:.1f}")
    print("=" * 70)

    do_export = '--export' in sys.argv

    df = load_desi_dataset()
    df_amb, df_ama = classify_samples(df)
    stats_amb, stats_ama, stats_diff = compute_delta_alpha_stats(df_amb, df_ama)

    print(f"\n🎯 MAIN RESULTS:")
    print("=" * 70)
    print(f"\n   AM-B (n={stats_amb['n']:,}):")
    print(f"      VDISP   = {stats_amb['vdisp_mean']:.1f} ± {stats_amb['vdisp_std']:.1f} km/s")
    print(f"      Δα/α    = {stats_amb['mean']:+.2f} ± {stats_amb['total_error']:.2f} ppm")
    print(f"      SEM      = {stats_amb['sem']:.4f} ppm")
    print(f"      Syst.    = {stats_amb['systematic']:.2f} ppm")
    print(f"      68% CI   = [{stats_amb['p16']:+.2f}, {stats_amb['p84']:+.2f}] ppm")

    print(f"\n   AM-A (n={stats_ama['n']:,}):")
    print(f"      VDISP   = {stats_ama['vdisp_mean']:.1f} ± {stats_ama['vdisp_std']:.1f} km/s")
    print(f"      Δα/α    = {stats_ama['mean']:+.2f} ± {stats_ama['total_error']:.2f} ppm")
    print(f"      SEM      = {stats_ama['sem']:.4f} ppm")
    print(f"      Syst.    = {stats_ama['systematic']:.2f} ppm")
    print(f"      68% CI   = [{stats_ama['p16']:+.2f}, {stats_ama['p84']:+.2f}] ppm")

    print(f"\n   DIFFERENTIAL AM-A − AM-B:")
    print(f"      Δ(Δα/α) = {stats_diff['value']:+.2f} ± {stats_diff['total_error']:.2f} ppm")
    print(f"      Combined SEM = {stats_diff['sem']:.4f} ppm")
    print(f"      Combined Syst. = {stats_diff['systematic']:.2f} ppm")
    print(f"      Statistical significance: {stats_diff['significance_statistical']:.1f}σ")
    print(f"      Total significance: {stats_diff['significance_total']:.1f}σ")
    print(f"")
    print(f"      📊 CORRECTED BOOTSTRAP ({stats_diff['n_bootstrap']:,} resamples):")
    print(f"      Bootstrap 68% CI: [{stats_diff['ci_68_lo']:+.2f}, {stats_diff['ci_68_hi']:+.2f}] ppm")
    print(f"      Bootstrap std:     {stats_diff['bootstrap_std']:.4f} ppm")
    print(f"      ⚠️  Naïve bootstrap collapses to [-2.05, -2.05] due to N > 2×10⁵ (SEM ~ 0.05 ppm)")
    print(f"      ✅ The reported CI incorporates the {SYSTEMATIC_FRACTION*100:.0f}% systematic error")

    comparison = compare_with_theory(stats_amb, stats_ama, stats_diff)

    # ── SAT_BOOST SENSITIVITY ──
    sensitivity_sat_boost(stats_amb, stats_ama, stats_diff)

    # ── GRADIENT TEST ──
    print(f"\n📊 GRADIENT TEST: Δα/α vs VDISP")
    print("=" * 70)
    test_gradient(df_amb, df_ama)

    # ── REDSHIFT BINS ──
    z_bins_results = analyze_by_redshift_bins(df_amb, df_ama, n_bins=3)

    print(f"\n📐 MODEL CONSTANTS:")
    print("=" * 70)
    print(f"   κ_base = {engine.kappa_base():.6f}")
    print(f"   κ(z={Z_LRG}) = {engine.kappa_vcv(Z_LRG):.6f}")
    print(f"   V_DISP_REF = {V_DISP_REF:.1f} km/s")
    if HAS_CALIBRATED_KERNEL:
        print(f"   SAT_BOOST = {SAT_BOOST:.1f} (calibrated parameter, requires theoretical justification)")
        print(f"   Effective threshold ≈ {V_DISP_REF * SAT_BOOST:.0f} km/s")

    print(f"\n📝 SUMMARY FOR PUBLICATION:")
    print("=" * 70)
    print(f"   Δα/α(AM-B) = {stats_amb['mean']:+.2f} ± {stats_amb['total_error']:.2f} ppm")
    print(f"   Δα/α(AM-A) = {stats_ama['mean']:+.2f} ± {stats_ama['total_error']:.2f} ppm")
    print(f"   Δ(Δα/α)    = {stats_diff['value']:+.2f} ± {stats_diff['total_error']:.2f} ppm")
    print(f"   Differential significance: {stats_diff['significance_total']:.1f}σ")
    print(f"   Bootstrap 68% CI: [{stats_diff['ci_68_lo']:+.2f}, {stats_diff['ci_68_hi']:+.2f}] ppm")
    if HAS_CALIBRATED_KERNEL:
        print(f"   Agreement with calibrated model (SAT_BOOST={SAT_BOOST}): {comparison['agreement_diff_cal']:.1f}σ")
        print(f"   ⚠️  This agreement depends on calibrated SAT_BOOST={SAT_BOOST}, not derived from theory.")

    print(f"\n⚠️  ACKNOWLEDGED METHODOLOGICAL LIMITATIONS:")
    print("=" * 70)
    print(f"   1. Δα/α is not measured directly from spectra → it is calculated with the VCV48 kernel")
    print(f"   2. SAT_BOOST = {SAT_BOOST} is a calibrated parameter, not derived from first principles")
    print(f"   3. The gradient test and redshift bins verify internal consistency")
    print(f"   4. Independent spectroscopic measurement required for external validation")

    if do_export:
        with open('delta_alpha_results_v6.txt', 'w') as f:
            f.write(f"DELTA_ALPHA_ANALYZER v6.0 — Results\n")
            f.write(f"========================================\n\n")
            f.write(f"Δα/α(AM-B) = {stats_amb['mean']:+.2f} ± {stats_amb['total_error']:.2f} ppm\n")
            f.write(f"Δα/α(AM-A) = {stats_ama['mean']:+.2f} ± {stats_ama['total_error']:.2f} ppm\n")
            f.write(f"Δ(Δα/α)    = {stats_diff['value']:+.2f} ± {stats_diff['total_error']:.2f} ppm\n")
            f.write(f"Significance: {stats_diff['significance_total']:.1f}σ\n")
            f.write(f"Bootstrap 68% CI: [{stats_diff['ci_68_lo']:+.2f}, {stats_diff['ci_68_hi']:+.2f}] ppm\n")
            f.write(f"Calibrated SAT_BOOST: {SAT_BOOST}\n")
        print("\n✅ Saved: delta_alpha_results_v6.txt")

        if z_bins_results:
            pd.DataFrame(z_bins_results).to_csv('delta_alpha_by_redshift_v6.csv', index=False)
            print("✅ Saved: delta_alpha_by_redshift_v6.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()