#!/usr/bin/env python3        
"""
CROSS-VALIDATION OF THERMAL β(z) — SDSS + DESI LRG
==============================================================================            
Test of thermal correction on two independent datasets.                             
Strategy:
  1. Load SDSS (sdss_vdisp_calidad.npz) and DESI (FITS).
  2. For each dataset, build proxy M_lum/M_dyn ∝ F·dL²/σ⁴.
  3. Bin by redshift and extract residual trend.
  4. Fit ξ_iso(z) and ξ_ter(z) against the residual.
  5. Measure ΔR² and its significance via bootstrap.
  6. Compare results between SDSS and DESI.

Frozen parameters from the Rust kernel vpm_wave.rs.
"""

import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import pearsonr
from numpy.polynomial import Polynomial
import sys
import os
import json
from datetime import datetime

# ==============================================================================
# FROZEN PARAMETERS (Rust kernel vpm_wave.rs)
# ==============================================================================
XI_0 = 0.084
Z_C = 1.5
BETA_0 = 0.03
THETA_D = 35.772
T_CMB_0 = 2.725

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.315)

# ==============================================================================
# ξ(z) FUNCTIONS
# ==============================================================================

def xi_isotermo(z):
    return XI_0 * np.exp(-z / Z_C) - BETA_0 * np.sqrt(z)

def xi_termico(z):
    t_cmb = T_CMB_0 * (1.0 + z)
    beta_z = BETA_0 * (1.0 + t_cmb / THETA_D)
    return XI_0 * np.exp(-z / Z_C) - beta_z * np.sqrt(z)

def r_squared(obs, pred):
    ss_res = np.sum((obs - pred)**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

# ==============================================================================
# MAIN ANALYSIS FUNCTION PER DATASET
# ==============================================================================

def analizar_dataset(name, z_data, vdisp_data, flux_data=None, n_bins=40,
                     z_min=0.05, z_max=0.80, vdisp_min=50, vdisp_max=500):
    """
    Complete analysis for a dataset.

    If flux_data is None, uses M_dyn ∝ σ⁴ as mass proxy
    and constructs the observable as ln(σ⁴ / dL²) ~ ln(M_dyn / dL²).
    """
    print(f"\n{'='*70}")
    print(f"  ANALYZING: {name}")
    print(f"{'='*70}")

    # Filter
    mask = (
        (z_data >= z_min) & (z_data <= z_max) &
        (vdisp_data >= vdisp_min) & (vdisp_data <= vdisp_max)
    )

    if flux_data is not None:
        mask = mask & (flux_data > 0) & np.isfinite(flux_data)

    z = z_data[mask]
    vdisp = vdisp_data[mask]
    n_gal = len(z)

    print(f"  Galaxies after filtering: {n_gal}")
    print(f"  z:     [{z.min():.4f}, {z.max():.4f}]")
    print(f"  σ:     [{vdisp.min():.0f}, {vdisp.max():.0f}] km/s")

    if n_gal < 1000:
        print(f"  ❌ Insufficient sample")
        return None

    # Build observable
    d_L = COSMO.luminosity_distance(z).value

    if flux_data is not None:
        flux = flux_data[mask]
        proxy_raw = (flux * d_L**2) / (vdisp**4)
        observable_label = "ln[(F·dL²/σ⁴) / median]"
    else:
        # Without flux: use σ⁴ as dynamical mass proxy
        # observable = ln(σ⁴) normalized
        proxy_raw = vdisp**4
        observable_label = "ln[σ⁴ / median]"

    proxy_norm = proxy_raw / np.median(proxy_raw)
    observable = np.log(proxy_norm)

    print(f"  Observable: {observable_label}")
    print(f"  Mean: {np.mean(observable):.4f}, Std: {np.std(observable):.4f}")

    # Bin
    bins = np.linspace(z_min, z_max, n_bins + 1)
    z_centers = 0.5 * (bins[:-1] + bins[1:])

    obs_binned = np.zeros(n_bins)
    obs_sem = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    xi_iso_b = np.zeros(n_bins)
    xi_ter_b = np.zeros(n_bins)

    for i in range(n_bins):
        mask_bin = (z >= bins[i]) & (z < bins[i+1])
        n = np.sum(mask_bin)
        counts[i] = n
        if n >= 15:
            obs_binned[i] = np.mean(observable[mask_bin])
            obs_sem[i] = np.std(observable[mask_bin]) / np.sqrt(n)
        else:
            obs_binned[i] = np.nan
            obs_sem[i] = np.nan
        xi_iso_b[i] = xi_isotermo(z_centers[i])
        xi_ter_b[i] = xi_termico(z_centers[i])

    valid = ~np.isnan(obs_binned)
    z_bin = z_centers[valid]
    obs_bin = obs_binned[valid]
    sem_bin = obs_sem[valid]
    xi_iso_bin = xi_iso_b[valid]
    xi_ter_bin = xi_ter_b[valid]

    print(f"  Valid bins: {len(z_bin)} (median: {np.median(counts[valid]):.0f} gal/bin)")

    # Remove systematics with cubic polynomial
    poly = Polynomial.fit(z_bin, obs_bin, deg=3)
    obs_smooth = poly(z_bin)
    obs_residual = obs_bin - obs_smooth

    var_explicada = 1 - np.var(obs_residual) / np.var(obs_bin)
    print(f"  Variance explained by polynomial: {var_explicada:.1%}")

    # Fit amplitudes
    A_iso = np.sum(obs_residual * xi_iso_bin) / np.sum(xi_iso_bin**2)
    A_ter = np.sum(obs_residual * xi_ter_bin) / np.sum(xi_ter_bin**2)

    pred_iso = A_iso * xi_iso_bin
    pred_ter = A_ter * xi_ter_bin

    r2_iso = r_squared(obs_residual, pred_iso)
    r2_ter = r_squared(obs_residual, pred_ter)
    delta_r2 = r2_ter - r2_iso

    r_iso, p_iso = pearsonr(obs_residual, xi_iso_bin)
    r_ter, p_ter = pearsonr(obs_residual, xi_ter_bin)

    # Bootstrap
    n_boot = 10000
    rng = np.random.default_rng(42)
    delta_r2_boot = np.zeros(n_boot)
    n_valid = len(z_bin)

    for b in range(n_boot):
        idx = rng.choice(n_valid, size=n_valid, replace=True)
        o_b = obs_residual[idx]
        xi_i = xi_iso_bin[idx]
        xi_t = xi_ter_bin[idx]

        A_i = np.sum(o_b * xi_i) / max(np.sum(xi_i**2), 1e-30)
        A_t = np.sum(o_b * xi_t) / max(np.sum(xi_t**2), 1e-30)

        r2_i = r_squared(o_b, A_i * xi_i)
        r2_t = r_squared(o_b, A_t * xi_t)
        delta_r2_boot[b] = r2_t - r2_i

    delta_r2_mean = np.mean(delta_r2_boot)
    delta_r2_std = np.std(delta_r2_boot)
    delta_r2_ci = np.percentile(delta_r2_boot, [2.5, 97.5])
    p_mejora = np.mean(delta_r2_boot > 0)
    sigma = delta_r2_mean / delta_r2_std if delta_r2_std > 0 else 0

    # Determine if improvement is significant
    if sigma > 3 and p_mejora > 0.99:
        verdict = "✅ ROBUST DETECTION"
    elif sigma > 2 and p_mejora > 0.95:
        verdict = "✅ STRONG EVIDENCE"
    elif delta_r2 > 0 and p_mejora > 0.5:
        verdict = "🔍 HINT"
    else:
        verdict = "❌ Not significant"

    # Results
    resultados = {
        'dataset': name,
        'n_galaxias': int(n_gal),
        'n_bins': int(n_valid),
        'z_range': [float(z_bin[0]), float(z_bin[-1])],
        'var_explicada_polinomio': float(var_explicada),
        'observable': observable_label,
        'amplitud_iso': float(A_iso),
        'amplitud_ter': float(A_ter),
        'r_iso': float(r_iso),
        'p_iso': float(p_iso),
        'r_ter': float(r_ter),
        'p_ter': float(p_ter),
        'r2_iso': float(r2_iso),
        'r2_ter': float(r2_ter),
        'delta_r2': float(delta_r2),
        'bootstrap': {
            'delta_r2_mean': float(delta_r2_mean),
            'delta_r2_std': float(delta_r2_std),
            'ci_95': [float(delta_r2_ci[0]), float(delta_r2_ci[1])],
            'p_mejora': float(p_mejora),
            'sigma': float(sigma)
        },
        'veredicto': verdict
    }

    # Display
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │ RESULTS: {name:<44s} │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │ R² isothermal: {r2_iso:8.6f}                         │")
    print(f"  │ R² thermal:    {r2_ter:8.6f}                         │")
    print(f"  │ ΔR²:           {delta_r2:+8.6f}                         │")
    print(f"  │ Bootstrap:     {delta_r2_mean:+8.6f} ± {delta_r2_std:.6f}                     │")
    print(f"  │ Significance:  {sigma:6.1f}σ                                   │")
    print(f"  │ P(improvement):{p_mejora:6.1%}                                    │")
    print(f"  │ Verdict:       {verdict:<42s} │")
    print(f"  └─────────────────────────────────────────────────────┘")

    return resultados


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

print("=" * 78)
print("  CROSS-VALIDATION OF THERMAL β(z) — SDSS + DESI LRG")
print(f"  Parameters: ξ₀={XI_0}, Z_c={Z_C}, β₀={BETA_0}, Θ_D={THETA_D}K")
print("=" * 78)

resultados_todos = []

# ==============================================================================
# 1. SDSS
# ==============================================================================

print(f"\n  [Loading SDSS]")
sdss_path = 'data/sdss_vdisp_calidad.npz'
if os.path.exists(sdss_path):
    data_sdss = np.load(sdss_path)
    z_sdss = data_sdss['Z']
    vdisp_sdss = data_sdss['VDISP']
    print(f"  ✅ SDSS loaded: {len(z_sdss):,} galaxies")
    print(f"     Columns: {list(data_sdss.keys())}")

    # SDSS does not have FLUX_R, use only σ
    res_sdss = analizar_dataset(
        "SDSS", z_sdss, vdisp_sdss, flux_data=None,
        n_bins=35, z_min=0.05, z_max=0.50, vdisp_min=150, vdisp_max=500
    )
    if res_sdss:
        resultados_todos.append(res_sdss)
else:
    print(f"  ❌ {sdss_path} not found")

# ==============================================================================
# 2. DESI LRG
# ==============================================================================

print(f"\n  [Loading DESI LRG]")
desi_path = 'data/DATASET_LRG_VDISP_FLUXR_FINAL.fits'
if os.path.exists(desi_path):
    data_desi = Table.read(desi_path)
    z_desi = data_desi['Z'].data.astype(np.float64)
    vdisp_desi = data_desi['VDISP'].data.astype(np.float64)
    flux_desi = data_desi['FLUX_R'].data.astype(np.float64)
    print(f"  ✅ DESI loaded: {len(z_desi):,} galaxies")

    res_desi = analizar_dataset(
        "DESI LRG", z_desi, vdisp_desi, flux_data=flux_desi,
        n_bins=50, z_min=0.05, z_max=0.50, vdisp_min=50, vdisp_max=500
    )
    if res_desi:
        resultados_todos.append(res_desi)
else:
    print(f"  ❌ {desi_path} not found")

# ==============================================================================
# COMPARATIVE SUMMARY
# ==============================================================================

print(f"\n{'='*78}")
print(f"  COMPARATIVE SUMMARY — CROSS-VALIDATION")
print(f"{'='*78}")

if len(resultados_todos) >= 2:
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │ SDSS vs DESI COMPARISON                                      │
  ├──────────────────┬────────────┬────────────┬─────────────────┤
  │ Metric           │ SDSS       │ DESI LRG   │ Consistent?     │
  ├──────────────────┼────────────┼────────────┼─────────────────┤""")

    r2 = resultados_todos
    for key, label in [('delta_r2', 'ΔR²'), ('amplitud_ter', 'A_ter'),
                         ('r_ter', 'r(ξ_ter)')]:
        v0 = r2[0].get(key, 0) if key in r2[0] else r2[0]['bootstrap'].get(key, 0)
        v1 = r2[1].get(key, 0) if key in r2[1] else r2[1]['bootstrap'].get(key, 0)
        if isinstance(v0, dict): v0 = v0.get('delta_r2_mean', 0)
        if isinstance(v1, dict): v1 = v1.get('delta_r2_mean', 0)
        consistente = "✅" if np.sign(v0) == np.sign(v1) else "⚠️"
        print(f"  │ {label:<16s} │ {v0:+.6f} │ {v1:+.6f} │ {consistente:<15s} │")

    print(f"""  └──────────────────┴────────────┴────────────┴─────────────────┘
""")

# Global verdict
n_mejora = sum(1 for r in resultados_todos if r['delta_r2'] > 0)
n_total = len(resultados_todos)

print(f"  GLOBAL VERDICT:")
print(f"    Datasets favoring thermal model: {n_mejora}/{n_total}")
if n_mejora == n_total:
    print(f"    ✅ TOTAL CONSISTENCY: all datasets show improvement")
elif n_mejora > n_total/2:
    print(f"    🔍 MAJORITY: most datasets favor the thermal model")
else:
    print(f"    ❌ No clear evidence in favor of the thermal model")

# ==============================================================================
# SAVE
# ==============================================================================

output = {
    'fecha': datetime.now().isoformat(),
    'parametros': {
        'XI_0': XI_0, 'Z_C': Z_C, 'BETA_0': BETA_0,
        'THETA_D': THETA_D, 'T_CMB_0': T_CMB_0
    },
    'n_datasets': len(resultados_todos),
    'n_mejora': n_mejora,
    'resultados': resultados_todos
}

with open('validacion_cruzada_sdss_desi.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n  💾 Saved to validacion_cruzada_sdss_desi.json")
print(f"{'='*78}\n")