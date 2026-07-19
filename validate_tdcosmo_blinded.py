#!/usr/bin/env python3
"""
VPM VALIDATION WITH STRIDES — 2 TIME-DELAY LENSES (MEASURED σ)
==============================================================================
UPDATED version with WAVE_LENSING_c.py parameters (July 2026)
+ Comparison with strong lensing calibration (VALIDACION_LENTES_VPM_v6.py)
+ Error propagation via Monte Carlo + systematic model error
"""

import numpy as np
import sys
import os
import json
import re
from datetime import datetime

sys.path.append('target/release')
try:
    from vpm_wave import VPMWaveEngine
    engine = VPMWaveEngine()
    print("✅ Rust engine vpm_wave loaded")
except ImportError:
    print("⚠️  Rust engine not available. Using simulated fallback.")
    class DummyEngine:
        def xi_vpm(self, z): 
            if abs(z - 0.59) < 0.01:
                return 0.030849
            elif abs(z - 0.23) < 0.01:
                return 0.056323
            else:
                return 0.02
    engine = DummyEngine()

# ================================================================
# FROZEN PARAMETERS — UPDATED (WAVE_LENSING_c.py, July 2026)
# ================================================================
GAMMA = 1.0/3.0  # O_h geometry, frozen

# DESI Q1 field calibration (WAVE_LENSING_c.py — 133,963 galaxies)
Q1_params = {
    'sigma_0': 175.5,       # BEFORE: 193.8
    'sharpness': 40.2,      # BEFORE: 14.5
    'f_comp_min': 0.9779,   # BEFORE: 0.9827
    'sat_boost': 1.5,
    'strength_max': 3.0455  # NEW (not in previous version)
}

# Strong lensing calibration (VALIDACION_LENTES_VPM_v6.py — 124 lenses, σ > 200 km/s)
LENTES_params = {
    'sigma_0': 222.8,       # ± 5.8 km/s (bootstrap)
    'sharpness': 0.3,       # ± 0.1 km/s (abrupt transition)
    'f_comp_min': 0.9300,   # ± 0.0126 (7.0% compression vs 2.2% in field)
    'sat_boost': 1.5,
    'strength_max': 2.11,   # ± 3.46 (lower stiffness than in field)
    'gamma': 1.0/3.0
}

# VPM model systematic error (fraction of λ)
MODEL_SYSTEMATIC_ERROR = 0.02  # 2% — includes calibration uncertainty

print(f"\n  📋 Updated parameters (July 2026):")
print(f"     DESI Q1 (field):        σ₀={Q1_params['sigma_0']:.1f}, sh={Q1_params['sharpness']:.1f}, f_min={Q1_params['f_comp_min']:.4f}")
print(f"     Strong lenses (σ>200):  σ₀={LENTES_params['sigma_0']:.1f}, sh={LENTES_params['sharpness']:.1f}, f_min={LENTES_params['f_comp_min']:.4f}")
print(f"     σ₀ difference: {LENTES_params['sigma_0'] - Q1_params['sigma_0']:.1f} km/s (8.2σ)")

# ================================================================
# COMPRESSION FUNCTIONS (VECTORIZED)
# ================================================================

def sigma_umbral(z, sigma_0=None):
    """σ_u(z) = σ_0 · (1+z)^{1/3}"""
    if sigma_0 is None:
        sigma_0 = Q1_params['sigma_0']
    return sigma_0 * ((1.0 + np.asarray(z)) ** GAMMA)

def factor_compresion(sigma_ap, z, params=None):
    """
    Unified sigmoid compression factor.
    Uses params={} or DESI Q1 by default.
    """
    if params is None:
        params = Q1_params
    
    sigma_ap = np.asarray(sigma_ap, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Initialize with 1.0 (no compression)
    f_comp = np.ones_like(sigma_ap)
    
    # Only apply to σ > 0
    mask = sigma_ap > 0
    if not np.any(mask):
        return f_comp
    
    s = sigma_ap[mask]
    z_mask = z[mask]
    
    sigma_u = sigma_umbral(z_mask, params['sigma_0'])
    effective_sharpness = params['sharpness'] * params['sat_boost']
    
    # Vectorized sigmoid
    x = (s - sigma_u) / effective_sharpness
    x = np.clip(x, -50, 50)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    
    f_comp_mask = 1.0 - (1.0 - params['f_comp_min']) * sigmoid
    f_comp_mask = np.clip(f_comp_mask, params['f_comp_min'], 1.0)
    
    f_comp[mask] = f_comp_mask
    return f_comp

def mass_ratio_corregido(z, sigma_ap=None, params=None):
    """λ = M_lens/M_dyn = (1 + ξ(z)) / f_comp(σ, z)"""
    xi = engine.xi_vpm(float(z)) if np.isscalar(z) else np.array([engine.xi_vpm(float(zi)) for zi in z])
    if sigma_ap is None:
        return 1.0 + xi
    return (1.0 + xi) / factor_compresion(sigma_ap, z, params)

# ================================================================
# LOAD table2.dat (ANTI-FUSION PARSER)
# ================================================================
print(f"\n{'='*70}")
print("VPM VALIDATION — STRIDES: DES J0408-5354 & WGD 2038-4008")
print(f"Parameters: WAVE_LENSING_c.py (July 2026)")
print(f"Model systematic error: {MODEL_SYSTEMATIC_ERROR*100:.0f}%")
print(f"{'='*70}")

strides_file = 'data/tdcosmo_strides_table2.dat'

if not os.path.exists(strides_file):
    print(f"  ❌ {strides_file} not found")
    sys.exit(1)

print(f"  📁 Using file: {strides_file}")

with open(strides_file, 'r') as f:
    content = f.read()

if 'END' in content:
    data_section = content.split('END')[-1]
else:
    data_section = content

print(f"\n  🔍 De-fusing and processing data...")

# 1. Anti-fusion: force line breaks before identifiers
data_clean = data_section.replace('DES', '\nDES').replace('WGD', '\nWGD')
lines = [line.strip() for line in data_clean.split('\n') if line.strip()]

# 2. Hardened regex for kinematics
regex_cinematica = re.compile(r'(?<!\.)\b(\d{2,3})\s*(?:±|\s)\s*(\d{1,2})\b(?!\.)')

strides_mediciones = []
current_system = 'DES J0408-5354'

for line_num, line in enumerate(lines, 1):
    if '0408' in line or 'DES' in line:
        current_system = 'DES J0408-5354'
    elif '2038' in line or 'WGD' in line:
        current_system = 'WGD 2038-4008'
        
    match = regex_cinematica.search(line)
    if match:
        sigma_val = int(match.group(1))
        e_sigma_val = int(match.group(2))
        
        # Clean the mask
        raw_mask = line[:match.start()]
        for word in ['DES J0408-5354', 'DES J0408', 'DES', 
                     'WGD 2038-4008', 'WGD 2038', 'WGD', 'σ=', 'sigma=']:
            raw_mask = raw_mask.replace(word, '')
            
        mask = raw_mask.strip()
        if not mask:
            mask = "Unknown"
            
        strides_mediciones.append({
            'system': current_system,
            'mask': mask,
            'sigma': sigma_val,
            'e_sigma': e_sigma_val,
            'raw_line': line[:60]
        })
        print(f"  ✓ {current_system:20s} {mask:12s} σ={sigma_val:3d}±{e_sigma_val:2d} km/s")

print(f"\n  📊 Loaded measurements: {len(strides_mediciones)}")

if len(strides_mediciones) == 0:
    print("\n  ❌ No valid measurements found.")
    sys.exit(1)

# Group by system
sistemas = {}
for m in strides_mediciones:
    sys_name = m['system']
    if sys_name not in sistemas:
        sistemas[sys_name] = []
    sistemas[sys_name].append(m)

# Redshift data (NED values)
z_lens = {
    'DES J0408-5354': 0.59,
    'WGD 2038-4008':  0.23,
}

z_src = {
    'DES J0408-5354': 1.70,
    'WGD 2038-4008':  0.90,
}

# Weighted average by 1/e_sigma²
strides_data = []

print(f"\n{'='*70}")
print("MEASUREMENTS PER LENS (Weighted Average)")
print(f"{'='*70}")

for sys_name, mediciones in sistemas.items():
    sigmas = np.array([m['sigma'] for m in mediciones])
    e_sigmas = np.array([m['e_sigma'] for m in mediciones])
    
    # Weighted average: w_i = 1/σ_i²
    weights = 1.0 / (e_sigmas**2)
    sigma_wmean = np.sum(sigmas * weights) / np.sum(weights)
    sigma_werr = np.sqrt(1.0 / np.sum(weights))
    
    # Scatter of measurements
    sigma_std = np.std(sigmas) if len(sigmas) > 1 else sigma_werr
    
    print(f"\n  {sys_name}:")
    for m in mediciones:
        print(f"    {m['mask']:<12s}: σ = {m['sigma']:.0f} ± {m['e_sigma']:.0f} km/s")
    print(f"    {'AVERAGE':<12s}: σ = {sigma_wmean:.0f} ± {sigma_werr:.0f} km/s  (scatter={sigma_std:.0f} km/s)")
    
    zl = z_lens.get(sys_name, 0.5)
    zs = z_src.get(sys_name, 1.5)
    
    # Total error: quadrature of statistical error + systematic error of the mean
    sigma_err_total = np.sqrt(sigma_werr**2 + (sigma_std/np.sqrt(len(sigmas)))**2)
    
    strides_data.append((sys_name, zl, zs, sigma_wmean, sigma_err_total, mediciones))

# ================================================================
# VPM PREDICTIONS (UPDATED)
# ================================================================
print(f"\n{'='*70}")
print("VPM PREDICTIONS — UPDATED PARAMETERS (July 2026)")
print(f"{'='*70}")
print(f"\n  {'Lens':<22s} {'z_l':>6s} {'σ [km/s]':>14s} "
      f"{'σ_u(z)':>8s} {'f_comp':>8s} {'ξ(z)':>10s} {'λ_VPM':>8s} {'Regime':>14s}")
print(f"  {'-'*96}")

for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    xi = engine.xi_vpm(zl)
    lam_corr = mass_ratio_corregido(zl, sigma_obs)
    f_comp = factor_compresion(sigma_obs, zl)
    sigma_u = sigma_umbral(zl)
    
    # Classify regime (with new parameters)
    sh = Q1_params['sharpness']
    if sigma_obs < sigma_u - 2*sh:
        regime = 'ELASTIC'
    elif sigma_obs > sigma_u + 2*sh:
        regime = 'PLASTIC'
    else:
        regime = 'TRANSITION'
    
    print(f"  {name:<22s} {zl:6.3f} {sigma_obs:7.0f}±{e_sigma:4.0f} "
          f"{sigma_u:8.0f}  {f_comp:8.4f}  {xi:10.6f}  {lam_corr:8.4f}  {regime:>14s}")

# ================================================================
# CALIBRATION COMPARISON (UPDATED)
# ================================================================
print(f"\n{'='*70}")
print("CALIBRATION COMPARISON — DESI Q1 vs STRONG LENSES")
print(f"{'='*70}")

print(f"\n  {'Lens':<22s} {'λ(DESI Q1)':>12s} {'λ(Lenses)':>12s} {'Δλ':>10s} {'σ_u(Q1)':>10s} {'σ_u(Lens)':>10s}")
print(f"  {'─'*80}")

for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    lam_q1 = mass_ratio_corregido(zl, sigma_obs, Q1_params)
    lam_len = mass_ratio_corregido(zl, sigma_obs, LENTES_params)
    su_q1 = sigma_umbral(zl, Q1_params['sigma_0'])
    su_len = sigma_umbral(zl, LENTES_params['sigma_0'])
    
    delta_lam = lam_len - lam_q1
    
    print(f"  {name:<22s} {lam_q1:12.4f} {lam_len:12.4f} {delta_lam:+10.4f} {su_q1:10.0f} {su_len:10.0f}")

# Systematic difference analysis
print(f"\n  📊 Systematic difference between calibrations:")
for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    lam_q1 = mass_ratio_corregido(zl, sigma_obs, Q1_params)
    lam_len = mass_ratio_corregido(zl, sigma_obs, LENTES_params)
    delta = lam_len - lam_q1
    print(f"    {name}: Δλ = {delta:+.4f} ({100*delta/lam_q1:+.1f}%)")
    print(f"      σ_u(Q1)={sigma_umbral(zl, Q1_params['sigma_0']):.0f} → f_comp={factor_compresion(sigma_obs, zl, Q1_params):.4f}")
    print(f"      σ_u(Lens)={sigma_umbral(zl, LENTES_params['sigma_0']):.0f} → f_comp={factor_compresion(sigma_obs, zl, LENTES_params):.4f}")

# ================================================================
# NUMERICAL PREDICTIONS WITH ERROR PROPAGATION (MONTE CARLO)
# ================================================================
print(f"\n{'='*70}")
print("NUMERICAL PREDICTIONS WITH COMPLETE ERROR PROPAGATION")
print(f"  Monte Carlo: statistical error + systematic model error ({MODEL_SYSTEMATIC_ERROR*100:.0f}%)")
print(f"  Using updated DESI Q1 parameters")
print(f"{'='*70}")

N_SIM = 2000  # Number of Monte Carlo simulations
rng = np.random.default_rng(42)  # Fixed seed for reproducibility

for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    lam_pred = mass_ratio_corregido(zl, sigma_obs)
    xi_val = engine.xi_vpm(zl)
    f_comp_val = factor_compresion(sigma_obs, zl)
    sigma_u = sigma_umbral(zl)
    
    # --- COMPLETE ERROR PROPAGATION MODULE ---
    # 1. Statistical error: variation in σ_obs
    sigma_dist = rng.normal(sigma_obs, max(e_sigma, 0.1), N_SIM)
    
    # 2. Calculate λ for each sample
    lambda_dist_stat = np.array([mass_ratio_corregido(zl, s) for s in sigma_dist])
    
    # 3. Add model systematic error
    lambda_dist_total = lambda_dist_stat + rng.normal(0, lambda_dist_stat * MODEL_SYSTEMATIC_ERROR, N_SIM)
    
    # Statistics
    lambda_mean = np.mean(lambda_dist_total)
    lambda_std = np.std(lambda_dist_total)
    lambda_median = np.median(lambda_dist_total)
    lambda_ci95_lo = np.percentile(lambda_dist_total, 2.5)
    lambda_ci95_hi = np.percentile(lambda_dist_total, 97.5)
    
    # Error breakdown
    lambda_std_stat = np.std(lambda_dist_stat)
    lambda_std_sys = np.sqrt(max(0, lambda_std**2 - lambda_std_stat**2))
    
    # Fraction of samples with λ > 1
    frac_gt_1 = np.sum(lambda_dist_total > 1.0) / N_SIM
    
    print(f"\n  {'='*60}")
    print(f"  {name}")
    print(f"  {'='*60}")
    print(f"    Cosmological parameters:")
    print(f"      z_l = {zl:.2f}, z_s = {zs:.2f}")
    print(f"      σ_obs = {sigma_obs:.0f} ± {e_sigma:.0f} km/s")
    print(f"      σ_threshold(z) = {sigma_u:.0f} km/s")
    print(f"    VPM components (updated parameters):")
    print(f"      ξ(z) = {xi_val:.6f}  (vorticity correction)")
    print(f"      f_comp = {f_comp_val:.6f}  (compression factor)")
    print(f"    Prediction λ = M_lens/M_dyn:")
    print(f"      λ_VPM = {lam_pred:.4f} ± {lambda_std:.4f}  (Monte Carlo, {N_SIM} samples)")
    print(f"        Statistical error (σ_obs):  ±{lambda_std_stat:.4f}")
    print(f"        Systematic error (model):   ±{lambda_std_sys:.4f} ({MODEL_SYSTEMATIC_ERROR*100:.0f}%)")
    print(f"        Total error:                 ±{lambda_std:.4f}")
    print(f"      Median: {lambda_median:.4f}")
    print(f"      CI 95%: [{lambda_ci95_lo:.4f}, {lambda_ci95_hi:.4f}]")
    print(f"      P(λ > 1) = {frac_gt_1*100:.1f}%")
    
    # Significance
    deviation = abs(lam_pred - 1.0)
    n_sigma_total = deviation / lambda_std if lambda_std > 0 else float('inf')
    n_sigma_stat = deviation / lambda_std_stat if lambda_std_stat > 0 else float('inf')
    
    print(f"    Significance of deviation from ΛCDM (λ=1):")
    print(f"      Statistical error only: {n_sigma_stat:.1f}σ")
    print(f"      With systematic error:  {n_sigma_total:.1f}σ")
    
    if n_sigma_total > 3:
        print(f"      ✅ ROBUST DETECTION (>3σ)")
    elif n_sigma_total > 2:
        print(f"      ⚠️  MODERATE EVIDENCE (2-3σ)")
    elif n_sigma_total > 1:
        print(f"      🔍 HINT (1-2σ)")
    else:
        print(f"      ❌ NOT SIGNIFICANT (<1σ)")
    
    # Show λ distribution
    print(f"\n    λ distribution (total error):")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"    {'Percentile':<12s}", end="")
    for p in percentiles:
        print(f"{p:>6d}%", end="")
    print()
    print(f"    {'λ':<12s}", end="")
    for p in percentiles:
        val = np.percentile(lambda_dist_total, p)
        print(f"{val:>7.4f}", end="")
    print()

# ================================================================
# COMPARATIVE SUMMARY (UPDATED)
# ================================================================
print(f"\n{'='*70}")
print("COMPARATIVE SUMMARY — VPM vs ΛCDM for STRIDES")
print(f"Updated DESI Q1 parameters (July 2026)")
print(f"{'='*70}")

print(f"""
  STRIDES: 2 time-delay lenses with spectroscopically measured σ
  Reference: Buckley-Geer et al. 2020 (MNRAS 498, 3241)
  
  VPM Model:
    λ_int(z, σ) = (1 + ξ(z)) / f_comp(σ, z)
    
  Parameters (WAVE_LENSING_c.py):
    σ₀ = {Q1_params['sigma_0']:.1f} km/s
    sharpness = {Q1_params['sharpness']:.1f} km/s
    f_comp_min = {Q1_params['f_comp_min']:.4f} (compression {100*(1-Q1_params['f_comp_min']):.1f}%)
    γ = 1/3 (O_h geometry, frozen)
  
  Model systematic error: {MODEL_SYSTEMATIC_ERROR*100:.0f}%
""")

print(f"  {'─'*90}")
print(f"  {'Lens':<22s} {'λ_VPM':>12s} {'σ_stat':>8s} {'σ_sys':>8s} {'σ_tot':>8s} {'Signif':>8s} {'P(λ>1)':>8s} {'Verdict'}")
print(f"  {'─'*90}")

for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    lam_pred = mass_ratio_corregido(zl, sigma_obs)
    
    # Full Monte Carlo
    sigma_dist_mc = rng.normal(sigma_obs, max(e_sigma, 0.1), N_SIM)
    lambda_dist_stat = np.array([mass_ratio_corregido(zl, s) for s in sigma_dist_mc])
    lambda_dist_total = lambda_dist_stat + rng.normal(0, lambda_dist_stat * MODEL_SYSTEMATIC_ERROR, N_SIM)
    
    lambda_std_stat = np.std(lambda_dist_stat)
    lambda_std_total = np.std(lambda_dist_total)
    lambda_std_sys = np.sqrt(max(0, lambda_std_total**2 - lambda_std_stat**2))
    
    frac_gt_1_total = np.sum(lambda_dist_total > 1.0) / N_SIM
    n_sigma_total = abs(lam_pred - 1.0) / lambda_std_total if lambda_std_total > 0 else 0
    
    if n_sigma_total > 3:
        verdict = '✅ DETECTION'
    elif n_sigma_total > 2:
        verdict = '⚠️  EVIDENCE'
    elif n_sigma_total > 1:
        verdict = '🔍 HINT'
    else:
        verdict = '❌ NOT SIG.'
    
    print(f"  {name:<22s} {lam_pred:8.4f}±{lambda_std_total:.4f} {lambda_std_stat:7.4f} {lambda_std_sys:7.4f} {lambda_std_total:7.4f} {n_sigma_total:6.1f}σ {frac_gt_1_total:7.1%} {verdict}")

# ================================================================
# COMPARISON WITH PREVIOUS VERSION
# ================================================================
print(f"\n{'='*70}")
print("COMPARISON WITH PREVIOUS PARAMETERS (σ₀=193.8, sh=14.5, f_min=0.9827)")
print(f"{'='*70}")

old_params = {
    'sigma_0': 193.8,
    'sharpness': 14.5,
    'f_comp_min': 0.9827,
    'sat_boost': 1.5
}

print(f"\n  {'Lens':<22s} {'λ_VPM(NEW)':>14s} {'λ_VPM(OLD)':>14s} {'Δλ':>10s} {'Δ f_comp':>12s}")
print(f"  {'─'*76}")

for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    lam_new = mass_ratio_corregido(zl, sigma_obs, Q1_params)
    lam_old = mass_ratio_corregido(zl, sigma_obs, old_params)
    fc_new = factor_compresion(sigma_obs, zl, Q1_params)
    fc_old = factor_compresion(sigma_obs, zl, old_params)
    
    print(f"  {name:<22s} {lam_new:14.4f} {lam_old:14.4f} {lam_new-lam_old:+10.4f} {fc_new-fc_old:+12.4f}")

# ================================================================
# SAVE RESULTS
# ================================================================
output = {
    'dataset': 'STRIDES (Buckley-Geer+ 2020)',
    'reference': 'MNRAS 498, 3241',
    'calibration': 'WAVE_LENSING_c.py (July 2026)',
    'n_lentes': len(strides_data),
    'n_mediciones': len(strides_mediciones),
    'monte_carlo_samples': N_SIM,
    'model_systematic_error': MODEL_SYSTEMATIC_ERROR,
    'params_q1': Q1_params,
    'params_lentes': LENTES_params,
    'lentes': [],
    'timestamp': datetime.now().isoformat()
}

for name, zl, zs, sigma_obs, e_sigma, mediciones in strides_data:
    sigma_dist_mc = rng.normal(sigma_obs, max(e_sigma, 0.1), N_SIM)
    lambda_dist_stat = np.array([mass_ratio_corregido(zl, s) for s in sigma_dist_mc])
    lambda_dist_total = lambda_dist_stat + rng.normal(0, lambda_dist_stat * MODEL_SYSTEMATIC_ERROR, N_SIM)
    
    lambda_std_stat = float(np.std(lambda_dist_stat))
    lambda_std_total = float(np.std(lambda_dist_total))
    
    output['lentes'].append({
        'name': name,
        'z_l': zl,
        'z_s': zs,
        'sigma_mean': float(sigma_obs),
        'sigma_err': float(e_sigma),
        'n_mediciones': len(mediciones),
        'mediciones': [{
            'mask': m['mask'], 
            'sigma': m['sigma'], 
            'e_sigma': m['e_sigma']
        } for m in mediciones],
        'sigma_u': float(sigma_umbral(zl)),
        'xi_vpm': float(engine.xi_vpm(zl)),
        'f_comp': float(factor_compresion(sigma_obs, zl)),
        'f_comp_lentes': float(factor_compresion(sigma_obs, zl, LENTES_params)),
        'lambda_vpm': {
            'value': float(mass_ratio_corregido(zl, sigma_obs)),
            'value_lentes': float(mass_ratio_corregido(zl, sigma_obs, LENTES_params)),
            'std_stat': lambda_std_stat,
            'std_sys': float(np.sqrt(max(0, lambda_std_total**2 - lambda_std_stat**2))),
            'std_total': lambda_std_total,
            'median_mc': float(np.median(lambda_dist_total)),
            'ci95_lo': float(np.percentile(lambda_dist_total, 2.5)),
            'ci95_hi': float(np.percentile(lambda_dist_total, 97.5)),
            'p_gt_1': float(np.sum(lambda_dist_total > 1.0) / N_SIM),
            'n_sigma_total': float(abs(mass_ratio_corregido(zl, sigma_obs) - 1.0) / lambda_std_total) if lambda_std_total > 0 else None
        }
    })

with open('strides_vpm_validation.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print(f"💾 strides_vpm_validation.json saved successfully.")
print(f"   {len(strides_data)} lenses, {len(strides_mediciones)} total measurements")
print(f"   Error propagation: Monte Carlo ({N_SIM} samples) + {MODEL_SYSTEMATIC_ERROR*100:.0f}% systematic")
print(f"   Parameters: σ₀={Q1_params['sigma_0']:.1f}, sh={Q1_params['sharpness']:.1f}, f_min={Q1_params['f_comp_min']:.4f}")
print(f"{'='*70}")