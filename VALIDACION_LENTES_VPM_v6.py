#!/usr/bin/env python3
"""
VPM VALIDATION — QUALITY CUT σ > 200 km/s
VERSION 6: OPTIMIZED (VECTORIZED + FAST)
=========================================================================
- D_A(z): polynomial approximation (original calibration)
- G(z): linear interpolation (original calibration)
- ξ(z): Rust kernel (exact)
- gamma = 1/3 frozen (O_h geometry)
- Comparison with DESI Q1 parameters (WAVE_LENSING_c.py)
- VECTORIZED for performance
"""

import numpy as np
from scipy.stats import ttest_1samp, pearsonr
from scipy.optimize import minimize, differential_evolution
import sys
import json
import time

sys.path.append('target/release')
from vpm_wave import VPMWaveEngine

engine = VPMWaveEngine()

t_start = time.time()

print("=" * 70)
print("VPM VALIDATION — QUALITY CUT σ > 200 km/s")
print("VERSION 6: VECTORIZED + FAST OPTIMIZATION")
print("gamma = 1/3 FROZEN — KERNEL + APPROXIMATIONS")
print("=" * 70)

c_kms = 299792.458
H0 = 70.0

# ================================================================
# O_h LATTICE PARAMETERS (FROZEN)
# ================================================================
Phi_alpha_0 = 0.98022575
eta1_0 = 0.25229889
eta2_0 = 0.15000000
ratio_eta_0 = eta2_0 / eta1_0

Phi_alpha_z258 = 0.98452
eta1_z258 = 0.2693
eta2_z258 = 0.0993
ratio_eta_z258 = eta2_z258 / eta1_z258

GAMMA_O_h = 1.0 / 3.0

# ================================================================
# DESI Q1 PARAMETERS (WAVE_LENSING_c.py) — REFERENCE
# ================================================================
DESI_Q1 = {
    'sigma_0': 175.5,
    'strength_max': 3.0455,
    'f_comp_min': 0.9779,
    'sharpness': 40.2,
    'sat_boost': 1.5
}

# ================================================================
# ORIGINAL APPROXIMATIONS (VECTORIZED)
# ================================================================

def D_A_kpc(z):
    """Angular diameter distance with polynomial approximation (vectorized)"""
    z = np.asarray(z)
    comoving_Mpc = (c_kms / H0) * (z - 0.35*z**2 + 0.14*z**3)
    return (comoving_Mpc / (1.0 + z)) * 1000.0

def evolucion_G_cosmologica(z):
    """G evolution with linear interpolation (vectorized)"""
    z = np.asarray(z)
    z_ref = 2.578
    
    # Initialize arrays
    Phi = np.zeros_like(z)
    ratio = np.zeros_like(z)
    
    # z <= 0
    mask0 = z <= 0
    Phi[mask0] = Phi_alpha_0
    ratio[mask0] = ratio_eta_0
    
    # 0 < z <= z_ref
    mask_ref = (z > 0) & (z <= z_ref)
    frac = z[mask_ref] / z_ref
    Phi[mask_ref] = Phi_alpha_0 + frac * (Phi_alpha_z258 - Phi_alpha_0)
    ratio[mask_ref] = ratio_eta_0 + frac * (ratio_eta_z258 - ratio_eta_0)
    
    # z > z_ref
    mask_high = z > z_ref
    dz = z[mask_high] - z_ref
    Phi[mask_high] = Phi_alpha_z258 + (0.987 - Phi_alpha_z258) * (1 - np.exp(-dz/3.0))
    ratio[mask_high] = ratio_eta_z258 + (0.25 - ratio_eta_z258) * (1 - np.exp(-dz/3.0))
    
    return (Phi_alpha_0 / Phi) * (ratio / ratio_eta_0)

# ================================================================
# KERNEL FUNCTIONS (EXACT ξ(z))
# ================================================================

def xi_vpm(z):
    """Exact ξ(z) from Rust kernel (vectorized)"""
    z = np.asarray(z)
    if z.ndim == 0:
        return engine.xi_vpm(float(z))
    return np.array([engine.xi_vpm(float(zi)) for zi in z])

# ================================================================
# GEOMETRIC COMPRESSION MODEL (VECTORIZED)
# ================================================================

def sigma_umbral_efectivo(z, sigma_0):
    """σ_u(z) = σ_0 · (1+z)^{1/3} (vectorized)"""
    return sigma_0 * ((1.0 + np.asarray(z)) ** GAMMA_O_h)

def factor_compresion_geometrico_vectorizado(sigma_ap, z, params):
    """
    VECTORIZED geometric compression factor.
    Operates on full arrays without Python loop.
    """
    sigma_ap = np.asarray(sigma_ap, dtype=float)
    z = np.asarray(z, dtype=float)
    
    strength_max, f_comp_min, sigma_0, sharpness = params
    
    # Initialize with 1.0 (no compression)
    f_comp = np.ones_like(sigma_ap)
    
    # Only apply to σ > 150
    mask = sigma_ap > 150
    if not np.any(mask):
        return f_comp
    
    s = sigma_ap[mask]
    z_mask = z[mask]
    
    G_ratio = evolucion_G_cosmologica(z_mask)
    stiffness = 1.0 / G_ratio - 1.0
    
    sigma_u = sigma_umbral_efectivo(z_mask, sigma_0)
    
    # Vectorized sigmoid
    arg = (s - sigma_u) / sharpness
    arg = np.clip(arg, -50, 50)
    compression_weight = 1.0 / (1.0 + np.exp(-arg))
    
    # Maximum compression
    mass_excess = (s - 150.0) / 200.0
    max_compression = strength_max * mass_excess * (1.0 + stiffness)
    f_comp_unsaturated = 1.0 / (1.0 + max_compression)
    
    # Interpolation
    f_comp_mask = f_comp_unsaturated + compression_weight * (f_comp_min - f_comp_unsaturated)
    
    # Safe clipping
    floor = np.maximum(0.70, f_comp_min - 0.05)
    f_comp_mask = np.clip(f_comp_mask, floor, 1.0)
    
    f_comp[mask] = f_comp_mask
    return f_comp

def calc_fE_obs(thetaE_arcsec, sigma_ap, zl, zs, theta_ap=1.5):
    """Observed f_E (vectorized)"""
    thetaE_arcsec = np.asarray(thetaE_arcsec)
    sigma_ap = np.asarray(sigma_ap)
    zl = np.asarray(zl)
    zs = np.asarray(zs)
    
    thetaE_rad = thetaE_arcsec * np.pi / (180.0 * 3600.0)
    D_l = D_A_kpc(zl)
    D_s = D_A_kpc(zs)
    D_ls = np.maximum(D_s - D_l, D_s * 0.1)
    factor = (c_kms**2 / (4.0 * np.pi * sigma_ap**2)) * thetaE_rad
    geom = D_s / D_ls
    ap_corr = (thetaE_arcsec / theta_ap)**(0.08)
    return np.sqrt(factor * geom * ap_corr)

def calc_fE_vpm_geometrico(zl, zs, sigma_ap, params):
    """Predicted f_E: (1+ξ)/f_comp (vectorized)"""
    xi = xi_vpm(zl)
    f_comp = factor_compresion_geometrico_vectorizado(sigma_ap, zl, params)
    return (1.0 + xi) / f_comp

def funcion_costo_simple(params, zl, zs, sigma_ap, fE_obs):
    """Cost: mean |Δ| (vectorized)"""
    fE_pred = calc_fE_vpm_geometrico(zl, zs, sigma_ap, params)
    delta = fE_obs - fE_pred
    return np.mean(np.abs(delta))

# ================================================================
# DATA LOADING AND QUALITY CUT
# ================================================================

print("\n[1/6] Loading Chen+ 2019...")
data = []
with open('data/chen2019_tablea1.dat', 'r') as f:
    for line in f:
        line = line.rstrip()
        if len(line) < 70:
            continue
        try:
            zl = float(line[19:25].strip())
            zs = float(line[26:32].strip())
            thetaE = float(line[33:37].strip())
            thetaap = float(line[57:61].strip())
            sigap = float(line[62:65].strip())
            survey = line[69:82].strip()
            if all(v > 0 for v in [zl, zs, thetaE, thetaap, sigap]):
                data.append({
                    'zl': zl, 'zs': zs, 'thetaE': thetaE,
                    'thetaap': thetaap, 'sigap': sigap, 'survey': survey
                })
        except (ValueError, IndexError):
            continue

print(f"  Total lenses loaded: {len(data)}")

# QUALITY CUT
data_corte = [d for d in data if d['sigap'] > 200]
print(f"  Lenses with σ > 200 km/s: {len(data_corte)}")
print(f"  Excluded lenses (σ ≤ 200): {len(data) - len(data_corte)}")

zl = np.array([d['zl'] for d in data_corte])
zs = np.array([d['zs'] for d in data_corte])
thetaE = np.array([d['thetaE'] for d in data_corte])
thetaap = np.array([d['thetaap'] for d in data_corte])
sigap = np.array([d['sigap'] for d in data_corte])
surveys = np.array([d['survey'] for d in data_corte])

fE_obs = calc_fE_obs(thetaE, sigap, zl, zs, thetaap)
fE_vpm_original = 1.0 + xi_vpm(zl)

print(f"  σ_ap: {np.min(sigap):.0f}–{np.max(sigap):.0f} km/s (median: {np.median(sigap):.0f})")
print(f"  z_l: {np.min(zl):.3f}–{np.max(zl):.3f} (median: {np.median(zl):.3f})")
print(f"  f_E obs: {np.min(fE_obs):.3f}–{np.max(fE_obs):.3f} (median: {np.median(fE_obs):.3f})")

# ================================================================
# PRELIMINARY COMPARISON: ORIGINAL VPM vs DESI Q1
# ================================================================

print(f"\n[2/6] Preliminary comparison...")

params_desi = [DESI_Q1['strength_max'], DESI_Q1['f_comp_min'],
               DESI_Q1['sigma_0'], DESI_Q1['sharpness']]
fE_desi = calc_fE_vpm_geometrico(zl, zs, sigap, params_desi)

delta_orig = fE_obs - fE_vpm_original
delta_desi = fE_obs - fE_desi

mae_orig = np.mean(np.abs(delta_orig))
mae_desi = np.mean(np.abs(delta_desi))

print(f"  Original VPM (ξ only):       |Δ| = {mae_orig:.4f}")
print(f"  VPM + DESI Q1 (frozen):      |Δ| = {mae_desi:.4f}")
if mae_desi < mae_orig:
    print(f"  → DESI Q1 parameters already improve the fit ({100*(mae_orig-mae_desi)/mae_orig:.1f}%)")
else:
    print(f"  → DESI Q1 parameters do not improve. Recalibration required.")

# ================================================================
# OPTIMIZATION ON STRONG LENSES
# ================================================================

print(f"\n[3/6] Optimization on strong lenses (vectorized)...")

bounds_de = [
    (0.01, 15.0),    # strength_max
    (0.80, 0.99),    # f_comp_min
    (150.0, 280.0),  # sigma_0
    (0.5, 80.0)      # sharpness
]

print(f"  Lenses: {len(zl)}")
print(f"  Bounds: sm∈[0.01,15], fc∈[0.80,0.99], σ₀∈[150,280], sh∈[0.5,80]")
print(f"  Differential Evolution: 200 iter, pop=10, seed=42...")

t_de_start = time.time()
result_de = differential_evolution(
    funcion_costo_simple,
    bounds_de,
    args=(zl, zs, sigap, fE_obs),
    seed=42,
    maxiter=200,       # REDUCED: 200 is sufficient for 4 params
    popsize=10,        # Small but adequate population
    tol=1e-8,
    polish=False
)
t_de = time.time() - t_de_start
print(f"  DE completed in {t_de:.1f}s")
print(f"  DE: |Δ| = {result_de.fun:.6f}, x = {result_de.x}")

print(f"\n  Local refinement (Nelder-Mead, max 5000 iter)...")
t_nm_start = time.time()
result_nm = minimize(
    funcion_costo_simple,
    result_de.x,
    args=(zl, zs, sigap, fE_obs),
    method='Nelder-Mead',
    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
)
t_nm = time.time() - t_nm_start
print(f"  NM completed in {t_nm:.1f}s")

if result_nm.fun < result_de.fun:
    params_opt = result_nm.x
    best_fun = result_nm.fun
    print(f"  NM improved DE: |Δ| = {best_fun:.6f}")
else:
    params_opt = result_de.x
    best_fun = result_de.fun
    print(f"  DE optimal: |Δ| = {best_fun:.6f}")

# ================================================================
# MAIN RESULTS
# ================================================================

print(f"\n[4/6] Main results...")

fE_geo = calc_fE_vpm_geometrico(zl, zs, sigap, params_opt)
delta_geo = fE_obs - fE_geo

mae_geo = np.mean(np.abs(delta_geo))
improvement_abs = mae_orig - mae_geo
improvement_pct = 100 * improvement_abs / mae_orig

print(f"\n  {'='*60}")
print(f"  MODEL COMPARISON")
print(f"  {'='*60}")
print(f"  Model                          Mean |Δ|     Improvement")
print(f"  {'-'*60}")
print(f"  Original VPM (ξ only)          {mae_orig:.4f}        ---")
print(f"  VPM + DESI Q1 (frozen)         {mae_desi:.4f}        {100*(mae_orig-mae_desi)/mae_orig:+.1f}%")
print(f"  Geometric VPM (optimal)        {mae_geo:.4f}        {improvement_pct:+.1f}%")

print(f"\n  OPTIMAL PARAMETERS (strong lenses, σ > 200 km/s):")
print(f"  ┌────────────────────────────────────────────┐")
print(f"  │ strength_max = {params_opt[0]:.4f}                      │")
print(f"  │ f_comp_min   = {params_opt[1]:.4f}  → compression {1-params_opt[1]:.4f} ({100*(1-params_opt[1]):.1f}%) │")
print(f"  │ sigma_0      = {params_opt[2]:.1f} km/s                  │")
print(f"  │ sharpness    = {params_opt[3]:.1f} km/s                   │")
print(f"  └────────────────────────────────────────────┘")

print(f"\n  COMPARISON WITH DESI Q1 (field, 133,963 galaxies):")
print(f"  ┌─────────────────┬────────────┬────────────┐")
print(f"  │ Parameter       │ DESI Q1    │ Lenses     │")
print(f"  ├─────────────────┼────────────┼────────────┤")
print(f"  │ sigma_0 [km/s]  │ {DESI_Q1['sigma_0']:.1f}      │ {params_opt[2]:.1f}      │")
print(f"  │ f_comp_min      │ {DESI_Q1['f_comp_min']:.4f}      │ {params_opt[1]:.4f}      │")
print(f"  │ strength_max    │ {DESI_Q1['strength_max']:.4f}      │ {params_opt[0]:.4f}      │")
print(f"  │ sharpness [km/s]│ {DESI_Q1['sharpness']:.1f}       │ {params_opt[3]:.1f}       │")
print(f"  └─────────────────┴────────────┴────────────┘")

# ================================================================
# RESIDUAL ANALYSIS
# ================================================================

print(f"\n[5/6] Residual analysis...")

t_stat, p_value = ttest_1samp(delta_geo, 0.0)
print(f"\n  t-test (H₀: μ_Δ = 0):")
print(f"  t = {t_stat:.4f}, p = {p_value:.4f}")
if p_value > 0.05:
    print(f"  ✅ H₀ not rejected: residuals consistent with zero mean")
else:
    print(f"  ⚠️  H₀ rejected: significant residual bias (p = {p_value:.4f})")

sigma_delta = np.std(delta_geo)
frac_1sig = np.mean(np.abs(delta_geo) < sigma_delta)
frac_2sig = np.mean(np.abs(delta_geo) < 2*sigma_delta)
print(f"\n  Fraction within ±1σ: {frac_1sig:.3f} ({100*frac_1sig:.1f}%)")
print(f"  Fraction within ±2σ: {frac_2sig:.3f} ({100*frac_2sig:.1f}%)")
print(f"  (Gaussian expectation: 68.3% / 95.4%)")

r_zl, p_zl = pearsonr(delta_geo, zl)
r_sig, p_sig = pearsonr(delta_geo, sigap)
r_fE, p_fE = pearsonr(delta_geo, fE_obs)
print(f"\n  Correlation Δ vs z_l:    r = {r_zl:+.4f} (p = {p_zl:.4f})")
print(f"  Correlation Δ vs σ_ap:   r = {r_sig:+.4f} (p = {p_sig:.4f})")
print(f"  Correlation Δ vs f_E:    r = {r_fE:+.4f} (p = {p_fE:.4f})")

# ================================================================
# ANALYSIS BY DEFORMATION REGIME
# ================================================================

print(f"\n[6/6] Analysis by deformation regime...")

f_comps = factor_compresion_geometrico_vectorizado(sigap, zl, params_opt)
sigma_umbrales = sigma_umbral_efectivo(zl, params_opt[2])

regimes = []
for s, su in zip(sigap, sigma_umbrales):
    if s < su - params_opt[3]:
        regimes.append('elastic')
    elif s > su + params_opt[3]:
        regimes.append('plastic')
    else:
        regimes.append('transition')
regimes = np.array(regimes)

print(f"\n  Regime distribution (σ₀ = {params_opt[2]:.1f} km/s):")
for reg in ['elastic', 'transition', 'plastic']:
    mask = regimes == reg
    n_reg = np.sum(mask)
    if n_reg > 0:
        mae_reg = np.mean(np.abs(delta_geo[mask]))
        fcomp_med = np.median(f_comps[mask])
        sig_med = np.median(sigap[mask])
        print(f"    {reg:>12s}: {n_reg:>3d} lenses ({100*n_reg/len(regimes):.1f}%)")
        print(f"               |Δ| = {mae_reg:.4f}, median f_comp = {fcomp_med:.4f}, median σ = {sig_med:.0f} km/s")

# ================================================================
# BOOTSTRAP (FAST WITH LOCAL NM)
# ================================================================

print(f"\n{'='*70}")
print("BOOTSTRAP (50 resamples, local optimization)")
print(f"{'='*70}")

n_boot = 50  # Reduced for speed
np.random.seed(12345)
n_lenses = len(zl)

params_boot = []
mae_boot = []

t_boot_start = time.time()
for i in range(n_boot):
    idx = np.random.choice(n_lenses, size=n_lenses, replace=True)
    result_boot = minimize(
        funcion_costo_simple,
        params_opt,  # Starts from optimum
        args=(zl[idx], zs[idx], sigap[idx], fE_obs[idx]),
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    params_boot.append(result_boot.x)
    mae_boot.append(result_boot.fun)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{n_boot}...")

t_boot = time.time() - t_boot_start
print(f"  Bootstrap completed in {t_boot:.1f}s")

params_boot = np.array(params_boot)
mae_boot = np.array(mae_boot)

print(f"\n  Bootstrap uncertainties (68% CI):")
param_names = ['strength_max', 'f_comp_min', 'sigma_0 [km/s]', 'sharpness [km/s]']
for j, name in enumerate(param_names):
    med = np.median(params_boot[:, j])
    low = np.percentile(params_boot[:, j], 16)
    high = np.percentile(params_boot[:, j], 84)
    print(f"    {name:>20s} = {med:.4f}  [+{high-med:.4f} / -{med-low:.4f}]")

# ================================================================
# COMPATIBILITY WITH DESI Q1
# ================================================================

print(f"\n{'='*70}")
print("COMPATIBILITY TEST DESI Q1 vs STRONG LENSES")
print(f"{'='*70}")

sigma0_boot = params_boot[:, 2]
sigma0_desi = DESI_Q1['sigma_0']
diff_sigma = sigma0_boot - sigma0_desi

# Normalized difference (how many sigma apart)
sigma_boot_std = np.std(sigma0_boot)
n_sigma_diff = abs(np.median(sigma0_boot) - sigma0_desi) / sigma_boot_std if sigma_boot_std > 0 else 0

print(f"\n  σ₀(DESI Q1) = {sigma0_desi:.1f} km/s (field, 133,963 galaxies)")
print(f"  σ₀(lenses)  = {np.median(sigma0_boot):.1f} ± {sigma_boot_std:.1f} km/s")
print(f"  Difference   = {np.median(sigma0_boot) - sigma0_desi:.1f} km/s ({n_sigma_diff:.1f}σ)")
print(f"  95% CI bootstrap: [{np.percentile(sigma0_boot, 2.5):.1f}, {np.percentile(sigma0_boot, 97.5):.1f}] km/s")

if n_sigma_diff < 2:
    print(f"  ✅ σ₀ is compatible between DESI Q1 and strong lenses (< 2σ)")
else:
    print(f"  ⚠️  σ₀ differs between the two samples (≥ 2σ)")
    print(f"      DESI Q1 calibrates in the field (low density)")
    print(f"      Strong lenses are systems in denser environments")
    print(f"      → Consistent with environmental phase transition (Sec. 7)")

# ================================================================
# FINAL SUMMARY
# ================================================================

t_total = time.time() - t_start

print(f"\n{'='*70}")
print("SUMMARY FOR PUBLICATION")
print(f"{'='*70}")

print(f"""
  SAMPLE: {len(data_corte)} strong lenses with σ_ap > 200 km/s
          Chen+ (2019) catalog: SLACS, BELLS, S4TM, SL2S

  GEOMETRIC MODEL (γ = 1/3 frozen):
  ┌──────────────────────────────────────────────────────┐
  │ strength_max = {params_opt[0]:.2f} ± {np.std(params_boot[:,0]):.2f}                          │
  │ f_comp_min   = {params_opt[1]:.4f} ± {np.std(params_boot[:,1]):.4f}                      │
  │ σ₀           = {params_opt[2]:.1f} ± {np.std(params_boot[:,2]):.1f} km/s                  │
  │ sharpness    = {params_opt[3]:.1f} ± {np.std(params_boot[:,3]):.1f} km/s                    │
  └──────────────────────────────────────────────────────┘

  GOODNESS OF FIT:
  ┌──────────────────────────────────────────────────────┐
  │ Mean |Δ|    = {mae_geo:.4f} ± {np.std(mae_boot):.4f}                         │
  │ Median Δ    = {np.median(delta_geo):+.4f}                              │
  │ Improvement = {improvement_pct:.1f}% over original VPM                    │
  │ μ_Δ = 0?    p = {p_value:.4f}                                    │
  │ ±1σ / ±2σ   = {frac_1sig:.1%} / {frac_2sig:.1%}                          │
  └──────────────────────────────────────────────────────┘

  CONSISTENCY WITH DESI Q1:
  ┌──────────────────────────────────────────────────────┐
  │ σ₀(DESI Q1)  = {DESI_Q1['sigma_0']:.1f} km/s                        │
  │ σ₀(lenses)   = {np.median(sigma0_boot):.1f} ± {sigma_boot_std:.1f} km/s                  │
  │ Difference   = {n_sigma_diff:.1f}σ                               │
  │ Compatible?   {'YES (< 2σ)' if n_sigma_diff < 2 else 'NO (≥ 2σ, see text)'}     │
  └──────────────────────────────────────────────────────┘

  TOTAL TIME: {t_total:.1f}s (DE: {t_de:.1f}s, NM: {t_nm:.1f}s, Bootstrap: {t_boot:.1f}s)
""")

# Save results
resultados = {
    'n_lentes': len(data_corte),
    'corte_sigma': 200,
    'params_opt': params_opt.tolist(),
    'params_boot_median': np.median(params_boot, axis=0).tolist(),
    'params_boot_std': np.std(params_boot, axis=0).tolist(),
    'mae_orig': float(mae_orig),
    'mae_desi': float(mae_desi),
    'mae_geo': float(mae_geo),
    'mejora_pct': float(improvement_pct),
    'p_value_ttest': float(p_value),
    'frac_1sig': float(frac_1sig),
    'frac_2sig': float(frac_2sig),
    'n_sigma_diff_desi': float(n_sigma_diff),
    'tiempo_total': float(t_total)
}

with open('resultados_lentes_v6.json', 'w') as f:
    json.dump(resultados, f, indent=2)

print(f"\n  💾 Results saved to resultados_lentes_v6.json")
print(f"\n{'='*70}")
print(f"COMPLETE ANALYSIS — {len(data_corte)} lenses — CUT σ > 200 km/s")
print(f"gamma = 1/3 FROZEN — VERSION 6 (VECTORIZED)")
print(f"{'='*70}")