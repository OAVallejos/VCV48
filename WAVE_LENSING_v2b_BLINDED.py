#!/usr/bin/env python3
"""
WAVE_LENSING_v2b_BLINDED.py — 3 DATASET VALIDATION WITH O_h COMPRESSION                    ==============================================================================            
VPM VALIDATION WITH 3 DATASETS + O_h PROFILE COMPRESSION   Complete parameter traceability

BLINDED VERSION (v5.0) — UNIFIED PARAMETERS:
  - Unified continuous sigmoid (no ad-hoc hybrid logic)
  - SAT_BOOST = 1.5 integrated into compression
  - Consistent regime classification (±2×sharpness)
  - Updated parameters from WAVE_LENSING_c.py (real calibration)
  - Corrected fundamental harmonic = 10×ω₀

Datasets: Euclid Q1 MER, DESI LRG, eROSITA
Test: M_lens/M_dyn with O_h profile compression
==============================================================================
"""

import numpy as np
import gzip
import sys
import os
from astropy.table import Table
from scipy.stats import linregress

sys.path.append('target/release')
from vpm_wave import VPMWaveEngine
engine = VPMWaveEngine()

DATA_DIR = 'data'

# ================================================================
# BLOCK A: PARAMETERS FROZEN BY O_h GEOMETRY (Rust kernel)
# ================================================================
GAMMA = 1.0/3.0       # O_h geometry: a(z) ∝ (1+z)^(-1) → σ_u ∝ (1+z)^(1/3)

# Rust kernel constants (frozen, 0 free parameters)
XI_0    = 0.084       # primordial vorticity coupling
Z_C     = 1.5         # vorticity decay scale
BETA_0  = 0.03        # universal kinematic dilation
THETA_D = 35.772      # Debye temperature (harmonic 10: ω₀ = 1.146 Gyr⁻¹)
T_CMB_0 = 2.725       # Planck 2018

# ================================================================
# BLOCK B: FITTED PARAMETERS — UNIFIED WITH WAVE_LENSING_c.py
# ================================================================
# Origin: optimization on 133,963 DESI Q1 galaxies (field)
# Method: Differential Evolution minimizing ρ(corrected_ratio, σ_ap)
# Script: WAVE_LENSING_c.py (vFinal, 2026-07-06)
# Bootstrap: 50 resamples, 95% CI reported
Q1_params = {
    'sigma_0':      175.5,     # elastic threshold at z=0 [km/s] — 95% CI: [140.6, 226.0]
    'strength_max': 3.0455,    # O_h lattice stiffness — 95% CI: [1.89, 4.01]
    'f_comp_min':   0.9779,    # maximum asymptotic compression (2.21%) — 95% CI: [0.92, 0.98]
    'sharpness':    40.2,      # transition width [km/s] — 95% CI: [29.0, 52.8]
    'sat_boost':    1.5        # calibrated saturation threshold (fixed by geometry)
}

print("=" * 70)
print("VPM VALIDATION — 3 DATASETS")
print("WITH O_h PROFILE COMPRESSION (unified Q1 field parameters)")
print("BLINDED VERSION v5.0 — UNIFIED PARAMETERS")
print("=" * 70)

# ================================================================
# BLOCK C: COMPRESSION FUNCTIONS — UNIFIED WITH JWST
# ================================================================

def sigma_umbral(z):
    """
    Evolutionary elastic threshold — O_h geometry.
    σ_u(z) = σ₀ · (1+z)^γ    with γ = 1/3

    Returns:
      σ_u(z) in km/s
    """
    return Q1_params['sigma_0'] * ((1.0 + z) ** GAMMA)


def factor_compresion(sigma_ap, z):
    """
    Profile compression due to vacuum stiffness — CONTINUOUS SIGMOID.

      - Elastic→plastic transition via logistic sigmoid
      - Effective width scales with sat_boost
      - No ad-hoc hybrid logic

    Args:
      sigma_ap: velocity dispersion [km/s] (None → 1.0)
      z: redshift

    Returns:
      f_comp ∈ [f_comp_min, 1.0]
    """
    if sigma_ap is None or sigma_ap <= 0:
        return 1.0

    sigma_u = sigma_umbral(z)

    # Effective width scales with sat_boost
    effective_sharpness = Q1_params['sharpness'] * Q1_params.get('sat_boost', 1.0)

    # Reduced variable for the sigmoid
    x = (sigma_ap - sigma_u) / effective_sharpness

    # Logistic sigmoid: 0 (elastic) → 1 (plastic)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    # Compression factor
    f_comp = 1.0 - (1.0 - Q1_params['f_comp_min']) * sigmoid

    return np.clip(f_comp, Q1_params['f_comp_min'], 1.0)


def mass_ratio_corregido(z, sigma_ap=None):
    """
    M_lens/M_dyn ratio with profile compression.

    Args:
      z: redshift
      sigma_ap: velocity dispersion [km/s] (None → only ξ)

    Returns:
      corrected_ratio = (1 + ξ(z)) / f_comp(σ, z)
    """
    xi = engine.xi_vpm(z)
    if sigma_ap is None:
        return 1.0 + xi
    return (1.0 + xi) / factor_compresion(sigma_ap, z)


def clasificar_regimen(sigma_ap, z):
    """
    Classifies the vacuum deformation regime.

    Args:
      sigma_ap: velocity dispersion [km/s]
      z: redshift

    Returns:
      'elastic' | 'transition' | 'plastic' | 'unknown'
    """
    if sigma_ap is None:
        return 'unknown'
    su = sigma_umbral(z)
    sh = Q1_params['sharpness']
    if sigma_ap < su - 2 * sh:
        return 'elastic'
    elif sigma_ap > su + 2 * sh:
        return 'plastic'
    else:
        return 'transition'


# ================================================================
# Display parameters
# ================================================================
print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ MODEL PARAMETERS — FULL TRACEABILITY (UNIFIED v5.0)                  │
├─────────────────────────────────────────────────────────────────────┤
│ FROZEN (O_h geometry + Rust kernel vpm_wave.rs):                     │
│   GAMMA   = 1/3        ← a(z) ∝ (1+z)^(-1) → σ_u ∝ (1+z)^(1/3)    │
│   XI_0    = {XI_0:.3f}       ← vorticity coupling                          │
│   Z_C     = {Z_C:.1f}         ← decay scale                                  │
│   BETA_0  = {BETA_0:.2f}       ← kinematic dilation                            │
│   THETA_D = {THETA_D:.2f} K  ← Debye temperature (harmonic 10: ω₀=1.146) │
│   T_CMB_0 = {T_CMB_0:.3f} K  ← Planck 2018                                  │
│                                                                     │
│ FITTED (WAVE_LENSING_c.py, DESI Q1, 133,963 galaxies):               │
│   sigma_0      = {Q1_params['sigma_0']:.1f} km/s  ← elastic threshold at z=0     │
│   strength_max = {Q1_params['strength_max']:.4f}      ← lattice stiffness            │
│   f_comp_min   = {Q1_params['f_comp_min']:.4f}      ← maximum compression ({100*(1-Q1_params['f_comp_min']):.1f}%) │
│   sharpness    = {Q1_params['sharpness']:.1f} km/s  ← transition width            │
│   sat_boost    = {Q1_params['sat_boost']:.1f}        ← saturation threshold         │
│                                                                     │
│ COMPRESSION FUNCTION: UNIFIED CONTINUOUS SIGMOID                     │
│   f_comp(σ,z) = 1 - (1-f_min) / [1 + exp(-(σ-σ_u)/(sh·sat_boost))]                │
│                                                                     │
│ FREE PARAMETERS: k = 0 (all transferred from Rust + DESI Q1)         │
└─────────────────────────────────────────────────────────────────────┘
""")

# ================================================================
# DATASET 1: EUCLID Q1
# ================================================================
print("=" * 70)
print("1. EUCLID Q1 — PHOTOMETRY AND SHEAR")
print("=" * 70)

mer_file = os.path.join(DATA_DIR, 'mer1.sam.gz')
data_euclid = []
try:
    with gzip.open(mer_file, 'rt') as f:
        for line in f:
            if len(line) < 100: continue
            try:
                data_euclid.append({
                    'id': line[0:19].strip(),
                    'ra': float(line[20:35]),
                    'dec': float(line[36:51]),
                    'vis_det': int(line[68]),
                    'flag_vis': int(line[854:856]) if line[854:856].strip() else 0,
                    'ppl': float(line[884:892]) if line[884:892].strip() else -1,
                    'pspur': float(line[895:906]) if line[895:906].strip() else -1,
                    'mag': float(line[907:916]) if line[907:916].strip() else 99,
                    'ell': float(line[971:982]) if line[971:982].strip() else -1,
                    'ebv': float(line[1004:1012]) if line[1004:1012].strip() else 0
                })
            except (ValueError, IndexError):
                continue
    print(f"  Objects read: {len(data_euclid)}")
except FileNotFoundError:
    print(f"  ❌ {mer_file} not found")
    data_euclid = []

if data_euclid:
    galaxias = [g for g in data_euclid
                if g['vis_det'] == 1 and g['pspur'] < 0.5
                and g['ppl'] < 0.5 and g['mag'] < 25 and g['ell'] > 0]
    print(f"  Reliable galaxies: {len(galaxias)}")
    if galaxias:
        ell_values = [g['ell'] for g in galaxias]
        mag_values = [g['mag'] for g in galaxias]
        print(f"  Mean ellipticity: {np.mean(ell_values):.4f}")
        print(f"  Mean mag: {np.mean(mag_values):.1f}")
        print(f"  Ellipticity range: [{np.min(ell_values):.4f}, {np.max(ell_values):.4f}]")
        print(f"  NOTE: No σ_ap → no profile compression for Euclid")
        print(f"        Base VPM applicable: M_lens/M_dyn(z_med) = {engine.mass_ratio_vpm(0.8):.4f}")

# ================================================================
# DATASET 2: DESI LRG — WITH COMPRESSION
# ================================================================
print("\n" + "=" * 70)
print("2. DESI LRG — M_dyn WITH PROFILE COMPRESSION")
print("=" * 70)

lrg_file = os.path.join(DATA_DIR, 'DATASET_LRG_VDISP_FLUXR_FINAL.fits')
try:
    lrg = Table.read(lrg_file)
    mask = (lrg['VDISP'] > 50) & (lrg['VDISP'] < 500) & (lrg['Z'] > 0.05) & (lrg['Z'] < 0.5)
    lrg_good = lrg[mask]

    R_eff = 10.0  # kpc, typical LRG effective radius
    G = 4.302e-6  # (km/s)² kpc / M☉
    M_dyn = (R_eff * lrg_good['VDISP']**2) / G
    z_lrg = np.array(lrg_good['Z'], dtype=np.float64)
    vdisp_lrg = np.array(lrg_good['VDISP'], dtype=np.float64)

    # Ratios with unified function
    ratios_base = np.array([engine.mass_ratio_vpm(float(z)) for z in z_lrg])
    ratios_corr = np.array([mass_ratio_corregido(float(z), float(s)) for s, z in zip(vdisp_lrg, z_lrg)])
    f_comps = np.array([factor_compresion(float(s), float(z)) for s, z in zip(vdisp_lrg, z_lrg)])
    regimenes = np.array([clasificar_regimen(float(s), float(z)) for s, z in zip(vdisp_lrg, z_lrg)])

    print(f"  Galaxies: {len(lrg_good):,}")
    print(f"  Median z: {np.median(z_lrg):.3f}")
    print(f"  Median M_dyn: {np.median(M_dyn):.1e} M☉")
    print(f"  Median σ: {np.median(vdisp_lrg):.0f} km/s")
    print(f"  σ_threshold(z_med): {sigma_umbral(np.median(z_lrg)):.0f} km/s")
    print(f"  Base ratio (ξ only):          {np.median(ratios_base):.4f}")
    print(f"  Corrected ratio (with f_comp): {np.median(ratios_corr):.4f}")
    print(f"  Median f_comp:                 {np.median(f_comps):.4f}")

    # Regimes (consistent classification with JWST: ±2×sharpness)
    for reg in ['elastic', 'transition', 'plastic']:
        m = regimenes == reg
        if m.sum() > 0:
            print(f"\n  {reg} regime: {m.sum():,} galaxies ({100*m.sum()/len(regimenes):.0f}%)")
            print(f"    Median corrected ratio: {np.median(ratios_corr[m]):.4f}")
            print(f"    Median f_comp: {np.median(f_comps[m]):.4f}")
            print(f"    Median σ: {np.median(vdisp_lrg[m]):.0f} km/s")
            print(f"    Median z: {np.median(z_lrg[m]):.3f}")

    # Cuts by VDISP
    vdisp_low = np.percentile(vdisp_lrg, 33)
    vdisp_high = np.percentile(vdisp_lrg, 67)
    mask_low = vdisp_lrg <= vdisp_low
    mask_mid = (vdisp_lrg > vdisp_low) & (vdisp_lrg <= vdisp_high)
    mask_high = vdisp_lrg > vdisp_high

    print(f"\n  CUTS BY VDISP:")
    for label, mask_v in [('LOW MASS', mask_low), ('MEDIUM MASS', mask_mid), ('HIGH MASS', mask_high)]:
        r_base = np.median(ratios_base[mask_v])
        r_corr = np.median(ratios_corr[mask_v])
        fc = np.median(f_comps[mask_v])
        su = sigma_umbral(np.median(z_lrg[mask_v]))
        print(f"    {label}: base_ratio={r_base:.4f} | corr_ratio={r_corr:.4f} | f_comp={fc:.4f} | σ_u={su:.0f} km/s")

    # Slopes
    s_base, _, r2_base, _, _ = linregress(z_lrg, ratios_base)
    s_corr, _, r2_corr, _, _ = linregress(z_lrg, ratios_corr)
    print(f"\n  d(ratio)/dz slope:")
    print(f"    Base:      {s_base:+.4f}  R²={r2_base:.4f}")
    print(f"    Corrected: {s_corr:+.4f}  R²={r2_corr:.4f}")

    # Bootstrap for uncertainty
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_ratios = []
    for _ in range(n_boot):
        idx = rng.choice(len(ratios_corr), len(ratios_corr), replace=True)
        boot_ratios.append(np.median(ratios_corr[idx]))
    ratio_std = np.std(boot_ratios)
    print(f"    Corrected ratio: {np.median(ratios_corr):.4f} ± {ratio_std:.4f} (bootstrap)")

except FileNotFoundError:
    print(f"  ❌ {lrg_file} not found")

# ================================================================
# DATASET 3: eROSITA — M_hydro
# ================================================================
print("\n" + "=" * 70)
print("3. eROSITA — M_hydro")
print("=" * 70)

erosita_file = os.path.join(DATA_DIR, 'DL1_spec_SDSSV_eROSITA_eRASS1-v1_0_2.fits')
try:
    erosita = Table.read(erosita_file)
    z_ero = np.array(erosita['sdss_z'], dtype=np.float64)
    mask_ero = (z_ero > 0.05) & (z_ero < 1.0)
    z_ero_good = z_ero[mask_ero]

    # For clusters, σ ~ 500 km/s typical
    sigma_est_erosita = 500.0

    ratios_base = np.array([engine.mass_ratio_vpm(float(z)) for z in z_ero_good[:5000]])
    ratios_corr = np.array([mass_ratio_corregido(float(z), sigma_est_erosita) for z in z_ero_good[:5000]])
    f_comps = np.array([factor_compresion(sigma_est_erosita, float(z)) for z in z_ero_good[:5000]])

    z_med_ero = float(np.median(z_ero_good))
    su_med = sigma_umbral(z_med_ero)
    reg = clasificar_regimen(sigma_est_erosita, z_med_ero)

    print(f"  Total clusters: {len(erosita)}")
    print(f"  Clusters with z (0.05-1.0): {len(z_ero_good)}")
    print(f"  Median z: {z_med_ero:.3f}")
    print(f"  Estimated cluster σ: {sigma_est_erosita:.0f} km/s")
    print(f"  σ_threshold(z_med): {su_med:.0f} km/s")
    print(f"  σ_max(z_med) = σ_u × sat_boost: {su_med * Q1_params['sat_boost']:.0f} km/s")
    print(f"  Regime: {reg}")
    print(f"  Median base ratio:      {np.median(ratios_base):.4f}")
    print(f"  Median corrected ratio: {np.median(ratios_corr):.4f}")
    print(f"  Median f_comp: {np.median(f_comps):.4f}")

except FileNotFoundError:
    print(f"  ❌ {erosita_file} not found")
except Exception as e:
    print(f"  ⚠️  Error: {e}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("SUMMARY — 3 DATASETS WITH PROFILE COMPRESSION")
print("=" * 70)

# Collect values for the table (handle cases where there is no data)
try:
    euclid_n = len(data_euclid)
except:
    euclid_n = 0

try:
    desi_n = int(len(lrg_good))
    desi_base = f"{np.median(ratios_base):.4f}"
    desi_corr = f"{np.median(ratios_corr):.4f}"
except:
    desi_n = 0
    desi_base = "N/A"
    desi_corr = "N/A"

try:
    erosita_n = int(len(z_ero_good))
    erosita_base = f"{np.median(ratios_base):.4f}"
    erosita_corr = f"{np.median(ratios_corr):.4f}"
except:
    erosita_n = 0
    erosita_base = "N/A"
    erosita_corr = "N/A"

print(f"""
┌──────────────────────────┬───────────┬────────────────────────────────┐
│ Dataset                  │ Objects   │ Ratios M_lens/M_dyn             │
│                          │           │ base → corrected                │
├──────────────────────────┼───────────┼────────────────────────────────┤
│ Euclid Q1 MER            │ {euclid_n:>8}  │ Ellipticity (no σ_ap)            │
│ DESI LRG                 │ {desi_n:>8,}  │ {desi_base} → {desi_corr}               │
│ eROSITA (z<1)            │ {erosita_n:>8}  │ {erosita_base} → {erosita_corr}               │
└──────────────────────────┴───────────┴────────────────────────────────┘
""")

print(f"""
TESTABLE PREDICTIONS:
  1. Local BOOST (z < 1.0):    M_lens/M_dyn > 1  →  +3 to +7% (base)
                                With compression  →  +4 to +8% (corrected)
  2. High-z DEFICIT (z > 1.5): M_lens/M_dyn < 1  →  -2 to -4%
  3. TRANSITION:               z ≈ 1.19  (M_lens = M_dyn, Rust kernel)
  4. COMPRESSION:              Massive galaxies show higher ratio
                                due to f_comp < 1 (compressed profile)
  5. REGIME:                   σ_u(z) = {Q1_params['sigma_0']:.1f}·(1+z)^(1/3) km/s
                                σ_max(z) = σ_u(z) × {Q1_params['sat_boost']:.1f} (sat_boost)
                                Galaxies with σ > σ_u enter plastic regime.

METHODOLOGICAL NOTE:
  The factor_compresion() function uses a unified continuous sigmoid,
  identical to the JWST script. This ensures cross-consistency
  between the 3 local datasets and the 7 JWST fields.

  Free parameters for these datasets: k = 0
  (all transferred from Rust + DESI Q1 — WAVE_LENSING_c.py)
""")

print("=" * 70)
print("VALIDATION COMPLETE — 3 DATASETS — BLINDED SCRIPT v5.0")
print("UNIFIED PARAMETERS WITH WAVE_LENSING_c.py")
print("=" * 70)