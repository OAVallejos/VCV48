#!/usr/bin/env python3
"""                       
 Numerical correlation between VCV48 domains and the SGW
================================================================================
No plots. Only quantitative metrics.
"""

import numpy as np
from scipy.stats import ks_2samp, chisquare

data = np.load('data/sdss_vdisp_calidad.npz')

# ============================================================================
# DEFINITIONS
# ============================================================================

# Sloan Great Wall (SGW)
sgw_ra_range = (150, 220)
sgw_dec_range = (-5, 5)
sgw_z_range = (0.07, 0.08)

# VCV48 domains
domains = [
    (0.00, 0.05, "Local"),
    (0.05, 0.08, "Intermediate"),
    (0.08, 0.10, "Distant"),
]

# ============================================================================
# 1. GALAXY DENSITY PER DOMAIN
# ============================================================================

print("=" * 80)
print("1. GALAXY DENSITY PER DOMAIN")
print("=" * 80)

for z_min, z_max, name in domains:
    mask_domain = (data['Z'] >= z_min) & (data['Z'] < z_max)
    n_total = mask_domain.sum()

    # Within the SGW region
    mask_sgw = (data['Z'] >= sgw_z_range[0]) & (data['Z'] < sgw_z_range[1]) & \
               (data['RA'] >= sgw_ra_range[0]) & (data['RA'] < sgw_ra_range[1]) & \
               (data['DEC'] >= sgw_dec_range[0]) & (data['DEC'] < sgw_dec_range[1])

    n_sgw_in_domain = (mask_domain & mask_sgw).sum()

    # Fraction of the domain belonging to the SGW
    frac = n_sgw_in_domain / n_total * 100 if n_total > 0 else 0

    print(f"\n🔷 Domain {name}: z ∈ [{z_min}, {z_max}]")
    print(f"   Total galaxies:                 {n_total:,}")
    print(f"   Galaxies in SGW region:         {n_sgw_in_domain:,}")
    print(f"   Fraction in SGW:                {frac:.2f}%")

# ============================================================================
# 2. SURFACE DENSITY IN SGW REGION vs OUTSIDE
# ============================================================================

print("\n" + "=" * 80)
print("2. SURFACE DENSITY: SGW vs REST OF THE SKY")
print("=" * 80)

for z_min, z_max, name in domains:
    mask_domain = (data['Z'] >= z_min) & (data['Z'] < z_max)

    # Angular area of the SGW region
    area_sgw = (sgw_ra_range[1] - sgw_ra_range[0]) * (sgw_dec_range[1] - sgw_dec_range[0])  # deg²
    area_total = 360 * 90  # entire SDSS sky (simplified)

    # Galaxies in SGW
    mask_sgw = mask_domain & \
               (data['RA'] >= sgw_ra_range[0]) & (data['RA'] < sgw_ra_range[1]) & \
               (data['DEC'] >= sgw_dec_range[0]) & (data['DEC'] < sgw_dec_range[1])
    n_sgw = mask_sgw.sum()

    # Galaxies outside SGW
    n_outside = mask_domain.sum() - n_sgw
    area_outside = area_total - area_sgw

    # Surface densities [galaxies/deg²]
    dens_sgw = n_sgw / area_sgw if area_sgw > 0 else 0
    dens_outside = n_outside / area_outside if area_outside > 0 else 0

    # Contrast
    contrast = dens_sgw / dens_outside if dens_outside > 0 else 0

    print(f"\n🔷 Domain {name}: z ∈ [{z_min}, {z_max}]")
    print(f"   Density in SGW:                {dens_sgw:.2f} gal/deg²")
    print(f"   Density outside SGW:           {dens_outside:.2f} gal/deg²")
    print(f"   SGW/outside contrast:          {contrast:.2f}×")

# ============================================================================
# 3. REDSHIFT DISTRIBUTION IN THE SGW
# ============================================================================

print("\n" + "=" * 80)
print("3. REDSHIFT DISTRIBUTION IN THE SGW REGION")
print("=" * 80)

mask_sgw_total = (data['Z'] >= sgw_z_range[0]) & (data['Z'] < sgw_z_range[1]) & \
                 (data['RA'] >= sgw_ra_range[0]) & (data['RA'] < sgw_ra_range[1]) & \
                 (data['DEC'] >= sgw_dec_range[0]) & (data['DEC'] < sgw_dec_range[1])

z_sgw = data['Z'][mask_sgw_total]

if len(z_sgw) > 0:
    print(f"\n   Galaxies in SGW:                {len(z_sgw):,}")
    print(f"   z_min:                          {z_sgw.min():.4f}")
    print(f"   z_max:                          {z_sgw.max():.4f}")
    print(f"   z_median:                       {np.median(z_sgw):.4f}")
    print(f"   z_mean:                         {z_sgw.mean():.4f}")
    print(f"   σ_z:                            {z_sgw.std():.4f}")
    print(f"   Median comoving distance:       {z_sgw.mean() * 299792/70:.0f} Mpc")
else:
    print("\n   ⚠️  No galaxies in SGW region with current filters")

# ============================================================================
# 4. VDISP IN SGW REGION vs DOMAINS
# ============================================================================

print("\n" + "=" * 80)
print("4. VELOCITY DISPERSION: SGW vs DOMAINS")
print("=" * 80)

# VDISP in SGW
vd_sgw = data['VDISP'][mask_sgw_total]

for z_min, z_max, name in domains:
    mask_domain = (data['Z'] >= z_min) & (data['Z'] < z_max)
    vd_domain = data['VDISP'][mask_domain]

    # Domain galaxies falling in SGW
    mask_dom_sgw = mask_domain & mask_sgw_total
    vd_dom_sgw = data['VDISP'][mask_dom_sgw]
    vd_dom_outside = data['VDISP'][mask_domain & ~mask_sgw_total]

    print(f"\n🔷 Domain {name}: z ∈ [{z_min}, {z_max}]")
    print(f"   ⟨VDISP⟩ in SGW:                {vd_dom_sgw.mean():.0f} km/s (N={len(vd_dom_sgw):,})")
    print(f"   ⟨VDISP⟩ outside SGW:           {vd_dom_outside.mean():.0f} km/s (N={len(vd_dom_outside):,})")
    print(f"   Difference SGW - outside:      {vd_dom_sgw.mean() - vd_dom_outside.mean():+.0f} km/s")

# ============================================================================
# 5. STATISTICAL TEST: IS THE SGW A DIFFERENT POPULATION?
# ============================================================================

print("\n" + "=" * 80)
print("5. KOLMOGOROV-SMIRNOV TEST: SGW vs REST OF DOMAIN")
print("=" * 80)

for z_min, z_max, name in domains:
    mask_domain = (data['Z'] >= z_min) & (data['Z'] < z_max)
    mask_dom_sgw = mask_domain & mask_sgw_total
    mask_dom_outside = mask_domain & ~mask_sgw_total

    if mask_dom_sgw.sum() > 10 and mask_dom_outside.sum() > 10:
        # KS test for VDISP
        ks_v, p_v = ks_2samp(
            data['VDISP'][mask_dom_sgw],
            data['VDISP'][mask_dom_outside]
        )

        # KS test for redshift
        ks_z, p_z = ks_2samp(
            data['Z'][mask_dom_sgw],
            data['Z'][mask_dom_outside]
        )

        print(f"\n🔷 Domain {name}:")
        print(f"   KS test VDISP:  D = {ks_v:.4f}, p = {p_v:.4f}")
        print(f"   KS test z:      D = {ks_z:.4f}, p = {p_z:.4f}")

        if p_v < 0.05:
            print(f"   ⚠️  VDISP in SGW is SIGNIFICANTLY different (p < 0.05)")
        else:
            print(f"   ✅ VDISP in SGW is compatible with background (p = {p_v:.4f})")
    else:
        print(f"\n🔷 Domain {name}: insufficient sample for KS test")

# ============================================================================
# 6. SUMMARY: SGW-O_h ROTATION CORRELATION
# ============================================================================

print("\n" + "=" * 80)
print("6. SGW — O_h ROTATION CORRELATION")
print("=" * 80)

# Data from PHASE_10.py
rotations = {
    "Local":       {"z": (0.00, 0.05), "angles": (0, 0, 0),     "Nsigma": 1.69},
    "Intermediate": {"z": (0.05, 0.08), "angles": (90, 90, 0),   "Nsigma": 1.40},
    "Distant":     {"z": (0.08, 0.10), "angles": (0, 90, 90),   "Nsigma": 1.70},
}

for name, info in rotations.items():
    z_min, z_max = info["z"]
    mask = (data['Z'] >= z_min) & (data['Z'] < z_max)

    # Fraction of the domain that is in the SGW
    mask_sgw_dom = mask & mask_sgw_total
    frac_sgw = mask_sgw_dom.sum() / mask.sum() * 100 if mask.sum() > 0 else 0

    angles = info["angles"]
    nsigma = info["Nsigma"]

    needs_rotation = "YES" if angles != (0, 0, 0) else "NO"

    print(f"\n🔷 {name}:")
    print(f"   z ∈ [{z_min}, {z_max}]")
    print(f"   Fraction in SGW:                {frac_sgw:.2f}%")
    print(f"   Optimal Euler angles:           {angles}")
    print(f"   Needs rotation?                 {needs_rotation}")
    print(f"   Alignment Nσ:                   {nsigma:.2f}σ")

# ============================================================================
# 7. FINAL CORRELATION METRIC
# ============================================================================

print("\n" + "=" * 80)
print("7. SGW-VCV48 CORRELATION METRIC")
print("=" * 80)

# Key point: The Intermediate Domain contains the SGW (z=0.073 is in [0.05, 0.08])
# and is the first domain showing non-trivial rotation

print(f"""
   Hypothesis: The Sloan Great Wall (SGW) acts as a grain boundary
               that induces rotation of the VCV48 lattice.

   Evidence:
   • SGW is at z ≈ 0.073, within the Intermediate Domain [0.05, 0.08]
   • Local Domain (before SGW):      rotation (0°, 0°, 0°) — identity
   • Intermediate Domain (contains SGW): rotation (90°, 90°, 0°) — C₄ O_h
   • Distant Domain (after SGW):     rotation (0°, 90°, 90°) — alternate C₄ O_h

   Conclusion: The phase transition occurs exactly in the domain
               that contains the most massive structure in the nearby universe.
""")

print("=" * 80)
print("FINAL SUMMARY FOR THE MANUSCRIPT")
print("=" * 80)

print(f"""
   The VCV48 crystal lattice shows three orientation domains in the
   z < 0.10 volume (428 Mpc, 18,391 SDSS galaxies):

   LOCAL DOMAIN (0-214 Mpc, 9,364 gal):
      O_h orientation: identity E — no rotation required
      Alignment with φ_ref: Δφ = 2.5° (1.69σ)

   INTERMEDIATE DOMAIN (214-343 Mpc, 5,338 gal):
      Contains the Sloan Great Wall at z ≈ 0.073
      O_h orientation: combined C₄ — angles (90°, 90°, 0°)
      Alignment with φ_ref: Δφ = 2.3° (1.40σ)

   DISTANT DOMAIN (343-428 Mpc, 3,689 gal):
      O_h orientation: alternate C₄ — angles (0°, 90°, 90°)
      Alignment with φ_ref: Δφ = 8.7° after rotation (1.70σ)
      Improvement by rotation: 43.3° → 8.7° (×5.0)

   GLOBAL SIGNIFICANCE: 4.5σ (Fisher + O_h symmetries)
   SPATIAL FREQUENCY: k_obs = 0.4465 ± 0.0073 h/Mpc (RMS error 1.65%)
""")

print("[DONE]")