#!/usr/bin/env python3
"""
Domain-Based Phase Alignment with O_h Rotation
========================================================================
Divides the volume into concentric shells and applies Euler rotation
to align the phase of each domain with φ_ref before stacking.
"""

import numpy as np
import vpm_core

# ============================================================================
# O_h ROTATION FUNCTIONS
# ============================================================================

def rotation_matrix_euler(alpha, beta, gamma):
    """Euler rotation matrix Z(alpha) * X(beta) * Z(gamma)"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    return np.array([
        [ca*cg - sa*cb*sg, -ca*sg - sa*cb*cg,  sa*sb],
        [sa*cg + ca*cb*sg, -sa*sg + ca*cb*cg, -ca*sb],
        [sb*sg,             sb*cg,              cb   ]
    ])

def spherical_to_cartesian(ra, dec):
    """Converts spherical coordinates (rad) to Cartesian"""
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

def cartesian_to_spherical(coords):
    """Converts Cartesian coordinates to spherical (rad)"""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ra = np.arctan2(y, x)
    ra = np.where(ra < 0, ra + 2*np.pi, ra)
    dec = np.arcsin(z)
    return ra, dec

def apply_euler_rotation(ra, dec, alpha, beta, gamma):
    """Applies Euler rotation to coordinates (ra, dec in degrees)"""
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    coords = spherical_to_cartesian(ra_rad, dec_rad)
    R = rotation_matrix_euler(alpha, beta, gamma)
    rotated = coords @ R.T
    
    ra_rot_rad, dec_rot_rad = cartesian_to_spherical(rotated)
    return np.degrees(ra_rot_rad), np.degrees(dec_rot_rad)

# ============================================================================
# OPTIMAL ORIENTATION SEARCH FOR A DOMAIN
# ============================================================================

def find_optimal_orientation(ra, dec, z, weights, engine_func, n_angles=6):
    """
    Finds the Euler angles that minimize Δφ with respect to φ_ref
    for a set of galaxies (a domain).
    """
    best_dphi = np.inf
    best_angles = (0.0, 0.0, 0.0)
    
    alphas = np.linspace(0, 2*np.pi, n_angles)
    betas = np.linspace(0, np.pi, max(3, n_angles//2))
    gammas = np.linspace(0, 2*np.pi, n_angles)
    
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                # Apply rotation
                ra_rot, dec_rot = apply_euler_rotation(ra, dec, alpha, beta, gamma)
                
                # Re-initialize engine with rotated data
                engine = vpm_core.VPMEngine()
                engine.inicializar(
                    ra_rot.tolist(),
                    dec_rot.tolist(),
                    z.tolist(),
                    weights.tolist()
                )
                
                # Alignment test
                k_obs, delta_phi, bg_mean, bg_std, n_sigma, p_align = \
                    engine.test_alineacion_fase(128, 80)
                
                if delta_phi < best_dphi:
                    best_dphi = delta_phi
                    best_angles = (alpha, beta, gamma)
    
    return best_angles, best_dphi

# ============================================================================
# DATA LOADING
# ============================================================================
print("[1/4] Loading SDSS...")
data = np.load('data/sdss_vdisp_calidad.npz')

# Concentric shells
shells = [
    (0.00, 0.05, "Local Domain (0-214 Mpc)"),
    (0.05, 0.08, "Intermediate Domain (214-343 Mpc)"),
    (0.08, 0.10, "Distant Domain (343-428 Mpc)"),
]

for z_min, z_max, name in shells:
    mask = (data['Z'] >= z_min) & (data['Z'] < z_max) & (data['VDISP'] > 262)
    ra = data['RA'][mask]
    dec = data['DEC'][mask]
    z = data['Z'][mask]
    vd = data['VDISP'][mask]
    n_gal = len(ra)
    
    if n_gal < 1000:
        print(f"\n  ⚠️  {name}: {n_gal:,} galaxies — insufficient sample")
        continue
    
    print(f"\n{'='*70}")
    print(f"🔷 {name}")
    print(f"   z ∈ [{z_min}, {z_max}], {n_gal:,} galaxies")
    print(f"   ⟨z⟩ = {z.mean():.4f}, ⟨VDISP⟩ = {vd.mean():.0f} km/s")
    print(f"{'='*70}")
    
    # Weights
    ratio = vd / 373.0
    excess = ratio**4 - 1.0
    sat = np.ones_like(excess)
    m1 = (excess > 0) & (excess < 0.5)
    m2 = excess >= 0.5
    sat[m1] = 1.0 - 0.1 * (excess[m1] / 0.5)**2
    sat[m2] = 0.9 / (1.0 + np.exp((excess[m2] - 0.5) / 0.3)) + 0.1
    weights = 1.0 + excess * sat
    
    # 1. Analysis WITHOUT rotation
    engine = vpm_core.VPMEngine()
    engine.inicializar(ra.tolist(), dec.tolist(), z.tolist(), weights.tolist())
    k_obs, dphi, bg_m, bg_s, n_sig, p_val = engine.test_alineacion_fase(128, 80)
    
    print(f"\n   📊 WITHOUT ROTATION:")
    print(f"      k_obs    = {k_obs:.4f} h/Mpc (error: {abs(k_obs - vpm_core.K_VCV)/vpm_core.K_VCV*100:.3f}%)")
    print(f"      Δφ       = {dphi:.4f} rad ({np.degrees(dphi):.1f}°)")
    print(f"      Nσ       = {n_sig:+.2f}σ")
    
    # 2. Find optimal orientation
    print(f"\n   🔍 Searching for optimal O_h orientation...")
    best_angles, best_dphi = find_optimal_orientation(
        ra, dec, z, weights, engine, n_angles=5
    )
    
    # 3. Analysis WITH optimal rotation
    alpha, beta, gamma = best_angles
    ra_rot, dec_rot = apply_euler_rotation(ra, dec, alpha, beta, gamma)
    
    engine_rot = vpm_core.VPMEngine()
    engine_rot.inicializar(ra_rot.tolist(), dec_rot.tolist(), z.tolist(), weights.tolist())
    k_obs_r, dphi_r, bg_mr, bg_sr, n_sig_r, p_val_r = engine_rot.test_alineacion_fase(128, 80)
    
    print(f"\n   📊 WITH OPTIMAL ROTATION:")
    print(f"      Angles   = ({np.degrees(alpha):.0f}°, {np.degrees(beta):.0f}°, {np.degrees(gamma):.0f}°)")
    print(f"      k_obs    = {k_obs_r:.4f} h/Mpc (error: {abs(k_obs_r - vpm_core.K_VCV)/vpm_core.K_VCV*100:.3f}%)")
    print(f"      Δφ       = {dphi_r:.4f} rad ({np.degrees(dphi_r):.1f}°)")
    print(f"      Nσ       = {n_sig_r:+.2f}σ")
    print(f"      Improvement = Δφ: {dphi:.4f} → {dphi_r:.4f} rad")

print(f"\n[DONE]")