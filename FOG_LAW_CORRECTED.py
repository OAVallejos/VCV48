"""                        FOG_LAW_CORRECTED.py
Corrected derivation of the FOG Law using the bin closest to the theoretical value.
SDSS VERSION: Direct reading from sdss_vdisp_calidad.npz
"""                                                   
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import json
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

A0_TEO = 14.075  # Mpc - fundamental scale of VCV48
H0 = 70.0  # km/s/Mpc
C_LIGHT = 299792.458  # km/s

# SDSS Configuration
VDISP_CUTS = {
    'AM-Full': 262.0,      # high mass (like in DESI)
    'AM-A': 350.0,         # strict high mass
    'AM-B': 262.0          # reference
}

Z_MIN = 0.03
Z_MAX = 0.10  # local SDSS to avoid evolution

# Estimator configuration
R_BINS = np.arange(0, 210, 2.0)  # 2 Mpc bins up to 210 Mpc
N_RANDOMS_FACTOR = 1.5

# ==============================================================================
# SDSS DATA LOADING
# ==============================================================================

def load_sdss(filepath='data/sdss_vdisp_calidad.npz', vdisp_min=262.0, z_min=Z_MIN, z_max=Z_MAX):
    """Loads the SDSS catalog from the .npz file."""
    print(f"📂 Loading SDSS from: {filepath}")
    
    try:
        data = np.load(filepath)
        
        # Check keys
        keys = list(data.keys())
        print(f"   Keys in file: {keys}")
        
        # Extract data (standard format)
        ra = data.get('RA', data.get('ra', None))
        dec = data.get('DEC', data.get('dec', None))
        z = data.get('Z', data.get('z', None))
        vdisp = data.get('VDISP', data.get('vdisp', None))
        
        if ra is None or dec is None or z is None or vdisp is None:
            raise KeyError(f"Expected keys not found. Keys: {keys}")
        
        # Apply filters
        mask = (vdisp >= vdisp_min) & (z >= z_min) & (z < z_max)
        
        ra_filt = ra[mask]
        dec_filt = dec[mask]
        z_filt = z[mask]
        vdisp_filt = vdisp[mask]
        
        n_gal = len(ra_filt)
        print(f"\n✅ SDSS loaded:")
        print(f"   • VDISP ≥ {vdisp_min} km/s: {n_gal:,} galaxies")
        print(f"   • Redshift range: {z_filt.min():.4f} - {z_filt.max():.4f}")
        print(f"   • Mean VDISP: {vdisp_filt.mean():.1f} ± {vdisp_filt.std():.1f} km/s")
        
        return ra_filt, dec_filt, z_filt, vdisp_filt
        
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {filepath}")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None, None, None, None

# ==============================================================================
# CONVERSION TO CARTESIAN COORDINATES
# ==============================================================================

def ra_dec_z_to_xyz(ra, dec, z):
    """Converts spherical coordinates to Cartesian (Mpc)."""
    # Approximate comoving distance for small z (local Hubble)
    d = (C_LIGHT / H0) * z  # Mpc
    
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    x = d * np.cos(dec_rad) * np.cos(ra_rad)
    y = d * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d * np.sin(dec_rad)
    
    return np.column_stack([x, y, z_cart])

def generate_randoms(ra, dec, z, factor=1.5):
    """Generates random points within the same volume."""
    n_data = len(ra)
    n_randoms = int(n_data * factor)
    
    # RA uniform in [0, 360)
    ra_rand = np.random.uniform(0, 360, n_randoms)
    
    # DEC uniform in area (uniform sine)
    sin_dec_rand = np.random.uniform(-1, 1, n_randoms)
    dec_rand = np.arcsin(sin_dec_rand) * 180 / np.pi
    
    # Z uniform in the observed range
    z_min, z_max = z.min(), z.max()
    z_rand = np.random.uniform(z_min, z_max, n_randoms)
    
    return ra_rand, dec_rand, z_rand

# ==============================================================================
# CORRELATION FUNCTION - LANDAY-SZALAY ESTIMATOR
# ==============================================================================

def count_pairs_3d(points, bins):
    """Counts 3D pairs using cKDTree."""
    tree = cKDTree(points)
    counts = np.zeros(len(bins) - 1)
    r_max = bins[-1]
    
    for i, point in enumerate(points):
        indices = tree.query_ball_point(point, r_max, return_length=False)
        neighbors = [idx for idx in indices if idx != i]
        if neighbors:
            dists = np.linalg.norm(points[neighbors] - point, axis=1)
            hist, _ = np.histogram(dists, bins=bins)
            counts += hist
    
    return counts

def compute_xi(ra, dec, z, bins=R_BINS, n_randoms_factor=N_RANDOMS_FACTOR):
    """Calculates the correlation function ξ(r) using Landy-Szalay."""
    print("   Calculating correlation function...")
    
    # Convert to Cartesian
    points_data = ra_dec_z_to_xyz(ra, dec, z)
    
    # Generate randoms
    ra_rand, dec_rand, z_rand = generate_randoms(ra, dec, z, n_randoms_factor)
    points_rand = ra_dec_z_to_xyz(ra_rand, dec_rand, z_rand)
    
    n_data = len(points_data)
    n_rand = len(points_rand)
    
    print(f"   N_data = {n_data:,}, N_rand = {n_rand:,}")
    
    # Count pairs
    print("   Counting DD...")
    DD = count_pairs_3d(points_data, bins)
    
    print("   Counting RR...")
    RR = count_pairs_3d(points_rand, bins)
    
    print("   Counting DR...")
    DR = np.zeros(len(bins)-1)
    tree_rand = cKDTree(points_rand)
    for point in points_data:
        indices = tree_rand.query_ball_point(point, bins[-1], return_length=False)
        if indices:
            dists = np.linalg.norm(points_rand[indices] - point, axis=1)
            hist, _ = np.histogram(dists, bins=bins)
            DR += hist
    
    # Normalize
    nrr = n_rand * (n_rand - 1) / 2
    ndd = n_data * (n_data - 1) / 2
    ndr = n_data * n_rand
    
    DD_norm = DD / ndd
    RR_norm = RR / nrr
    DR_norm = DR / ndr
    
    # Landy-Szalay
    xi = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
    
    # Bin centers
    r_centers = (bins[:-1] + bins[1:]) / 2
    
    return r_centers, xi

# ==============================================================================
# EXTRACTION OF VALUES AT THEORETICAL HARMONICS
# ==============================================================================

def get_value_at_nearest_bin(r, xi, r_teo):
    """Gets the value of xi at the bin closest to the theoretical position."""
    r_obs = []
    xi_obs = []
    for rt in r_teo:
        idx = np.argmin(np.abs(r - rt))
        r_obs.append(r[idx])
        xi_obs.append(xi[idx])
    return np.array(r_obs), np.array(xi_obs)

# ==============================================================================
# PROCESS A SAMPLE
# ==============================================================================

def process_sample(sample_name, vdisp_min, ra, dec, z, vdisp):
    """Processes a complete sample: computes ξ and extracts harmonics."""
    print(f"\n{'='*60}")
    print(f"PROCESSING: {sample_name} (VDISP ≥ {vdisp_min} km/s)")
    print(f"{'='*60}")
    
    # Filter by vdisp
    mask = vdisp >= vdisp_min
    ra_f = ra[mask]
    dec_f = dec[mask]
    z_f = z[mask]
    vdisp_f = vdisp[mask]
    
    print(f"Galaxies in sample: {len(ra_f):,}")
    
    if len(ra_f) < 100:
        print(f"⚠️ Sample too small (n={len(ra_f)}), skipping...")
        return None
    
    # Compute ξ(r)
    r_centers, xi = compute_xi(ra_f, dec_f, z_f)
    
    # Extract values at harmonics
    n_array = np.arange(1, 15)
    r_teo = n_array * A0_TEO
    r_obs, xi_obs = get_value_at_nearest_bin(r_centers, xi, r_teo)
    delta_r = r_obs - r_teo
    
    return {
        'n': n_array,
        'r_teo': r_teo,
        'r_obs': r_obs,
        'delta_r': delta_r,
        'xi': xi_obs,
        'vdisp_mean': vdisp_f.mean(),
        'vdisp_std': vdisp_f.std(),
        'n_galaxies': len(ra_f)
    }

# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

def main():
    print("=" * 80)
    print("CORRECTED DERIVATION OF THE FOG LAW - SDSS VERSION")
    print("Direct reading from sdss_vdisp_calidad.npz")
    print("=" * 80)
    
    # 1. Load SDSS base data (without vdisp filters)
    ra, dec, z, vdisp = load_sdss(
        filepath='data/sdss_vdisp_calidad.npz',
        vdisp_min=0,  # no initial filter
        z_min=Z_MIN,
        z_max=Z_MAX
    )
    
    if ra is None:
        return
    
    # 2. Process each sample
    results = {}
    
    for name, vdisp_min in VDISP_CUTS.items():
        res = process_sample(name, vdisp_min, ra, dec, z, vdisp)
        if res is not None:
            results[name] = res
    
    # 3. Display results
    print("\n" + "=" * 80)
    print("OBSERVED VALUES AT VCV48 HARMONICS (SDSS)")
    print("=" * 80)
    
    for name, res in results.items():
        print(f"\n{name} (n={res['n_galaxies']:,}, ⟨vdisp⟩={res['vdisp_mean']:.1f} km/s):")
        print(f"{'n':<4} {'r_teo':<12} {'r_obs':<12} {'Δr':<12} {'ξ':<10}")
        print("-" * 55)
        for i in range(len(res['n'])):
            print(f"{res['n'][i]:<4} {res['r_teo'][i]:<12.3f} {res['r_obs'][i]:<12.3f} "
                  f"{res['delta_r'][i]:<12.3f} {res['xi'][i]:<10.6f}")
    
    # 4. Derive FOG law using AM-Full (or the one with most galaxies)
    main_sample = 'AM-Full' if 'AM-Full' in results else list(results.keys())[0]
    res_main = results[main_sample]
    
    print("\n" + "=" * 80)
    print(f"REAL FOG LAW DERIVATION (using {main_sample})")
    print("=" * 80)
    
    n_array = res_main['n']
    r_obs_main = res_main['r_obs']
    delta_r_main = res_main['delta_r']
    
    # Regression of r_obs vs n
    m, c, r_value, p_value, std_err = stats.linregress(n_array, r_obs_main)
    
    print(f"\nRegression r_obs(n) = m * n + c:")
    print(f"   m = {m:.6f} Mpc/harmonic")
    print(f"   c = {c:.6f} Mpc")
    print(f"   R2 = {r_value**2:.10f}")
    
    # Regression of delta_r vs n
    m_dr, c_dr, r_dr, _, _ = stats.linregress(n_array, delta_r_main)
    
    print(f"\nRegression Δr(n) = m_dr * n + c_dr:")
    print(f"   m_dr = {m_dr:.6f} Mpc/harmonic")
    print(f"   c_dr = {c_dr:.6f} Mpc")
    print(f"   R2 = {r_dr**2:.10f}")
    
    print(f"\nOBSERVED LAW (SDSS):")
    print(f"   r_obs(n) = {m:.3f} * n + {c:.3f}")
    print(f"   Δr(n) = {m_dr:.3f} * n + {c_dr:.3f}")
    
    # 5. Compression analysis
    print("\n" + "=" * 80)
    print("COMPRESSION/EXPANSION ANALYSIS")
    print("=" * 80)
    
    compression = A0_TEO - m
    print(f"\n   Theoretical a0     = {A0_TEO:.3f} Mpc")
    print(f"   Slope m    = {m:.3f} Mpc")
    print(f"   Δa = a0 - m    = {compression:.3f} Mpc")
    
    if compression > 0:
        print(f"   Interpretation: FOG COMPRESSION of {compression:.3f} Mpc/harmonic")
    else:
        print(f"   Interpretation: Apparent EXPANSION of {-compression:.3f} Mpc/harmonic")
    
    # 6. Cosmological milestones
    print("\n" + "=" * 80)
    print("COSMOLOGICAL MILESTONE VALIDATION")
    print("=" * 80)
    
    idx_bao = 9  # n=10
    idx_14 = 13  # n=14
    
    print(f"\nBAO (n=10):")
    print(f"   VCV48 Theoretical: {10 * A0_TEO:.3f} Mpc")
    print(f"   SDSS Observed: {r_obs_main[idx_bao]:.3f} Mpc")
    print(f"   Δr: {delta_r_main[idx_bao]:.3f} Mpc")
    
    print(f"\nHarmonic 14 (n=14):")
    print(f"   VCV48 Theoretical: {14 * A0_TEO:.3f} Mpc")
    print(f"   SDSS Observed: {r_obs_main[idx_14]:.3f} Mpc")
    print(f"   Δr: {delta_r_main[idx_14]:.3f} Mpc")
    
    # 7. Save results
    print("\n" + "=" * 80)
    print("EXPORTING CORRECTED RESULTS")
    print("=" * 80)
    
    results_json = {
        "dataset": "SDSS",
        "z_range": [Z_MIN, Z_MAX],
        "observed_law": {
            "formula_r": f"r_obs(n) = {m:.6f} * n + {c:.6f}",
            "formula_dr": f"Δr(n) = {m_dr:.6f} * n + {c_dr:.6f}",
            "m": float(m),
            "c": float(c),
            "m_dr": float(m_dr),
            "c_dr": float(c_dr),
            "R2": float(r_value**2)
        },
        "effective_compression": {
            "theoretical_a0": A0_TEO,
            "slope_m": float(m),
            "delta_a": float(compression),
            "type": "compression" if compression > 0 else "expansion"
        },
        "milestones": {
            "bao_n10": {
                "r_teo": float(10 * A0_TEO),
                "r_obs": float(r_obs_main[idx_bao]),
                "delta_r": float(delta_r_main[idx_bao])
            },
            "harmonic_n14": {
                "r_teo": float(14 * A0_TEO),
                "r_obs": float(r_obs_main[idx_14]),
                "delta_r": float(delta_r_main[idx_14])
            }
        },
        "samples": {}
    }
    
    for name, res in results.items():
        results_json["samples"][name] = {
            "n_galaxies": int(res['n_galaxies']),
            "vdisp_mean": float(res['vdisp_mean']),
            "vdisp_std": float(res['vdisp_std']),
            "data": {
                f"n{i+1}": {
                    "r_teo": float(res['r_teo'][i]),
                    "r_obs": float(res['r_obs'][i]),
                    "delta_r": float(res['delta_r'][i]),
                    "xi": float(res['xi'][i])
                } for i in range(len(res['n']))
            }
        }
    
    with open('FOG_LAW_SDSS_RESULTS.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print("Results saved to 'FOG_LAW_SDSS_RESULTS.json'")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()