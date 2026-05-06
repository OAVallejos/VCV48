#!/usr/bin/env python3     
"""                       
Segments the DESI LRG dataset by VDISP and (Unified Field Theory).             
Uses the weighted correlation method.
"""

import numpy as np
from astropy.table import Table
import time
import vpm_core
import psutil
import os
from scipy.stats import gaussian_kde
import sys

# ============================================================================
# AUXILIARY FUNCTIONS
# ============================================================================

def print_memory_usage():
    """Prints current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"   💾 Memory: {mem:.1f} MB")

def generate_footprint_randoms(data_table, factor=3, cache_file=None):
    """
    Generates a random catalog that replicates the survey footprint
    and smooths N(z) with KDE.
    """
    n_ran = len(data_table) * factor
    print(f"   🛠️  Generating {n_ran:,} randoms ({factor}x data)...")
    start = time.time()

    # 1. Angular Footprint (resampling with replacement)
    print("   🗺️  [STEP 1/2] Replicating survey angular footprint (RA, DEC)...")
    indices_ran = np.random.choice(len(data_table), size=n_ran, replace=True)
    ra_ran = data_table['RA'][indices_ran]
    dec_ran = data_table['DEC'][indices_ran]

    # 2. Redshift (KDE to break 3D clustering)
    print("   🧠 [STEP 2/2] Generating independent Redshifts (Z) via KDE...")
    kde = gaussian_kde(data_table['Z'], bw_method=0.05)
    z_ran = kde.resample(n_ran)[0]
    z_ran = np.clip(z_ran, data_table['Z'].min(), data_table['Z'].max())

    randoms = Table([ra_ran, dec_ran, z_ran], names=('RA', 'DEC', 'Z'))

    if cache_file:
        randoms.write(cache_file, overwrite=True)
        print(f"   💾 Saved to cache: {cache_file}")

    elapsed = time.time() - start
    print(f"   ✅ Randoms generated in {elapsed:.2f} seconds")
    return randoms

def analyze_sample_v7(table, sample_name, engine, randoms_table, a0, fraction=0.20):
    """
    Executes the WEIGHTED correlation analysis for a specific sample
    using the VCV48 v7.0 engine.
    """
    print("\n" + "=" * 70)
    print(f"🎯 ANALYZING SAMPLE: {sample_name} with VCV48 v7.0 Kernel")
    print(f"   Total size: {len(table):,} galaxies")
    print("=" * 70)

    # 1. Data subsampling
    n_sample = int(len(table) * fraction)
    if n_sample == 0:
        print("   ⚠️  Empty sample after subsampling. Skipping.")
        return None
    idx_data = np.random.choice(len(table), n_sample, replace=False)
    table_sample = table[idx_data]

    # 2. Randoms subsampling (same fraction to maintain ratio)
    n_ran_sample = int(len(randoms_table) * fraction)
    idx_ran = np.random.choice(len(randoms_table), n_ran_sample, replace=False)
    randoms_sample = randoms_table[idx_ran]

    print(f"   📊 TEST WITH {fraction*100:.1f}% OF DATA")
    print(f"   Data: {n_sample:,} galaxies | Randoms: {n_ran_sample:,}")
    print(f"   Mean VDISP: {np.mean(table_sample['VDISP']):.1f} km/s")
    print(f"   Mean Redshift: {np.mean(table_sample['Z']):.3f}")

    # 3. Prepare data for Rust
    ra_data = table_sample['RA'].astype(np.float64).tolist()
    dec_data = table_sample['DEC'].astype(np.float64).tolist()
    z_data = table_sample['Z'].astype(np.float64).tolist()
    vdisp_data = table_sample['VDISP'].astype(np.float64).tolist()

    ra_ran = randoms_sample['RA'].astype(np.float64).tolist()
    dec_ran = randoms_sample['DEC'].astype(np.float64).tolist()
    z_ran = randoms_sample['Z'].astype(np.float64).tolist()

    # 4. CALL TO THE KERNEL!
    print("   🔬 Calling VPMEngine.weighted_correlation()...")
    start_total = time.time()

    # --- NEW CALL TO THE KERNEL METHOD ---
    centers, xi, mean_kappa, predicted_delta_ns = engine.weighted_correlation(
        ra_data, dec_data, z_data, vdisp_data,
        ra_ran, dec_ran, z_ran,
        r_min=0.0, r_max=200.0, n_bins=100
    )
    # -------------------------------------------

    elapsed_total = time.time() - start_total

    # 5. Results analysis
    centers_np = np.array(centers)
    xi_np = np.array(xi)

    # Value at a₀
    idx_a0 = np.argmin(np.abs(centers_np - a0))
    xi_a0_measured = xi_np[idx_a0]
    r_a0_real = centers_np[idx_a0]

    # Global maximum peak
    idx_max = np.argmax(xi_np)
    xi_max = xi_np[idx_max]
    r_max = centers_np[idx_max]

    print(f"\n   📈 RESULTS FOR {sample_name}:")
    print(f"      ⏱️  Total time: {elapsed_total:.2f} seconds")
    print(f"      🔍 ξ at r = {r_a0_real:.2f} Mpc (~a₀): {xi_a0_measured:.6f}")
    print(f"      📊 Mean κ = {mean_kappa:.6f}")
    print(f"      🎯 Predicted Δn_s = {predicted_delta_ns:.4f}")
    print(f"      📈 Global maximum at r = {r_max:.2f} Mpc: ξ = {xi_max:.6f}")

    # Save CSV
    output_file = f'correlation_V7_{sample_name.replace(" ", "_").replace("(", "").replace(")", "")}.csv'
    np.savetxt(output_file, np.column_stack([centers_np, xi_np]),
               header='r_Mpc,xi', delimiter=',', comments='')
    print(f"      💾 Results saved to '{output_file}'")

    return {
        'name': sample_name,
        'n_galaxies': n_sample,
        'n_randoms': n_ran_sample,
        'time': elapsed_total,
        'z_mean': np.mean(z_data),
        'vdisp_mean': np.mean(vdisp_data),
        'xi_a0_measured': xi_a0_measured,
        'r_a0_real': r_a0_real,
        'xi_max': xi_max,
        'r_max': r_max,
        'mean_kappa': mean_kappa,
        'predicted_delta_ns': predicted_delta_ns
    }

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    print("=" * 85)
    print("🚀 VCV48 v7.0 - SEGMENTED HIGH MASS ANALYSIS")
    print("   Unified Field Theory - Matter-Red Coupling Validation")
    print("=" * 85)
    print(f"⏱️  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_memory_usage()

    # -------------------------------------------------------------------------
    # 1. LOAD THE COMPLETE DATASET
    # -------------------------------------------------------------------------
    fits_file = 'data/DATASET_LRG_VDISP_FLUXR_FINAL.fits'
    print(f"\n📂 LOADING COMPLETE DATASET from {fits_file}...")
    try:
        table_full = Table.read(fits_file)
    except FileNotFoundError:
        print(f"❌ Error: File {fits_file} not found")
        print("   Make sure the path is correct and the file exists.")
        sys.exit(1)
    print(f"   ✅ Dataset loaded: {len(table_full):,} galaxies")

    # Filter by minimum VDISP to ensure measurement quality
    print("\n🔪 FILTERING BY VDISP QUALITY...")
    vdisp_min = 150.0
    quality_mask = table_full['VDISP'] > vdisp_min
    table_full = table_full[quality_mask]
    print(f"   Galaxies with VDISP > {vdisp_min} km/s: {len(table_full):,}")

    # -------------------------------------------------------------------------
    # 2. SEGMENT INTO HIGH MASS SUB-SAMPLES
    # -------------------------------------------------------------------------
    print("\n🔪 SEGMENTING BY HALO MASS (VDISP)...")
    # Define the HIGH MASS group limits
    vdisp_am_min = 262.2
    vdisp_am_max = 1000.0

    high_mass_mask = (table_full['VDISP'] >= vdisp_am_min) & (table_full['VDISP'] <= vdisp_am_max)
    table_am = table_full[high_mass_mask]
    print(f"   HIGH MASS group ({vdisp_am_min}-{vdisp_am_max} km/s): {len(table_am):,} galaxies")

    if len(table_am) == 0:
        print("❌ Error: No galaxies in High Mass range. Aborting.")
        sys.exit(1)

    # Calculate the median VDISP WITHIN the high mass group to split it
    median_vdisp_am = np.median(table_am['VDISP'])
    print(f"   📊 Median VDISP in High Mass: {median_vdisp_am:.1f} km/s")

    # Create the two sub-samples
    mask_am_b = table_am['VDISP'] < median_vdisp_am
    mask_am_a = table_am['VDISP'] >= median_vdisp_am

    table_am_low = table_am[mask_am_b]   # High Mass "Low"
    table_am_high = table_am[mask_am_a]  # High Mass "High"

    print(f"   ▶️ Sub-Sample AM-B (Low Mass): {len(table_am_low):,} galaxies")
    print(f"   ▶️ Sub-Sample AM-A (High Mass): {len(table_am_high):,} galaxies")

    # -------------------------------------------------------------------------
    # 3. INITIALIZE VCV48 v7.0 ENGINE (Rust)
    # -------------------------------------------------------------------------
    print("\n🔬 INITIALIZING VCV48 v7.0 ENGINE (Rust)...")
    try:
        engine = vpm_core.VPMEngine()
    except AttributeError:
        print("❌ Error: The 'vpm_core' module does not have the 'VPMEngine' class.")
        print("   Make sure you compiled vpm_core_v4.rs and that the .so/.pyd is in the path.")
        sys.exit(1)

    # Verify the consistency of the unified theory
    print("\n🔮 VERIFYING UNIFIED FIELD THEORY...")
    engine.check_consistency()  # This prints the table of derived constants

    a0 = engine.get_a0()
    print(f"\n📏 Model constants:")
    print(f"   a₀ = {a0:.3f} Mpc")
    print(f"   κ_base = {engine.get_kappa_base():.6f}")

    # -------------------------------------------------------------------------
    # 4. GENERATE OR LOAD RANDOM CATALOG
    # -------------------------------------------------------------------------
    randoms_file = 'data/LRG_RANDOMS_V7.fits'
    print(f"\n🎲 PREPARING RANDOM CATALOG...")

    if os.path.exists(randoms_file):
        try:
            randoms = Table.read(randoms_file)
            print(f"   ✅ Randoms loaded from cache: {len(randoms):,}")
        except:
            print("   ⚠️  Corrupt cache file. Regenerating...")
            randoms = generate_footprint_randoms(table_full, cache_file=randoms_file)
    else:
        randoms = generate_footprint_randoms(table_full, cache_file=randoms_file)

    # -------------------------------------------------------------------------
    # 5. EXECUTE ANALYSIS FOR EACH SUB-SAMPLE
    # -------------------------------------------------------------------------
    results = []
    analysis_fraction = 0.20  # Use 20% of each sub-sample for quick tests

    # 5.1 High Mass "Low" (AM-B)
    res_am_b = analyze_sample_v7(table_am_low, "AM-B (Low Mass)",
                                   engine, randoms, a0, analysis_fraction)
    if res_am_b:
        results.append(res_am_b)

    # 5.2 High Mass "High" (AM-A)
    res_am_a = analyze_sample_v7(table_am_high, "AM-A (High Mass)",
                                   engine, randoms, a0, analysis_fraction)
    if res_am_a:
        results.append(res_am_a)

    # 5.3 Complete High Mass sample for reference
    res_am_full = analyze_sample_v7(table_am, "AM-Full (Complete)",
                                      engine, randoms, a0, analysis_fraction)
    if res_am_full:
        results.append(res_am_full)

    # -------------------------------------------------------------------------
    # 6. FINAL SUMMARY AND COMPARISON
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("📊 FINAL SUMMARY: UNIFIED FIELD THEORY VALIDATION")
    print("=" * 100)

    if not results:
        print("❌ No results generated. Aborting.")
        sys.exit(1)

    print(f"{'Sample':<20} | {'ξ(a₀)':>12} | {'Mean κ':>12} | {'Predicted Δn_s':>14} | {'Δn_s sign':<12}")
    print("-" * 80)

    for r in results:
        # The sign of Δn_s should be more positive for AM-A than for AM-B
        sign = "+" if r['predicted_delta_ns'] > 0 else "-"
        print(f"{r['name']:<20} | {r['xi_a0_measured']:>12.6f} | {r['mean_kappa']:>12.6f} | {r['predicted_delta_ns']:>14.4f} | {sign:<12}")

    print("\n🎯 VCV48 v7.0 PREDICTION:")
    print("   • Elastic coupling κ must be positive (κ > 0).")
    print("   • Δn_s must be more positive (less negative) in AM-A than in AM-B.")
    print("   • This validates the doping and saturation model of vacuum rigidity.")

    # Direct comparison AM-A vs AM-B
    if len(results) >= 2:
        res_b = next((r for r in results if 'AM-B' in r['name']), None)
        res_a = next((r for r in results if 'AM-A' in r['name']), None)
        if res_b and res_a:
            print("\n📈 DIRECT COMPARISON AM-A vs AM-B:")
            print(f"   Δn_s(AM-A) - Δn_s(AM-B) = {res_a['predicted_delta_ns'] - res_b['predicted_delta_ns']:.4f}")
            print(f"   ξ(a₀)(AM-A) / ξ(a₀)(AM-B) = {res_a['xi_a0_measured'] / res_b['xi_a0_measured']:.2f}x")

    print("\n" + "=" * 100)
    print(f"✅ SEGMENTED ANALYSIS COMPLETED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_memory_usage()
    print("=" * 100)

if __name__ == "__main__":
    main()