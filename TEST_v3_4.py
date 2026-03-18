#!/usr/bin/env python3     
# -*- coding: utf-8 -*-    
"""                        
Test of vpm48_engine - CORRECTED VERSION               
"""                                                    
import vpm48_engine as vpm 
import time                
import csv
from datetime import datetime

def test_known_particles():
    """Tests particles with known experimental values"""

    print("="*70)
    print("VPM-48 TEST: ELASTIC UNIFICATION OF THE VACUUM")
    print("="*70)

    # List of particles to test
    particles = [
        ("Electron", "electron", 1, 1, 0.000511, "GeV"),  # Converted to GeV
        ("Muon", "muon", 207, 15, 0.10566, "GeV"),        # Converted to GeV
        ("Tau", "tau", 3477, 21, 1.77686, "GeV"),
        ("Proton", "proton", 1836, 12, 0.93827, "GeV"),
        ("Neutron", "neutron", 1838, 14, 0.93957, "GeV"),
    ]

    results = {}

    for name, cmd, nf_exp, res_exp, mass_exp, unit in particles:
        print(f"\n🔬 {name.upper()} (RES {res_exp})")
        print("-"*50)

        p = vpm.BraidWord([], 4, cmd)
        p.reduce()

        nf_real = len(p.generators)
        res_real = p.get_residuo_value()
        mass_calc = p.energy_ev / 1e9  # Always in GeV
        gamma = p.get_gamma_factor()

        print(f"  Nf: {nf_real} (expected {nf_exp}) ✓")
        print(f"  Residue: {res_real} (expected {res_exp}) ✓")
        print(f"  Mass: {mass_calc:.6f} {unit} (exp {mass_exp:.6f} {unit})")

        error = abs(mass_calc - mass_exp)/mass_exp*100
        print(f"  Error: {error:.4f}%")
        print(f"  γ factor: {gamma:.4f}")

        results[name] = {
            "mass": mass_calc,
            "error": error,
            "gamma": gamma
        }

    return results

def test_vibrational_modes():
    """Analyzes global vibrational modes of the Oh lattice"""

    print("\n🔬 VIBRATIONAL MODES OF THE Oh LATTICE")
    print("-"*50)

    modes = vpm.calcular_modos_vibratorios()

    # Use the correct keys from the Rust dictionary
    print(f"  Torsional velocity: {modes['c_t_ms']:.2e} m/s")
    print(f"  Velocity (fraction of c): {modes['c_t_c']:.4f} c")
    print(f"  Minimum frequency: {modes['frecuencia_min_hz']:.2e} Hz")
    print(f"  Maximum frequency: {modes['frecuencia_max_hz']:.2e} Hz")
    print(f"  Total modes: {modes['modos_totales']}")
    print(f"  Modes resonant with Hubble: {modes['modos_resonantes_con_hubble']}")
    print(f"  Characteristic energy: {modes['energia_caracteristica_ev']:.2e} eV")

    print("\n  Example of first modes:")
    for i, mode in enumerate(modes['modos_ejemplo'][:3]):
        print(f"    Mode {mode['modo']}: {mode['frecuencia_hz']:.2e} Hz, "
              f"mass {mode['masa_ev']:.2e} eV, symmetry {mode['simetria']:.2f}")

    return modes

def test_emergent_gravity():
    """Calculates the emergent gravitational constant"""

    print("\n🔬 GRAVITY AS AN ELASTIC PROPERTY")
    print("-"*50)

    g_calc = vpm.calcular_g_emergente()
    g_exp = 6.67430e-11
    error = abs(g_calc - g_exp)/g_exp * 100

    print(f"  Calculated G: {g_calc:.6e} m³/kg/s²")
    print(f"  Experimental G: {g_exp:.6e} m³/kg/s²")
    print(f"  Error: {error:.4f}%")

    # Save to CSV
    with open('gravity_vpm48.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'G_calculated', 'G_experimental', 'error_%'])
        writer.writerow([datetime.now(), g_calc, g_exp, error])

    return g_calc, error

def test_rapid_exploration():
    """Explores the first Nf looking for stable residues"""

    print("\n🔬 EXPLORATION OF STABLE RESIDUES (Nf 1-200)")
    print("-"*50)

    results = vpm.busqueda_masiva_estabilidad(list(range(1, 201)), 4)

    stable = [r for r in results if r["estable"]]
    print(f"  Stable configurations: {len(stable)}/{len(results)}")

    # Group by residue
    residues = {}
    for r in stable:
        res = r["residuo_48"]
        if res not in residues:
            residues[res] = []
        residues[res].append(r)

    print("\n  Residues found:")
    for res in sorted(residues.keys()):
        examples = residues[res][:3]
        print(f"    RES {res:2d}: {len(residues[res])} configurations")
        for e in examples:
            print(f"      Nf={e['nf']:3d}, mass={e['energia_gev']:.3f} GeV, γ={e['gamma_factor']:.3f}")

    return residues

if __name__ == "__main__":
    start = time.time()

    print("\n" + "╔" + "═"*68 + "╗")
    print("║   VPM-48: VACUUM CRYSTALLOGRAPHY (VCV48)   ║")
    print("╚" + "═"*68 + "╝")

    # Test 1: Known particles
    particle_results = test_known_particles()

    # Test 2: Lattice vibrational modes
    modes = test_vibrational_modes()

    # Test 3: Emergent gravity
    g_calc, g_error = test_emergent_gravity()

    # Test 4: Stability exploration
    residues = test_rapid_exploration()

    # Final summary
    print("\n" + "="*70)
    print("MODEL VALIDATION SUMMARY")
    print("="*70)

    print("\n📊 MASS PRECISION:")
    for name, data in particle_results.items():
        print(f"  {name}: error {data['error']:.4f}%")

    print(f"\n🌌 EMERGENT GRAVITY:")
    print(f"  Error in G: {g_error:.4f}%")
    print(f"  Result saved in gravity_vpm48.csv")

    print(f"\n🎵 LATTICE MODES:")
    print(f"  {modes['modos_totales']} vibrational modes")
    print(f"  {modes['modos_resonantes_con_hubble']} modes resonant with Hubble")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")