#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMPLETE VCV48 TEST - VACUUM CRYSTALLOGRAPHY      PUBLIC VERSION - FOR PASADENA 248 (STRUCTURE AND MASSES)
Validates all model predictions: masses, constants, angles
"""

import vpm48_engine_optimized as vpm
import numpy as np
from datetime import datetime
import time
import json
import sys

# ============================================================================
# TERMINAL COLOR CONFIGURATION
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}📌 {text}{Colors.END}")

def print_result(label, value, expected=None, unit="", error=None):
    if expected is not None and error is not None:
        status = f"{Colors.GREEN}✓{Colors.END}" if error < 1.0 else f"{Colors.YELLOW}⚠{Colors.END}" if error < 5.0 else f"{Colors.RED}✗{Colors.END}"
        print(f"  {status} {label:25}: {value:>12.6f} {unit:<5} (expected: {expected:>12.6f}) error: {error:>6.3f}%")
    else:
        print(f"     {label:25}: {value:>12.6f} {unit}")

# ============================================================================
# TEST 1: FUNDAMENTAL PARTICLES
# ============================================================================

def test_fundamental_particles():
    """Verifies masses of all known particles"""

    print_header("🔬 TEST 1: FUNDAMENTAL PARTICLES")

    particles = [
        ("electron", "Electron", 1, 0.000511),
        ("muon", "Muon", 207, 0.10566),
        ("tau", "Tau", 3477, 1.77686),
        ("proton", "Proton", 1836, 0.93827),
        ("neutron", "Neutron", 1838, 0.93957),
        ("w", "W Boson", 15563000, 80.379),
        ("z", "Z Boson", 19996000, 91.1876),
        ("higgs", "Higgs Boson", 37445000, 125.18),
    ]

    results = []
    errors = []

    for cmd, name, nf, exp_mass in particles:
        try:
            res = vpm.analizar_particula(cmd)

            calc_mass = res['masa_gev']
            residue = res['residuo_48']
            stability = res['estabilidad_oh']
            gamma = res['gamma_factor']
            ptype = res['tipo']
            representation = res.get('representacion', 'N/A')

            error = abs(calc_mass - exp_mass) / exp_mass * 100.0

            print(f"\n{Colors.BOLD}{name}{Colors.END} (Nf={nf:,}, residue {residue})")
            print(f"  📊 Mass: {calc_mass:.6f} GeV (exp: {exp_mass:.6f})")
            print(f"  📊 Error: {error:.4f}%")
            print(f"  📊 Stability: {stability:.6f}")
            print(f"  📊 γ factor: {gamma:.4f}")
            print(f"  📊 Representation: {representation}")

            results.append({
                'name': name,
                'nf': nf,
                'residue': residue,
                'calc_mass': calc_mass,
                'exp_mass': exp_mass,
                'error': error,
                'stability': stability,
                'gamma': gamma
            })
            errors.append(error)

        except Exception as e:
            print_error(f"Error analyzing {name}: {e}")

    if errors:
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)

        print(f"\n{Colors.BOLD}📊 STATISTICS:{Colors.END}")
        print(f"  Average error: {avg_error:.4f}%")
        print(f"  Maximum error: {max_error:.4f}%")
        print(f"  Minimum error: {min_error:.4f}%")

    return results

# ============================================================================
# TEST 2: COUPLING CONSTANTS
# ============================================================================

def test_coupling_constants():
    """Verifies coupling constants from geometry"""

    print_header("🔬 TEST 2: COUPLING CONSTANTS FROM GEOMETRY")

    try:
        const = vpm.calcular_constantes_acoplamiento()

        print(f"\n{Colors.BOLD}Weinberg Angle:{Colors.END}")
        print_result("θ_W (rad)", const['theta_w_rad'], np.pi/8, "rad",
                    abs(const['theta_w_rad'] - np.pi/8)/np.pi*8*100)
        print_result("θ_W (degrees)", const['theta_w_deg'], 22.5, "°",
                    abs(const['theta_w_deg'] - 22.5))
        print_result("sin θ_W", const['sin_theta_w'], np.sin(np.pi/8), "",
                    abs(const['sin_theta_w'] - np.sin(np.pi/8))/np.sin(np.pi/8)*100)

        print(f"\n{Colors.BOLD}Coupling Constants:{Colors.END}")
        print_result("e (charge)", const['e_carga'], "", "")
        print_result("g (SU(2))", const['g_acoplamiento'], "", "")
        print_result("g' (U(1))", const['g_prime'], "", "")

        print(f"\n{Colors.BOLD}Fermi Constant:{Colors.END}")
        print_result("G_F (GeV⁻²)", const['g_fermi_gev2'], const['g_fermi_exp'], "",
                    const['error_gf_percent'])

        print(f"\n{Colors.BOLD}W Mass from Geometry:{Colors.END}")
        print_result("M_W (GeV)", const['m_w_geom_gev'], const['m_w_exp_gev'], "",
                    abs(const['m_w_geom_gev'] - const['m_w_exp_gev'])/const['m_w_exp_gev']*100)

        return const

    except Exception as e:
        print_error(f"Error calculating constants: {e}")
        return None

# ============================================================================
# TEST 3: CABIBBO ANGLE
# ============================================================================

def test_cabibbo_angle():
    """Verifies Cabibbo angle from Nf(d) and Nf(s) with Pasadena factor"""

    print_header("🔬 TEST 3: CABIBBO ANGLE (PASADENA FACTOR 50/48)")

    try:
        cb = vpm.calcular_angulo_cabibbo()

        print(f"\n{Colors.BOLD}Burgers Numbers:{Colors.END}")
        print_result("Nf(d)", cb['nf_d'], 9, "")
        print_result("Nf(s)", cb['nf_s'], 183, "")
        print_result("φ_pasadena", cb['phi_pasadena'], 50/48, "")

        print(f"\n{Colors.BOLD}Cabibbo Angle:{Colors.END}")
        print_result("tan θ_C", cb['tan_theta_c'], 0.22177, "",
                    abs(cb['tan_theta_c'] - 0.22177)/0.22177*100)
        print_result("θ_C (degrees)", cb['theta_c_deg'], 13.02, "°",
                    abs(cb['theta_c_deg'] - 13.02)/13.02*100)

        print(f"\n{Colors.BOLD}CKM Elements:{Colors.END}")
        print_result("V_ud", cb['v_ud'], 0.974, "")
        print_result("V_us", cb['v_us'], 0.225, "")

        print(f"\n{Colors.CYAN}📌 Note: {cb.get('nota', '')}{Colors.END}")

        return cb

    except Exception as e:
        print_error(f"Error calculating Cabibbo angle: {e}")
        return None

# ============================================================================
# TEST 4: FINE STRUCTURE CONSTANT
# ============================================================================

def test_geometric_alpha():
    """Verifies fine structure constant from geometry"""

    print_header("🔬 TEST 4: FINE STRUCTURE CONSTANT")

    try:
        alpha = vpm.calcular_alpha_geometrico()

        print(f"\n{Colors.BOLD}Fine Structure Constant:{Colors.END}")
        print_result("α⁻¹ geometric", alpha['alpha_inv_geom'], 137.036, "",
                    abs(alpha['alpha_inv_geom'] - 137.036)/137.036*100)
        print_result("α geometric", alpha['alpha_geom'], 1/137.036, "",
                    alpha['error_percent'])
        print_result("α experimental", alpha['alpha_exp'], "", "")

        print(f"\n{Colors.BOLD}Geometric Factors:{Colors.END}")
        print_result("48 × 2π", alpha['factor_48'] * alpha['factor_2pi'], 48*2*np.pi, "")

        return alpha

    except Exception as e:
        print_error(f"Error calculating α: {e}")
        return None

# ============================================================================
# TEST 5: RESIDUE TABLE
# ============================================================================

def test_residue_table():
    """Generates complete table of residues and representations"""

    print_header("🔬 TEST 5: VCV48 RESIDUE TABLE")

    residues = [
        (0, "E_g", "Graviton", 0),
        (1, "A₁g", "Electron/Photon", 0.000511),
        (2, "T₁u", "π⁻/π⁰", 0.13498),
        (8, "E_g", "W Boson / Higgs", 80.379),
        (12, "T₁u", "Proton", 0.93827),
        (13, "?", "ν_μ", 0),
        (14, "T₂g", "Neutron", 0.93957),
        (15, "G_g", "Muon", 0.10566),
        (16, "T₁u⊕A₁g", "Z Boson", 91.1876),
        (19, "?", "ν_τ", 0),
        (21, "G_g", "Tau", 1.77686),
        (24, "H_u", "Axion/Dark Matter", 1e-5),
        (28, "T₁u", "π⁺", 0.13957),
        (29, "?", "ν̄_τ", 0),
        (35, "?", "ν̄_μ", 0),
    ]

    print(f"\n{Colors.BOLD}{'RES':>4} | {'Representation':<15} | {'Particle':<25} | {'Mass (GeV)':>12}{Colors.END}")
    print("-" * 70)

    for res, rep, name, mass in sorted(residues):
        mass_str = f"{mass:.6f}" if mass < 1 else f"{mass:.3f}"
        print(f"{res:4} | {rep:<15} | {name:<25} | {mass_str:>12}")

    print("\n" + "="*70)
    print("Neutrino pattern:")
    print("  ν_μ (13) + ν̄_μ (35) = 48")
    print("  ν_τ (19) + ν̄_τ (29) = 48")
    print("  Annihilation returns to vacuum!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Runs all public tests"""

    start_total = time.time()

    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     VCV48 - VACUUM CRYSTALLOGRAPHY (PUBLIC VERSION)           ║")
    print("║               FOR PASADENA 248 - AAS JUNE 2025                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

    print(f"\n{Colors.CYAN}Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.CYAN}Engine: vpm48_engine_optimized{Colors.END}")

    results = {}

    # TEST 1: Fundamental particles
    results['particles'] = test_fundamental_particles()

    # TEST 2: Coupling constants
    results['constants'] = test_coupling_constants()

    # TEST 3: Cabibbo angle (with Pasadena factor)
    results['cabibbo'] = test_cabibbo_angle()

    # TEST 4: Fine structure constant
    results['alpha'] = test_geometric_alpha()

    # TEST 5: Residue table
    test_residue_table()

    elapsed = time.time() - start_total

    print_header("📊 FINAL VALIDATION SUMMARY")

    points = 0
    total = 0

    if results.get('particles'):
        errors = [p['error'] for p in results['particles'] if p['error'] < 5.0]
        points += len(errors)
        total += len(results['particles'])
        print(f"  {Colors.GREEN}✓{Colors.END} Particles: {len(errors)}/{len(results['particles'])} OK")

    if results.get('constants'):
        gf_error = results['constants'].get('error_gf_percent', 100)
        if gf_error < 5.0:
            points += 1
            print(f"  {Colors.GREEN}✓{Colors.END} Coupling constants: OK")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} G_F: pure theoretical error ({gf_error:.1f}%)")
        total += 1

    if results.get('cabibbo'):
        error_cab = results['cabibbo'].get('error_percent', 100)
        if error_cab < 1.0:
            points += 1
            print(f"  {Colors.GREEN}✓{Colors.END} Cabibbo angle: OK")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} Cabibbo angle: error >1%")
        total += 1

    if results.get('alpha'):
        if results['alpha'].get('error_percent', 100) < 1.0:
            points += 1
            print(f"  {Colors.GREEN}✓{Colors.END} Fine structure constant: OK")
        else:
            print(f"  {Colors.YELLOW}⚠{Colors.END} α: error >1%")
        total += 1

    percentage = (points / total) * 100 if total > 0 else 0

    print(f"\n{Colors.BOLD}Score: {points}/{total} ({percentage:.1f}%){Colors.END}")

    if percentage > 95:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 COMPLETE STRUCTURE VALIDATION!{Colors.END}")
        print(f"{Colors.GREEN}VCV48 is consistent in its geometry and masses.{Colors.END}")
    elif percentage > 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Partial validation - check structure.{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ Structural problems detected.{Colors.END}")

    print(f"\n{Colors.CYAN}Total time: {elapsed:.2f} seconds{Colors.END}")
    print(f"{Colors.CYAN}End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")

    with open('vcv48_public_validation.json', 'w') as f:
        serializable_results = {}
        for k, v in results.items():
            if k == 'particles' and v:
                serializable_results[k] = [
                    {kk: vv for kk, vv in p.items() if not callable(vv)}
                    for p in v
                ]
            elif v is not None and hasattr(v, 'items'):
                serializable_results[k] = {kk: vv for kk, vv in v.items()}
            else:
                serializable_results[k] = v
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"{Colors.GREEN}💾 Results saved in 'vcv48_public_validation.json'{Colors.END}")

if __name__ == "__main__":
    main()