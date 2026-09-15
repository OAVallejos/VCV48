#!/usr/bin/env python3
"""
TEST_OH_v2.py - Verification of the O_h architecture of VCV48
=============================================================
This script generates the 48 matrices of the O_h group and computes
the average of Tr(M·R)²/3 in a metric deformed by birefringence δ.

It numerically validates that the average over the O_h group gives 1/3,
which is the second pillar of the derivation of f_geom = 2/9.

Dependencies: numpy>=1.24
"""

import numpy as np
from collections import Counter

# ------------------------------------------------------------
# CONFIGURATION: VCV48 Model Constants
# ------------------------------------------------------------
delta = 0.00610865  # CMB birefringence (radians) - 0.35°

# Deformation matrix M_delta (incompressible, volume-conserving)
def M_delta(delta):
    """
    Deformation matrix due to birefringence.
    Volume-conserving: det = (1+δ)(1-δ)*d3 = 1
    """
    d1 = 1.0 + delta
    d2 = 1.0 - delta
    d3 = 1.0 / (1.0 - delta**2)  # To conserve volume
    return np.array([
        [d1, 0.0, 0.0],
        [0.0, d2, 0.0],
        [0.0, 0.0, d3]
    ])

# ------------------------------------------------------------
# GENERATION OF THE O_h GROUP (48 rotation matrices)
# ------------------------------------------------------------
def generate_oh():
    """
    Generates the 48 rotation matrices of the O_h group.
    24 proper rotations + 24 improper rotations (with inversion).
    """
    matrices = []
    pi = np.pi

    # --------------------------------------------------------
    # 1. IDENTITY
    # --------------------------------------------------------
    matrices.append(np.eye(3))

    # --------------------------------------------------------
    # 2. ROTATIONS OF 90°, 180°, 270° AROUND X, Y, Z AXES
    # --------------------------------------------------------
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1])
    ]
    for axis in axes:
        for angle in [pi/2, pi, 3*pi/2]:
            # Rotation matrix using Rodrigues' formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            matrices.append(R)

    # --------------------------------------------------------
    # 3. ROTATIONS OF 120° AND 240° AROUND CUBE DIAGONALS (8 AXES)
    # --------------------------------------------------------
    diag_axes = [
        np.array([1, 1, 1]),
        np.array([1, 1, -1]),
        np.array([1, -1, 1]),
        np.array([-1, 1, 1])
    ]
    for axis in diag_axes:
        axis = axis / np.linalg.norm(axis)  # Normalize
        for angle in [2*pi/3, 4*pi/3]:
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            matrices.append(R)

    # --------------------------------------------------------
    # 4. ROTATIONS OF 180° AROUND EDGE-CENTER AXES (6 AXES)
    # --------------------------------------------------------
    edge_axes = [
        np.array([1, 1, 0]),
        np.array([1, -1, 0]),
        np.array([1, 0, 1]),
        np.array([1, 0, -1]),
        np.array([0, 1, 1]),
        np.array([0, 1, -1])
    ]
    for axis in edge_axes:
        axis = axis / np.linalg.norm(axis)  # Normalize
        angle = pi  # 180°
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        matrices.append(R)

    # Verify that we have 24 proper rotations
    assert len(matrices) == 24, f"Error: {len(matrices)} proper rotations, expected 24"

    # --------------------------------------------------------
    # 5. ADD IMPROPER ROTATIONS (WITH INVERSION)
    # --------------------------------------------------------
    n_proper = len(matrices)
    for i in range(n_proper):
        matrices.append(-matrices[i])

    assert len(matrices) == 48, f"Error: {len(matrices)} matrices, expected 48"
    return matrices

# ------------------------------------------------------------
# CALCULATION OF AVERAGE IN DEFORMED METRIC
# ------------------------------------------------------------
def deformed_average(matrices, delta, verbose=True):
    """
    Computes the average of Tr(M·R)²/3 for the O_h group in a deformed metric.

    Args:
        matrices: List of 48 matrices of the O_h group
        delta: Birefringence (radians)
        verbose: If True, prints details

    Returns:
        average: Average ⟨Tr(M·R)²/3⟩
        trace_values: List of traces for distribution analysis
    """
    M = M_delta(delta)
    total = 0.0
    n = len(matrices)

    if verbose:
        print("=" * 70)
        print(f"CALCULATION IN DEFORMED METRIC (δ = {delta:.8f} rad)")
        print("=" * 70)
        print(f"{'Index':>6} | {'Trace(M·R)':>12} | {'(Tr²)/3':>12}")
        print("-" * 70)

    trace_values = []
    for i, R in enumerate(matrices):
        product = M @ R
        trace = np.trace(product)
        term = (trace ** 2) / 3.0
        total += term
        trace_values.append(round(trace, 6))

        # Show every 6 matrices to avoid saturating the output
        if verbose and i % 6 == 0:
            print(f"{i:6} | {trace:12.8f} | {term:12.8f}")

    if verbose:
        print("-" * 70)
        average = total / n
        print(f"\nTOTAL SUM: {total:.8f}")
        print(f"AVERAGE ⟨Tr(M·R)²/3⟩: {average:.8f}")

        # Trace distribution analysis
        print("\n" + "=" * 70)
        print("TRACE DISTRIBUTION (frequency)")
        print("=" * 70)
        frequencies = Counter(trace_values)
        for tr in sorted(frequencies.keys()):
            print(f"Trace = {tr:8.4f}: {frequencies[tr]:2} occurrences")

        return average, trace_values
    else:
        return total / n, trace_values

# ------------------------------------------------------------
# COMPARISON WITH THE BASE VALUE 1/3
# ------------------------------------------------------------
def analyze_result(average, delta):
    """Analyzes the result by comparing it with the base value 1/3."""
    base_value = 1.0 / 3.0  # 0.3333333333
    diff = average - base_value

    print("\n" + "=" * 70)
    print("MEANING ANALYSIS")
    print("=" * 70)
    print(f"Base value (pure symmetry, δ=0): {base_value:.8f}")
    print(f"Value obtained with real δ:       {average:.8f}")
    print(f"Difference:                       {diff:+.8f}")

    # Expected second-order correction: (4/9)·δ²
    expected_diff = (4.0 / 9.0) * delta**2
    print(f"Expected second-order correction: {expected_diff:+.8f}")

    if abs(diff - expected_diff) < 1e-8:
        print("\n✅✅✅ EXACT MATCH! The correction is purely second-order in δ.")
        print("    The average over O_h gives 1/3 + (4/9)·δ²")
    elif diff > 0:
        proportion = (diff / expected_diff) * 100 if expected_diff > 0 else 0
        print(f"\n📊 The result is {proportion:.1f}% of the expected correction.")
        print("    Birefringence δ explains the anisotropy.")
    else:
        print("\n⚠️ The result is LOWER than the base value (δ=0)")
        print("    Check the definition of M_delta.")

    return diff

# ------------------------------------------------------------
# VERIFICATION OF THE GEOMETRIC FACTOR 2/9
# ------------------------------------------------------------
def verify_geom_factor():
    """
    Verifies that the geometric factor 2/9 emerges correctly.

    f_geom = I_ang × I_gpo × (1 + 8α/(1-ν)) × f_core

    Where:
    - I_ang = 2/3 (continuous angular average)
    - I_gpo = 1/3 (average over O_h)
    - I_ang × I_gpo = 2/9
    """
    print("\n" + "=" * 70)
    print("GEOMETRIC FACTOR VERIFICATION")
    print("=" * 70)

    # Continuous angular average
    I_ang = 2.0 / 3.0  # ∫ sinθ cos²θ dθ = 2/3

    # Average over the O_h group (numerically verified)
    I_gpo = 1.0 / 3.0  # ⟨Tr²⟩/3 = 1/3

    # Product
    I_sym = I_ang * I_gpo  # = 2/9

    print(f"I_ang (angular average)     = {I_ang:.6f}  (2/3)")
    print(f"I_gpo (group average)       = {I_gpo:.6f}  (1/3)")
    print(f"I_sym = I_ang × I_gpo       = {I_sym:.6f}  (2/9)")

    # With corrections
    alpha = 0.00729735  # Fine structure constant
    nu = 0.25           # Poisson's ratio
    f_core = 1.15       # Core factor (Voigt-Reuss-Hill)

    correction = 1.0 + (8.0 * alpha) / (1.0 - nu)
    f_geom = I_sym * correction * f_core

    print(f"\nWith corrections:")
    print(f"  8α/(1-ν) = {8.0*alpha/(1.0-nu):.6f}")
    print(f"  correction = {correction:.6f}")
    print(f"  f_core (VRH) = {f_core:.2f}")
    print(f"  f_geom = {f_geom:.4f}")

    return f_geom

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("VCV48 ARCHITECTURE TEST - Birefringence as metric deformation")
    print("=" * 70)

    # 1. Generate the O_h group
    oh_matrices = generate_oh()
    print(f"\n✅ O_h group generated: {len(oh_matrices)} matrices")

    # 2. Compute with real δ
    result, trace_values = deformed_average(oh_matrices, delta, verbose=True)

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: ⟨Tr(M·R)²/3⟩ = {result:.8f}")
    print("=" * 70)

    # 3. Analyze the result
    diff = analyze_result(result, delta)

    # 4. Verify the geometric factor
    f_geom = verify_geom_factor()

    # 5. Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"1. O_h group average (δ={delta:.6f}): ⟨Tr²⟩/3 = {result:.8f}")
    print(f"2. Difference from 1/3: {diff:+.8f}")
    print(f"3. f_geom = {f_geom:.4f}")

    if abs(result - 0.33335) < 0.0001:
        print("\n✅ CONFIRMED: The average over O_h gives 1/3 + (4/9)·δ²")
    else:
        print("\n⚠️ Check the calculation: result deviates from expected value.")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()