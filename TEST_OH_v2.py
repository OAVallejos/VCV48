import numpy as np
from collections import Counter

# ------------------------------------------------------------
# CONFIGURATION: VCV48 Model Constants
# ------------------------------------------------------------
delta = 0.00610865  # CMB birefringence (radians)

# Deformation matrix M_delta (incompressible, volume-conserving)
def M_delta(delta):
    """Deformation matrix due to birefringence."""
    d1 = 1.0 + delta
    d2 = 1.0 - delta
    d3 = 1.0 / (1.0 - delta**2)  # To conserve volume: det = (1+δ)(1-δ)*d3 = 1
    return np.array([
        [d1, 0.0, 0.0],
        [0.0, d2, 0.0],
        [0.0, 0.0, d3]
    ])

# ------------------------------------------------------------
# GENERATION OF THE O_h GROUP (48 rotation matrices)
# ------------------------------------------------------------
def generate_oh():
    """Generates the 48 rotation matrices of the O_h group."""
    matrices = []
    pi = np.pi

    # Identity
    matrices.append(np.eye(3))

    # 90°, 180°, 270° rotations around x, y, z axes
    axes = [
        np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
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

    # Cube diagonals (8 axes)
    diag_axes = [
        np.array([1,1,1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([-1,1,1])
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

    # Edge-center axes (6 axes)
    edge_axes = [
        np.array([1,1,0]), np.array([1,-1,0]),
        np.array([1,0,1]), np.array([1,0,-1]),
        np.array([0,1,1]), np.array([0,1,-1])
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

    # Verify we have 24 proper rotations
    assert len(matrices) == 24, f"Error: {len(matrices)} proper rotations, should be 24"

    # Add improper rotations (with inversion)
    n_proper = len(matrices)
    for i in range(n_proper):
        matrices.append(-matrices[i])

    assert len(matrices) == 48, f"Error: {len(matrices)} matrices, should be 48"
    return matrices

# ------------------------------------------------------------
# CALCULATION OF AVERAGE IN DEFORMED METRIC
# ------------------------------------------------------------
def deformed_average(matrices, delta):
    """Calculates the average of Tr(M·R)²/3 for the O_h group."""
    M = M_delta(delta)
    total = 0.0
    n = len(matrices)

    print("=" * 70)
    print(f"CALCULATION IN DEFORMED METRIC (δ = {delta:.8f} rad)")
    print("=" * 70)
    print(f"{'Index':>6} | {'Trace(M·R)':>12} | {'(Tr²)/3':>12}")
    print("-" * 70)

    trace_values = []
    for i, R in enumerate(matrices):
        product = M @ R
        trace = np.trace(product)
        term = (trace**2) / 3.0
        total += term
        trace_values.append(round(trace, 6))

        # Show every 6 matrices to avoid saturation
        if i % 6 == 0:
            print(f"{i:6} | {trace:12.8f} | {term:12.8f}")

    print("-" * 70)
    average = total / n
    print(f"\nTOTAL SUM: {total:.8f}")
    print(f"AVERAGE ⟨Tr(M·R)²/3⟩: {average:.8f}")

    # Trace frequency analysis
    print("\n" + "=" * 70)
    print("TRACE DISTRIBUTION (frequency)")
    print("=" * 70)
    frequencies = Counter(trace_values)
    for tr in sorted(frequencies.keys()):
        print(f"Trace = {tr:8.4f}: {frequencies[tr]:2} occurrences")

    return average

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VCV48 ARCHITECTURE TEST - Birefringence as metric deformation")
    print("=" * 70)

    # Generate O_h group
    oh_matrices = generate_oh()
    print(f"\n✅ O_h group generated: {len(oh_matrices)} matrices")

    # Calculate with δ = 0.00610865
    result = deformed_average(oh_matrices, delta)

    print("\n" + "=" * 70)
    print(f"FINAL RESULT: ⟨Tr(M·R)²/3⟩ = {result:.8f}")
    print("=" * 70)

    # Comparison with reference values
    print("\n" + "=" * 70)
    print("MEANING ANALYSIS")
    print("=" * 70)
    print(f"Base value (pure symmetry, δ=0): 0.33333333")
    print(f"Value obtained with real δ:        {result:.8f}")
    print(f"Difference:                         {result - 0.33333333:+.8f}")

    # Relationship with factor 0.4125 (if it appears)
    if abs(result - 0.4125) < 0.001:
        print("\n✅✅✅ EXACT MATCH! The value is 0.4125")
        print("    Birefringence δ is the only source of anisotropy.")
    elif result > 0.33333 and result < 0.4125:
        proportion = (result - 0.33333) / (0.4125 - 0.33333) * 100
        print(f"\n📊 The result covers {proportion:.1f}% of the way to 0.4125")
        print("    Birefringence explains part of the anisotropy.")
        print("    The remainder would come from α (fine structure constant).")
    elif result > 0.4125:
        print("\n⚠️ The result EXCEEDS 0.4125")
        print("    There is more anisotropy than δ can explain alone.")
    else:
        print("\n⚠️ The result is LOWER than the base value (δ=0)")
        print("    Check the definition of M_delta.")