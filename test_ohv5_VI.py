# test_ohv5.py
import vcv48_kernel
import math


def main():
    print("=" * 60)
    print("Test of the vcv48_kernel kernel (Kernel 5/5)")
    print("=" * 60)
    print()

    # 1. Complete report
    print(vcv48_kernel.report_full())

    # 2. Verify group O_h
    print("=== Verification of the group O_h ===")
    assert vcv48_kernel.oh_order() == 48
    ok, msg = vcv48_kernel.oh_verify()
    assert ok, f"Group O_h fails: {msg}"
    print(f"  O_h: {vcv48_kernel.oh_order()} elements, verification OK")
    print()

    # 3. Verify stabilizer
    print("=== Verification of the stabilizer ===")
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt3 = 1.0 / math.sqrt(3.0)

    assert vcv48_kernel.oh_stabilizer(0.2833, 0.2833, 0.9162) == 2
    assert vcv48_kernel.oh_stabilizer(1.0, 0.0, 0.0) == 8
    assert vcv48_kernel.oh_stabilizer(inv_sqrt2, inv_sqrt2, 0.0) == 4
    assert vcv48_kernel.oh_stabilizer(inv_sqrt3, inv_sqrt3, inv_sqrt3) == 6
    print("  Correct stabilizers: maximum=2, <100>=8, <110>=4, <111>=6")
    print()

    # 4. Verify G_T (with EXACT values)
    print("=== Verification of G_T ===")
    tol = 1e-12
    assert abs(vcv48_kernel.gt_value(1.0, 0.0, 0.0)) < tol
    assert abs(vcv48_kernel.gt_value(inv_sqrt2, inv_sqrt2, 0.0)) < tol
    assert abs(vcv48_kernel.gt_value(inv_sqrt3, inv_sqrt3, inv_sqrt3)) < tol
    g_max = vcv48_kernel.gt_value(0.283299274553765, 0.283299274553765, 0.916233071917087)
    print(f"  G_T = 0 at <100>, <110>, <111> (using exact values)")
    print(f"  G_T at maximum: {g_max:.10f}")
    print()

    # 5. Verify A_max
    print("=== Verification of A_max ===")
    result = vcv48_kernel.compute_a_max(0.25229889, 0.15)
    print(f"  Direction of the maximum: ({result['u']:.6f}, {result['v']:.6f}, {result['w']:.6f})")
    print(f"  G_T_max = {result['gt_max']:.10f}")
    print(f"  |T|^2 = {result['T2']:.10f}")
    print(f"  S = {result['S']:.10f}")
    print(f"  A_max = {result['A_max']:.10f} ({result['A_max']*100:.4f}%)")
    assert abs(result['A_max'] - 0.0227) < 0.001
    print()

    # 6. Verify solver
    print("=== Verification of the solver ===")
    sol = vcv48_kernel.solver_newton(0.2833, 0.2833, 0.9162)
    assert sol is not None
    res = vcv48_kernel.solver_residual_norm(sol[0], sol[1], sol[2], sol[3])
    print(f"  Residual at the maximum: {res:.2e}")
    assert res < 1e-10
    print()

    # 7. Count points by classification
    print("=== Count of points by classification ===")
    points = vcv48_kernel.analyze_all()
    n_max = sum(1 for p in points if p["classification"] == "maximo")
    n_saddle = sum(1 for p in points if p["classification"] == "punto_de_silla")
    n_min = sum(1 for p in points if p["classification"] == "minimo")
    print(f"  Maxima: {n_max}")
    print(f"  Saddle points: {n_saddle}")
    print(f"  Minima: {n_min}")
    print(f"  Total: {len(points)}")
    assert n_max == 24
    assert n_saddle == 48
    assert n_min == 26
    print()

    print("=" * 60)
    print("ALL TESTS OF KERNEL 5/5 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()