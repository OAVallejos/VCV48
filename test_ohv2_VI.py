# test_ohv2.py

import gt_kernel              
import math                   

def newton_raphson_cubic(x0=0.08, tol=1e-15, max_iter=100):
    """Solves 36x^3 - 47x^2 + 16x - 1 = 0."""
    x = x0
    for _ in range(max_iter):         f = 36.0 * x**3 - 47.0 * x**2 + 16.0 * x - 1.0
        df = 108.0 * x**2 - 94.0 * x + 16.0
        x_new = x - f / df
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def main():
    print("=" * 60)
    print("Test of the gt_kernel kernel (Kernel 2/5)")
    print("=" * 60)
    print()

    # ============================================================
    # 1. Verification of the analytical gradient
    # ============================================================
    print("=== 1. Verification of the analytical gradient ===")
    test_points = [
        (1.0, 0.0, 0.0),
        (0.7071067811865476, 0.7071067811865476, 0.0),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
        (0.2824, 0.2824, 0.9169),
        (0.3, 0.4, 0.8660254037844386),
        (0.5, 0.5, 0.7071067811865476),
        (0.1, 0.2, 0.9746794344808963),
    ]
    all_ok = True
    for (u, v, w) in test_points:
        ok, err = gt_kernel.gt_verify_gradient(u, v, w)
        print(f"  ({u:.4f}, {v:.4f}, {w:.4f}): ok={ok}, err={err:.2e}")
        if not ok:
            all_ok = False
    assert all_ok, "Gradient fails"
    print("OK Analytical gradient verified")
    print()

    # ============================================================
    # 2. Verification of the analytical Jacobian
    # ============================================================
    print("=== 2. Verification of the analytical Jacobian ===")
    test_points_lam = [
        (1.0, 0.0, 0.0, 0.5),
        (0.7071067811865476, 0.7071067811865476, 0.0, 0.1),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.2),
        (0.2824, 0.2824, 0.9169, 0.05),
        (0.3, 0.4, 0.8660254037844386, 0.0),
        (0.5, 0.5, 0.7071067811865476, 0.15),
        (0.1, 0.2, 0.9746794344808963, 0.3),
    ]
    all_ok = True
    for (u, v, w, lam) in test_points_lam:
        ok, err = gt_kernel.gt_verify_jacobian(u, v, w, lam)
        print(f"  ({u:.4f}, {v:.4f}, {w:.4f}, lam={lam}): ok={ok}, err={err:.2e}")
        if not ok:
            all_ok = False
    assert all_ok, "Jacobian fails"
    print("OK Analytical Jacobian verified")
    print()

    # ============================================================
    # 3. Values of G_T at high-symmetry points
    # ============================================================
    print("=== 3. Values of G_T at high-symmetry points ===")

    g = gt_kernel.gt_value(1.0, 0.0, 0.0)
    print(f"  G_T(1,0,0) = {g:.10e} (expected: 0)")
    assert abs(g) < 1e-15

    g = gt_kernel.gt_value(0.7071067811865476, 0.7071067811865476, 0.0)
    print(f"  G_T(1,1,0)/sqrt(2) = {g:.10e} (expected: 0)")
    assert abs(g) < 1e-15

    g = gt_kernel.gt_value(0.5773502691896258, 0.5773502691896258, 0.5773502691896258)
    print(f"  G_T(1,1,1)/sqrt(3) = {g:.10e} (expected: 0)")
    assert abs(g) < 1e-15

    print("OK G_T = 0 at the three high-symmetry directions")
    print()

    # ============================================================
    # 4. Exact maximum (Newton-Raphson)
    # ============================================================
    print("=== 4. Exact maximum (Newton-Raphson) ===")

    x_max = newton_raphson_cubic()
    u_max = math.sqrt(x_max)
    v_max = u_max
    w_max = math.sqrt(1.0 - 2.0 * x_max)

    print(f"  x_max = {x_max:.15f}")
    print(f"  (u, v, w) = ({u_max:.15f}, {v_max:.15f}, {w_max:.15f})")

    g_max = gt_kernel.gt_value(u_max, v_max, w_max)
    print(f"  G_T = {g_max:.15f}")
    print(f"  Expected: 0.06570599...")

    grad = gt_kernel.gt_gradient(u_max, v_max, w_max)
    dot_gp = grad["du"] * u_max + grad["dv"] * v_max + grad["dw"] * w_max
    dot_pp = u_max * u_max + v_max * v_max + w_max * w_max
    lam_max = dot_gp / (2.0 * dot_pp)
    print(f"  lambda = {lam_max:.15f}")
    print()

    # ============================================================
    # 5. Residual of the Lagrange system at the exact maximum
    # ============================================================
    print("=== 5. Residual of the Lagrange system at the maximum ===")

    res = gt_kernel.lagrange_residual(u_max, v_max, w_max, lam_max)
    print(f"  F1 = {res['F1']:.2e}")
    print(f"  F2 = {res['F2']:.2e}")
    print(f"  F3 = {res['F3']:.2e}")
    print(f"  F4 = {res['F4']:.2e}")
    print(f"  ||F|| = {res['norm']:.2e}")
    assert res['norm'] < 1e-10, f"Residual too large: {res['norm']}"
    print("OK The point is an exact critical point")
    print()

    # ============================================================
    # 6. Verification of the sphere
    # ============================================================
    print("=== 6. Verification of the sphere ===")
    assert gt_kernel.gt_on_sphere(u_max, v_max, w_max)
    assert not gt_kernel.gt_on_sphere(1.0, 1.0, 1.0)
    assert gt_kernel.gt_on_sphere(1.0, 0.0, 0.0)
    print("OK gt_on_sphere works")
    print()

    # ============================================================
    # 7. Values of G_T at additional points
    # ============================================================
    print("=== 7. Values of G_T at additional points ===")

    # Saddle point (second root of the cubic)
    def newton_raphson_cubic_saddle(x0=0.44, tol=1e-15, max_iter=100):
        x = x0
        for _ in range(max_iter):
            f = 36.0 * x**3 - 47.0 * x**2 + 16.0 * x - 1.0
            df = 108.0 * x**2 - 94.0 * x + 16.0
            x_new = x - f / df
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        return x

    x_saddle = newton_raphson_cubic_saddle()
    u_s = math.sqrt(x_saddle)
    v_s = u_s
    w_s = math.sqrt(1.0 - 2.0 * x_saddle)
    g_saddle = gt_kernel.gt_value(u_s, v_s, w_s)
    print(f"  Saddle point x = {x_saddle:.10f}")
    print(f"  (u, v, w) = ({u_s:.6f}, {v_s:.6f}, {w_s:.6f})")
    print(f"  G_T = {g_saddle:.10f} (expected: ~0.003393)")

    g = gt_kernel.gt_value(0.3, 0.4, 0.8660254037844386)
    print(f"  G_T(0.3, 0.4, 0.8660) = {g:.10f}")
    print()

    # ============================================================
    # 8. Verification of the O_h symmetry of G_T
    # ============================================================
    print("=== 8. Verification of the O_h symmetry of G_T ===")

    u, v, w = 0.2824, 0.2824, 0.9169
    g_orig = gt_kernel.gt_value(u, v, w)

    perms = [
        (u, v, w),
        (u, w, v),
        (v, u, w),
        (v, w, u),
        (w, u, v),
        (w, v, u),
    ]
    for p in perms:
        g_p = gt_kernel.gt_value(p[0], p[1], p[2])
        diff = abs(g_p - g_orig)
        assert diff < 1e-10, f"Symmetry fails: diff = {diff}"

    sign_perms = [
        (-u, v, w),
        (u, -v, w),
        (u, v, -w),
        (-u, -v, w),
        (-u, v, -w),
        (u, -v, -w),
        (-u, -v, -w),
    ]
    for p in sign_perms:
        g_p = gt_kernel.gt_value(p[0], p[1], p[2])
        diff = abs(g_p - g_orig)
        assert diff < 1e-10, f"Sign symmetry fails: diff = {diff}"

    print("OK G_T is invariant under O_h (permutations + sign changes)")
    print()

    # ============================================================
    # 9. Gradient at high-symmetry points
    # ============================================================
    print("=== 9. Gradient at high-symmetry points ===")

    g = gt_kernel.gt_gradient(1.0, 0.0, 0.0)
    print(f"  Gradient at (1,0,0): ({g['du']:.6f}, {g['dv']:.6f}, {g['dw']:.6f})")
    assert abs(g['dv']) < 1e-10
    assert abs(g['dw']) < 1e-10
    print("  dv, dw ~ 0 (by symmetry) OK")

    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    g = gt_kernel.gt_gradient(inv_sqrt3, inv_sqrt3, inv_sqrt3)
    print(f"  Gradient at (1,1,1)/sqrt(3): ({g['du']:.6f}, {g['dv']:.6f}, {g['dw']:.6f})")
    assert abs(g['du'] - g['dv']) < 1e-10
    assert abs(g['du'] - g['dw']) < 1e-10
    print("  du = dv = dw (by symmetry) OK")
    print()

    # ============================================================
    # 10. Test of lagrange_system_flat
    # ============================================================
    print("=== 10. Test of lagrange_system_flat ===")

    f = gt_kernel.lagrange_system_flat(u_max, v_max, w_max, lam_max)
    print(f"  F = {[f'{x:.2e}' for x in f]}")
    assert len(f) == 4
    print("OK lagrange_system_flat works")
    print()

    # ============================================================
    # 11. Test of lagrange_jacobian_flat
    # ============================================================
    print("=== 11. Test of lagrange_jacobian_flat ===")

    j = gt_kernel.lagrange_jacobian_flat(u_max, v_max, w_max, lam_max)
    print(f"  Jacobian (16 elements):")
    for i in range(4):
        row = j[i*4:(i+1)*4]
        print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}, {row[3]:+.6f}]")
    assert len(j) == 16
    print("OK lagrange_jacobian_flat works")
    print()

    print("=" * 60)
    print("ALL TESTS OF KERNEL 2/5 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()