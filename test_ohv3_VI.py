# test_ohv3.py
import solver_kernel
import math
                              
def main():
    print("=" * 60)
    print("Test of the solver_kernel kernel (Kernel 3/5)")
    print("=" * 60)
    print()                   
    # ============================================================
    # 1. Complete report
    # ============================================================
    print("=== 1. Solver report ===")
    print(solver_kernel.solver_report())
    print()

    # ============================================================
    # 2. Number of solutions
    # ============================================================
    print("=== 2. Number of solutions found ===")
    sols = solver_kernel.solver_solve()
    n = len(sols)
    print(f"Total unique solutions: {n}")
    # We expect at least: maximum, saddle point, zeros
    # (plus the zeros at <100>, <110>, <111>)
    assert n >= 3, f"Expected >= 3, obtained {n}"
    print("OK Reasonable number of solutions")
    print()

    # ============================================================
    # 3. Verify that the maximum is among the solutions
    # ============================================================
    print("=== 3. Verify maximum among the solutions ===")
    # x_max = 0.080258... => u = v = sqrt(x), w = sqrt(1-2x)
    x_max = 0.080258478962689
    u_max = math.sqrt(x_max)
    v_max = u_max
    w_max = math.sqrt(1.0 - 2.0 * x_max)

    # Search among the solutions
    found_max = False
    for sol in sols:
        if (abs(abs(sol[0]) - u_max) < 1e-4 and
            abs(abs(sol[1]) - v_max) < 1e-4 and
            abs(abs(sol[2]) - w_max) < 1e-4):
            found_max = True
            break
    assert found_max, "The maximum is not among the solutions"
    print(f"OK Maximum (u,v,w)=({u_max:.4f},{v_max:.4f},{w_max:.4f}) found")
    print()

    # ============================================================
    # 4. Verify that the saddle point is among the solutions
    # ============================================================
    print("=== 4. Verify saddle point among the solutions ===")
    x_saddle = 0.4416682275
    u_s = math.sqrt(x_saddle)
    v_s = u_s
    w_s = math.sqrt(1.0 - 2.0 * x_saddle)

    found_saddle = False
    for sol in sols:
        if (abs(abs(sol[0]) - u_s) < 1e-4 and
            abs(abs(sol[1]) - v_s) < 1e-4 and
            abs(abs(sol[2]) - w_s) < 1e-4):
            found_saddle = True
            break
    assert found_saddle, "The saddle point is not among the solutions"
    print(f"OK Saddle point (u,v,w)=({u_s:.4f},{v_s:.4f},{w_s:.4f}) found")
    print()

    # ============================================================
    # 5. Verify that the zeros are among the solutions
    # ============================================================
    print("=== 5. Verify zeros among the solutions ===")
    # <100>: (1, 0, 0)
    # <110>: (1,1,0)/sqrt(2)
    # <111>: (1,1,1)/sqrt(3)

    targets = [
        (1.0, 0.0, 0.0),
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0),
        (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
    ]
    for (tu, tv, tw) in targets:
        found = False
        for sol in sols:
            if (abs(abs(sol[0]) - abs(tu)) < 1e-4 and
                abs(abs(sol[1]) - abs(tv)) < 1e-4 and
                abs(abs(sol[2]) - abs(tw)) < 1e-4):
                found = True
                break
        assert found, f"Zero ({tu},{tv},{tw}) not found"
        print(f"  Zero ({tu:.4f},{tv:.4f},{tw:.4f}) OK")

    print("OK The three zeros are among the solutions")
    print()

    # ============================================================
    # 6. Verify the residual of all solutions
    # ============================================================
    print("=== 6. Verify residual of all solutions ===")
    max_res = 0.0
    for sol in sols:
        res = solver_kernel.solver_residual_norm(sol[0], sol[1], sol[2], sol[3])
        if res > max_res:
            max_res = res
    print(f"Maximum residual: {max_res:.2e}")
    assert max_res < 1e-8, f"Maximum residual too large: {max_res}"
    print("OK All critical points have residual < 1e-8")
    print()

    # ============================================================
    # 7. Verify classification: maximum, saddle, zeros
    # ============================================================
    print("=== 7. Classification by value of G_T ===")

    # Group by value of G_T
    gt_values = {}
    for sol in sols:
        g = solver_kernel.solver_gt(sol[0], sol[1], sol[2])
        key = round(g, 6)
        gt_values[key] = gt_values.get(key, 0) + 1

    print(f"Unique values of G_T: {len(gt_values)}")
    for g_val, count in sorted(gt_values.items(), reverse=True):
        print(f"  G_T = {g_val:.6f} (multiplicity: {count})")

    # There must be at least: G_T = 0.0657 (maximum), G_T ~ 0.0034 (saddle), G_T = 0 (zeros)
    assert any(abs(g - 0.065706) < 1e-4 for g in gt_values), "Missing the maximum"
    assert any(abs(g - 0.003393) < 1e-4 for g in gt_values), "Missing the saddle point"
    assert any(abs(g) < 1e-6 for g in gt_values), "Missing the zeros"
    print("OK Correct classification")
    print()

    # ============================================================
    # 8. Additional verification: Newton from the maximum converges
    # ============================================================
    print("=== 8. Test of Newton from the maximum ===")
    result = solver_kernel.solver_newton(u_max, v_max, w_max)
    assert result is not None, "Newton did not converge from the maximum"
    sol = result
    res = solver_kernel.solver_residual_norm(sol[0], sol[1], sol[2], sol[3])
    print(f"  Newton from maximum: residual = {res:.2e}")
    assert res < 1e-10, f"Residual too large: {res}"
    print("OK Newton converges from the maximum")
    print()

    # ============================================================
    # 9. Test of lambda_from_grad
    # ============================================================
    print("=== 9. Test of lambda_from_grad ===")
    lam = solver_kernel.solver_lambda_from_grad(u_max, v_max, w_max)
    print(f"  lambda from gradient at the maximum: {lam:.10f}")
    # Compare with the known value
    assert abs(lam - 0.328529928206077) < 1e-10, f"incorrect lambda: {lam}"
    print("OK lambda_from_grad works")
    print()

    print("=" * 60)
    print("ALL TESTS OF KERNEL 3/5 PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()