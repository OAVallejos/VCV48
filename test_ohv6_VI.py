#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ohv6.py — Test of the a_nu_kernel kernel (Kernel 6/6)

Computes the sidereal modulation amplitude of the M87 neutrino flux
according to Annex VI of the VCV48 model.

IT DOES NOT INVENT ANYTHING. All values come from the Rust kernel.

New features with respect to the previous version:
  - jet_geometry(): verifies n_LOS, k_jet, and the angle between them
  - gt_circle_extrema(): computes G_T_max, G_T_min over the jet circle
"""

import a_nu_kernel as anu

SEP = "=" * 60

def header(title):
    print(SEP)
    print(title)
    print(SEP)

def main():
    header("Test of the a_nu_kernel kernel (Kernel 6/6)")
    print()

    # ========================================================
    # 1. Geometry of the M87 jet
    # ========================================================
    print("=== 1. Geometry of the M87 jet ===")
    g = anu.jet_geometry()
    print(f"  n_LOS   = ({g['n_los_x']:+.9f}, {g['n_los_y']:+.9f}, {g['n_los_z']:+.9f})")
    print(f"  |n_LOS| = {g['norm_n_los']:.12f}")
    print(f"  k_jet^E = ({g['k_jet_x']:+.9f}, {g['k_jet_y']:+.9f}, {g['k_jet_z']:+.9f})")
    print(f"  |k_jet| = {g['norm_k_jet']:.12f}")
    print(f"  cos(jet-LOS angle) = {g['cos_theta']:.9f}")
    print(f"  jet-LOS angle      = {g['theta_deg']:.4f} deg")
    print()
    # Verifications
    assert abs(g['norm_n_los'] - 1.0) < 1e-9, "n_LOS is not unitary"
    assert abs(g['norm_k_jet'] - 1.0) < 1e-9, "k_jet is not unitary"
    assert abs(g['theta_deg'] - 17.0) < 0.5, "jet-LOS angle is not ~17 deg"
    print("  OK: n_LOS and k_jet are unitary, angle ~17 deg")
    print()

    # ========================================================
    # 2. Verification Kernel 5/5 (corrected)
    # ========================================================
    print("=== 2. Verification of the maximum of Kernel 5/5 (corrected) ===")
    v = anu.a_nu_verify_kernel5()
    print(f"  k_max         = ({v['u']:.9f}, {v['v']:.9f}, {v['w']:.9f})")
    print(f"  G_T(k_max)    = {v['gt_max']:.15f}")
    print(f"  phi6(k_max)   = {v['phi6_max']:.15f}")
    print(f"  S(k_max)      = {v['s_max']:.15f}")
    print(f"  |T|^2(k_max)  = {v['t2_max']:.15f}")
    print(f"  A_Rabi(k_max) = {v['a_rabi']:.15f}")
    print(f"  A_Rabi_max(K5)= {v['a_rabi_max_kernel5']:.15f}")
    print(f"  Difference    = {v['diff']:.3e}")
    if v['diff'] < 1e-12:
        print("  OK: matches the corrected Kernel 5/5")
    else:
        print("  WARN: difference > 1e-12")
    print()

    # ========================================================
    # 3. G_T over the jet circle (canonical orientation)
    # ========================================================
    print("=== 3. G_T over the jet circle (canonical orientation) ===")
    gt_ext = anu.gt_circle_extrema(0.0, 0.0, 0.0, 4096)
    print(f"  G_T_max    = {gt_ext['gt_max']:.10f}")
    print(f"    at k = ({gt_ext['u_at_max']:+.6f}, {gt_ext['v_at_max']:+.6f}, {gt_ext['w_at_max']:+.6f})")
    print(f"  G_T_min    = {gt_ext['gt_min']:.10f}")
    print(f"  G_T_median = {gt_ext['gt_median']:.10f}")
    print(f"  Range      = {gt_ext['gt_max'] - gt_ext['gt_min']:.10f}")
    print()

    # ========================================================
    # 4. A_nu for the canonical orientation
    # ========================================================
    print("=== 4. A_nu for the canonical orientation (0,0,0) ===")
    a_canon = anu.a_nu_canonical(4096)
    print(f"  A_nu_canonical = {a_canon:.10f} ({100*a_canon:.4f}%)")
    print()

    # ========================================================
    # 5. Average over SO(3)
    # ========================================================
    print("=== 5. A_nu averaged over 5000 orientations ===")
    avg = anu.a_nu_average(5000, 1024, 48)
    print(f"  mean   = {avg['mean']:.10f} ({100*avg['mean']:.4f}%)")
    print(f"  median = {avg['median']:.10f} ({100*avg['median']:.4f}%)")
    print(f"  std    = {avg['std']:.10f}")
    print(f"  min    = {avg['min']:.10f} ({100*avg['min']:.4f}%)")
    print(f"  max    = {avg['max']:.10f} ({100*avg['max']:.4f}%)")
    print()

    # ========================================================
    # 6. Verification of bounds
    # ========================================================
    print("=== 6. Verification of bounds ===")
    A_RABI_MAX_AT_GTMAX = v['a_rabi_max_kernel5']
    A_NU_MAX_SAMPLED = 0.0244839380
    print(f"  A_Rabi_max (at k_max of G_T) = {A_RABI_MAX_AT_GTMAX:.10f}")
    print(f"  A_nu_max (sampled SO3)       = {avg['max']:.10f}")
    print(f"  A_nu_max (documented)        = {A_NU_MAX_SAMPLED:.10f}")
    if avg['max'] <= A_NU_MAX_SAMPLED + 1e-6:
        print(f"  OK: A_nu_max <= sampled bound")
    else:
        print(f"  WARN: A_nu_max exceeds the documented value")
    print()

    # ========================================================
    # 7. Harmonic structure
    # ========================================================
    print("=== 7. Harmonic structure (canonical orientation) ===")
    harm = anu.a_nu_harmonics(0.0, 0.0, 0.0, 4096)
    for n, amp in enumerate(harm, start=1):
        allowed = n in (2, 3, 4, 6, 8, 12)
        marker = " <-- O_h allows" if allowed else ""
        print(f"  n = {n:2d}: {amp:.6e}{marker}")
    print()

    # ========================================================
    # 8. Complete report of the kernel
    # ========================================================
    print("=== 8. Complete report of the kernel ===")
    report = anu.a_nu_report()
    print(report)


if __name__ == "__main__":
    main()