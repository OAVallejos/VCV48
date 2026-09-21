# test_ohv1.py   

import oh_group
import math                   
print("=" * 60)
print("Test of the oh_group kernel (Kernel 1/5)")
print("=" * 60)
print()

# ============================================================
# 1. Group order
# ============================================================
n = oh_group.oh_order()
print(f"Order of O_h: {n}")
assert n == 48, f"Error: O_h must have 48 elements, it has {n}"
print("OK Correct order")
print()

# ============================================================
# 2. Complete verification (orthogonality, determinant, closure)
# ============================================================
ok, msg = oh_group.oh_verify()
print(f"Verification: {ok}")
print(f"Message: {msg}")
assert ok, "Error: verification failed"
print()

# ============================================================
# 3. Report with stabilizers
# ============================================================
print(oh_group.oh_report())
print()

# ============================================================
# 4. Stabilizer of the point <100>
# ============================================================
s = oh_group.oh_stabilizer(1.0, 0.0, 0.0)
print(f"Stabilizer of (1,0,0): {s} (expected: 8)")
assert s == 8, f"Error: expected 8, obtained {s}"

# ============================================================
# 5. Stabilizer of the point <110>
# ============================================================
inv_sqrt2 = 1.0 / math.sqrt(2.0)
s = oh_group.oh_stabilizer(inv_sqrt2, inv_sqrt2, 0.0)
print(f"Stabilizer of (1,1,0)/sqrt(2): {s} (expected: 4)")
assert s == 4, f"Error: expected 4, obtained {s}"

# ============================================================
# 6. Stabilizer of the point <111>
# ============================================================
inv_sqrt3 = 1.0 / math.sqrt(3.0)
s = oh_group.oh_stabilizer(inv_sqrt3, inv_sqrt3, inv_sqrt3)
print(f"Stabilizer of (1,1,1)/sqrt(3): {s} (expected: 6)")
assert s == 6, f"Error: expected 6, obtained {s}"

# ============================================================
# 7. Stabilizer of the maximum point (Annex VI)
# ============================================================
s = oh_group.oh_stabilizer(0.2824, 0.2824, 0.9169)
print(f"Stabilizer of (0.2824,0.2824,0.9169): {s} (expected: 2)")
assert s == 2, f"Error: expected 2, obtained {s}"

# ============================================================
# 8. Triviality
# ============================================================
has_triv = oh_group.oh_has_trivial_stabilizer(0.3, 0.4, 0.5)
print(f"Does (0.3, 0.4, 0.5) have trivial stabilizer? {has_triv} (expected: True)")
assert has_triv == True

# ============================================================
# 9. Orbit of the maximum (must have 24 elements)
# ============================================================
orbit = oh_group.oh_orbit(0.2824, 0.2824, 0.9169)
print(f"Orbit of (0.2824,0.2824,0.9169): {len(orbit)} elements (expected: 24)")
assert len(orbit) == 24, f"Error: expected 24, obtained {len(orbit)}"

# ============================================================
# 10. Orbit of <111> (must have 8 elements)
# ============================================================
orbit = oh_group.oh_orbit(inv_sqrt3, inv_sqrt3, inv_sqrt3)
print(f"Orbit of (1,1,1)/sqrt(3): {len(orbit)} elements (expected: 8)")
assert len(orbit) == 8, f"Error: expected 8, obtained {len(orbit)}"

# ============================================================
# 11. Orbit of <100> (must have 6 elements)
# ============================================================
orbit = oh_group.oh_orbit(1.0, 0.0, 0.0)
print(f"Orbit of (1,0,0): {len(orbit)} elements (expected: 6)")
assert len(orbit) == 6, f"Error: expected 6, obtained {len(orbit)}"

# ============================================================
# 12. Orbit of <110> (must have 12 elements)
# ============================================================
orbit = oh_group.oh_orbit(inv_sqrt2, inv_sqrt2, 0.0)
print(f"Orbit of (1,1,0)/sqrt(2): {len(orbit)} elements (expected: 12)")
assert len(orbit) == 12, f"Error: expected 12, obtained {len(orbit)}"

print()
print("=" * 60)
print("ALL TESTS OF KERNEL 1/5 PASSED")
print("=" * 60)