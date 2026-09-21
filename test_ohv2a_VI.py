# test_ohv3a.py
import gt_kernel
import numpy as np

# New point
u, v, w = 0.382683, 0.0, 0.923880
lam = 0.3125

# Hessian of G_T
h = gt_kernel.gt_hessian(u, v, w)
H = np.array([
    [h["uu"], h["uv"], h["uw"]],
    [h["uv"], h["vv"], h["vw"]],
    [h["uw"], h["vw"], h["ww"]],
])

print("Hessian of G_T:")
print(H)
print()

# Restricted Hessian = H - 2*lambda*I
H_rest = H - 2 * lam * np.eye(3)
print("Restricted Hessian (H - 2*lambda*I):")
print(H_rest)
print()

# Eigenvalues of the restricted Hessian
eigenvalues = np.linalg.eigvalsh(H_rest)
print("Eigenvalues of the restricted Hessian:")
print(eigenvalues)
print()

# Classification
n_pos = sum(1 for e in eigenvalues if e > 1e-10)
n_neg = sum(1 for e in eigenvalues if e < -1e-10)
n_zero = sum(1 for e in eigenvalues if abs(e) < 1e-10)

print(f"Positive eigenvalues: {n_pos}")
print(f"Negative eigenvalues: {n_neg}")
print(f"Zero eigenvalues: {n_zero}")
print()

if n_pos == 0 and n_neg == 0:
    print("CLASSIFICATION: Degenerate point (all eigenvalues zero)")
elif n_pos == 0:
    print("CLASSIFICATION: LOCAL MINIMUM")
elif n_neg == 0:
    print("CLASSIFICATION: LOCAL MAXIMUM")
else:
    print("CLASSIFICATION: SADDLE POINT")