# test_ohv3b.py
import gt_kernel
import numpy as np            import math

# Global maximum               x_max = 0.080258478962689
u = math.sqrt(x_max)
v = u                         
w = math.sqrt(1.0 - 2.0 * x_max)                            
lam = 0.328529928206077

h = gt_kernel.gt_hessian(u, v, w)
H = np.array([
    [h["uu"], h["uv"], h["uw"]],
    [h["uv"], h["vv"], h["vw"]],
    [h["uw"], h["vw"], h["ww"]],
])
H_rest = H - 2 * lam * np.eye(3)
eigenvalues = np.linalg.eigvalsh(H_rest)
print(f"Global maximum: eigenvalues = {eigenvalues}")