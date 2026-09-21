use pyo3::prelude::*;
use pyo3::types::PyList;
use std::f64::consts::PI;

// ============================================================
// BASIC FUNCTIONS OF G_T AND THE LAGRANGE SYSTEM
// ============================================================
// (Reimplemented here so that the kernel is self-contained)

#[inline]
fn sigma6(u: f64, v: f64, w: f64) -> f64 {
    u.powi(6) + v.powi(6) + w.powi(6)
}

pub fn gt(u: f64, v: f64, w: f64) -> f64 {
    let s6 = sigma6(u, v, w);
    let a = u.powi(4) - s6;
    let b = v.powi(4) - s6;
    let c = w.powi(4) - s6;
    u * u * a * a + v * v * b * b + w * w * c * c
}

pub fn grad_gt(u: f64, v: f64, w: f64) -> [f64; 3] {
    let u2 = u * u;
    let v2 = v * v;
    let w2 = w * w;
    let u3 = u2 * u;
    let v3 = v2 * v;
    let w3 = w2 * w;
    let u4 = u2 * u2;
    let v4 = v2 * v2;
    let w4 = w2 * w2;
    let u5 = u4 * u;
    let v5 = v4 * v;
    let w5 = w4 * w;

    let s6 = u4 * u2 + v4 * v2 + w4 * w2;

    let a = u4 - s6;
    let b = v4 - s6;
    let c = w4 - s6;

    let da_du = 4.0 * u3 - 6.0 * u5;
    let db_dv = 4.0 * v3 - 6.0 * v5;
    let dc_dw = 4.0 * w3 - 6.0 * w5;

    let ds6_du = 6.0 * u5;
    let ds6_dv = 6.0 * v5;
    let ds6_dw = 6.0 * w5;

    let da_dv = -ds6_dv;
    let da_dw = -ds6_dw;
    let db_du = -ds6_du;
    let db_dw = -ds6_dw;
    let dc_du = -ds6_du;
    let dc_dv = -ds6_dv;

    let dgt_du = 2.0 * u * a * a
        + u2 * 2.0 * a * da_du
        + v2 * 2.0 * b * db_du
        + w2 * 2.0 * c * dc_du;

    let dgt_dv = 2.0 * v * b * b
        + v2 * 2.0 * b * db_dv
        + u2 * 2.0 * a * da_dv
        + w2 * 2.0 * c * dc_dv;

    let dgt_dw = 2.0 * w * c * c
        + w2 * 2.0 * c * dc_dw
        + u2 * 2.0 * a * da_dw
        + v2 * 2.0 * b * db_dw;

    [dgt_du, dgt_dv, dgt_dw]
}

pub fn hess_gt(u: f64, v: f64, w: f64) -> [f64; 6] {
    let u2 = u * u;
    let v2 = v * v;
    let w2 = w * w;
    let u3 = u2 * u;
    let v3 = v2 * v;
    let w3 = w2 * w;
    let u4 = u2 * u2;
    let v4 = v2 * v2;
    let w4 = w2 * w2;
    let u5 = u4 * u;
    let v5 = v4 * v;
    let w5 = w4 * w;

    let s6 = u4 * u2 + v4 * v2 + w4 * w2;

    let a = u4 - s6;
    let b = v4 - s6;
    let c = w4 - s6;

    let da_du = 4.0 * u3 - 6.0 * u5;
    let db_dv = 4.0 * v3 - 6.0 * v5;
    let dc_dw = 4.0 * w3 - 6.0 * w5;

    let ds6_du = 6.0 * u5;
    let ds6_dv = 6.0 * v5;
    let ds6_dw = 6.0 * w5;

    let da_dv = -ds6_dv;
    let da_dw = -ds6_dw;
    let db_du = -ds6_du;
    let db_dw = -ds6_dw;
    let dc_du = -ds6_du;
    let dc_dv = -ds6_dv;

    let d2s6_du2 = 30.0 * u4;
    let d2s6_dv2 = 30.0 * v4;
    let d2s6_dw2 = 30.0 * w4;

    let d2a_du2 = 12.0 * u2 - 30.0 * u4;
    let d2b_dv2 = 12.0 * v2 - 30.0 * v4;
    let d2c_dw2 = 12.0 * w2 - 30.0 * w4;

    let d2a_dv2 = -d2s6_dv2;
    let d2a_dw2 = -d2s6_dw2;
    let d2b_du2 = -d2s6_du2;
    let d2b_dw2 = -d2s6_dw2;
    let d2c_du2 = -d2s6_du2;
    let d2c_dv2 = -d2s6_dv2;

    let h_uu = 2.0 * a * a
        + 4.0 * u * a * da_du
        + 4.0 * u * a * da_du
        + 2.0 * u2 * da_du * da_du
        + 2.0 * u2 * a * d2a_du2
        + 2.0 * v2 * db_du * db_du
        + 2.0 * v2 * b * d2b_du2
        + 2.0 * w2 * dc_du * dc_du
        + 2.0 * w2 * c * d2c_du2;

    let h_vv = 2.0 * b * b
        + 4.0 * v * b * db_dv
        + 4.0 * v * b * db_dv
        + 2.0 * v2 * db_dv * db_dv
        + 2.0 * v2 * b * d2b_dv2
        + 2.0 * u2 * da_dv * da_dv
        + 2.0 * u2 * a * d2a_dv2
        + 2.0 * w2 * dc_dv * dc_dv
        + 2.0 * w2 * c * d2c_dv2;

    let h_ww = 2.0 * c * c
        + 4.0 * w * c * dc_dw
        + 4.0 * w * c * dc_dw
        + 2.0 * w2 * dc_dw * dc_dw
        + 2.0 * w2 * c * d2c_dw2
        + 2.0 * u2 * da_dw * da_dw
        + 2.0 * u2 * a * d2a_dw2
        + 2.0 * v2 * db_dw * db_dw
        + 2.0 * v2 * b * d2b_dw2;

    let h_uv = 4.0 * u * a * da_dv
        + 2.0 * u2 * da_dv * da_du
        + 4.0 * v * b * db_du
        + 2.0 * v2 * db_dv * db_du
        + 2.0 * w2 * dc_dv * dc_du;

    let h_uw = 4.0 * u * a * da_dw
        + 2.0 * u2 * da_dw * da_du
        + 4.0 * w * c * dc_du
        + 2.0 * w2 * dc_dw * dc_du
        + 2.0 * v2 * db_dw * db_du;

    let h_vw = 4.0 * v * b * db_dw
        + 2.0 * v2 * db_dw * db_dv
        + 4.0 * w * c * dc_dv
        + 2.0 * w2 * dc_dw * dc_dv
        + 2.0 * u2 * da_dw * da_dv;

    [h_uu, h_uv, h_uw, h_vv, h_vw, h_ww]
}

pub fn lagrange_system(x: &[f64; 4]) -> [f64; 4] {
    let u = x[0];
    let v = x[1];
    let w = x[2];
    let lam = x[3];

    let grad = grad_gt(u, v, w);

    [
        grad[0] - 2.0 * lam * u,
        grad[1] - 2.0 * lam * v,
        grad[2] - 2.0 * lam * w,
        u * u + v * v + w * w - 1.0,
    ]
}

fn lagrange_jacobian(x: &[f64; 4]) -> [f64; 16] {
    let u = x[0];
    let v = x[1];
    let w = x[2];
    let lam = x[3];

    let h = hess_gt(u, v, w);

    [
        h[0] - 2.0 * lam,  h[1],              h[2],              -2.0 * u,
        h[1],              h[3] - 2.0 * lam,  h[4],              -2.0 * v,
        h[2],              h[4],              h[5] - 2.0 * lam,  -2.0 * w,
        2.0 * u,           2.0 * v,           2.0 * w,           0.0,
    ]
}

// ============================================================
// LINEAR ALGEBRA: SOLVE 4x4 SYSTEM
// ============================================================
// Gaussian elimination with partial pivoting.

fn solve_4x4(a: &[f64; 16], b: &[f64; 4]) -> Option<[f64; 4]> {
    // Copy to augmented matrix 4x5 (row-major)
    let mut aug = [[0.0f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = a[i * 4 + j];
        }
        aug[i][4] = b[i];
    }

    // Forward elimination
    for col in 0..4 {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        for row in (col + 1)..4 {
            let factor = aug[row][col] / aug[col][col];
            for j in col..5 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut sum = aug[i][4];
        for j in (i + 1)..4 {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() < 1e-15 {
            return None;
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

// ============================================================
// NEWTON-RAPHSON FOR THE LAGRANGE SYSTEM
// ============================================================

pub fn newton_refine(x0: &[f64; 4], tol: f64, max_iter: usize) -> Option<[f64; 4]> {
    let mut x = *x0;

    for _ in 0..max_iter {
        let f = lagrange_system(&x);
        let f_norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt();

        if f_norm < tol {
            return Some(x);
        }

        let jac = lagrange_jacobian(&x);
        let dx = solve_4x4(&jac, &f)?;

        for i in 0..4 {
            x[i] -= dx[i];
        }
    }

    // Verify final convergence
    let f = lagrange_system(&x);
    let f_norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt();
    if f_norm < tol * 100.0 {
        Some(x)
    } else {
        None
    }
}

// ============================================================
// SYSTEMATIC EXPLORATION OF THE SPHERE
// ============================================================
// Generates points on the sphere using a grid in spherical
// coordinates: (theta, phi) with theta in [0, pi], phi in [0, 2pi).
//
// For each point, we compute the implicit lambda and refine with
// Newton-Raphson.

fn lambda_from_grad(u: f64, v: f64, w: f64) -> f64 {
    let grad = grad_gt(u, v, w);
    // lambda = (grad . (u,v,w)) / (2 |(u,v,w)|^2)
    let dot = grad[0] * u + grad[1] * v + grad[2] * w;
    let n2 = u * u + v * v + w * w;
    dot / (2.0 * n2)
}

/// Explores the sphere with a grid of n_theta x n_phi points.
/// Returns all critical points found (refined).
fn explore_sphere(n_theta: usize, n_phi: usize, tol: f64) -> Vec<[f64; 4]> {
    let mut solutions: Vec<[f64; 4]> = Vec::new();

    for i_theta in 0..=n_theta {
        let theta = PI * (i_theta as f64) / (n_theta as f64);
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        // If sin_theta is ~0, there is only one point at phi
        let n_phi_eff = if sin_theta.abs() < 1e-10 { 1 } else { n_phi };

        for i_phi in 0..n_phi_eff {
            let phi = 2.0 * PI * (i_phi as f64) / (n_phi as f64);
            let u = sin_theta * phi.cos();
            let v = sin_theta * phi.sin();
            let w = cos_theta;

            // Verify that it is on the sphere
            let n2 = u * u + v * v + w * w;
            if (n2 - 1.0).abs() > 1e-10 {
                continue;
            }

            // Implicit lambda
            let lam = lambda_from_grad(u, v, w);

            // Refine with Newton
            let x0 = [u, v, w, lam];
            if let Some(x_ref) = newton_refine(&x0, tol, 50) {
                // Verify that it is on the sphere
                let n2 = x_ref[0] * x_ref[0] + x_ref[1] * x_ref[1] + x_ref[2] * x_ref[2];
                if (n2 - 1.0).abs() < 1e-8 {
                    solutions.push(x_ref);
                }
            }
        }
    }

    solutions
}

// ============================================================
// DEDUPLICATION
// ============================================================

fn deduplicate(solutions: &mut Vec<[f64; 4]>, tol: f64) {
    let mut unique: Vec<[f64; 4]> = Vec::new();

    for sol in solutions.iter() {
        let is_dup = unique.iter().any(|u| {
            (u[0] - sol[0]).abs() < tol
                && (u[1] - sol[1]).abs() < tol
                && (u[2] - sol[2]).abs() < tol
        });
        if !is_dup {
            unique.push(*sol);
        }
    }

    *solutions = unique;
}

// ============================================================
// MULTI-GRID EXPLORATION
// ============================================================

pub fn solve_lagrange_multigrid() -> Vec<[f64; 4]> {
    // Multiple grid resolutions to capture all solutions
    let grids = [
        (12, 24),
        (24, 48),
        (48, 96),
        (96, 192),
    ];

    let mut all_solutions: Vec<[f64; 4]> = Vec::new();

    for &(n_theta, n_phi) in &grids {
        let sols = explore_sphere(n_theta, n_phi, 1e-12);
        all_solutions.extend(sols);
    }

    // Deduplicate with tolerance
    deduplicate(&mut all_solutions, 1e-6);

    all_solutions
}

// ============================================================
// FUNCTIONS EXPOSED TO PYTHON
// ============================================================

#[pyfunction]
fn solver_solve() -> PyResult<Vec<Vec<f64>>> {
    let sols = solve_lagrange_multigrid();
    Ok(sols.into_iter().map(|s| s.to_vec()).collect())
}

#[pyfunction]
fn solver_newton(u: f64, v: f64, w: f64) -> PyResult<Option<Vec<f64>>> {
    let lam = lambda_from_grad(u, v, w);
    let x0 = [u, v, w, lam];
    match newton_refine(&x0, 1e-12, 100) {
        Some(x) => Ok(Some(x.to_vec())),
        None => Ok(None),
    }
}

#[pyfunction]
fn solver_lambda_from_grad(u: f64, v: f64, w: f64) -> f64 {
    lambda_from_grad(u, v, w)
}

#[pyfunction]
fn solver_gt(u: f64, v: f64, w: f64) -> f64 {
    gt(u, v, w)
}

#[pyfunction]
fn solver_residual_norm(u: f64, v: f64, w: f64, lam: f64) -> f64 {
    let f = lagrange_system(&[u, v, w, lam]);
    (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt()
}

#[pyfunction]
fn solver_report() -> PyResult<String> {
    let sols = solve_lagrange_multigrid();
    let mut s = String::new();
    s.push_str(&format!("Found {} unique critical points\n", sols.len()));
    s.push_str("\nDetails:\n");
    s.push_str("----------------------------------------------------------\n");
    s.push_str("   idx |   u          v          w          lambda     |  G_T\n");
    s.push_str("----------------------------------------------------------\n");
    for (i, sol) in sols.iter().enumerate() {
        let g = gt(sol[0], sol[1], sol[2]);
        s.push_str(&format!(
            "  {:4} | {:+.6} {:+.6} {:+.6} {:+.6} | {:.8}\n",
            i, sol[0], sol[1], sol[2], sol[3], g
        ));
    }
    s.push_str("----------------------------------------------------------\n");
    Ok(s)
}

#[pymodule]
fn solver_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solver_solve, m)?)?;
    m.add_function(wrap_pyfunction!(solver_newton, m)?)?;
    m.add_function(wrap_pyfunction!(solver_lambda_from_grad, m)?)?;
    m.add_function(wrap_pyfunction!(solver_gt, m)?)?;
    m.add_function(wrap_pyfunction!(solver_residual_norm, m)?)?;
    m.add_function(wrap_pyfunction!(solver_report, m)?)?;
    Ok(())
}