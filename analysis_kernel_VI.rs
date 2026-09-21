use pyo3::prelude::*;
use pyo3::types::PyList;

// ============================================================
// 3x3 MATRIX IN ROW-MAJOR
// ============================================================
type Mat3 = [f64; 9];

fn mat_identity() -> Mat3 {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            r[i * 3 + j] = s;
        }
    }
    r
}

fn mat_det(m: &Mat3) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

fn mat_transpose(m: &Mat3) -> Mat3 {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

fn mat_apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

// ============================================================
// GENERATION OF THE 48 OPERATIONS OF O_h
// ============================================================

fn generate_oh_group() -> Vec<Mat3> {
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2],
        [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ];
    let signs: [[f64; 3]; 8] = [
        [ 1.0,  1.0,  1.0], [ 1.0,  1.0, -1.0],
        [ 1.0, -1.0,  1.0], [ 1.0, -1.0, -1.0],
        [-1.0,  1.0,  1.0], [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0], [-1.0, -1.0, -1.0],
    ];
    let mut ops: Vec<Mat3> = Vec::with_capacity(48);
    for p in &perms {
        for s in &signs {
            let mut m = [0.0; 9];
            for i in 0..3 {
                m[i * 3 + p[i]] = s[i];
            }
            ops.push(m);
        }
    }
    ops
}

// ============================================================
// ACTUAL STABILIZER
// ============================================================

fn stabilizer_order(ops: &[Mat3], u: f64, v: f64, w: f64) -> usize {
    let point = [u, v, w];
    let tol = 1e-6;
    let mut count = 0;
    for m in ops {
        let t = mat_apply(m, point);
        let dx = t[0] - point[0];
        let dy = t[1] - point[1];
        let dz = t[2] - point[2];
        if dx * dx + dy * dy + dz * dz < tol * tol {
            count += 1;
        }
    }
    count
}

// ============================================================
// G_T AND HESSIAN
// ============================================================

fn sigma6(u: f64, v: f64, w: f64) -> f64 {
    u.powi(6) + v.powi(6) + w.powi(6)
}

fn gt(u: f64, v: f64, w: f64) -> f64 {
    let s6 = sigma6(u, v, w);
    let a = u.powi(4) - s6;
    let b = v.powi(4) - s6;
    let c = w.powi(4) - s6;
    u * u * a * a + v * v * b * b + w * w * c * c
}

fn hess_gt(u: f64, v: f64, w: f64) -> [f64; 6] {
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
        + 8.0 * u * a * da_du
        + 2.0 * u2 * da_du * da_du
        + 2.0 * u2 * a * d2a_du2
        + 2.0 * v2 * db_du * db_du
        + 2.0 * v2 * b * d2b_du2
        + 2.0 * w2 * dc_du * dc_du
        + 2.0 * w2 * c * d2c_du2;

    let h_vv = 2.0 * b * b
        + 8.0 * v * b * db_dv
        + 2.0 * v2 * db_dv * db_dv
        + 2.0 * v2 * b * d2b_dv2
        + 2.0 * u2 * da_dv * da_dv
        + 2.0 * u2 * a * d2a_dv2
        + 2.0 * w2 * dc_dv * dc_dv
        + 2.0 * w2 * c * d2c_dv2;

    let h_ww = 2.0 * c * c
        + 8.0 * w * c * dc_dw
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

// ============================================================
// EIGENVALUES OF A SYMMETRIC 3x3 MATRIX
// ============================================================
// Uses the Jacobi method for eigenvalues.

fn jacobi_eigenvalues(a_in: [[f64; 3]; 3], max_iter: usize) -> [f64; 3] {
    let mut a = a_in;
    let tol = 1e-14;

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut p = 0;
        let mut q = 1;
        let mut max_val = a[0][1].abs();
        if a[0][2].abs() > max_val {
            max_val = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > max_val {
            max_val = a[1][2].abs();
            p = 1;
            q = 2;
        }

        if max_val < tol {
            break;
        }

        // Compute the rotation angle
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-15 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * (2.0 * a[p][q] / (a[q][q] - a[p][p])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation
        let mut a_new = a;
        for i in 0..3 {
            if i != p && i != q {
                a_new[i][p] = c * a[i][p] - s * a[i][q];
                a_new[p][i] = a_new[i][p];
                a_new[i][q] = s * a[i][p] + c * a[i][q];
                a_new[q][i] = a_new[i][q];
            }
        }
        a_new[p][p] = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q];
        a_new[q][q] = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q];
        a_new[p][q] = 0.0;
        a_new[q][p] = 0.0;

        a = a_new;
    }

    [a[0][0], a[1][1], a[2][2]]
}

fn classify_point(u: f64, v: f64, w: f64, lam: f64) -> (&'static str, [f64; 3]) {
    let h = hess_gt(u, v, w);
    let h_mat = [
        [h[0], h[1], h[2]],
        [h[1], h[3], h[4]],
        [h[2], h[4], h[5]],
    ];

    // Restricted Hessian = H - 2*lambda*I
    let h_rest = [
        [h_mat[0][0] - 2.0 * lam, h_mat[0][1], h_mat[0][2]],
        [h_mat[1][0], h_mat[1][1] - 2.0 * lam, h_mat[1][2]],
        [h_mat[2][0], h_mat[2][1], h_mat[2][2] - 2.0 * lam],
    ];

    let eigenvalues = jacobi_eigenvalues(h_rest, 100);

    // Classify: count positive and negative eigenvalues
    // (excluding the radial one, which is the large positive eigenvalue)
    let mut sorted = eigenvalues;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // The two tangential eigenvalues are the two smallest
    let tangent = [sorted[0], sorted[1]];

    let n_pos = tangent.iter().filter(|&&e| e > 1e-6).count();
    let n_neg = tangent.iter().filter(|&&e| e < -1e-6).count();

    let classification = if n_neg == 2 {
        "maximo"
    } else if n_pos == 2 {
        "minimo"
    } else if n_pos == 1 && n_neg == 1 {
        "punto_de_silla"
    } else {
        "degenerado"
    };

    (classification, eigenvalues)
}

// ============================================================
// SOLVER (from Kernel 3, reimplemented)
// ============================================================

fn lambda_from_grad(u: f64, v: f64, w: f64) -> f64 {
    // Quick reimplementation
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

    let dgt_du = 2.0 * u * a * a + u2 * 2.0 * a * da_du
        + v2 * 2.0 * b * (-ds6_du) + w2 * 2.0 * c * (-ds6_du);
    let dgt_dv = 2.0 * v * b * b + v2 * 2.0 * b * db_dv
        + u2 * 2.0 * a * (-ds6_dv) + w2 * 2.0 * c * (-ds6_dv);
    let dgt_dw = 2.0 * w * c * c + w2 * 2.0 * c * dc_dw
        + u2 * 2.0 * a * (-ds6_dw) + v2 * 2.0 * b * (-ds6_dw);

    let dot = dgt_du * u + dgt_dv * v + dgt_dw * w;
    let n2 = u * u + v * v + w * w;
    dot / (2.0 * n2)
}

fn solve_4x4(a: &[f64; 16], b: &[f64; 4]) -> Option<[f64; 4]> {
    let mut aug = [[0.0f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = a[i * 4 + j];
        }
        aug[i][4] = b[i];
    }
    for col in 0..4 {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
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

fn lagrange_system(x: &[f64; 4]) -> [f64; 4] {
    let u = x[0];
    let v = x[1];
    let w = x[2];
    let lam = x[3];

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

    let dgt_du = 2.0 * u * a * a + u2 * 2.0 * a * da_du
        + v2 * 2.0 * b * (-ds6_du) + w2 * 2.0 * c * (-ds6_du);
    let dgt_dv = 2.0 * v * b * b + v2 * 2.0 * b * db_dv
        + u2 * 2.0 * a * (-ds6_dv) + w2 * 2.0 * c * (-ds6_dv);
    let dgt_dw = 2.0 * w * c * c + w2 * 2.0 * c * dc_dw
        + u2 * 2.0 * a * (-ds6_dw) + v2 * 2.0 * b * (-ds6_dw);

    [
        dgt_du - 2.0 * lam * u,
        dgt_dv - 2.0 * lam * v,
        dgt_dw - 2.0 * lam * w,
        u2 + v2 + w2 - 1.0,
    ]
}

fn lagrange_jacobian(x: &[f64; 4]) -> [f64; 16] {
    let u = x[0];
    let v = x[1];
    let w = x[2];
    let lam = x[3];
    let h = hess_gt(u, v, w);
    [
        h[0] - 2.0 * lam, h[1], h[2], -2.0 * u,
        h[1], h[3] - 2.0 * lam, h[4], -2.0 * v,
        h[2], h[4], h[5] - 2.0 * lam, -2.0 * w,
        2.0 * u, 2.0 * v, 2.0 * w, 0.0,
    ]
}

fn newton_refine(x0: &[f64; 4], tol: f64, max_iter: usize) -> Option<[f64; 4]> {
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
    let f = lagrange_system(&x);
    let f_norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt();
    if f_norm < tol * 100.0 {
        Some(x)
    } else {
        None
    }
}

// ============================================================
// DATA STRUCTURE
// ============================================================

#[derive(Debug, Clone)]
struct CriticalPoint {
    u: f64,
    v: f64,
    w: f64,
    lambda: f64,
    gt_value: f64,
    stabilizer: usize,
    orbit_size: usize,
    classification: String,
    eigenvalues: [f64; 3],
}

fn find_and_classify() -> Vec<CriticalPoint> {
    let ops = generate_oh_group();

    // Exploration with several grids
    let grids = [(12, 24), (24, 48), (48, 96), (96, 192)];
    let mut raw_points: Vec<[f64; 4]> = Vec::new();

    for &(n_theta, n_phi) in &grids {
        for i_theta in 0..=n_theta {
            let theta = std::f64::consts::PI * (i_theta as f64) / (n_theta as f64);
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let n_phi_eff = if sin_theta.abs() < 1e-10 { 1 } else { n_phi };

            for i_phi in 0..n_phi_eff {
                let phi = 2.0 * std::f64::consts::PI * (i_phi as f64) / (n_phi as f64);
                let u = sin_theta * phi.cos();
                let v = sin_theta * phi.sin();
                let w = cos_theta;
                let lam = lambda_from_grad(u, v, w);
                if let Some(x_ref) = newton_refine(&[u, v, w, lam], 1e-12, 50) {
                    let n2 = x_ref[0] * x_ref[0] + x_ref[1] * x_ref[1] + x_ref[2] * x_ref[2];
                    if (n2 - 1.0).abs() < 1e-8 {
                        raw_points.push(x_ref);
                    }
                }
            }
        }
    }

    // Deduplicate
    let mut unique: Vec<[f64; 4]> = Vec::new();
    for sol in &raw_points {
        let is_dup = unique.iter().any(|u| {
            (u[0] - sol[0]).abs() < 1e-6
                && (u[1] - sol[1]).abs() < 1e-6
                && (u[2] - sol[2]).abs() < 1e-6
        });
        if !is_dup {
            unique.push(*sol);
        }
    }

    // Classify each point
    let mut points: Vec<CriticalPoint> = Vec::new();
    for sol in &unique {
        let u = sol[0];
        let v = sol[1];
        let w = sol[2];
        let lam = sol[3];
        let g = gt(u, v, w);
        let stab = stabilizer_order(&ops, u, v, w);
        let orbit_size = 48 / stab;
        let (class, eigs) = classify_point(u, v, w, lam);

        points.push(CriticalPoint {
            u, v, w,
            lambda: lam,
            gt_value: g,
            stabilizer: stab,
            orbit_size,
            classification: class.to_string(),
            eigenvalues: eigs,
        });
    }

    points
}

// ============================================================
// FUNCTIONS EXPOSED TO PYTHON
// ============================================================

#[pyfunction]
fn analyze_py(py: Python<'_>) -> PyResult<Py<PyList>> {
    let points = find_and_classify();
    let list = PyList::empty(py);

    for p in points {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("u", p.u)?;
        dict.set_item("v", p.v)?;
        dict.set_item("w", p.w)?;
        dict.set_item("lambda", p.lambda)?;
        dict.set_item("gt", p.gt_value)?;
        dict.set_item("stabilizer", p.stabilizer)?;
        dict.set_item("orbit_size", p.orbit_size)?;
        dict.set_item("classification", p.classification)?;
        dict.set_item("eig1", p.eigenvalues[0])?;
        dict.set_item("eig2", p.eigenvalues[1])?;
        dict.set_item("eig3", p.eigenvalues[2])?;
        list.append(dict)?;
    }

    Ok(list.into())
}

#[pyfunction]
fn report_py() -> PyResult<String> {
    let points = find_and_classify();
    let mut s = String::new();

    s.push_str("============================================================\n");
    s.push_str(" COMPLETE ANALYSIS OF CRITICAL POINTS OF G_T\n");
    s.push_str("============================================================\n\n");

    s.push_str(&format!("Total unique critical points: {}\n\n", points.len()));

    // Group by value of G_T
    let mut by_gt: Vec<(f64, Vec<&CriticalPoint>)> = Vec::new();
    for p in &points {
        let key = (p.gt_value * 1e6).round() / 1e6;
        let found = by_gt.iter_mut().find(|(g, _)| (*g - key).abs() < 1e-6);
        match found {
            Some((_, v)) => v.push(p),
            None => by_gt.push((key, vec![p])),
        }
    }

    by_gt.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    s.push_str("------------------------------------------------------------\n");
    s.push_str(" SUMMARY BY TYPE\n");
    s.push_str("------------------------------------------------------------\n");
    s.push_str(&format!("{:<15} {:>12} {:>8} {:>10} {:>12}\n",
        "Type", "G_T", "Stab", "Orbit", "Count"));
    s.push_str("------------------------------------------------------------\n");

    for (g_val, group) in &by_gt {
        let p = group[0];
        let class = &p.classification;
        s.push_str(&format!("{:<15} {:>12.6} {:>8} {:>10} {:>12}\n",
            class, g_val, p.stabilizer, p.orbit_size, group.len()));
    }

    s.push_str("------------------------------------------------------------\n\n");

    // Detail of each orbit
    s.push_str("------------------------------------------------------------\n");
    s.push_str(" DETAIL OF EACH ORBIT (representative)\n");
    s.push_str("------------------------------------------------------------\n");
    s.push_str(&format!("{:<3} {:<10} {:<10} {:<10} {:<10} {:<8} {:<6} {:<16}\n",
        "idx", "u", "v", "w", "G_T", "Stab", "Orb", "Classification"));
    s.push_str("------------------------------------------------------------\n");

    for (i, (g_val, group)) in by_gt.iter().enumerate() {
        let p = group[0];
        s.push_str(&format!("{:<3} {:+.6} {:+.6} {:+.6} {:<10.6} {:<8} {:<6} {:<16}\n",
            i, p.u, p.v, p.w, g_val, p.stabilizer, p.orbit_size, &p.classification));
    }

    s.push_str("------------------------------------------------------------\n");

    // Verification: sum of orbit sizes
    let total_from_orbits: usize = by_gt.iter().map(|(_, g)| g.len()).sum();
    s.push_str(&format!("\nVerification: total points = {}\n", total_from_orbits));

    Ok(s)
}

#[pyfunction]
fn stabilize(u: f64, v: f64, w: f64) -> usize {
    let ops = generate_oh_group();
    stabilizer_order(&ops, u, v, w)
}

#[pyfunction]
fn orbit_size(u: f64, v: f64, w: f64) -> usize {
    let stab = stabilize(u, v, w);
    48 / stab
}

#[pyfunction]
fn classify(u: f64, v: f64, w: f64, lam: f64) -> (String, Vec<f64>) {
    let (class, eigs) = classify_point(u, v, w, lam);
    (class.to_string(), eigs.to_vec())
}

#[pymodule]
fn analysis_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_py, m)?)?;
    m.add_function(wrap_pyfunction!(report_py, m)?)?;
    m.add_function(wrap_pyfunction!(stabilize, m)?)?;
    m.add_function(wrap_pyfunction!(orbit_size, m)?)?;
    m.add_function(wrap_pyfunction!(classify, m)?)?;
    Ok(())
}