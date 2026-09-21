use pyo3::prelude::*;
use pyo3::types::PyDict;

// ============================================================
// G_T FUNCTION AND ITS ANALYTICAL DERIVATIVES
// ============================================================
//
// Sigma6(u,v,w) = u^6 + v^6 + w^6
//
// G_T(u,v,w) = u^2 (u^4 - Sigma6)^2
//            + v^2 (v^4 - Sigma6)^2
//            + w^2 (w^4 - Sigma6)^2
//
// Let A = u^4 - Sigma6,  B = v^4 - Sigma6,  C = w^4 - Sigma6
//
// dSigma6/du = 6 u^5,  dSigma6/dv = 6 v^5,  dSigma6/dw = 6 w^5
//
// dA/du = 4 u^3 - 6 u^5,  dA/dv = -6 v^5,  dA/dw = -6 w^5
// dB/du = -6 u^5,         dB/dv = 4 v^3 - 6 v^5,  dB/dw = -6 w^5
// dC/du = -6 u^5,         dC/dv = -6 v^5,         dC/dw = 4 w^3 - 6 w^5

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

/// Analytical gradient of G_T.
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

// ============================================================
// ANALYTICAL HESSIAN OF G_T 
// ============================================================
// Returns [H_uu, H_uv, H_uw, H_vv, H_vw, H_ww]
//
// Derivation:
//   dG_T/du = 2u A^2 + 2u^2 A da_du + 2v^2 B db_du + 2w^2 C dc_du
//
//   d2G_T/du2 = d/du [2u A^2] + d/du [2u^2 A da_du]
//             + d/du [2v^2 B db_du] + d/du [2w^2 C dc_du]
//
//   d/du [2u A^2]        = 2 A^2 + 4u A da_du
//   d/du [2u^2 A da_du]  = 4u A da_du + 2u^2 (da_du)^2 + 2u^2 A d2a_du2
//   d/du [2v^2 B db_du]  = 2v^2 (db_du)^2 + 2v^2 B d2b_du2
//   d/du [2w^2 C dc_du]  = 2w^2 (dc_du)^2 + 2w^2 C d2c_du2
//
//   Where d2a_du2 = 12u^2 - 30u^4, d2b_du2 = -30u^4, d2c_du2 = -30u^4

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

    // First derivatives
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

    // Second diagonal derivatives of Sigma6
    let d2s6_du2 = 30.0 * u4;
    let d2s6_dv2 = 30.0 * v4;
    let d2s6_dw2 = 30.0 * w4;

    // Second diagonal derivatives of A, B, C
    let d2a_du2 = 12.0 * u2 - 30.0 * u4;
    let d2b_dv2 = 12.0 * v2 - 30.0 * v4;
    let d2c_dw2 = 12.0 * w2 - 30.0 * w4;

    // Second cross derivatives of A, B, C with respect to the other variables
    // (d2A/dv2, d2A/dw2, d2B/du2, etc.)
    let d2a_dv2 = -d2s6_dv2;
    let d2a_dw2 = -d2s6_dw2;
    let d2b_du2 = -d2s6_du2;
    let d2b_dw2 = -d2s6_dw2;
    let d2c_du2 = -d2s6_du2;
    let d2c_dv2 = -d2s6_dv2;

    // ============================================================
    // H_uu = d^2 G_T / du^2
    // ============================================================
    let h_uu = 2.0 * a * a
        + 4.0 * u * a * da_du
        + 4.0 * u * a * da_du
        + 2.0 * u2 * da_du * da_du
        + 2.0 * u2 * a * d2a_du2
        + 2.0 * v2 * db_du * db_du
        + 2.0 * v2 * b * d2b_du2
        + 2.0 * w2 * dc_du * dc_du
        + 2.0 * w2 * c * d2c_du2;

    // ============================================================
    // H_vv = d^2 G_T / dv^2
    // ============================================================
    let h_vv = 2.0 * b * b
        + 4.0 * v * b * db_dv
        + 4.0 * v * b * db_dv
        + 2.0 * v2 * db_dv * db_dv
        + 2.0 * v2 * b * d2b_dv2
        + 2.0 * u2 * da_dv * da_dv
        + 2.0 * u2 * a * d2a_dv2
        + 2.0 * w2 * dc_dv * dc_dv
        + 2.0 * w2 * c * d2c_dv2;

    // ============================================================
    // H_ww = d^2 G_T / dw^2
    // ============================================================
    let h_ww = 2.0 * c * c
        + 4.0 * w * c * dc_dw
        + 4.0 * w * c * dc_dw
        + 2.0 * w2 * dc_dw * dc_dw
        + 2.0 * w2 * c * d2c_dw2
        + 2.0 * u2 * da_dw * da_dw
        + 2.0 * u2 * a * d2a_dw2
        + 2.0 * v2 * db_dw * db_dw
        + 2.0 * v2 * b * d2b_dw2;

    // ============================================================
    // H_uv = d^2 G_T / du dv
    // ============================================================
    // d/dv [2u A^2]        = 4u A da_dv
    // d/dv [2u^2 A da_du]  = 2u^2 da_dv da_du  (d2a_dudv = 0)
    // d/dv [2v^2 B db_du]  = 4v B db_du + 2v^2 db_dv db_du  (d2b_dudv = 0)
    // d/dv [2w^2 C dc_du]  = 2w^2 dc_dv dc_du  (d2c_dudv = 0)
    let h_uv = 4.0 * u * a * da_dv
        + 2.0 * u2 * da_dv * da_du
        + 4.0 * v * b * db_du
        + 2.0 * v2 * db_dv * db_du
        + 2.0 * w2 * dc_dv * dc_du;

    // ============================================================
    // H_uw = d^2 G_T / du dw
    // ============================================================
    let h_uw = 4.0 * u * a * da_dw
        + 2.0 * u2 * da_dw * da_du
        + 4.0 * w * c * dc_du
        + 2.0 * w2 * dc_dw * dc_du
        + 2.0 * v2 * db_dw * db_du;

    // ============================================================
    // H_vw = d^2 G_T / dv dw
    // ============================================================
    let h_vw = 4.0 * v * b * db_dw
        + 2.0 * v2 * db_dw * db_dv
        + 4.0 * w * c * dc_dv
        + 2.0 * w2 * dc_dw * dc_dv
        + 2.0 * u2 * da_dw * da_dv;

    [h_uu, h_uv, h_uw, h_vv, h_vw, h_ww]
}

// ============================================================
// LAGRANGE SYSTEM
// ============================================================
// Unknowns: x = (u, v, w, lambda)
//
// F1 = dG_T/du - 2 lambda u = 0
// F2 = dG_T/dv - 2 lambda v = 0
// F3 = dG_T/dw - 2 lambda w = 0
// F4 = u^2 + v^2 + w^2 - 1 = 0

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

// ============================================================
// ANALYTICAL JACOBIAN OF THE LAGRANGE SYSTEM
// ============================================================
// J[i][j] = dF_i / dx_j
// Row-major 4x4: [J00, J01, J02, J03, J10, J11, J12, J13, ...]

pub fn lagrange_jacobian_analytic(x: &[f64; 4]) -> [f64; 16] {
    let u = x[0];
    let v = x[1];
    let w = x[2];
    let lam = x[3];

    let h = hess_gt(u, v, w);

    [
        h[0] - 2.0 * lam,  h[1],               h[2],               -2.0 * u,
        h[1],              h[3] - 2.0 * lam,   h[4],               -2.0 * v,
        h[2],              h[4],               h[5] - 2.0 * lam,   -2.0 * w,
        2.0 * u,           2.0 * v,            2.0 * w,            0.0,
    ]
}

// ============================================================
// FUNCTIONS EXPOSED TO PYTHON
// ============================================================

#[pyfunction]
fn gt_value(u: f64, v: f64, w: f64) -> f64 {
    gt(u, v, w)
}

#[pyfunction]
fn gt_gradient(py: Python<'_>, u: f64, v: f64, w: f64) -> PyResult<Py<PyDict>> {
    let g = grad_gt(u, v, w);
    let dict = PyDict::new(py);
    dict.set_item("du", g[0])?;
    dict.set_item("dv", g[1])?;
    dict.set_item("dw", g[2])?;
    Ok(dict.into())
}

#[pyfunction]
fn gt_hessian(py: Python<'_>, u: f64, v: f64, w: f64) -> PyResult<Py<PyDict>> {
    let h = hess_gt(u, v, w);
    let dict = PyDict::new(py);
    dict.set_item("uu", h[0])?;
    dict.set_item("uv", h[1])?;
    dict.set_item("uw", h[2])?;
    dict.set_item("vv", h[3])?;
    dict.set_item("vw", h[4])?;
    dict.set_item("ww", h[5])?;
    Ok(dict.into())
}

#[pyfunction]
fn lagrange_residual(py: Python<'_>, u: f64, v: f64, w: f64, lam: f64) -> PyResult<Py<PyDict>> {
    let f = lagrange_system(&[u, v, w, lam]);
    let dict = PyDict::new(py);
    dict.set_item("F1", f[0])?;
    dict.set_item("F2", f[1])?;
    dict.set_item("F3", f[2])?;
    dict.set_item("F4", f[3])?;
    let norm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2] + f[3] * f[3]).sqrt();
    dict.set_item("norm", norm)?;
    Ok(dict.into())
}

/// Verifies the analytical gradient against finite differences.
#[pyfunction]
fn gt_verify_gradient(u: f64, v: f64, w: f64) -> PyResult<(bool, f64)> {
    let eps = 1e-7;
    let g_analytic = grad_gt(u, v, w);

    let g_num = [
        (gt(u + eps, v, w) - gt(u - eps, v, w)) / (2.0 * eps),
        (gt(u, v + eps, w) - gt(u, v - eps, w)) / (2.0 * eps),
        (gt(u, v, w + eps) - gt(u, v, w - eps)) / (2.0 * eps),
    ];

    let err = ((g_analytic[0] - g_num[0]).powi(2)
        + (g_analytic[1] - g_num[1]).powi(2)
        + (g_analytic[2] - g_num[2]).powi(2))
    .sqrt();

    Ok((err < 1e-5, err))
}

/// Verifies the analytical Jacobian against finite differences.
#[pyfunction]
fn gt_verify_jacobian(u: f64, v: f64, w: f64, lam: f64) -> PyResult<(bool, f64)> {
    let eps = 1e-7;
    let x = [u, v, w, lam];
    let j_analytic = lagrange_jacobian_analytic(&x);

    let mut j_num = [0.0; 16];
    for j in 0..4 {
        let mut xp = x;
        let mut xm = x;
        xp[j] += eps;
        xm[j] -= eps;
        let fp = lagrange_system(&xp);
        let fm = lagrange_system(&xm);
        for i in 0..4 {
            j_num[i * 4 + j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
    }

    let mut err2 = 0.0;
    for i in 0..16 {
        err2 += (j_analytic[i] - j_num[i]).powi(2);
    }
    let err = err2.sqrt();

    Ok((err < 1e-5, err))
}

/// Verifies that the point (u, v, w) is on the sphere.
#[pyfunction]
fn gt_on_sphere(u: f64, v: f64, w: f64) -> bool {
    let n2 = u * u + v * v + w * w;
    (n2 - 1.0).abs() < 1e-8
}

/// Returns the Jacobian as a list of 16 elements (row-major).
#[pyfunction]
fn lagrange_jacobian_flat(u: f64, v: f64, w: f64, lam: f64) -> Vec<f64> {
    let j = lagrange_jacobian_analytic(&[u, v, w, lam]);
    j.to_vec()
}

/// Returns the residual as a vector of 4 elements.
#[pyfunction]
fn lagrange_system_flat(u: f64, v: f64, w: f64, lam: f64) -> Vec<f64> {
    let f = lagrange_system(&[u, v, w, lam]);
    f.to_vec()
}

#[pymodule]
fn gt_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gt_value, m)?)?;
    m.add_function(wrap_pyfunction!(gt_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(gt_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(lagrange_residual, m)?)?;
    m.add_function(wrap_pyfunction!(gt_verify_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(gt_verify_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(gt_on_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(lagrange_jacobian_flat, m)?)?;
    m.add_function(wrap_pyfunction!(lagrange_system_flat, m)?)?;
    Ok(())
}