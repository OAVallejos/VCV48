use pyo3::prelude::*;
use pyo3::types::PyList;

// ============================================================
// 3x3 MATRIX IN ROW-MAJOR: [m00, m01, m02, m10, m11, m12, m20, m21, m22]
// ============================================================
type Mat3 = [f64; 9];

fn mat_identity() -> Mat3 {
    [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
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
    [
        m[0], m[3], m[6],
        m[1], m[4], m[7],
        m[2], m[5], m[8],
    ]
}

fn mat_apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn mat_sub(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [0.0; 9];
    for i in 0..9 {
        r[i] = a[i] - b[i];
    }
    r
}

fn mat_norm(m: &Mat3) -> f64 {
    let mut s = 0.0;
    for i in 0..9 {
        s += m[i] * m[i];
    }
    s.sqrt()
}

// ============================================================
// GENERATION OF THE 48 OPERATIONS OF O_h
// ============================================================
// The 48 operations of O_h are the orthogonal 3×3 matrices
// with entries in {-1, 0, +1}, with exactly one ±1 in each
// row and column.
//
// They are generated as: 6 permutations × 8 signs = 48 matrices.
// Of these, 24 have det = +1 (proper rotations, group O)
// and 24 have det = -1 (improper operations).

fn build_permutations() -> [[usize; 3]; 6] {
    [
        [0, 1, 2], // identity
        [0, 2, 1], // swap 1,2
        [1, 0, 2], // swap 0,1
        [1, 2, 0], // cycle (0,1,2)
        [2, 0, 1], // cycle (0,2,1)
        [2, 1, 0], // swap 0,2
    ]
}

fn build_all_signs() -> [[f64; 3]; 8] {
    [
        [ 1.0,  1.0,  1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [-1.0, -1.0, -1.0],
    ]
}

fn generate_oh_group() -> Vec<Mat3> {
    let perms = build_permutations();
    let signs = build_all_signs();
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

    // Verification
    let n_proper = ops.iter().filter(|m| (mat_det(m) - 1.0).abs() < 1e-10).count();
    let n_improper = ops.iter().filter(|m| (mat_det(m) + 1.0).abs() < 1e-10).count();

    assert_eq!(ops.len(), 48, "Expected 48 matrices, obtained {}", ops.len());
    assert_eq!(n_proper, 24, "Expected 24 proper rotations, obtained {}", n_proper);
    assert_eq!(n_improper, 24, "Expected 24 improper rotations, obtained {}", n_improper);

    ops
}

// ============================================================
// GROUP VERIFICATION
// ============================================================
fn verify_group(ops: &[Mat3]) -> (bool, String) {
    let id = mat_identity();
    let tol = 1e-10;

    // 1. The identity is in the group
    let has_id = ops.iter().any(|m| mat_norm(&mat_sub(m, &id)) < tol);
    if !has_id {
        return (false, "Identity not found".to_string());
    }

    // 2. All matrices are orthogonal: M^T M = I
    for (i, m) in ops.iter().enumerate() {
        let mt = mat_transpose(m);
        let prod = mat_mul(&mt, m);
        if mat_norm(&mat_sub(&prod, &id)) > tol {
            return (false, format!("Matrix {} is not orthogonal", i));
        }
    }

    // 3. Determinant ±1
    for (i, m) in ops.iter().enumerate() {
        let d = mat_det(m);
        if (d.abs() - 1.0).abs() > tol {
            return (false, format!("Matrix {} has det = {}", i, d));
        }
    }

    // 4. Closure under composition (complete verification 48×48 = 2304 products)
    for i in 0..48 {
        for j in 0..48 {
            let prod = mat_mul(&ops[i], &ops[j]);
            let found = ops.iter().any(|m| mat_norm(&mat_sub(m, &prod)) < tol);
            if !found {
                return (false, format!("Closure fails: {} × {} is not in the group", i, j));
            }
        }
    }

    (true, "Group O_h verified: 48 elements, orthogonal, closed, det = ±1".to_string())
}

// ============================================================
// COMPUTATION OF THE ACTUAL STABILIZER
// ============================================================
fn stabilizer_order(ops: &[Mat3], u: f64, v: f64, w: f64) -> usize {
    let point = [u, v, w];
    let tol = 1e-8;
    let mut count = 0;

    for m in ops {
        let t = mat_apply(m, point);
        let dx = t[0] - point[0];
        let dy = t[1] - point[1];
        let dz = t[2] - point[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 < tol * tol {
            count += 1;
        }
    }
    count
}

// ============================================================
// FUNCTIONS EXPOSED TO PYTHON
// ============================================================

/// Returns the number of elements of the group (must be 48).
#[pyfunction]
fn oh_order() -> usize {
    generate_oh_group().len()
}

/// Verifies the group O_h. Returns (bool, message).
#[pyfunction]
fn oh_verify() -> PyResult<(bool, String)> {
    let ops = generate_oh_group();
    Ok(verify_group(&ops))
}

/// Computes the stabilizer of a point (u, v, w).
#[pyfunction]
fn oh_stabilizer(u: f64, v: f64, w: f64) -> usize {
    let ops = generate_oh_group();
    stabilizer_order(&ops, u, v, w)
}

/// Verifies whether the stabilizer is trivial.
#[pyfunction]
fn oh_has_trivial_stabilizer(u: f64, v: f64, w: f64) -> bool {
    oh_stabilizer(u, v, w) == 1
}

/// Applies all 48 operations to a point and returns the unique orbit.
#[pyfunction]
fn oh_orbit(py: Python<'_>, u: f64, v: f64, w: f64) -> PyResult<Py<PyList>> {
    let ops = generate_oh_group();
    let point = [u, v, w];
    let tol = 1e-8;

    let mut orbit: Vec<[f64; 3]> = Vec::with_capacity(48);
    for m in &ops {
        let t = mat_apply(m, point);
        let is_new = orbit.iter().all(|p| {
            let dx = p[0] - t[0];
            let dy = p[1] - t[1];
            let dz = p[2] - t[2];
            dx * dx + dy * dy + dz * dz > tol * tol
        });
        if is_new {
            orbit.push(t);
        }
    }

    let list = PyList::empty(py);
    for p in orbit {
        list.append((p[0], p[1], p[2]))?;
    }
    Ok(list.into())
}

/// Complete report of the group.
#[pyfunction]
fn oh_report() -> PyResult<String> {
    let ops = generate_oh_group();
    let (ok, msg) = verify_group(&ops);
    let status = if ok { "OK" } else { "FAILED" };

    let mut s = String::new();
    s.push_str("=== Group O_h ===\n");
    s.push_str(&format!("Elements: {}\n", ops.len()));
    s.push_str(&format!("Verification: {}\n", status));
    s.push_str(&format!("Message: {}\n", msg));
    s.push_str("\nStabilizer of test points:\n");
    s.push_str(&format!("  (1, 0, 0)                -> {}\n", stabilizer_order(&ops, 1.0, 0.0, 0.0)));
    s.push_str(&format!("  (1, 1, 0)/sqrt(2)        -> {}\n", stabilizer_order(&ops, 0.7071067811865476, 0.7071067811865476, 0.0)));
    s.push_str(&format!("  (1, 1, 1)/sqrt(3)        -> {}\n", stabilizer_order(&ops, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258)));
    s.push_str(&format!("  (0.2824, 0.2824, 0.9169) -> {}\n", stabilizer_order(&ops, 0.2824, 0.2824, 0.9169)));
    s.push_str(&format!("  (0.3, 0.4, 0.8660)       -> {}\n", stabilizer_order(&ops, 0.3, 0.4, 0.8660254037844386)));
    Ok(s)
}

#[pymodule]
fn oh_group(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(oh_order, m)?)?;
    m.add_function(wrap_pyfunction!(oh_verify, m)?)?;
    m.add_function(wrap_pyfunction!(oh_stabilizer, m)?)?;
    m.add_function(wrap_pyfunction!(oh_has_trivial_stabilizer, m)?)?;
    m.add_function(wrap_pyfunction!(oh_orbit, m)?)?;
    m.add_function(wrap_pyfunction!(oh_report, m)?)?;
    Ok(())
}