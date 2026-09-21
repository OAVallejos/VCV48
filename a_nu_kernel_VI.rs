use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

// ============================================================
// KERNEL 6/6: A_nu — SIDEREAL MODULATION OF NEUTRINOS
// ============================================================
//
// Computes the sidereal modulation amplitude of the M87 neutrino flux
// according to the VCV48 model (Annex VI).
//

//   1. The jet vector is constructed from observational data:
//      RA, Dec of M87, position angle PA=288 deg, and 3D angle
//      with the line of sight theta_jet = 17 deg (Walker 2018).
//      The previous vector (-0.9672, -0.1308, 0.2176) was actually
//      the line-of-sight vector, mislabeled as the jet.
//   2. The function gt_extrema_on_circle is added, which computes
//      G_T_max and G_T_min over the jet circle.
//
// All physical parameters come from public sources:
//   - eta_1, eta_2: 19487810.tex (App. A, calibration of 131 galaxies)
//   - A_Rabi_max: corrected Kernel 5/5 (with exact S)
//   - M87 data: NED (RA, Dec) + Walker et al. 2018 (PA, theta_jet)
//   - T_sid: sidereal day (23h 56m 4.0905s)
//
// Everything is direct and reproducible calculation.

// ============================================================
// SECTION 1: PHYSICAL CONSTANTS (all from explicit sources)
// ============================================================

/// Isotropic coefficient of the O_h vacuum (19487810.tex, App. A)
const ETA1: f64 = 0.25229889;

/// Anisotropic coefficient of the O_h vacuum (19487810.tex, App. A)
const ETA2: f64 = 0.15000000;

// --- Observational data of M87 ---

/// RA of M87 in decimal degrees (NED: 12h30m49.42338s)
const RA_M87_DEG: f64 = 187.7059308;

/// Dec of M87 in decimal degrees (NED: +12d23'28.0439")
const DEC_M87_DEG: f64 = 12.39112330;

/// Position angle of the M87 jet in degrees (Walker 2018)
const PA_JET_DEG: f64 = 288.0;

/// 3D angle of the jet with the line of sight in degrees (Walker 2018)
const THETA_JET_DEG: f64 = 17.0;

/// Earth's sidereal period in seconds (23h 56m 4.0905s)
const T_SID: f64 = 86164.0905;

/// Upper bound of A_Rabi at the global maximum of G_T.
/// EXACT value from the corrected Kernel 5/5.
const A_RABI_MAX_AT_GTMAX: f64 = 0.022893155543155;

/// Sampled bound of A_nu over 5000 random orientations.
const A_NU_MAX_SAMPLED: f64 = 0.0244839380;

/// Components of the global maximum of G_T (Kernel 5/5)
const K_MAX_K5: [f64; 3] = [
    0.283299274553765,
    0.283299274553765,
    0.916233071917087,
];

// ============================================================
// SECTION 2: 3x3 MATRIX IN ROW-MAJOR
// ============================================================

type Mat3 = [f64; 9];

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

/// Rotation around the z axis.
fn mat_rz(angle: f64) -> Mat3 {
    let (c, s) = (angle.cos(), angle.sin());
    [c, -s, 0.0,
     s,  c, 0.0,
     0.0, 0.0, 1.0]
}

/// Rotation around the x axis.
fn mat_rx(angle: f64) -> Mat3 {
    let (c, s) = (angle.cos(), angle.sin());
    [1.0, 0.0, 0.0,
     0.0, c, -s,
     0.0, s,  c]
}

/// Euler matrix R_EC(alpha, beta, gamma) with ZXZ convention.
fn euler_matrix(alpha: f64, beta: f64, gamma: f64) -> Mat3 {
    let rz_a = mat_rz(alpha);
    let rx_b = mat_rx(beta);
    let rz_g = mat_rz(gamma);
    mat_mul(&rz_a, &mat_mul(&rx_b, &rz_g))
}

fn vec_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn vec_normalize(v: [f64; 3]) -> [f64; 3] {
    let n = vec_norm(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn vec_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ============================================================
// SECTION 3: GEOMETRY OF THE M87 JET
// ============================================================
//
// The M87 jet is at theta_jet = 17 deg from the line of sight,
// with position angle PA = 288 deg (from north toward east).
//
// We construct the jet vector in the celestial frame as:
//     k_jet = cos(theta_jet) * n_LOS + sin(theta_jet) * t
// where:
//     n_LOS = direction from Earth toward M87
//     t     = tangent vector on the sky in the PA direction

/// Direction from Earth toward M87 (normalized).
fn n_los_vec() -> [f64; 3] {
    let ra = RA_M87_DEG.to_radians();
    let dec = DEC_M87_DEG.to_radians();
    let x = dec.cos() * ra.cos();
    let y = dec.cos() * ra.sin();
    let z = dec.sin();
    vec_normalize([x, y, z])
}

/// Tangent vector on the sky in the direction of the position angle PA.
/// Convention: PA measured from north toward east.
fn tangent_pa_vec(n_los: [f64; 3]) -> [f64; 3] {
    let ra = RA_M87_DEG.to_radians();
    let dec = DEC_M87_DEG.to_radians();
    let pa = PA_JET_DEG.to_radians();

    // Unit vectors of the equatorial frame
    let e_north = [
        -dec.sin() * ra.cos(),
        -dec.sin() * ra.sin(),
        dec.cos(),
    ];
    let e_east = [-ra.sin(), ra.cos(), 0.0];

    // Vector in the PA direction
    let t = [
        pa.cos() * e_north[0] + pa.sin() * e_east[0],
        pa.cos() * e_north[1] + pa.sin() * e_east[1],
        pa.cos() * e_north[2] + pa.sin() * e_east[2],
    ];

    // Project onto the plane perpendicular to n_LOS (for safety)
    let proj = vec_dot(t, n_los);
    let t_perp = [
        t[0] - proj * n_los[0],
        t[1] - proj * n_los[1],
        t[2] - proj * n_los[2],
    ];

    vec_normalize(t_perp)
}

/// Direction of the M87 jet in the celestial frame (normalized).
fn k_jet_E_vec() -> [f64; 3] {
    let n_los = n_los_vec();
    let t = tangent_pa_vec(n_los);
    let theta_jet = THETA_JET_DEG.to_radians();

    let k = [
        theta_jet.cos() * n_los[0] + theta_jet.sin() * t[0],
        theta_jet.cos() * n_los[1] + theta_jet.sin() * t[1],
        theta_jet.cos() * n_los[2] + theta_jet.sin() * t[2],
    ];

    vec_normalize(k)
}

/// Direction of the jet in the crystallographic frame at sidereal time t.
fn k_jet_C(t: f64, r_ec: &Mat3, k_jet_e: &[f64; 3], omega_oplus: f64) -> [f64; 3] {
    let r_t = mat_rz(omega_oplus * t);
    let k_e_rot = mat_apply(&r_t, *k_jet_e);
    let r_ec_t = mat_transpose(r_ec);
    mat_apply(&r_ec_t, k_e_rot)
}

// ============================================================
// SECTION 4: G_T, S, |T|^2, A_RABI
// ============================================================

#[inline]
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

/// phi_6 = u^6 + v^6 + w^6 - 3/5
fn phi6(u: f64, v: f64, w: f64) -> f64 {
    sigma6(u, v, w) - 0.6
}

/// S = eta_1 + eta_2 * phi_6  (EXACT)
fn s_of_k(u: f64, v: f64, w: f64) -> f64 {
    ETA1 + ETA2 * phi6(u, v, w)
}

/// |T|^2 = eta_2^2 * G_T
fn t2_of_k(u: f64, v: f64, w: f64) -> f64 {
    ETA2 * ETA2 * gt(u, v, w)
}

/// A_Rabi = |T|^2 / (|T|^2 + S^2)
fn a_rabi_of_k(u: f64, v: f64, w: f64) -> f64 {
    let s = s_of_k(u, v, w);
    let t2 = t2_of_k(u, v, w);
    t2 / (t2 + s * s)
}

// ============================================================
// SECTION 5: COMPUTATION OF A_nu FOR ONE ORIENTATION
// ============================================================

fn a_nu_for_orientation(
    alpha: f64,
    beta: f64,
    gamma: f64,
    n_t: usize,
    omega_oplus: f64,
    k_jet_e: &[f64; 3],
) -> (f64, Vec<f64>) {
    let r_ec = euler_matrix(alpha, beta, gamma);
    let mut a_rabi = Vec::with_capacity(n_t);

    for i in 0..n_t {
        let ti = (i as f64) * T_SID / (n_t as f64);
        let kc = k_jet_C(ti, &r_ec, k_jet_e, omega_oplus);
        a_rabi.push(a_rabi_of_k(kc[0], kc[1], kc[2]));
    }

    let a_max = a_rabi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let a_min = a_rabi.iter().cloned().fold(f64::INFINITY, f64::min);

    (a_max - a_min, a_rabi)
}

/// Computes G_T_max and G_T_min over the jet circle for one orientation.
/// Returns (gt_max, gt_min, gt_median, u_at_max, v_at_max, w_at_max).
fn gt_extrema_on_circle(
    alpha: f64,
    beta: f64,
    gamma: f64,
    n_t: usize,
    omega_oplus: f64,
    k_jet_e: &[f64; 3],
) -> (f64, f64, f64, [f64; 3]) {
    let r_ec = euler_matrix(alpha, beta, gamma);
    let mut gt_values = Vec::with_capacity(n_t);
    let mut k_at_max = [0.0; 3];
    let mut gt_max = f64::NEG_INFINITY;
    let mut gt_min = f64::INFINITY;

    for i in 0..n_t {
        let ti = (i as f64) * T_SID / (n_t as f64);
        let kc = k_jet_C(ti, &r_ec, k_jet_e, omega_oplus);
        let g = gt(kc[0], kc[1], kc[2]);
        gt_values.push(g);
        if g > gt_max {
            gt_max = g;
            k_at_max = kc;
        }
        if g < gt_min {
            gt_min = g;
        }
    }

    let mut sorted = gt_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gt_median = sorted[n_t / 2];

    (gt_max, gt_min, gt_median, k_at_max)
}

// ============================================================
// SECTION 6: UNIFORM SAMPLING IN SO(3) — SHOEMAKE (1992)
// ============================================================

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_orientation(&mut self) -> (f64, f64, f64) {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let u3 = self.next_f64();
        (
            2.0 * PI * u1,
            (2.0 * u2 - 1.0).acos(),
            2.0 * PI * u3,
        )
    }
}

// ============================================================
// SECTION 7: HARMONIC STRUCTURE (DIRECT FFT)
// ============================================================

fn harmonic_structure(a_rabi: &[f64], n_max: usize) -> Vec<f64> {
    let n = a_rabi.len();
    let mean: f64 = a_rabi.iter().sum::<f64>() / n as f64;
    let mut amps = vec![0.0; n_max + 1];

    for k in 1..=n_max {
        let omega = 2.0 * PI * (k as f64) / (n as f64);
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, &ai) in a_rabi.iter().enumerate() {
            let ai_c = ai - mean;
            let angle = omega * (i as f64);
            re += ai_c * angle.cos();
            im -= ai_c * angle.sin();
        }
        let mag = (re * re + im * im).sqrt();
        amps[k] = 2.0 * mag / (n as f64);
    }

    amps
}

// ============================================================
// SECTION 8: FUNCTIONS EXPOSED TO PYTHON
// ============================================================

#[pyfunction]
fn a_nu_canonical(n_t: usize) -> f64 {
    let omega_oplus = 2.0 * PI / T_SID;
    let k_jet_e = k_jet_E_vec();
    let (a_nu, _) = a_nu_for_orientation(0.0, 0.0, 0.0, n_t, omega_oplus, &k_jet_e);
    a_nu
}

#[pyfunction]
fn a_nu_for_euler(alpha: f64, beta: f64, gamma: f64, n_t: usize) -> f64 {
    let omega_oplus = 2.0 * PI / T_SID;
    let k_jet_e = k_jet_E_vec();
    let (a_nu, _) = a_nu_for_orientation(alpha, beta, gamma, n_t, omega_oplus, &k_jet_e);
    a_nu
}

#[pyfunction]
fn a_nu_average(py: Python<'_>, n_orient: usize, n_t: usize, seed: u64) -> PyResult<Py<PyDict>> {
    let omega_oplus = 2.0 * PI / T_SID;
    let k_jet_e = k_jet_E_vec();
    let mut rng = Lcg::new(seed);

    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut a_min = f64::INFINITY;
    let mut a_max = f64::NEG_INFINITY;
    let mut samples = Vec::with_capacity(n_orient);

    for _ in 0..n_orient {
        let (a, b, g) = rng.next_orientation();
        let (a_nu, _) = a_nu_for_orientation(a, b, g, n_t, omega_oplus, &k_jet_e);
        samples.push(a_nu);
        sum += a_nu;
        sumsq += a_nu * a_nu;
        if a_nu < a_min { a_min = a_nu; }
        if a_nu > a_max { a_max = a_nu; }
    }

    let mean = sum / n_orient as f64;
    let var = sumsq / n_orient as f64 - mean * mean;
    let std = var.sqrt();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[n_orient / 2];

    let dict = PyDict::new(py);
    dict.set_item("mean", mean)?;
    dict.set_item("median", median)?;
    dict.set_item("std", std)?;
    dict.set_item("min", a_min)?;
    dict.set_item("max", a_max)?;
    dict.set_item("n_orient", n_orient)?;
    dict.set_item("n_t", n_t)?;
    Ok(dict.into())
}

#[pyfunction]
fn a_nu_harmonics(alpha: f64, beta: f64, gamma: f64, n_t: usize) -> Vec<f64> {
    let omega_oplus = 2.0 * PI / T_SID;
    let k_jet_e = k_jet_E_vec();
    let (_, a_rabi) = a_nu_for_orientation(alpha, beta, gamma, n_t, omega_oplus, &k_jet_e);
    let amps = harmonic_structure(&a_rabi, 12);
    amps[1..=12].to_vec()
}

/// Reports the jet geometry: n_LOS, k_jet, and the angle between them.
#[pyfunction]
fn jet_geometry(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let n_los = n_los_vec();
    let k_jet = k_jet_E_vec();
    let cos_theta = vec_dot(n_los, k_jet).clamp(-1.0, 1.0);
    let theta_deg = cos_theta.acos().to_degrees();

    let dict = PyDict::new(py);
    dict.set_item("n_los_x", n_los[0])?;
    dict.set_item("n_los_y", n_los[1])?;
    dict.set_item("n_los_z", n_los[2])?;
    dict.set_item("k_jet_x", k_jet[0])?;
    dict.set_item("k_jet_y", k_jet[1])?;
    dict.set_item("k_jet_z", k_jet[2])?;
    dict.set_item("cos_theta", cos_theta)?;
    dict.set_item("theta_deg", theta_deg)?;
    dict.set_item("norm_n_los", vec_norm(n_los))?;
    dict.set_item("norm_k_jet", vec_norm(k_jet))?;
    Ok(dict.into())
}

/// Computes G_T_max and G_T_min over the jet circle.
#[pyfunction]
fn gt_circle_extrema(py: Python<'_>, alpha: f64, beta: f64, gamma: f64, n_t: usize) -> PyResult<Py<PyDict>> {
    let omega_oplus = 2.0 * PI / T_SID;
    let k_jet_e = k_jet_E_vec();
    let (gt_max, gt_min, gt_median, k_at_max) =
        gt_extrema_on_circle(alpha, beta, gamma, n_t, omega_oplus, &k_jet_e);

    let d = PyDict::new(py);
    d.set_item("gt_max", gt_max)?;
    d.set_item("gt_min", gt_min)?;
    d.set_item("gt_median", gt_median)?;
    d.set_item("u_at_max", k_at_max[0])?;
    d.set_item("v_at_max", k_at_max[1])?;
    d.set_item("w_at_max", k_at_max[2])?;
    Ok(d.into())
}

/// Verifies consistency with the maximum of Kernel 5/5 (corrected).
#[pyfunction]
fn a_nu_verify_kernel5(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let u = K_MAX_K5[0];
    let v = K_MAX_K5[1];
    let w = K_MAX_K5[2];
    let gt_max = gt(u, v, w);
    let phi6_max = phi6(u, v, w);
    let s_max = s_of_k(u, v, w);
    let t2_max = t2_of_k(u, v, w);
    let a_rabi = a_rabi_of_k(u, v, w);

    let dict = PyDict::new(py);
    dict.set_item("u", u)?;
    dict.set_item("v", v)?;
    dict.set_item("w", w)?;
    dict.set_item("gt_max", gt_max)?;
    dict.set_item("phi6_max", phi6_max)?;
    dict.set_item("s_max", s_max)?;
    dict.set_item("t2_max", t2_max)?;
    dict.set_item("a_rabi", a_rabi)?;
    dict.set_item("a_rabi_max_kernel5", A_RABI_MAX_AT_GTMAX)?;
    dict.set_item("diff", (a_rabi - A_RABI_MAX_AT_GTMAX).abs())?;
    Ok(dict.into())
}

/// Complete report of kernel 6.
#[pyfunction]
fn a_nu_report() -> PyResult<String> {
    let mut s = String::new();
    let omega_oplus = 2.0 * PI / T_SID;
    let k_jet_e = k_jet_E_vec();
    let n_los = n_los_vec();

    s.push_str("============================================================\n");
    s.push_str(" KERNEL 6/6: A_nu — SIDEREAL MODULATION OF NEUTRINOS\n");
    s.push_str("============================================================\n\n");

    s.push_str("GEOMETRY OF THE M87 JET:\n");
    s.push_str(&format!("  RA_M87  = {:.7} deg\n", RA_M87_DEG));
    s.push_str(&format!("  Dec_M87 = {:.7} deg\n", DEC_M87_DEG));
    s.push_str(&format!("  PA_jet  = {} deg (Walker 2018)\n", PA_JET_DEG));
    s.push_str(&format!("  theta_jet (3D with LOS) = {} deg\n", THETA_JET_DEG));
    s.push_str(&format!("  n_LOS   = ({:+.9}, {:+.9}, {:+.9})\n",
        n_los[0], n_los[1], n_los[2]));
    s.push_str(&format!("  k_jet^E = ({:+.9}, {:+.9}, {:+.9})\n",
        k_jet_e[0], k_jet_e[1], k_jet_e[2]));
    let cos_theta = vec_dot(n_los, k_jet_e).clamp(-1.0, 1.0);
    s.push_str(&format!("  cos(jet-LOS angle) = {:.9}\n", cos_theta));
    s.push_str(&format!("  jet-LOS angle = {:.4} deg\n", cos_theta.acos().to_degrees()));
    s.push_str("\n");

    s.push_str("PHYSICAL PARAMETERS:\n");
    s.push_str(&format!("  eta_1 = {:.8}\n", ETA1));
    s.push_str(&format!("  eta_2 = {:.8}\n", ETA2));
    s.push_str(&format!("  T_sid = {} s\n", T_SID));
    s.push_str(&format!("  omega_oplus = {:.6e} rad/s\n", omega_oplus));
    s.push_str(&format!("  A_Rabi_max (K5 corrected) = {:.15}\n", A_RABI_MAX_AT_GTMAX));
    s.push_str(&format!("  A_nu_max (sampled SO3) = {:.10}\n", A_NU_MAX_SAMPLED));
    s.push_str("\n");

    // Verification Kernel 5/5
    s.push_str("------------------------------------------------------------\n");
    s.push_str(" VERIFICATION: maximum of Kernel 5/5 (corrected)\n");
    s.push_str("------------------------------------------------------------\n");
    let u = K_MAX_K5[0];
    let v = K_MAX_K5[1];
    let w = K_MAX_K5[2];
    s.push_str(&format!("  k_max         = ({:.9}, {:.9}, {:.9})\n", u, v, w));
    s.push_str(&format!("  G_T(k_max)    = {:.15}\n", gt(u, v, w)));
    s.push_str(&format!("  A_Rabi(k_max) = {:.15}\n", a_rabi_of_k(u, v, w)));
    s.push_str(&format!("  Difference    = {:.3e}\n",
        (a_rabi_of_k(u, v, w) - A_RABI_MAX_AT_GTMAX).abs()));
    s.push_str("\n");

    // Canonical case: A_nu
    s.push_str("------------------------------------------------------------\n");
    s.push_str(" CANONICAL CASE (alpha = beta = gamma = 0)\n");
    s.push_str("------------------------------------------------------------\n");
    let (a_nu_canon, a_rabi_canon) =
        a_nu_for_orientation(0.0, 0.0, 0.0, 4096, omega_oplus, &k_jet_e);
    let a_max_canon = a_rabi_canon.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let a_min_canon = a_rabi_canon.iter().cloned().fold(f64::INFINITY, f64::min);
    s.push_str(&format!("  A_Rabi_max = {:.10}\n", a_max_canon));
    s.push_str(&format!("  A_Rabi_min = {:.10}\n", a_min_canon));
    s.push_str(&format!("  A_nu_canonical = {:.10} ({:.4}%)\n", a_nu_canon, 100.0 * a_nu_canon));
    s.push_str("\n");

    // Canonical case: G_T extrema
    s.push_str("------------------------------------------------------------\n");
    s.push_str(" CANONICAL CASE: G_T over the jet circle\n");
    s.push_str("------------------------------------------------------------\n");
    let (gt_max_c, gt_min_c, gt_med_c, k_at_max_c) =
        gt_extrema_on_circle(0.0, 0.0, 0.0, 4096, omega_oplus, &k_jet_e);
    s.push_str(&format!("  G_T_max = {:.10} (at k = ({:+.6}, {:+.6}, {:+.6}))\n",
        gt_max_c, k_at_max_c[0], k_at_max_c[1], k_at_max_c[2]));
    s.push_str(&format!("  G_T_min = {:.10}\n", gt_min_c));
    s.push_str(&format!("  G_T_median = {:.10}\n", gt_med_c));
    s.push_str(&format!("  Range = {:.10}\n", gt_max_c - gt_min_c));
    s.push_str("\n");

    // Harmonic structure
    s.push_str("Harmonic structure (normalized Fourier amplitude):\n");
    let harm = harmonic_structure(&a_rabi_canon, 12);
    for n in 1..=12 {
        let allowed = n == 2 || n == 3 || n == 4 || n == 6 || n == 8 || n == 12;
        let marker = if allowed { " <-- O_h allows" } else { "" };
        s.push_str(&format!("  n = {:2}: {:.6e}{}\n", n, harm[n], marker));
    }
    s.push_str("\n");

    // Average over SO(3)
    s.push_str("------------------------------------------------------------\n");
    s.push_str(" AVERAGE OVER 5000 RANDOM ORIENTATIONS\n");
    s.push_str("------------------------------------------------------------\n");
    let mut rng = Lcg::new(48);
    let n_orient = 5000usize;
    let n_t = 1024usize;
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut a_min = f64::INFINITY;
    let mut a_max = f64::NEG_INFINITY;
    let mut samples = Vec::with_capacity(n_orient);
    for _ in 0..n_orient {
        let (a, b, g) = rng.next_orientation();
        let (a_nu, _) = a_nu_for_orientation(a, b, g, n_t, omega_oplus, &k_jet_e);
        samples.push(a_nu);
        sum += a_nu;
        sumsq += a_nu * a_nu;
        if a_nu < a_min { a_min = a_nu; }
        if a_nu > a_max { a_max = a_nu; }
    }
    let mean = sum / n_orient as f64;
    let var = sumsq / n_orient as f64 - mean * mean;
    let std = var.sqrt();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[n_orient / 2];
    s.push_str(&format!("  N orientations = {}\n", n_orient));
    s.push_str(&format!("  A_nu mean      = {:.10} ({:.4}%)\n", mean, 100.0 * mean));
    s.push_str(&format!("  A_nu median    = {:.10} ({:.4}%)\n", median, 100.0 * median));
    s.push_str(&format!("  A_nu std       = {:.10}\n", std));
    s.push_str(&format!("  A_nu minimum   = {:.10} ({:.4}%)\n", a_min, 100.0 * a_min));
    s.push_str(&format!("  A_nu maximum   = {:.10} ({:.4}%)\n", a_max, 100.0 * a_max));
    s.push_str("\n");

    s.push_str("============================================================\n");

    Ok(s)
}

#[pymodule]
fn a_nu_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(a_nu_canonical, m)?)?;
    m.add_function(wrap_pyfunction!(a_nu_for_euler, m)?)?;
    m.add_function(wrap_pyfunction!(a_nu_average, m)?)?;
    m.add_function(wrap_pyfunction!(a_nu_harmonics, m)?)?;
    m.add_function(wrap_pyfunction!(jet_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(gt_circle_extrema, m)?)?;
    m.add_function(wrap_pyfunction!(a_nu_verify_kernel5, m)?)?;
    m.add_function(wrap_pyfunction!(a_nu_report, m)?)?;
    Ok(())
}