use pyo3::prelude::*;      
use pyo3::types::PyDict;
use std::f64::consts::PI;
use std::collections::HashMap;
use std::sync::Mutex;      
use lazy_static::lazy_static;

// ============================================================================
// CONSTANTES FUNDAMENTALES (CODATA 2022)
// ============================================================================
const ALPHA_EXP: f64 = 0.0072973525693;
const K_INVARIANT: f64 = 0.8025;
const DELTA_0: f64 = 0.00610865;
const OMEGA_K: f64 = -0.044;

// ============================================================================
// CONSTANTES GEOMÉTRICAS
// ============================================================================
const C_BIR_UNIVERSAL: f64 = 405.812;
const POISSON_RATIO: f64 = 0.25;

// ============================================================================
// PARÁMETROS DE INTEGRACIÓN
// ============================================================================
// Antes --> x2
const N_CELLS: usize = 128;
const N_SHIFTS: usize = 16;
const N_ANGULAR: usize = 1200;
const N_RADIAL: usize = 200;

// ============================================================================
// TOLERANCIAS
// ============================================================================
const CALIB_TOL: f64 = 1e-12;
const MAX_ITER: usize = 100;

// ============================================================================
// CACHÉ
// ============================================================================
lazy_static! {
    static ref PHI_CACHE: Mutex<HashMap<(i64, i64, i64, i64), f64>> = Mutex::new(HashMap::new());
}

// ============================================================================
// FUNCIÓN AUXILIAR: Invariante cúbico
// ============================================================================
fn cubic_invariant(kx: f64, ky: f64, kz: f64) -> f64 {
    let k2 = kx*kx + ky*ky + kz*kz;
    if k2 < 1e-18 { return 0.0; }
    let k4 = k2 * k2;
    let k4_sum = kx*kx*kx*kx + ky*ky*ky*ky + kz*kz*kz*kz;
    k4_sum / k4 - 0.6
}

// ============================================================================
// FUNCIÓN AUXILIAR: Drift lineal aproximado
// ============================================================================
fn drift_linear_approx(eta1: f64, eta2: f64) -> f64 {
    0.87 * eta1 + 0.12 * eta2 - 0.2375
}

// ============================================================================
// FUNCIÓN AUXILIAR: Puntos esfera Fibonacci
// ============================================================================
fn fibonacci_sphere_points(n: usize) -> Vec<(f64, f64)> {
    let phi = PI * (3.0 - 5.0_f64.sqrt());
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let y = 1.0 - (i as f64) / (n as f64 - 1.0) * 2.0;
        let theta = y.acos();
        let azimuth = 2.0 * PI * (i as f64) * phi;
        points.push((theta, azimuth));
    }
    points
}

// ============================================================================
// ESTRUCTURA FCC
// ============================================================================
#[pyclass]
pub struct FCCLattice {
    k0: f64,
    k_max: f64,
    v_mean: f64,
    bz_volume_base: f64,
    pub omega_k: f64,
    pub z: f64,
    scale_factor: f64,
    birefringence_factor: f64,
}

#[pymethods]
impl FCCLattice {
    #[new]
    pub fn new(omega_k: f64, z: f64) -> Self {
        let c = 299792458.0;
        let a0 = 4.342e-23;
        let k0 = 2.0 * PI / a0;

        let v_l = c;
        let v_t = c / (3.0_f64).sqrt();
        let v_inv3 = (1.0 / v_l.powi(3) + 2.0 / v_t.powi(3)) / 3.0;
        let v_mean = v_inv3.powf(-1.0 / 3.0);

        let bz_volume_base = 4.0 * k0.powi(3);
        let scale_factor = (1.0 + omega_k).abs().sqrt();

        let delta_z = (1.0 + z).sqrt() - 1.0;
        let birefringence_factor = 1.0 + C_BIR_UNIVERSAL * omega_k.abs().sqrt() * delta_z;
        let k_max = k0 * 1.35 * scale_factor;

        FCCLattice {
            k0, k_max, v_mean, bz_volume_base,
            omega_k, z, scale_factor, birefringence_factor,
        }
    }

    pub fn is_inside_bz(&self, kx: f64, ky: f64, kz: f64) -> bool {
        let s = self.scale_factor;
        let k_ref = self.k0 * s;
        let ux = kx.abs() / k_ref;
        let uy = ky.abs() / k_ref;
        let uz = kz.abs() / k_ref;
        (ux + uy + uz) <= 1.5 && ux <= 1.0 && uy <= 1.0 && uz <= 1.0
    }

    pub fn anisotropic_velocity(&self, kx: f64, ky: f64, kz: f64, eta1: f64, eta2: f64) -> f64 {
        let delta = cubic_invariant(kx, ky, kz);
        self.v_mean * (1.0 + eta1 * delta + eta2 * delta * delta)
    }

    pub fn bz_volume_exact(&self) -> f64 {
        self.bz_volume_base
    }

    pub fn bz_volume_deformed(&self) -> f64 {
        self.bz_volume_base * self.scale_factor.powi(3)
    }

    pub fn debye_radius(&self) -> f64 {
        (3.0 * self.bz_volume_deformed() / (4.0 * PI)).powf(1.0/3.0)
    }

    pub fn adaptive_quadrature(&self, n: usize, shift: f64, eta1: f64, eta2: f64) -> (f64, f64, usize) {
        let cell = 2.0 * self.k_max / n as f64;
        let off = shift * cell;
        let mut sum_f = 0.0;
        let mut sum_f2 = 0.0;
        let mut count = 0;

        for i in 0..n {
            let x = -self.k_max + (i as f64 + 0.5) * cell + off;
            for j in 0..n {
                let y = -self.k_max + (j as f64 + 0.5) * cell + off;
                for k in 0..n {
                    let z = -self.k_max + (k as f64 + 0.5) * cell + off;
                    if self.is_inside_bz(x, y, z) {
                        let v = self.anisotropic_velocity(x, y, z, eta1, eta2);
                        let f = 1.0 / v.powi(3);
                        sum_f += f;
                        sum_f2 += f * f;
                        count += 1;
                    }
                }
            }
        }

        if count == 0 { return (0.0, 0.0, 0); }
        let mean = sum_f / count as f64;
        let var = (sum_f2 / count as f64) - mean * mean;
        let integral = mean * self.bz_volume_deformed();
        let error = (var.max(0.0) / count as f64).sqrt() * self.bz_volume_deformed();
        (integral, error, count)
    }

    pub fn averaged_quadrature(&self, n: usize, n_shifts: usize, eta1: f64, eta2: f64) -> (f64, f64, usize) {
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        let mut total = 0;

        for i in 0..n_shifts {
            let shift = i as f64 / n_shifts as f64;
            let (integral, _, count) = self.adaptive_quadrature(n, shift, eta1, eta2);
            sum += integral;
            sum2 += integral * integral;
            total += count;
        }

        let mean = sum / n_shifts as f64;
        let var = (sum2 / n_shifts as f64) - mean * mean;
        let error = (var.max(0.0) / n_shifts as f64).sqrt();
        (mean, error, total)
    }

    /// INTEGRAL ANALÍTICA DE LA ESFERA (Denominador isótropo)
    pub fn integrate_debye_sphere(&self) -> f64 {
        let kd = self.debye_radius();
        let v0 = self.v_mean;
        (4.0 * PI / 3.0) * kd.powi(3) / v0.powi(3)
    }

    /// Φ_α = <1/v³>_BZ / <1/v₀³>_Debye
    /// Numerador: integral numérica sobre el octaedro truncado
    /// Denominador: integral analítica sobre la esfera isótropa
    pub fn compute_phi(&self, eta1: f64, eta2: f64) -> f64 {
        let (int_bz, _, _) = self.averaged_quadrature(N_CELLS, N_SHIFTS, eta1, eta2);
        let int_sph = self.integrate_debye_sphere();
        int_bz / int_sph
    }

    pub fn compute_alpha_from_eta(&self, eta1: f64, eta2: f64) -> f64 {
        let phi = self.compute_phi(eta1, eta2);
        let metric_factor = 1.0 + self.omega_k / 2.0;
        (DELTA_0 / K_INVARIANT) * metric_factor * phi
    }

    pub fn compute_drift(&self, eta1: f64, eta2: f64) -> f64 {
        let alpha_now = self.compute_alpha_from_eta(eta1, eta2);
        let lat_local = FCCLattice::new(self.omega_k, 0.0);
        let alpha_local = lat_local.compute_alpha_from_eta(eta1, eta2);
        (alpha_now - alpha_local) / alpha_local
    }
}

// ============================================================================
// FUNCIONES EXPORTADAS
// ============================================================================

#[pyfunction]
fn compute_alpha_from_eta(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<f64> {
    let lat = FCCLattice::new(omega_k, z);
    Ok(lat.compute_alpha_from_eta(eta1, eta2))
}

#[pyfunction]
fn compute_drift_from_eta(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<f64> {
    let lat = FCCLattice::new(omega_k, z);
    Ok(lat.compute_drift(eta1, eta2))
}

#[pyfunction]
fn calibrate_eta_from_alpha(omega_k: f64, z: f64) -> PyResult<(f64, f64, f64, usize)> {
    let lat = FCCLattice::new(omega_k, z);

    let mut eta1 = 0.25;
    let mut eta2 = 0.17;

    let mut iter = 0;
    let mut prev_alpha = 0.0;

    for i in 0..MAX_ITER {
        iter = i;
        let alpha_calc = lat.compute_alpha_from_eta(eta1, eta2);
        let error = alpha_calc - ALPHA_EXP;

        if error.abs() < CALIB_TOL {
            break;
        }

        let eps = 1e-7;
        let alpha_eta1 = (lat.compute_alpha_from_eta(eta1 + eps, eta2) - alpha_calc) / eps;
        let alpha_eta2 = (lat.compute_alpha_from_eta(eta1, eta2 + eps) - alpha_calc) / eps;

        let norm2 = alpha_eta1 * alpha_eta1 + alpha_eta2 * alpha_eta2;
        if norm2 < 1e-12 {
            break;
        }

        let step = 0.01 * error / norm2.sqrt();
        eta1 -= step * alpha_eta1;
        eta2 -= step * alpha_eta2;

        eta1 = eta1.clamp(0.1, 0.4);
        eta2 = eta2.clamp(0.05, 0.3);

        prev_alpha = alpha_calc;

        if i > 0 && (alpha_calc - prev_alpha).abs() < 1e-14 {
            break;
        }
    }

    let final_alpha = lat.compute_alpha_from_eta(eta1, eta2);
    Ok((eta1, eta2, final_alpha, iter))
}

#[pyfunction]
fn compute_phi_geometric() -> PyResult<f64> {
    let lat = FCCLattice::new(OMEGA_K, 0.0);
    Ok(lat.compute_phi(0.0, 0.0))
}

#[pyfunction]
fn verify_consistency(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<Py<PyDict>> {
    let lat = FCCLattice::new(omega_k, z);

    Python::with_gil(|py| {
        let dict = PyDict::new(py);

        let alpha_calc = lat.compute_alpha_from_eta(eta1, eta2);
        let phi = lat.compute_phi(eta1, eta2);
        let drift = lat.compute_drift(eta1, eta2);
        let drift_approx = drift_linear_approx(eta1, eta2);

        dict.set_item("alpha_calculated", alpha_calc)?;
        dict.set_item("alpha_target", ALPHA_EXP)?;
        dict.set_item("alpha_error_ppm", (alpha_calc - ALPHA_EXP) / ALPHA_EXP * 1e6)?;
        dict.set_item("phi", phi)?;
        dict.set_item("drift", drift)?;
        dict.set_item("drift_linear_approx", drift_approx)?;
        dict.set_item("drift_error", (drift - drift_approx).abs())?;
        dict.set_item("k_eff", K_INVARIANT * phi)?;
        dict.set_item("poisson_ratio", POISSON_RATIO)?;

        Ok(dict.into())
    })
}

#[pyfunction]
fn compute_phi_from_eta(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<f64> {
    let lat = FCCLattice::new(omega_k, z);
    let key = (
        (eta1 * 1e7) as i64,
        (eta2 * 1e7) as i64,
        (omega_k * 1e7) as i64,
        (z * 1e7) as i64,
    );

    let phi = {
        let mut cache = PHI_CACHE.lock().unwrap();
        if let Some(&v) = cache.get(&key) {
            v
        } else {
            let p = lat.compute_phi(eta1, eta2);
            cache.insert(key, p);
            p
        }
    };

    Ok(phi)
}

#[pyfunction]
fn compute_k_from_state(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<f64> {
    let lat = FCCLattice::new(omega_k, z);
    let phi = lat.compute_phi(eta1, eta2);
    Ok(K_INVARIANT * phi)
}

#[pyfunction]
fn delta_eff_with_omega_z(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<f64> {
    let lat = FCCLattice::new(omega_k, z);
    let phi = lat.compute_phi(eta1, eta2);
    Ok(K_INVARIANT * ALPHA_EXP * lat.birefringence_factor / phi)
}

#[pyfunction]
fn clear_cache() -> PyResult<()> {
    PHI_CACHE.lock().unwrap().clear();
    Ok(())
}

#[pyfunction]
fn lattice_info(omega_k: f64, z: f64) -> PyResult<Py<PyDict>> {
    let lat = FCCLattice::new(omega_k, z);
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("v_mean", lat.v_mean)?;
        dict.set_item("k0", lat.k0)?;
        dict.set_item("k_max", lat.k_max)?;
        dict.set_item("scale_factor", lat.scale_factor)?;
        dict.set_item("birefringence_factor", lat.birefringence_factor)?;
        dict.set_item("bz_volume_base", lat.bz_volume_base)?;
        dict.set_item("bz_volume_deformed", lat.bz_volume_deformed())?;
        dict.set_item("omega_k", lat.omega_k)?;
        dict.set_item("z", lat.z)?;
        dict.set_item("c_bir_universal", C_BIR_UNIVERSAL)?;
        dict.set_item("k_invariant", K_INVARIANT)?;
        dict.set_item("alpha_exp", ALPHA_EXP)?;
        dict.set_item("delta_cmb", DELTA_0)?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn debug_integrals(eta1: f64, eta2: f64, omega_k: f64, z: f64) -> PyResult<Py<PyDict>> {
    let lat = FCCLattice::new(omega_k, z);

    let (int_bz, err_bz, count) = lat.averaged_quadrature(N_CELLS, N_SHIFTS, eta1, eta2);
    let int_sph = lat.integrate_debye_sphere();
    let kd = lat.debye_radius();
    let bz_vol = lat.bz_volume_deformed();

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("int_bz", int_bz)?;
        dict.set_item("int_sph", int_sph)?;
        dict.set_item("phi", int_bz / int_sph)?;
        dict.set_item("bz_volume", bz_vol)?;
        dict.set_item("debye_radius", kd)?;
        dict.set_item("sphere_volume", (4.0 * PI / 3.0) * kd.powi(3))?;
        dict.set_item("v_mean", lat.v_mean)?;
        dict.set_item("count", count)?;
        dict.set_item("error_bz", err_bz)?;
        Ok(dict.into())
    })
}

// ============================================================================
// MÓDULO PYTHON
// ============================================================================
#[pymodule]
fn vcv48_k(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FCCLattice>()?;

    // Funciones de calibración y cálculo
    m.add_function(wrap_pyfunction!(compute_alpha_from_eta, m)?)?;
    m.add_function(wrap_pyfunction!(compute_drift_from_eta, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_eta_from_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(compute_phi_geometric, m)?)?;
    m.add_function(wrap_pyfunction!(verify_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(debug_integrals, m)?)?;

    // Funciones originales (compatibilidad)
    m.add_function(wrap_pyfunction!(compute_phi_from_eta, m)?)?;
    m.add_function(wrap_pyfunction!(compute_k_from_state, m)?)?;
    m.add_function(wrap_pyfunction!(delta_eff_with_omega_z, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    m.add_function(wrap_pyfunction!(lattice_info, m)?)?;

    // Constantes
    m.add("K_INVARIANT", K_INVARIANT)?;
    m.add("ALPHA_EXP", ALPHA_EXP)?;
    m.add("DELTA_CMB", DELTA_0)?;
    m.add("OMEGA_K", OMEGA_K)?;
    m.add("C_BIR_UNIVERSAL", C_BIR_UNIVERSAL)?;
    m.add("POISSON_RATIO", POISSON_RATIO)?;

    Ok(())
}