//! phase_stacking_core.rs — Kernel de Phase Stacking para VCV48
//!
//! Predicción de fase derivada de geometría O_h + compliancia elástica:
//!   φ_offset = arctan(1/√2) + (n/2) × (δ_CMB / K_inv) × √(2/3) × f_orient(n̂_CMB)
//!
//! = 0.615479 + 5 × (0.00610865 / 0.8025) × 0.816497 × 1.0
//! = 0.615479 + 0.031076
//! = 0.646555 rad (37.04°)

use rayon::prelude::*;
use std::f64::consts::PI;

// ============================================================================
// CONSTANTES DEL MODELO VCV48
// ============================================================================

/// Escala espacial de la red (Mpc) — derivada de R_Y/48
pub const A0: f64 = 14.075;

/// Vector de onda de la red VCV48 (h/Mpc)
pub const K_VCV: f64 = 0.4464;

/// Frecuencia fundamental observada (Gyr⁻¹)
pub const OMEGA_OBS: f64 = 0.1914;

// ============================================================================
// CONSTANTES PARA LA DERIVACIÓN GEOMÉTRICA DE φ_offset
// ============================================================================

/// Birrefringencia CMB (Planck 2018)
pub const DELTA_CMB: f64 = 0.00610865;

/// Factor elástico universal de la red
pub const K_INV: f64 = 0.8025;

/// Armónico fundamental
pub const N_HARMONIC: f64 = 10.0;

/// Ángulo diedro del octaedro truncado: arctan(1/√2)
pub const PHI_BASE: f64 = 0.6154797086706703;

/// Factor de proyección transversal analítico: √(2/3)
pub const PROJ_FACTOR: f64 = 0.816496580927726;

/// Factor de modo: n/2 = 5
pub const MODE_FACTOR: f64 = 5.0;

/// Fase de referencia calculada geométricamente (rad)
pub const PHASE_OFFSET: f64 = 0.646555;

// ============================================================================
// COSMOLOGÍA ESTÁNDAR (ΛCDM)
// ============================================================================

#[inline(always)]
fn hubble_factor(z: f64, omega_m: f64, omega_l: f64) -> f64 {
    (omega_m * (1.0 + z).powi(3) + omega_l).sqrt()
}

pub fn distancia_comovil_lcdm(z: f64, h0: f64, omega_m: f64, omega_l: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let c = 299792.458;
    let n_steps = 200;
    let dz = z / (n_steps as f64);
    let mut integral = 0.0;
    for i in 0..n_steps {
        let z_mid = (i as f64 + 0.5) * dz;
        integral += dz / hubble_factor(z_mid, omega_m, omega_l);
    }
    (c / h0) * integral
}

// ============================================================================
// CÁLCULO DE FASES
// ============================================================================

/// Calcula las fases predichas para un catálogo de galaxias.
/// φ_i = 2π × (d_comovil(z_i) % A0) / A0 + PHASE_OFFSET
pub fn compute_phases(z: &[f64], h0: f64, omega_m: f64, omega_l: f64) -> Vec<f64> {
    z.par_iter()
        .map(|&zi| {
            let d = distancia_comovil_lcdm(zi, h0, omega_m, omega_l);
            let phase = 2.0 * PI * (d % A0) / A0 + PHASE_OFFSET;
            let phase_norm = phase % (2.0 * PI);
            if phase_norm < 0.0 {
                phase_norm + 2.0 * PI
            } else {
                phase_norm
            }
        })
        .collect()
}

// ============================================================================
// TEST DE RAYLEIGH
// ============================================================================

#[derive(Debug, Clone)]
pub struct RayleighResult {
    pub r_coherence: f64,
    pub phi_mean: f64,
    pub z_statistic: f64,
    pub p_value: f64,
    pub n_samples: usize,
}

pub fn rayleigh_test(phases: &[f64]) -> RayleighResult {
    let n = phases.len();
    if n < 10 {
        return RayleighResult {
            r_coherence: 0.0,
            phi_mean: 0.0,
            z_statistic: 0.0,
            p_value: 1.0,
            n_samples: n,
        };
    }

    let sum_cos: f64 = phases.iter().map(|&p| p.cos()).sum();
    let sum_sin: f64 = phases.iter().map(|&p| p.sin()).sum();

    let r = (sum_cos.powi(2) + sum_sin.powi(2)).sqrt() / (n as f64);
    let phi_mean = sum_sin.atan2(sum_cos);
    let phi_mean_norm = if phi_mean < 0.0 {
        phi_mean + 2.0 * PI
    } else {
        phi_mean
    };
    let z = (n as f64) * r * r;
    let p = (-z).exp();

    RayleighResult {
        r_coherence: r,
        phi_mean: phi_mean_norm,
        z_statistic: z,
        p_value: p,
        n_samples: n,
    }
}

// ============================================================================
// PHASE STACKING COMPLETO
// ============================================================================

#[derive(Debug, Clone)]
pub struct PhaseStackingResult {
    pub r_coherence: f64,
    pub phi_obs: f64,
    pub delta_phi: f64,
    pub delta_phi_deg: f64,
    pub z_statistic: f64,
    pub p_value: f64,
    pub n_galaxies: usize,
    pub sigma: f64,
}

pub fn phase_stacking(
    z: &[f64],
    h0: f64,
    omega_m: f64,
    omega_l: f64,
) -> PhaseStackingResult {
    let phases = compute_phases(z, h0, omega_m, omega_l);
    let ray = rayleigh_test(&phases);

    let delta_phi = (ray.phi_mean - PHASE_OFFSET).abs();
    let delta_phi = if delta_phi > PI {
        2.0 * PI - delta_phi
    } else {
        delta_phi
    };
    let delta_phi_deg = delta_phi * 180.0 / PI;

    let sigma = if ray.p_value > 0.0 && ray.p_value < 1.0 {
        (-2.0 * ray.p_value.ln()).sqrt()
    } else {
        0.0
    };

    PhaseStackingResult {
        r_coherence: ray.r_coherence,
        phi_obs: ray.phi_mean,
        delta_phi,
        delta_phi_deg,
        z_statistic: ray.z_statistic,
        p_value: ray.p_value,
        n_galaxies: z.len(),
        sigma,
    }
}

// ============================================================================
// PHASE STACKING BINNEADO
// ============================================================================

pub fn phase_stacking_binned(
    z: &[f64],
    r_min: f64,
    r_max: f64,
    n_bins: usize,
    h0: f64,
    omega_m: f64,
    omega_l: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u64>) {
    let distances: Vec<f64> = z
        .par_iter()
        .map(|&zi| distancia_comovil_lcdm(zi, h0, omega_m, omega_l))
        .collect();

    let bin_width = (r_max - r_min) / (n_bins as f64);

    let mut r_centers = Vec::with_capacity(n_bins);
    let mut r_values = Vec::with_capacity(n_bins);
    let mut phi_values = Vec::with_capacity(n_bins);
    let mut p_values = Vec::with_capacity(n_bins);
    let mut n_per_bin = Vec::with_capacity(n_bins);

    for b in 0..n_bins {
        let r_lo = r_min + (b as f64) * bin_width;
        let r_hi = r_lo + bin_width;
        let r_center = (r_lo + r_hi) / 2.0;

        let bin_phases: Vec<f64> = distances
            .iter()
            .filter(|&&d| d >= r_lo && d < r_hi)
            .map(|&d| {
                let phase = 2.0 * PI * (d % A0) / A0 + PHASE_OFFSET;
                let phase_norm = phase % (2.0 * PI);
                if phase_norm < 0.0 {
                    phase_norm + 2.0 * PI
                } else {
                    phase_norm
                }
            })
            .collect();

        let n = bin_phases.len() as u64;
        let ray = rayleigh_test(&bin_phases);

        r_centers.push(r_center);
        r_values.push(ray.r_coherence);
        phi_values.push(ray.phi_mean);
        p_values.push(ray.p_value);
        n_per_bin.push(n);
    }

    (r_centers, r_values, phi_values, p_values, n_per_bin)
}

// ============================================================================
// GENERACIÓN DE CATÁLOGOS SINTÉTICOS
// ============================================================================

pub fn generate_modulated_catalog(
    z: &[f64],
    amplitude: f64,
    seed: u64,
    h0: f64,
    omega_m: f64,
    omega_l: f64,
) -> Vec<f64> {
    use rand::prelude::*;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);
    let n = z.len();

    let d_orig: Vec<f64> = z
        .par_iter()
        .map(|&zi| distancia_comovil_lcdm(zi, h0, omega_m, omega_l))
        .collect();

    let d_mod: Vec<f64> = d_orig
        .iter()
        .map(|&d| {
            let delta = amplitude * A0 * (2.0 * PI * (d % A0) / A0 + PHASE_OFFSET).cos();
            let noise = rng.gen_range(-0.1..0.1) * amplitude * A0;
            let d_new = d + delta + noise;
            if d_new < 0.0 { d } else { d_new }
        })
        .collect();

    d_mod
        .iter()
        .map(|&d| {
            if d <= 0.0 {
                return 0.0;
            }
            let mut z_lo = 0.0;
            let mut z_hi = 10.0;
            for _ in 0..50 {
                let z_mid = (z_lo + z_hi) / 2.0;
                let d_mid = distancia_comovil_lcdm(z_mid, h0, omega_m, omega_l);
                if d_mid < d {
                    z_lo = z_mid;
                } else {
                    z_hi = z_mid;
                }
            }
            (z_lo + z_hi) / 2.0
        })
        .collect()
}

// ============================================================================
// LOOK-ELSEWHERE EFFECT
// ============================================================================

pub fn lee_correction(p_local: f64, n_trials: usize) -> f64 {
    if p_local >= 1.0 {
        return 1.0;
    }
    let p_corr = 1.0 - (1.0 - p_local).powi(n_trials as i32);
    p_corr.min(1.0)
}

pub fn estimate_n_trials(r_min: f64, r_max: f64) -> usize {
    let range = r_max - r_min;
    let n_periods = (range / A0).ceil() as usize;
    n_periods * 14
}

// ============================================================================
// DIAGNÓSTICO DE φ_offset
// ============================================================================

pub fn diagnostic_phase_offset() -> Vec<(String, f64)> {
    vec![
        ("phi_base".to_string(), PHI_BASE),
        ("phi_base_deg".to_string(), PHI_BASE * 180.0 / PI),
        ("delta_cmb".to_string(), DELTA_CMB),
        ("K_inv".to_string(), K_INV),
        ("compliance".to_string(), 1.0 / K_INV),
        ("n_harmonic".to_string(), N_HARMONIC),
        ("mode_factor".to_string(), MODE_FACTOR),
        ("proj_factor".to_string(), PROJ_FACTOR),
        ("delta_phi".to_string(), MODE_FACTOR * DELTA_CMB / K_INV * PROJ_FACTOR),
        ("phi_offset_calc".to_string(), PHASE_OFFSET),
        ("phi_offset_deg".to_string(), PHASE_OFFSET * 180.0 / PI),
        ("sdss_measured".to_string(), 0.646466),
        ("difference_abs".to_string(), (PHASE_OFFSET - 0.646466).abs()),
    ]
}

// ============================================================================
// INTERFAZ PYTHON
// ============================================================================

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
#[pyclass]
pub struct PhaseStackingEngine {
    h0: f64,
    omega_m: f64,
    omega_l: f64,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PhaseStackingEngine {
    #[new]
    #[pyo3(signature = (h0=70.0, omega_m=0.315, omega_l=0.685))]
    pub fn new(h0: f64, omega_m: f64, omega_l: f64) -> Self {
        println!("🔧 Phase Stacking Engine — VCV48");
        println!("   Cosmología: H0={}, Ωm={}, ΩΛ={}", h0, omega_m, omega_l);
        println!("   A0 = {:.3} Mpc", A0);
        println!("   K_VCV = {:.4} h/Mpc", K_VCV);
        println!("   φ_offset = {:.6} rad ({:.2}°)", PHASE_OFFSET, PHASE_OFFSET * 180.0 / PI);
        println!("   Derivación: φ_base + (n/2)×(δ_CMB/K_inv)×√(2/3)");
        Self { h0, omega_m, omega_l }
    }

    pub fn compute_phases(&self, z: Vec<f64>) -> Vec<f64> {
        compute_phases(&z, self.h0, self.omega_m, self.omega_l)
    }

    pub fn phase_stacking(&self, z: Vec<f64>) -> PyResult<(f64, f64, f64, f64, f64, f64, usize, f64)> {
        let result = phase_stacking(&z, self.h0, self.omega_m, self.omega_l);
        Ok((
            result.r_coherence,
            result.phi_obs,
            result.delta_phi,
            result.delta_phi_deg,
            result.z_statistic,
            result.p_value,
            result.n_galaxies,
            result.sigma,
        ))
    }

    pub fn rayleigh_test(&self, phases: Vec<f64>) -> PyResult<(f64, f64, f64, f64)> {
        let ray = rayleigh_test(&phases);
        Ok((ray.r_coherence, ray.phi_mean, ray.z_statistic, ray.p_value))
    }

    #[pyo3(signature = (z, amplitude=0.015, seed=42))]
    pub fn generate_modulated_catalog(
        &self,
        z: Vec<f64>,
        amplitude: f64,
        seed: u64,
    ) -> Vec<f64> {
        generate_modulated_catalog(&z, amplitude, seed, self.h0, self.omega_m, self.omega_l)
    }

    #[pyo3(signature = (z, r_min=0.0, r_max=600.0, n_bins=40))]
    pub fn phase_stacking_binned(
        &self,
        z: Vec<f64>,
        r_min: f64,
        r_max: f64,
        n_bins: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u64>)> {
        Ok(phase_stacking_binned(&z, r_min, r_max, n_bins, self.h0, self.omega_m, self.omega_l))
    }

    pub fn diagnostic(&self) -> PyResult<Vec<(String, f64)>> {
        Ok(diagnostic_phase_offset())
    }

    // Getters
    pub fn get_a0(&self) -> f64 { A0 }
    pub fn get_phase_offset(&self) -> f64 { PHASE_OFFSET }
    pub fn get_phase_offset_deg(&self) -> f64 { PHASE_OFFSET * 180.0 / PI }
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn phase_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PhaseStackingEngine>()?;
    m.add("A0", A0)?;
    m.add("PHASE_OFFSET", PHASE_OFFSET)?;
    m.add("K_VCV", K_VCV)?;
    m.add("OMEGA_OBS", OMEGA_OBS)?;
    m.add("PHI_BASE", PHI_BASE)?;
    m.add("DELTA_CMB", DELTA_CMB)?;
    m.add("K_INV", K_INV)?;
    Ok(())
}