//! vpm_core_delta_alpha - Motor para análisis de Δα/α
//!                        
//! Versión 5.1 — Con umbral de saturación calibrable
//! Añade SATURATION_THRESHOLD_BOOST para ajustar el onset de plasticidad

use std::f64::consts::PI;

// ============================================================================
// CONSTANTES FUNDAMENTALES (PRIMEROS PRINCIPIOS)
// ============================================================================

const ALPHA: f64 = 7.2973525693e-3;
const OH_ORDER: f64 = 48.0;
const T_CMB_0: f64 = 2.72548;
const THETA_DEBYE: f64 = 35.776;
const V_DISP_REF: f64 = 373.0;

// ============================================================================
// CONSTANTES PARA ACOPLAMIENTO ALPHA
// ============================================================================

const ALPHA_RIGIDITY_COUPLING: f64 = -0.5;
const ALPHA_DIMENSIONAL_SCALE: f64 = 1.0e-3;

/// ← NUEVO: Factor de desplazamiento del umbral de saturación
/// Con 1.0 = modelo original (umbral ~373 km/s)
/// Con 1.5 = umbral ~560 km/s (mejor ajuste a datos)
const SATURATION_THRESHOLD_BOOST: f64 = 1.5;

/// Umbral de Frenkel para régimen plástico (adimensional)
const FRENKEL_THRESHOLD: f64 = 0.5;

/// Anchura de transición plástica
const PLASTIC_WIDTH: f64 = 0.3;

// ============================================================================
// FUNCIONES TERMODINÁMICAS (sin cambios)
// ============================================================================

#[inline(always)]
pub fn kappa_base() -> f64 {
    ((ALPHA / (OH_ORDER * PI)).sqrt()) * (2.0 * PI)
}

#[inline(always)]
pub fn t_cmb(z: f64) -> f64 {
    T_CMB_0 * (1.0 + z)
}

#[inline(always)]
pub fn debye_waller_factor(z: f64) -> f64 {
    let exponent = (t_cmb(z) / THETA_DEBYE) * 1.5;
    (-exponent).exp()
}

#[inline(always)]
pub fn kappa_vcv(z: f64) -> f64 {
    kappa_base() * debye_waller_factor(z)
}

// ============================================================================
// FUNCIONES DE RIGIDEZ (con umbral calibrado)
// ============================================================================

#[inline(always)]
pub fn rigidity_excess(vdisp: f64, vdisp_ref: f64) -> f64 {
    if vdisp <= 0.0 || vdisp_ref <= 0.0 {
        0.0
    } else {
        // ← MODIFICADO: la velocidad efectiva se reduce por el boost
        // Esto desplaza el onset de saturación a masas más altas
        let vdisp_scaled = vdisp / SATURATION_THRESHOLD_BOOST;
        (vdisp_scaled / vdisp_ref).powi(4) - 1.0
    }
}

#[inline(always)]
pub fn saturation_factor(delta_g: f64) -> f64 {
    if delta_g <= 0.0 {
        1.0
    } else if delta_g < FRENKEL_THRESHOLD {
        1.0 - 0.1 * (delta_g / FRENKEL_THRESHOLD).powi(2)
    } else {
        let x = (delta_g - FRENKEL_THRESHOLD) / PLASTIC_WIDTH;
        0.9 / (1.0 + x.exp()) + 0.1
    }
}

#[inline(always)]
pub fn rigidity_effective(vdisp: f64, vdisp_ref: f64) -> f64 {
    let delta_g_raw = rigidity_excess(vdisp, vdisp_ref);
    delta_g_raw * saturation_factor(delta_g_raw)
}

#[inline(always)]
pub fn redshift_coupling_factor(z: f64) -> f64 {
    (1.0 + z).powf(-0.5)
}

// ============================================================================
// CÁLCULO DE Δα/α (sin cambios en la fórmula)
// ============================================================================

#[inline]
pub fn delta_alpha_alpha(vdisp: f64, z: f64) -> f64 {
    let delta_g_eff = rigidity_effective(vdisp, V_DISP_REF);
    let kappa_z = kappa_vcv(z);
    let z_coupling = redshift_coupling_factor(z);

    ALPHA_RIGIDITY_COUPLING * delta_g_eff * kappa_z * z_coupling * ALPHA_DIMENSIONAL_SCALE
}

#[inline]
pub fn delta_alpha_ppm(vdisp: f64, z: f64) -> f64 {
    delta_alpha_alpha(vdisp, z) * 1_000_000.0
}

// ============================================================================
// PREDICCIONES PARA MUESTRAS (se recalibran automáticamente)
// ============================================================================

pub fn predict_delta_alpha_am_a() -> f64 {
    delta_alpha_alpha(450.0, 0.4)
}

pub fn predict_delta_alpha_am_b() -> f64 {
    delta_alpha_alpha(297.0, 0.4)
}

pub fn predict_delta_alpha_differential() -> f64 {
    predict_delta_alpha_am_a() - predict_delta_alpha_am_b()
}

pub fn amplification_factor(vdisp: f64) -> f64 {
    1.0 + rigidity_effective(vdisp, V_DISP_REF)
}

pub fn predict_amplification_ratio() -> f64 {
    amplification_factor(450.0) / amplification_factor(297.0)
}

// ============================================================================
// INTERFAZ PYTHON (pyo3) — con acceso al nuevo parámetro
// ============================================================================

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
#[pyclass]
pub struct DeltaAlphaEngine;

#[cfg(feature = "pyo3")]
#[pymethods]
impl DeltaAlphaEngine {
    #[new]
    pub fn new() -> Self {
        println!("🔧 DeltaAlphaEngine v5.1 — Umbral de saturación calibrado");
        println!("   Θ_D = {:.3} K", THETA_DEBYE);
        println!("   κ_base = {:.6}", kappa_base());
        println!("   κ(0) = {:.6}", kappa_vcv(0.0));
        println!("   κ(0.4) = {:.6}", kappa_vcv(0.4));
        println!("   V_DISP_REF = {:.1} km/s", V_DISP_REF);
        println!("   SAT_THRESHOLD_BOOST = {:.1}", SATURATION_THRESHOLD_BOOST);
        println!("   Umbral efectivo ≈ {:.0} km/s", V_DISP_REF * SATURATION_THRESHOLD_BOOST);
        Self
    }

    pub fn delta_alpha_alpha(&self, vdisp: f64, z: f64) -> f64 {
        delta_alpha_alpha(vdisp, z)
    }

    pub fn delta_alpha_ppm(&self, vdisp: f64, z: f64) -> f64 {
        delta_alpha_ppm(vdisp, z)
    }

    pub fn predict_delta_alpha_am_a(&self) -> f64 {
        predict_delta_alpha_am_a()
    }

    pub fn predict_delta_alpha_am_b(&self) -> f64 {
        predict_delta_alpha_am_b()
    }

    pub fn predict_delta_alpha_differential(&self) -> f64 {
        predict_delta_alpha_differential()
    }

    pub fn predict_amplification_ratio(&self) -> f64 {
        predict_amplification_ratio()
    }

    pub fn kappa_vcv(&self, z: f64) -> f64 {
        kappa_vcv(z)
    }

    pub fn kappa_base(&self) -> f64 {
        kappa_base()
    }

    pub fn debye_waller_factor(&self, z: f64) -> f64 {
        debye_waller_factor(z)
    }

    pub fn saturation_factor(&self, delta_g: f64) -> f64 {
        saturation_factor(delta_g)
    }

    pub fn rigidity_effective(&self, vdisp: f64) -> f64 {
        rigidity_effective(vdisp, V_DISP_REF)
    }

    pub fn get_constants(&self) -> (f64, f64, f64) {
        (THETA_DEBYE, kappa_base(), V_DISP_REF)
    }

    /// ← NUEVO: exponer el umbral de saturación a Python
    pub fn get_saturation_threshold_boost(&self) -> f64 {
        SATURATION_THRESHOLD_BOOST
    }
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn vpm_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DeltaAlphaEngine>()?;
    m.add("THETA_DEBYE", THETA_DEBYE)?;
    m.add("KAPPA_BASE", kappa_base())?;
    m.add("V_DISP_REF", V_DISP_REF)?;
    m.add("SATURATION_THRESHOLD_BOOST", SATURATION_THRESHOLD_BOOST)?;
    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwf_at_z0() {
        let dwf = debye_waller_factor(0.0);
        assert!((dwf - 0.892).abs() < 0.001);
    }

    #[test]
    fn test_kappa_values() {
        assert!((kappa_vcv(0.0) - 0.038988).abs() < 0.001);
        assert!((kappa_vcv(0.4) - 0.037246).abs() < 0.001);
    }

    #[test]
    fn test_delta_alpha_sign() {
        // Con el nuevo umbral, AM-A ya no es negativo
        let da_ama = delta_alpha_alpha(450.0, 0.4);
        let da_amb = delta_alpha_alpha(297.0, 0.4);
        println!("Δα/α AM-A (450): {:.6e}", da_ama);
        println!("Δα/α AM-B (297): {:.6e}", da_amb);
        // El diferencial debe ser negativo (AM-A menos positivo que AM-B)
        assert!(da_ama < da_amb, "AM-A debe ser menos positivo que AM-B");
    }

    #[test]
    fn test_saturation() {
        assert!(saturation_factor(0.2) > saturation_factor(1.0));
        assert!(saturation_factor(2.0) < 0.5);
    }

    #[test]
    fn test_threshold_boost_effect() {
        // Con boost=1.5, vdisp=373 produce rigidez ~ -0.3 (todavía régimen elástico)
        let excess_373 = rigidity_excess(373.0, V_DISP_REF);
        println!("Rigidity excess at 373 km/s (boost=1.5): {:.4}", excess_373);
        assert!(excess_373 < 0.0, "A 373 km/s debería estar en régimen elástico");

        // A 560 km/s (373*1.5) debería estar cerca del umbral
        let excess_560 = rigidity_excess(560.0, V_DISP_REF);
        println!("Rigidity excess at 560 km/s (boost=1.5): {:.4}", excess_560);
        assert!(excess_560.abs() < 0.2, "A 560 km/s debería estar cerca del umbral");
    }
}