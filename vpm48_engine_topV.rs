// vpm48_engine_top.rs - Motor VPM-48 para el quark top
// Compilar con: maturin develop --release

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ============================================================
// CONSTANTES DEL MODELO VCV48 PARA EL QUARK TOP
// ============================================================

/// Constante de estructura fina electromagnética (CODATA 2022)
const ALPHA_EM: f64 = 0.0072973525693;

/// Factor elástico Plateau I: K_I = 8α_em
const K_I: f64 = 8.0 * ALPHA_EM;

/// Factor elástico Plateau III: K_III
const K_III: f64 = 0.8025;

/// Masa del electrón en MeV (CODATA 2022)
const M_E_MEV: f64 = 0.5109989461;

/// Masa de referencia: m_ref = m_e / K_I
const M_REF_MEV: f64 = M_E_MEV / K_I;

/// Número de Burgers del quark top
const NF_TOP: i32 = 24603;

/// Residuo topológico: 24603 % 48 = 27
const RESIDUO_TOP: i32 = 27;

/// Masa experimental del quark top (PDG 2022)
const MASA_TOP_EXP_GEV: f64 = 172.76;

/// Incertidumbre experimental (PDG 2022)
const MASA_TOP_ERR_GEV: f64 = 0.30;

/// Orden del grupo O_h
const ORDEN_OH: i32 = 48;

/// Producto m_ref × K_III
const M_REF_K_III_MEV: f64 = M_REF_MEV * K_III;

// ============================================================
// FUNCIONES
// ============================================================

#[pyfunction]
fn calcular_masa_top(nf: i32, verbose: bool) -> PyResult<Py<PyDict>> {
    let nf_f64 = nf as f64;
    let residuo = nf % ORDEN_OH;
    
    if residuo != RESIDUO_TOP {
        if verbose {
            println!("⏭️ Saltando Nf={} (residuo {}, esperado {})", nf, residuo, RESIDUO_TOP);
        }
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("nf", nf)?;
            dict.set_item("residuo", residuo)?;
            dict.set_item("masa_final_gev", 0.0)?;
            dict.set_item("error_rel_pct", f64::INFINITY)?;
            dict.set_item("valido", false)?;
            Ok(dict.into())
        })
    } else {
        let masa_mev = M_REF_K_III_MEV * nf_f64;
        let masa_gev = masa_mev / 1000.0;
        
        let error_abs_gev = (masa_gev - MASA_TOP_EXP_GEV).abs();
        let error_rel_pct = error_abs_gev / MASA_TOP_EXP_GEV * 100.0;
        let dentro_incertidumbre = error_abs_gev <= MASA_TOP_ERR_GEV;
        
        if verbose {
            println!("🎯 Nf={:>8} (residuo {:2}) | Masa={:.6} GeV | Error={:.4}% | {}",
                     nf, residuo, masa_gev, error_rel_pct,
                     if dentro_incertidumbre { "✓" } else { "✗" });
        }
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("nf", nf)?;
            dict.set_item("residuo", residuo)?;
            dict.set_item("masa_final_gev", masa_gev)?;
            dict.set_item("error_rel_pct", error_rel_pct)?;
            dict.set_item("error_abs_gev", error_abs_gev)?;
            dict.set_item("dentro_incertidumbre", dentro_incertidumbre)?;
            dict.set_item("valido", true)?;
            Ok(dict.into())
        })
    }
}

#[pyfunction]
fn escaneo_ultra_fino(centro: i32, radio: i32, verbose: bool) -> PyResult<Py<PyDict>> {
    let inicio = centro - radio;
    let fin = centro + radio;
    
    println!("\n{}", "=".repeat(80));
    println!("🔬 ESCANEO ULTRA-FINO PARA EL QUARK TOP");
    println!("{}", "=".repeat(80));
    println!("📊 Masa objetivo: {:.6} ± {:.6} GeV", MASA_TOP_EXP_GEV, MASA_TOP_ERR_GEV);
    println!("📊 Nf teórico: {} (residuo {})", NF_TOP, RESIDUO_TOP);
    println!("📊 Rango: {} - {}", inicio, fin);
    println!("📊 Fórmula: M = m_ref × K_III × Nf");
    println!("📊 m_ref = {:.6} MeV, K_III = {:.6}", M_REF_MEV, K_III);
    println!("📊 Producto = {:.6} MeV", M_REF_K_III_MEV);
    println!("{}", "=".repeat(80));
    
    let mut mejores = Vec::new();
    let start = std::time::Instant::now();
    
    for nf in (inicio..=fin).step_by(1) {
        if nf % ORDEN_OH != RESIDUO_TOP {
            continue;
        }
        
        let res_dict = calcular_masa_top(nf, false)?;
        
        Python::with_gil(|py| {
            let dict = res_dict.as_ref(py);
            if let Ok(Some(valido_obj)) = dict.get_item("valido") {
                if let Ok(valido) = valido_obj.extract::<bool>() {
                    if valido {
                        if let Ok(Some(error_obj)) = dict.get_item("error_rel_pct") {
                            if let Ok(error) = error_obj.extract::<f64>() {
                                if let Ok(Some(masa_obj)) = dict.get_item("masa_final_gev") {
                                    if let Ok(masa) = masa_obj.extract::<f64>() {
                                        mejores.push((nf, masa, error));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    
    mejores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    
    println!("\n{}", "=".repeat(80));
    println!("🏆 TOP 5 MEJORES CANDIDATOS (residuo {}):", RESIDUO_TOP);
    for (i, (nf, masa, error)) in mejores.iter().take(5).enumerate() {
        println!("   {}. Nf={:>8} | Masa={:.6} GeV | Error={:.4}%", i+1, nf, masa, error);
    }
    println!("{}", "=".repeat(80));
    
    let masa_teorica_gev = M_REF_K_III_MEV * NF_TOP as f64 / 1000.0;
    let error_teorico_pct = (masa_teorica_gev - MASA_TOP_EXP_GEV).abs() / MASA_TOP_EXP_GEV * 100.0;
    let dentro = (masa_teorica_gev - MASA_TOP_EXP_GEV).abs() <= MASA_TOP_ERR_GEV;
    
    println!("\n📋 Verificación Nf = {}:", NF_TOP);
    println!("   Masa = {:.6} GeV (exp: {:.6} ± {:.6})", masa_teorica_gev, MASA_TOP_EXP_GEV, MASA_TOP_ERR_GEV);
    println!("   Error = {:.4}%", error_teorico_pct);
    println!("   {}", if dentro { "✅ DENTRO DE INCERTIDUMBRE" } else { "⚠️ FUERA DE INCERTIDUMBRE" });
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("mejores", mejores)?;
        dict.set_item("tiempo_segundos", start.elapsed().as_secs_f64())?;
        dict.set_item("nf_teorico", NF_TOP)?;
        dict.set_item("masa_teorica_gev", masa_teorica_gev)?;
        dict.set_item("error_teorico_pct", error_teorico_pct)?;
        dict.set_item("dentro_incertidumbre", dentro)?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn get_constantes() -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("alpha_em", ALPHA_EM)?;
        dict.set_item("k_i", K_I)?;
        dict.set_item("k_iii", K_III)?;
        dict.set_item("m_e_mev", M_E_MEV)?;
        dict.set_item("m_ref_mev", M_REF_MEV)?;
        dict.set_item("nf_top", NF_TOP)?;
        dict.set_item("residuo_top", RESIDUO_TOP)?;
        dict.set_item("masa_top_exp_gev", MASA_TOP_EXP_GEV)?;
        dict.set_item("masa_top_err_gev", MASA_TOP_ERR_GEV)?;
        dict.set_item("orden_oh", ORDEN_OH)?;
        dict.set_item("m_ref_k_iii_mev", M_REF_K_III_MEV)?;
        Ok(dict.into())
    })
}

#[pymodule]
fn vpm48_engine_top(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calcular_masa_top, m)?)?;
    m.add_function(wrap_pyfunction!(escaneo_ultra_fino, m)?)?;
    m.add_function(wrap_pyfunction!(get_constantes, m)?)?;
    Ok(())
}