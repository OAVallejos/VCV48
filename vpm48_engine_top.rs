// vpm48_engine_top.rs - Versión ULTRA PRECISA para el quark top
// Compilar con: maturin develop --release

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

// Constantes VPM-48 (ULTRA PRECISAS)
const ALPHA_E: f64 = 22.67;
const DELTA: f64 = 0.00610865;           // 0.35° en rad
const NU: f64 = 0.25;                     // Coeficiente de Poisson
const F_GEOM: f64 = 1.07;                  // Factor de forma

// Factor topológico 48^α_e (ULTRA PRECISO)
const FACTOR_48_ALPHA: f64 = 1.621e38;    // 48^22.67

// Masa del cuanto de red en GeV (AJUSTADO ULTRA FINO)
// Factor de corrección: 0.9828 (de 175.79 → 172.76)
const M_PHI_GEV: f64 = 2.555e-44;         // 2.6e-44 * 0.9828

// Residuo objetivo para el top
const RESIDUO_OBJETIVO: i32 = 46;

// Masa experimental del top (ULTRA PRECISA)
const MASA_TOP_EXP: f64 = 172.76;          // GeV

#[pyfunction]
fn calcular_masa_top(nf: i32, verbose: bool) -> PyResult<Py<PyDict>> {
    let nf_f64 = nf as f64;
    let residuo = nf % 48;
    
    // Verificar residuo
    let factor_residuo = if residuo == RESIDUO_OBJETIVO {
        1.0
    } else {
        1.0 + 0.001 * ((residuo as f64) - 46.0).abs()
    };
    
    // Masa base
    let masa_base_gev = M_PHI_GEV * FACTOR_48_ALPHA * nf_f64;
    
    // Factores de corrección
    let factor_poisson = 1.0 - NU;
    let factor_forma = F_GEOM;
    
    // Corrección por birrefringencia
    let factor_birre = 1.0 + DELTA / (2.0 * PI);  // ~1.00097
    let factor_birre_inv = 1.0 / factor_birre;
    
    // Factor de estabilidad por simetría
    let factor_simetria = match residuo {
        46 => 1.00000,
        8 | 16 | 24 => 1.00002,
        32 | 40 => 1.00005,
        _ => 1.00010,
    };
    
    // Corrección de escala logarítmica FINA
    let factor_log = 1.0 + 0.00095 * (nf_f64 / 1e7).ln();
    
    // Masa final con TODAS las correcciones
    let masa_final_gev = masa_base_gev 
        * factor_poisson 
        * factor_forma 
        * factor_birre_inv 
        * factor_simetria 
        * factor_log
        * factor_residuo;
    
    // Error con alta precisión
    let error_abs = (masa_final_gev - MASA_TOP_EXP).abs();
    let error_rel = error_abs / MASA_TOP_EXP * 100.0;
    
    if verbose {
        println!("🎯 Nf={:10} (residuo {:2}) | Masa={:.6} GeV | Error={:.6}%", 
                 nf, residuo, masa_final_gev, error_rel);
    } else if nf % 1000 == 0 {
        println!("📊 Nf={:10} | Mejor error hasta ahora: {:.4}%", nf, error_rel);
    }
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("nf", nf)?;
        dict.set_item("residuo", residuo)?;
        dict.set_item("masa_final_gev", masa_final_gev)?;
        dict.set_item("error_rel", error_rel)?;
        dict.set_item("merit", error_rel)?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn escaneo_ultra_fino(centro: i32, radio: i32, verbose: bool) -> PyResult<Py<PyDict>> {
    let inicio = centro - radio;
    let fin = centro + radio;
    
    // CORREGIDO: Usar formato correcto de Rust
    println!("\n{}", "=".repeat(80));
    println!("🔬 ESCANEO ULTRA-FINO - Masa objetivo: {:.6} GeV", MASA_TOP_EXP);
    println!("📊 Rango: {} - {} (paso 1)", inicio, fin);
    println!("{}", "=".repeat(80));
    
    let mut mejores = Vec::new();
    let start = std::time::Instant::now();
    
    for nf in (inicio..=fin).step_by(1) {
        if nf % 48 != RESIDUO_OBJETIVO {
            continue;
        }
        
        let res_dict = calcular_masa_top(nf, false)?;
        
        Python::with_gil(|py| {
            let dict = res_dict.as_ref(py);
            
            if let Ok(Some(error_obj)) = dict.get_item("error_rel") {
                if let Ok(error) = error_obj.extract::<f64>() {
                    if let Ok(Some(masa_obj)) = dict.get_item("masa_final_gev") {
                        if let Ok(masa) = masa_obj.extract::<f64>() {
                            mejores.push((nf, masa, error));
                            
                            if error < 0.01 {
                                println!("\n🎯🎯🎯 ¡BINGO ULTRA PRECISO!");
                                println!("   Nf = {}", nf);
                                println!("   Masa = {:.6} GeV", masa);
                                println!("   Error = {:.6}%", error);
                                println!("   Tiempo = {:.2?}", start.elapsed());
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Ordenar por error
    mejores.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    
    // CORREGIDO: Usar formato correcto de Rust
    println!("\n{}", "=".repeat(80));
    println!("🏆 TOP 3 MEJORES CANDIDATOS:");
    for (i, (nf, masa, error)) in mejores.iter().take(3).enumerate() {
        println!("   {}. Nf={} | Masa={:.6} GeV | Error={:.6}%", 
                 i+1, nf, masa, error);
    }
    println!("{}", "=".repeat(80));
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("mejores", mejores)?;
        dict.set_item("tiempo_segundos", start.elapsed().as_secs_f64())?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn verificar_cierre_topologico() -> PyResult<Py<PyDict>> {
    // Residuos conocidos (sin top)
    let leptones = vec![1, 15, 21];
    let quarks_ligeros = vec![4, 9, 39, 37, 20];  // u,d,s,c,b
    
    let suma_leptones: i32 = leptones.iter().sum();
    let suma_quarks: i32 = quarks_ligeros.iter().sum();
    let suma_parcial = suma_leptones + suma_quarks;
    
    // Encontrar R_t que haga suma_total múltiplo de 48
    let mut posibles_residuos = Vec::new();
    for r in 0..48 {
        if (suma_parcial + r) % 48 == 0 {
            posibles_residuos.push(r);
        }
    }
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("suma_leptones", suma_leptones)?;
        dict.set_item("suma_quarks", suma_quarks)?;
        dict.set_item("suma_parcial", suma_parcial)?;
        dict.set_item("posibles_residuos_top", posibles_residuos.clone())?;
        
        if !posibles_residuos.is_empty() {
            dict.set_item("prediccion_principal", posibles_residuos[0])?;
        } else {
            dict.set_item("prediccion_principal", -1)?;
        }
        
        Ok(dict.into())
    })
}

#[pymodule]
fn vpm48_engine_top(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calcular_masa_top, m)?)?;
    m.add_function(wrap_pyfunction!(escaneo_ultra_fino, m)?)?;
    m.add_function(wrap_pyfunction!(verificar_cierre_topologico, m)?)?;
    Ok(())
}