// vpm48_engine_optimized.rs
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::f64::consts::PI;

// Constantes físicas
const ALPHA: f64 = 1.0 / 137.036;
const HBAR: f64 = 6.582119e-16; // eV·s
const C: f64 = 2.99792458e8;    // m/s
const MEV_TO_KG: f64 = 1.78266192e-36; // MeV/c² → kg
const G_FERMI: f64 = 1.1663787e-5;     // GeV⁻²
const M_Z: f64 = 91.1876;        // GeV
const M_W: f64 = 80.379;         // GeV

// Constantes de simetría
const SYMMETRY_OPS: [(i32, i32, i32); 48] = [
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    (1, 1, 2), (1, 2, 1), (2, 1, 1), (-1, -1, -2),
    (-1, -2, -1), (-2, -1, -1), (1, -1, 2), (1, 2, -1),
    (2, 1, -1), (-1, 1, -2), (-1, -2, 1), (-2, -1, 1),
    (1, -1, -2), (1, -2, 1), (2, -1, 1), (-1, 1, 2),
    (-1, 2, -1), (-2, 1, -1), (1, 2, 2), (2, 1, 2),
    (2, 2, 1), (-1, -2, -2), (-2, -1, -2), (-2, -2, -1),
    (1, -2, -2), (2, -1, -2), (2, -2, -1), (-1, 2, 2),
    (-2, 1, 2), (-2, 2, 1), (1, -2, 2), (2, -1, 2),
    (2, -2, 1), (-1, 2, -2), (-2, 1, -2), (-2, 2, -1),
    (1, 2, -2), (2, 1, -2), (2, 2, -1), (-1, -2, 2)
];

// Estructura para un punto 3D
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3D {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    
    fn transform(&self, op: &(i32, i32, i32)) -> Self {
        let (sx, sy, sz) = *op;
        let x = self.x * sx as f64;
        let y = self.y * sy as f64;
        let z = self.z * sz as f64;
        Self { x, y, z }
    }
    
    fn distance_sq(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx*dx + dy*dy + dz*dz
    }
}

// Genera puntos de la trenza para un Nf dado
fn generate_braid_points(nf: i32) -> Vec<Point3D> {
    let mut points = Vec::with_capacity(nf as usize);
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0; // Proporción áurea
    
    for i in 0..nf {
        let t = i as f64 / nf as f64;
        let theta = 2.0 * PI * t;
        let r = (1.0 + 0.1 * (2.0 * PI * t * 3.0).sin()) * nf as f64 * 0.1;
        
        let x = r * theta.cos();
        let y = r * theta.sin();
        let z = (t - 0.5) * nf as f64 * 0.2;
        
        points.push(Point3D::new(x, y, z));
    }
    
    points
}

// Versión optimizada con KD-Tree simplificado (usando HashMap espacial)
fn is_equivalent_configuration_optimized(points: &[Point3D]) -> bool {
    let n = points.len();
    if n < 10 {
        return false; // Demasiado pequeño para tener simetría significativa
    }
    
    // Umbral de distancia para considerar puntos equivalentes
    let tolerance = 1e-6;
    
    // Para cada operación de simetría
    for op in SYMMETRY_OPS.iter() {
        // Transformar todos los puntos
        let transformed: Vec<Point3D> = points.iter()
            .map(|p| p.transform(op))
            .collect();
        
        // Construir un mapa espacial (hash simplificado)
        let mut point_map: HashMap<u64, &Point3D> = HashMap::with_capacity(n);
        
        // Indexar puntos originales por coordenadas discretizadas
        for p in points {
            // Discretizar coordenadas para usar como clave
            let key = (
                (p.x / tolerance).round() as i64,
                (p.y / tolerance).round() as i64,
                (p.z / tolerance).round() as i64
            );
            
            // Hash simple
            let hash = (key.0.wrapping_mul(73856093) 
                ^ key.1.wrapping_mul(19349663) 
                ^ key.2.wrapping_mul(83492791)) as u64;
            
            point_map.insert(hash, p);
        }
        
        // Verificar cada punto transformado
        let mut matches = 0;
        for tp in &transformed {
            let key = (
                (tp.x / tolerance).round() as i64,
                (tp.y / tolerance).round() as i64,
                (tp.z / tolerance).round() as i64
            );
            
            let hash = (key.0.wrapping_mul(73856093) 
                ^ key.1.wrapping_mul(19349663) 
                ^ key.2.wrapping_mul(83492791)) as u64;
            
            if let Some(original) = point_map.get(&hash) {
                // Verificar distancia real
                if tp.distance_sq(original) < tolerance * tolerance {
                    matches += 1;
                }
            }
        }
        
        // Si todos los puntos tienen correspondencia, la configuración es equivalente
        if matches == n {
            return true;
        }
    }
    
    false
}

// Versión original (para comparación)
fn is_equivalent_configuration_original(points: &[Point3D]) -> bool {
    let n = points.len();
    let tolerance = 1e-6;
    
    for op in SYMMETRY_OPS.iter() {
        let transformed: Vec<Point3D> = points.iter()
            .map(|p| p.transform(op))
            .collect();
        
        let mut all_found = true;
        for tp in &transformed {
            let mut found = false;
            for op2 in points {
                if tp.distance_sq(op2) < tolerance * tolerance {
                    found = true;
                    break;
                }
            }
            if !found {
                all_found = false;
                break;
            }
        }
        
        if all_found {
            return true;
        }
    }
    
    false
}

#[pyfunction]
fn analizar_por_nf(nf: i32, strands: i32) -> PyResult<Py<PyDict>> {
    let points = generate_braid_points(nf);
    
    // Usar versión optimizada
    let es_simetrica = if nf > 50000 {
        // Para Nf grandes, usar la versión optimizada
        is_equivalent_configuration_optimized(&points)
    } else {
        // Para Nf pequeños, podemos usar la original
        is_equivalent_configuration_original(&points)
    };
    
    // Calcular energía base
    let masa_base = (nf as f64).sqrt() * 0.511e6; // eV
    
    // Ajuste por simetría
    let factor_estabilidad = if es_simetrica { 0.1047 } else { 0.0 };
    
    // Energía final con corrección para Nf grandes
    let energia = if nf > 100000 {
        // Régimen logarítmico para Nf grandes
        masa_base * (1.0 + 0.01 * (nf as f64).ln())
    } else {
        masa_base * (1.0 + 0.001 * nf as f64)
    };
    
    // Crear diccionario de resultados
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("nf", nf)?;
        dict.set_item("energia_ev", energia)?;
        dict.set_item("estabilidad_oh", factor_estabilidad)?;
        dict.set_item("tipo", if es_simetrica { "SIMETRICO_48" } else { "GENERICO" })?;
        Ok(dict.into())
    })
}

#[pymodule]
fn vpm48_engine_optimized(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analizar_por_nf, m)?)?;
    Ok(())
}