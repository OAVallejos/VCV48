use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use nalgebra as na;
use na::{Unit, Vector3, Matrix3, Rotation3};

struct OhGroup {
    rotations: Vec<Matrix3<f64>>,
}

impl OhGroup {
    fn new() -> Self {
        let mut rotations = Vec::with_capacity(48);
        let pi = std::f64::consts::PI;

        // --- 1. Rotaciones propias (24 elementos) ---
        
        // Identidad - especificamos tipo explícitamente
        let identity: Matrix3<f64> = Matrix3::identity();
        rotations.push(identity);

        // Ejes coordenados
        let axes_coord = [
            Vector3::x_axis(),
            Vector3::y_axis(),
            Vector3::z_axis(),
        ];

        // Rotaciones de 90°, 180°, 270° alrededor de ejes x, y, z
        for axis_unit in &axes_coord {
            for &angle in &[pi/2.0, pi, 3.0*pi/2.0] {
                let rot = Rotation3::from_axis_angle(axis_unit, angle);
                rotations.push(rot.into());
            }
        }

        // Diagonales del cubo
        let axes_diag_vec = [
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
        ];
        let axes_diag: Vec<Unit<Vector3<f64>>> = axes_diag_vec.iter().map(|v| Unit::new_normalize(*v)).collect();

        // Rotaciones de 120° y 240° alrededor de diagonales
        for axis_unit in &axes_diag {
            for &angle in &[2.0*pi/3.0, 4.0*pi/3.0] {
                let rot = Rotation3::from_axis_angle(axis_unit, angle);
                rotations.push(rot.into());
            }
        }

        // Ejes por centros de aristas
        let axes_edge_vec = [
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, -1.0),
            Vector3::new(0.0, 1.0, 1.0),
            Vector3::new(0.0, 1.0, -1.0),
        ];
        let axes_edge: Vec<Unit<Vector3<f64>>> = axes_edge_vec.iter().map(|v| Unit::new_normalize(*v)).collect();

        // Rotaciones de 180° alrededor de estos ejes
        for axis_unit in &axes_edge {
            let rot = Rotation3::from_axis_angle(axis_unit, pi);
            rotations.push(rot.into());
        }

        // --- 2. Rotaciones impropias (24 elementos) ---
        // CORRECCIÓN CRÍTICA: Especificamos el tipo explícitamente
        let inversion: Matrix3<f64> = -Matrix3::identity();
        let n_proper = rotations.len();
        for i in 0..n_proper {
            // También necesitamos asegurar el tipo en la multiplicación
            let rot_proper: Matrix3<f64> = rotations[i];
            rotations.push(inversion * rot_proper);
        }

        assert_eq!(rotations.len(), 48, "O_h debe tener 48 elementos");
        println!("✅ Grupo O_h generado con {} matrices.", rotations.len());
        OhGroup { rotations }
    }

    /// Matriz de deformación por birrefringencia delta (conserva el volumen)
    fn matriz_deformacion(delta: f64) -> Matrix3<f64> {
        let d1 = 1.0 + delta;
        let d2 = 1.0 - delta;
        let d3 = 1.0 / (1.0 - delta * delta);

        Matrix3::new(
            d1, 0.0, 0.0,
            0.0, d2, 0.0,
            0.0, 0.0, d3
        )
    }

    /// Calcula el promedio de Tr(M·R)²/3 sobre las 48 matrices
    fn promedio_deformado(&self, delta: f64) -> f64 {
        let m = Self::matriz_deformacion(delta);
        let mut suma = 0.0;
        let n = self.rotations.len() as f64;

        println!("\n=== CÁLCULO EN MÉTRICA DEFORMADA (δ = {:.8} rad) ===", delta);
        println!("Índice | Traza(M·R)  | (Tr²/3)");
        println!("-------|-------------|----------");

        for (i, r) in self.rotations.iter().enumerate() {
            let producto = m * r;
            let traza = producto.trace();
            let termino = (traza * traza) / 3.0;
            suma += termino;
            
            // Mostrar solo cada 6 elementos para no saturar, o todos si quieres ver
            if i % 6 == 0 {
                println!("{:3}    | {:11.8} | {:9.8}", i, traza, termino);
            }
        }

        println!("-------|-------------|----------");
        let promedio = suma / n;
        println!("SUMA: {:.8}", suma);
        println!("PROMEDIO (Tr(M·R)²/3): {:.8}", promedio);
        
        promedio
    }
}

/// Función expuesta a Python
#[pyfunction]
fn calcular_promedio_oh_deformado(delta: f64) -> PyResult<f64> {
    let oh = OhGroup::new();
    let promedio = oh.promedio_deformado(delta);
    Ok(promedio)
}

/// Módulo Python
#[pymodule]
fn oh_group(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calcular_promedio_oh_deformado, m)?)?;
    Ok(())
}