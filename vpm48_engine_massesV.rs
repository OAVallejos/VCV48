// vpm48_engine_masses.rs - VCV48 kernel for mass calculation
// CORRECTED VERSION: M = N_f · m_e (no double K_I, no phantom QCD)
// Build with: maturin develop --release

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

// ============================================================
// FUNDAMENTAL CONSTANTS
// ============================================================

const ALPHA_EM: f64 = 0.0072973525693;
const M_E_MEV: f64 = 0.5109989461;

// Elastic factors
const K_I: f64 = 8.0 * ALPHA_EM;    // = 0.0583788205544
const K_II: f64 = 2.63;
const K_III: f64 = 0.8025;

// Reference mass (for Plateau III)
const M_REF_MEV: f64 = M_E_MEV / K_I; // = 8.753156 MeV

// ============================================================
// Nf OF EACH PARTICLE
// ============================================================

const NF_ELECTRON: f64 = 1.0;
const NF_MUON: f64 = 207.0;
const NF_TAU: f64 = 3477.0;
const NF_UP: f64 = 4.0;
const NF_DOWN: f64 = 9.0;
const NF_STRANGE: f64 = 183.0;
const NF_CHARM: f64 = 2485.0;
const NF_BOTTOM: f64 = 8180.0;
const NF_TOP: f64 = 24603.0;
const NF_PROTON: f64 = 1836.0;
const NF_NEUTRON: f64 = 1839.0;

// Bosons
const BOSON_OFFSET: f64 = 180000.0;
const NF_W_THEORETICAL: f64 = 15743000.0;
const NF_Z_THEORETICAL: f64 = 20176000.0;
const NF_H_THEORETICAL: f64 = 37625000.0;
const NF_W_REAL: f64 = NF_W_THEORETICAL - BOSON_OFFSET;
const NF_Z_REAL: f64 = NF_Z_THEORETICAL - BOSON_OFFSET;
const NF_H_REAL: f64 = NF_H_THEORETICAL - BOSON_OFFSET;

// Boson projection factors
const P_W: f64 = 260.23;
const P_Z: f64 = 294.70;
const P_H: f64 = 401.99;

// ============================================================
// NET RADIATIVE CORRECTIONS (Δ_net ≈ ±0.2 MeV)
// ============================================================

const DELTA_NET_ELECTRON: f64 = 0.0;
const DELTA_NET_MUON: f64 = -0.12;
const DELTA_NET_TAU: f64 = -0.04;
const DELTA_NET_UP: f64 = 0.116;
const DELTA_NET_DOWN: f64 = 0.101;
const DELTA_NET_STRANGE: f64 = -0.01;
const DELTA_NET_CHARM: f64 = 0.2;
const DELTA_NET_BOTTOM: f64 = 0.1;
const DELTA_NET_PROTON: f64 = 0.07;
const DELTA_NET_NEUTRON: f64 = -0.13;

// ============================================================
// MAIN FUNCTION: FERMION MASS (PLATEAU I)
// ============================================================

/// M = N_f · m_e + Δ_net
fn fermion_mass(nf: f64, delta_net: f64) -> f64 {
    nf * M_E_MEV + delta_net
}

// ============================================================
// FUNCTIONS EXPOSED TO PYTHON
// ============================================================

#[pyfunction]
fn calculate_particle_mass(name: String) -> PyResult<Py<PyDict>> {
    let (nf, mass_mev, plateau) = match name.as_str() {
        // Leptons
        "electron" => (NF_ELECTRON, fermion_mass(NF_ELECTRON, DELTA_NET_ELECTRON), "I"),
        "muon" => (NF_MUON, fermion_mass(NF_MUON, DELTA_NET_MUON), "I"),
        "tau" => (NF_TAU, fermion_mass(NF_TAU, DELTA_NET_TAU), "I"),
        
        // Light quarks
        "up" => (NF_UP, fermion_mass(NF_UP, DELTA_NET_UP), "I"),
        "down" => (NF_DOWN, fermion_mass(NF_DOWN, DELTA_NET_DOWN), "I"),
        "strange" => (NF_STRANGE, fermion_mass(NF_STRANGE, DELTA_NET_STRANGE), "I"),
        "charm" => (NF_CHARM, fermion_mass(NF_CHARM, DELTA_NET_CHARM), "I"),
        "bottom" => (NF_BOTTOM, fermion_mass(NF_BOTTOM, DELTA_NET_BOTTOM), "I"),
        
        // Top quark (Plateau III)
        "top" => (NF_TOP, M_REF_MEV * K_III * NF_TOP, "III"),
        
        // Nucleons
        "proton" => (NF_PROTON, fermion_mass(NF_PROTON, DELTA_NET_PROTON), "I"),
        "neutron" => (NF_NEUTRON, fermion_mass(NF_NEUTRON, DELTA_NET_NEUTRON), "I"),
        
        _ => (0.0, 0.0, "?"),
    };
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("name", name)?;
        dict.set_item("nf", nf)?;
        dict.set_item("mass_mev", mass_mev)?;
        dict.set_item("mass_gev", mass_mev / 1000.0)?;
        dict.set_item("plateau", plateau)?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn calculate_boson_mass(name: String) -> PyResult<Py<PyDict>> {
    let (nf_theoretical, nf_real, p) = match name.as_str() {
        "w" => (NF_W_THEORETICAL, NF_W_REAL, P_W),
        "z" => (NF_Z_THEORETICAL, NF_Z_REAL, P_Z),
        "higgs" => (NF_H_THEORETICAL, NF_H_REAL, P_H),
        _ => (0.0, 0.0, 1.0),
    };
    
    // M = Nf_real · m_e · K_II / P_B
    let mass_mev = nf_real * M_E_MEV * K_II / p;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("name", name)?;
        dict.set_item("nf_theoretical", nf_theoretical)?;
        dict.set_item("nf_real", nf_real)?;
        dict.set_item("projection_factor", p)?;
        dict.set_item("mass_mev", mass_mev)?;
        dict.set_item("mass_gev", mass_mev / 1000.0)?;
        dict.set_item("plateau", "II")?;
        Ok(dict.into())
    })
}

#[pyfunction]
fn calculate_all_masses() -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        
        let particles = [
            "electron", "muon", "tau",
            "up", "down", "strange", "charm", "bottom", "top",
            "proton", "neutron",
        ];
        
        for name in particles {
            let result = calculate_particle_mass(name.to_string())?;
            dict.set_item(name, result)?;
        }
        
        let bosons = ["w", "z", "higgs"];
        for name in bosons {
            let result = calculate_boson_mass(name.to_string())?;
            dict.set_item(name, result)?;
        }
        
        Ok(dict.into())
    })
}

#[pyfunction]
fn get_constants() -> PyResult<Py<PyDict>> {
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("alpha_em", ALPHA_EM)?;
        dict.set_item("m_e_mev", M_E_MEV)?;
        dict.set_item("k_i", K_I)?;
        dict.set_item("k_ii", K_II)?;
        dict.set_item("k_iii", K_III)?;
        dict.set_item("m_ref_mev", M_REF_MEV)?;
        dict.set_item("boson_offset", BOSON_OFFSET)?;
        Ok(dict.into())
    })
}

#[pymodule]
fn vpm48_engine_masses(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_particle_mass, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_boson_mass, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_all_masses, m)?)?;
    m.add_function(wrap_pyfunction!(get_constants, m)?)?;
    Ok(())
}