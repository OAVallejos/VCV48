// ============================================================================
// VCV48 KERNEL v10.2.1 — Targeted capture of the Ultralight Axion (ULA)
// Module: axion_ula2.rs
// ============================================================================
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const M_E_EV: f64 = 510998.9461;
const M_TARGET_EV: f64 = 1.8e-22;
const BAND: f64 = 0.03;
const STABILITY_QCD: f64 = 0.6262;
const SYMMETRY_PURE_BOSON: f64 = 0.82;
const CHARGE_PURE_BOSON: i32 = 2;

#[derive(Debug, Clone)]
pub struct UlaCandidate {
    pub p: u128, pub q: u128, pub mass_ev: f64, pub deviation: f64,
    pub stability: f64, pub symmetry: f64, pub charge: i32,
    pub factorization: String, pub a: u32, pub b: u32,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub candidates: Vec<UlaCandidate>, pub q_lo: u128, pub q_hi: u128,
    pub total_states_scanned: u64, pub best_candidate: Option<UlaCandidate>,
}

pub fn pow_u128(base: u128, exp: u32) -> u128 { base.checked_pow(exp).unwrap_or(u128::MAX) }
pub fn floor_log2(n: u128) -> u32 { if n == 0 { 0 } else { 127 - n.leading_zeros() } }

pub fn factorize_smooth(q: u128) -> (u32, u32, u128) {
    let mut n = q; let mut a = 0u32; let mut b = 0u32;
    while n % 2 == 0 { n /= 2; a += 1; }
    while n % 3 == 0 { n /= 3; b += 1; }
    (a, b, n)
}

pub fn format_factorization(q: u128) -> String {
    let (a, b, remainder) = factorize_smooth(q);
    if remainder == 1 {
        if a > 0 && b > 0 { format!("2^{} · 3^{}", a, b) }
        else if a > 0 { format!("2^{}", a) }
        else if b > 0 { format!("3^{}", b) }
        else { "1".to_string() }
    } else { format!("2^{} · 3^{} · {}", a, b, remainder) }
}

pub fn is_3_smooth(q: u128) -> bool { let (_, _, remainder) = factorize_smooth(q); remainder == 1 }

pub fn compute_stability(_p: u128, q: u128) -> f64 {
    let (a, b, remainder) = factorize_smooth(q);
    let q_qcd: f64 = 12230590464.0;
    let size_factor = ((q as f64).log10() / q_qcd.log10()).powf(0.01);
    let smooth_factor = if remainder == 1 { 1.0 } else { 0.98 };
    let parity_factor = if (a % 2 == 0) && (b % 2 == 0) { 1.0 } else { 0.998 };
    STABILITY_QCD * size_factor * smooth_factor * parity_factor
}

pub fn compute_symmetry(q: u128) -> f64 {
    let (_, _, remainder) = factorize_smooth(q);
    if remainder == 1 { SYMMETRY_PURE_BOSON }
    else { SYMMETRY_PURE_BOSON * (0.5 + 0.5 / (1.0 + (remainder as f64).log10())) }
}

// BUG FIX #2: correct destructuring with underscores
pub fn compute_charge(_p: u128, q: u128) -> i32 {
    let (_, _, remainder) = factorize_smooth(q);
    if remainder == 1 { CHARGE_PURE_BOSON } else { (remainder % 7) as i32 + 1 }
}

pub fn search_ula_axion(target_mass_ev: Option<f64>, band: Option<f64>) -> SearchResult {
    let target = target_mass_ev.unwrap_or(M_TARGET_EV);
    let band_width = band.unwrap_or(BAND);
    let q_lo = ((M_E_EV / (target * (1.0 + band_width))) as u128).max(1);
    let q_hi = ((M_E_EV / (target * (1.0 - band_width))) as u128).max(q_lo);

    let mut candidates = Vec::new();
    let mut total_scanned = 0u64;

    for b in 0u32..=64 {
        let p3 = pow_u128(3, b);
        if p3 > q_hi { break; }
        let a0 = if p3 > 0 { floor_log2(q_lo / p3) } else { 0 };

        for a in a0.saturating_sub(1)..=a0 + 2 {
            let p2 = pow_u128(2, a);
            let q = p2.saturating_mul(p3);
            if q == 0 { continue; }
            total_scanned += 1;
            if q < q_lo || q > q_hi { continue; }
            if !is_3_smooth(q) { continue; }

            let mass_ev = M_E_EV / (q as f64);
            let deviation = (mass_ev - target) / target;

            candidates.push(UlaCandidate {
                p: 1, q, mass_ev, deviation,
                stability: compute_stability(1, q), symmetry: compute_symmetry(q),
                charge: compute_charge(1, q), factorization: format_factorization(q), a, b,
            });
        }
    }
    candidates.sort_by(|a, b| a.deviation.abs().partial_cmp(&b.deviation.abs()).unwrap());
    SearchResult { candidates: candidates.clone(), q_lo, q_hi, total_states_scanned: total_scanned, best_candidate: candidates.first().cloned() }
}

// FUNCTION 1: Search
#[pyfunction]
fn search_ula(py: Python, target_ev: Option<f64>, band: Option<f64>) -> PyResult<Py<PyDict>> {
    let result = search_ula_axion(target_ev, band);
    let dict = PyDict::new(py);
    dict.set_item("q_lo", result.q_lo.to_string())?;
    dict.set_item("q_hi", result.q_hi.to_string())?;
    dict.set_item("total_states_scanned", result.total_states_scanned)?;

    let candidates_list = PyList::empty(py);
    for cand in &result.candidates {
        let cand_dict = PyDict::new(py);
        cand_dict.set_item("p", cand.p.to_string())?;
        cand_dict.set_item("q", cand.q.to_string())?;
        cand_dict.set_item("mass_ev", cand.mass_ev)?;
        cand_dict.set_item("deviation_percent", cand.deviation * 100.0)?;
        cand_dict.set_item("stability", cand.stability)?;
        cand_dict.set_item("symmetry", cand.symmetry)?;
        cand_dict.set_item("charge", cand.charge)?;
        cand_dict.set_item("factorization", &cand.factorization)?;
        cand_dict.set_item("a", cand.a)?;
        cand_dict.set_item("b", cand.b)?;
        candidates_list.append(cand_dict)?;
    }
    dict.set_item("candidates", candidates_list)?;

    if let Some(best) = &result.best_candidate {
        let best_dict = PyDict::new(py);
        best_dict.set_item("p", best.p.to_string())?;
        best_dict.set_item("q", best.q.to_string())?;
        best_dict.set_item("mass_ev", best.mass_ev)?;
        best_dict.set_item("deviation_percent", best.deviation * 100.0)?;
        best_dict.set_item("stability", best.stability)?;
        best_dict.set_item("symmetry", best.symmetry)?;
        best_dict.set_item("charge", best.charge)?;
        best_dict.set_item("factorization", &best.factorization)?;
        dict.set_item("best_candidate", best_dict)?;
    }
    Ok(dict.into_py(py))
}

// FUNCTION 2: Individual analysis
#[pyfunction]
fn analyze_ula(py: Python, p: u128, q: u128) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    let mass_ev = (p as f64 / q as f64) * M_E_EV;
    dict.set_item("p", p.to_string())?;
    dict.set_item("q", q.to_string())?;
    dict.set_item("mass_ev", mass_ev)?;
    dict.set_item("mass_ev_str", format!("{:.4e}", mass_ev))?;
    dict.set_item("stability", compute_stability(p, q))?;
    dict.set_item("symmetry", compute_symmetry(q))?;
    // BUG FIX #1: was compute_charge(p, 9), now it is compute_charge(p, q)
    dict.set_item("charge", compute_charge(p, q))?;
    dict.set_item("factorization", format_factorization(q))?;
    dict.set_item("is_3_smooth", is_3_smooth(q))?;

    let q_qcd: u128 = 12230590464;
    let ratio = q as f64 / q_qcd as f64;
    dict.set_item("ratio_to_qcd", ratio)?;
    dict.set_item("ratio_to_qcd_str", format!("{:.4e}", ratio))?;
    Ok(dict.into_py(py))
}

// MODULE REGISTRATION
#[pymodule]
fn axion_ula(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search_ula, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ula, m)?)?;
    m.add("M_TARGET_EV", M_TARGET_EV)?;
    m.add("BAND", BAND)?;
    Ok(())
}