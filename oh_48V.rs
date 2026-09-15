// ============================================================================
// VCV48 KERNEL — oh_48.rs
// Audit of the 48^k structure (O_h group signature) in the VCV48 catalog.
//
// Memory engineering (Android/Termux, 2 threads):
//   - No global Vec/HashSet. All scans are O(1) streaming.
//   - u128 with checked_pow / saturating_mul (no overflow-panic).
//   - Large q values are returned to Python as String.
// ============================================================================
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const M_E_EV: f64 = 510998.9461;      // eV
const GROUP_ORDER: u128 = 48;          // |O_h| = 2^4 · 3

// ---------- Basic arithmetic (no memory) ----------
pub fn pow_u128(base: u128, exp: u32) -> u128 {
    base.checked_pow(exp).unwrap_or(u128::MAX)
}
pub fn floor_log2(n: u128) -> u32 {
    if n == 0 { 0 } else { 127 - n.leading_zeros() }
}
pub fn ceil_log2(n: u128) -> u32 {
    if n <= 1 { return 0; }
    let f = floor_log2(n);
    if (1u128 << f) == n { f } else { f + 1 }
}

// Factorizes q = 2^a · 3^b · rest. Returns (a, b, rest). O(1) memory.
pub fn factorize_2_3(mut q: u128) -> (u32, u32, u128) {
    let mut a = 0u32; let mut b = 0u32;
    while q > 1 && q % 2 == 0 { q /= 2; a += 1; }
    while q > 1 && q % 3 == 0 { q /= 3; b += 1; }
    (a, b, q)
}

// 48-adic valuation: q = 48^k · rest, rest not divisible by 48.
// Since 48 = 2^4·3, k = min( floor(a/4), b ).
pub fn val_48(q: u128) -> (u32, u128) {
    if q < 1 { return (0, q); }
    let (a, b, _rest23) = factorize_2_3(q);
    let k = (a / 4).min(b);
    let divisor = pow_u128(GROUP_ORDER, k);
    let rest = if divisor > 0 && divisor != u128::MAX { q / divisor } else { q };
    (k, rest)
}

// ---------- Mass window classification ----------
fn classify_mass_window(m: f64) -> &'static str {
    if m >= 1e5      { ">=100 keV" }
    else if m >= 1e3 { "keV (sterile-DM candidate)" }
    else if m >= 1.0 { "eV (high neutrino)" }
    else if m >= 1e-3 { "meV (low neutrino / axion-like)" }
    else if m >= 1e-6 { "ueV (QCD AXION WINDOW)" }
    else if m >= 1e-9 { "neV (ultralight axion)" }
    else if m >= 1e-20 { "Fuzzy-DM candidate" }
    else { "below fuzzy" }
}

// ---------- Catalog (fixed, ~12 entries) ----------
struct CatEntry {
    name: &'static str,
    q: u128,
    anexo: &'static str,   // what Annex V claims (for auditing)
    class: &'static str,
}

fn build_catalog() -> Vec<CatEntry> {
    let mut v: Vec<CatEntry> = Vec::new();
    // QCD axion (Annex V, Prediction 1)
    v.push(CatEntry { name: "QCD Axion", q: 12_230_590_464,
        anexo: "2^24 · 3^6", class: "Pure boson" });
    // Neutrinos (Annex V) — factorization is audited
    v.push(CatEntry { name: "nu1 light", q: 80_621_568,
    anexo: "Annex: 2^12·3^9 = 48^3·729 (3-smooth, not 48-exact)", class: "Mixed fermion (Annex)" });
    v.push(CatEntry { name: "nu2 solar", q: 49_430_988,
        anexo: "Annex: 2^2·3·4119249", class: "Mixed fermion" });
    v.push(CatEntry { name: "nu3 atm", q: 10_223_244,
        anexo: "Annex: 2^2·3·851937", class: "Mixed fermion" });
    // Gauge bosons / Higgs (Annex V, topological signature table)
    v.push(CatEntry { name: "W boson", q: 64,  anexo: "q = 2^6", class: "Pure boson" });
    v.push(CatEntry { name: "Z boson", q: 128, anexo: "q = 2^7", class: "Pure boson" });
    v.push(CatEntry { name: "Higgs",   q: 91,  anexo: "q = 7·13", class: "Mixed boson" });
    // ULA: 5 3-smooth states  q = 2^a · 3^b  (Annex V, Table ula_candidates)
    let ula: [(&str, u32, u32); 5] = [
        ("ULA S1", 69, 14), ("ULA S2", 50, 26), ("ULA S3", 4, 55),
        ("ULA S4", 88, 2),  ("ULA S5", 31, 38),
    ];
    for (name, a, b) in ula.iter() {
        let q = pow_u128(2, *a).saturating_mul(pow_u128(3, *b));
        v.push(CatEntry { name, q,
            anexo: "q = 2^a·3^b (ULA)", class: "Pure boson (ULA)" });
    }
    v
}

// ============================================================================
// FUNCTION 1: Mass ladder  m = m_e / 48^k   (O(1), ~23 rungs)
// ============================================================================
#[pyfunction]
fn ladder_48(py: Python) -> PyResult<Py<PyDict>> {
    let mut rungs: Vec<(u32, u128, f64)> = Vec::with_capacity(24);
    let mut q: u128 = 1;
    let mut k: u32 = 0;
    while k <= 40 {
        let mass_ev = M_E_EV / (q as f64);
        rungs.push((k, q, mass_ev));
        if q > u128::MAX / GROUP_ORDER { break; }
        q *= GROUP_ORDER;
        k += 1;
    }
    let out = PyDict::new(py);
    let list = PyList::empty(py);
    for (kk, qq, mm) in &rungs {
        let d = PyDict::new(py);
        d.set_item("k", *kk)?;
        d.set_item("q", qq.to_string())?;
        d.set_item("mass_ev", *mm)?;
        d.set_item("mass_log10_ev", mm.log10())?;
        d.set_item("window", classify_mass_window(*mm))?;
        list.append(d)?;
    }
    out.set_item("rungs", list)?;
    out.set_item("count", rungs.len())?;
    Ok(out.into_py(py))
}

// ============================================================================
// FUNCTION 2: Catalog audit  (the key function)
// Refactorizes each q from scratch and reports v_48, remainder, and whether it
// is an exact 48^k power.
// ============================================================================
#[pyfunction]
fn analyze_catalog(py: Python) -> PyResult<Py<PyDict>> {
    let catalog = build_catalog();
    let out = PyDict::new(py);
    let list = PyList::empty(py);
    let mut exact_count = 0usize;
    let exact_names = PyList::empty(py);
    for e in &catalog {
        let (a, b, rest23) = factorize_2_3(e.q);
        let (k, rest) = val_48(e.q);
        let is_exact = (rest == 1) && (k > 0);
        let is_3smooth = rest23 == 1;
        if is_exact {
            exact_count += 1;
            exact_names.append(e.name)?;
        }
        let d = PyDict::new(py);
        d.set_item("name", e.name)?;
        d.set_item("class", e.class)?;
        d.set_item("q", e.q.to_string())?;
        d.set_item("a", a)?;                                  // v_2(q)
        d.set_item("b", b)?;                                  // v_3(q)
        d.set_item("residual_coprime6", rest23.to_string())?; // remainder after removing 2 and 3
        d.set_item("is_3smooth", is_3smooth)?;
        d.set_item("v48", k)?;                                // 48-adic valuation
        d.set_item("remainder_after_48", rest.to_string())?;
        d.set_item("is_exact_power_of_48", is_exact)?;
        d.set_item("mass_ev", M_E_EV / (e.q as f64))?;
        d.set_item("anexo_note", e.anexo)?;
        list.append(d)?;
    }
    out.set_item("entries", list)?;
    out.set_item("total", catalog.len())?;
    out.set_item("exact_power_count", exact_count)?;
    out.set_item("exact_power_names", exact_names)?;
    Ok(out.into_py(py))
}

// ============================================================================
// FUNCTION 3: 3-smooth scan in a band [q_lo, q_hi]  (O(1) streaming)
// Counts how many 3-smooth numbers there are and lists the EXACT powers of 48
// in the band.
// ============================================================================
fn scan_band_core(q_lo: u128, q_hi: u128) -> (u64, Vec<(u32, u128)>) {
    let mut total_smooth: u64 = 0;
    let mut exact_powers: Vec<(u32, u128)> = Vec::new(); // rare: at most ~1 per band
    if q_lo < 1 { return (0, exact_powers); }
    for b in 0u32..=90 {
        let p3 = pow_u128(3, b);
        if p3 == 0 || p3 > q_hi { break; }
        let lo_div = (q_lo + p3 - 1) / p3;      // ceil(q_lo / p3)
        let hi_div = q_hi / p3;
        if hi_div == 0 { continue; }
        let a_lo = ceil_log2(lo_div);
        let a_hi = floor_log2(hi_div);
        if a_lo > a_hi { continue; }
        for a in a_lo..=a_hi {
            let q = pow_u128(2, a).saturating_mul(p3);
            if q < q_lo || q > q_hi { continue; }
            total_smooth += 1;
            let (k, rest) = val_48(q);
            if rest == 1 && k > 0 { exact_powers.push((k, q)); }
        }
    }
    (total_smooth, exact_powers)
}

#[pyfunction]
fn scan_3smooth_band(py: Python, q_lo: u128, q_hi: u128) -> PyResult<Py<PyDict>> {
    let (total_smooth, exact_powers) = scan_band_core(q_lo, q_hi);
    let out = PyDict::new(py);
    out.set_item("q_lo", q_lo.to_string())?;
    out.set_item("q_hi", q_hi.to_string())?;
    out.set_item("total_3smooth", total_smooth)?;
    let ep = PyList::empty(py);
    for (k, q) in &exact_powers {
        let d = PyDict::new(py);
        d.set_item("k", *k)?;
        d.set_item("q", q.to_string())?;
        ep.append(d)?;
    }
    out.set_item("exact_48_powers_in_band", ep)?;
    out.set_item("exact_count", exact_powers.len())?;
    Ok(out.into_py(py))
}

// ============================================================================
// MODULE REGISTRATION
// ============================================================================
#[pymodule]
fn oh_48(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ladder_48, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(scan_3smooth_band, m)?)?;
    m.add("M_E_EV", M_E_EV)?;
    m.add("GROUP_ORDER", GROUP_ORDER)?;
    Ok(())
}