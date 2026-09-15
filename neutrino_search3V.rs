// neutrino_search — VCV48 KERNEL v10.1 (NEUTRINO + ν₁ REFINEMENT)
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::f64::consts::PI;

const M_E_EV: f64 = 510998.9461;
const ALPHA: f64 = 1.0 / 137.036;
const A0: f64 = 14.075;
const R_Y: f64 = 675.6;
const DELTA_M21_SQ: f64 = 7.53e-5;
const DELTA_M32_SQ: f64 = 2.453e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractionalNf {
    pub numerator: u64,
    pub denominator: u64,
}

impl FractionalNf {
    #[inline]
    pub fn new(numerator: u64, denominator: u64) -> Self {
        let gcd = Self::gcd(numerator, denominator);
        Self { numerator: numerator / gcd, denominator: denominator / gcd }
    }
    #[inline]
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 { let t = b; b = a % b; a = t; }
        a
    }
    #[inline]
    pub fn to_f64(&self) -> f64 { self.numerator as f64 / self.denominator as f64 }
    #[inline]
    pub fn mass_ev(&self) -> f64 { self.to_f64() * M_E_EV }
    #[inline]
    pub fn mass_gev(&self) -> f64 { self.mass_ev() / 1e9 }
}

#[derive(Debug, Clone)]
pub struct ParticleCandidate {
    pub nf: FractionalNf,
    pub mass_ev: f64,
    pub mass_gev: f64,
    pub energy_formation: f64,
    pub symmetry_score: f64,
    pub topological_charge: i32,
    pub winding_number: i32,
    pub stability_metric: f64,
}

impl PartialEq for ParticleCandidate {
    fn eq(&self, other: &Self) -> bool { self.stability_metric == other.stability_metric }
}
impl Eq for ParticleCandidate {}
impl Ord for ParticleCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.stability_metric.partial_cmp(&self.stability_metric).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for ParticleCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl ParticleCandidate {
    #[inline]
    pub fn analyze(nf: FractionalNf) -> Self {
        let mass_ev = nf.mass_ev();
        let symmetry_score = Self::calculate_symmetry(nf);
        let topological_charge = Self::calculate_charge(nf);
        let complexity = nf.numerator.saturating_add(nf.denominator) as f64;
        let simplicity = 1.0 / (1.0 + complexity.ln().max(0.0));
        let charge_factor = match topological_charge.abs() { 0 => 1.0, 1 => 0.85, 2 => 0.6, _ => 0.3 };
        
        let mut stability = (symmetry_score * 0.50) + (simplicity * 0.15) + (charge_factor * 0.35);
        if nf.numerator == 1 && nf.denominator == 1 {
            stability += 0.40;
            if stability > 1.0 { stability = 1.0; }
        }

        Self {
            nf, mass_ev, mass_gev: mass_ev / 1e9,
            energy_formation: (nf.to_f64() * A0).powi(2) * (1.0 / (8.0 * PI * ALPHA)) * (R_Y / (A0 / 48.0)).ln() / (4.0 * PI),
            symmetry_score, topological_charge,
            winding_number: (nf.to_f64() * 48.0).round() as i32,
            stability_metric: stability,
        }
    }

    #[inline]
    fn oh_purity(n: u64) -> f64 {
        let mut temp = n; let mut f2 = 0u32;
        while temp > 0 && temp % 2 == 0 { f2 += 1; temp /= 2; }
        let mut f3 = 0u32;
        while temp > 0 && temp % 3 == 0 { f3 += 1; temp /= 3; }
        let f2_score = (f2 as f64 / 4.0).min(1.0);
        let f3_score = if f3 > 0 { (f3 as f64 / 2.0).min(1.0) } else { 0.0 };
        let purity = if temp <= 1 { 1.0 } else { 1.0 / (1.0 + (temp as f64).ln()) };
        f2_score * 0.35 + f3_score * 0.25 + purity * 0.40
    }

    #[inline]
    fn calculate_symmetry(nf: FractionalNf) -> f64 {
        let sn = Self::oh_purity(nf.numerator);
        let sd = Self::oh_purity(nf.denominator);
        if nf.numerator <= nf.denominator { sn * 0.3 + sd * 0.7 } else { sn * 0.7 + sd * 0.3 }
    }

    #[inline]
    fn calculate_charge(nf: FractionalNf) -> i32 {
        if nf.numerator == 1 && nf.denominator <= 48 { 1 }
        else if nf.denominator % 48 == 0 { 2 }
        else if nf.numerator % 2 == 0 { -1 }
        else { 0 }
    }

    pub fn classify(&self) -> &'static str {
        if self.mass_ev < 1e-3 { "ULTRALIGHT" }
        else if self.mass_ev < 2.2 { "NEUTRINO_LIKE" }
        else if self.mass_ev < 5.12e5 { "ELECTRON_LIKE" }
        else if self.mass_ev < 1.06e8 { "MUON_LIKE" }
        else if self.mass_ev < 1.0e9 { "LIGHT_HADRONIC" }
        else if self.mass_ev < 1.78e9 { "TAU_LIKE" }
        else if self.mass_ev < 8.04e10 { "HEAVY_HADRONIC" }
        else if self.mass_ev < 9.12e10 { "W_BOSON_LIKE" }
        else if self.mass_ev < 1.26e11 { "Z_BOSON_LIKE" }
        else if self.mass_ev < 1.30e11 { "HIGGS_LIKE" }
        else if self.mass_ev < 1e12 { "HEAVY" }
        else if self.mass_ev < 1e15 { "TEV_TO_GUT_SCALE" }
        else { "PLANCK_SCALE" }
    }
}

pub struct StreamSearchEngine {
    top_peaks: BinaryHeap<ParticleCandidate>,
    top_wimps: BinaryHeap<ParticleCandidate>,
    top_axions: BinaryHeap<ParticleCandidate>,
    top_neutrinos: BinaryHeap<ParticleCandidate>,
    distribution: HashMap<&'static str, usize>,
    seen_fractions: Option<HashSet<(u64, u64)>>,
    total_analyzed: usize,
    unique_count: usize,
    capacity: usize,
    use_dedup: bool,
}

impl StreamSearchEngine {
    pub fn new(capacity: usize, use_dedup: bool) -> Self {
        Self {
            top_peaks: BinaryHeap::with_capacity(capacity + 1),
            top_wimps: BinaryHeap::with_capacity(capacity + 1),
            top_axions: BinaryHeap::with_capacity(capacity + 1),
            top_neutrinos: BinaryHeap::with_capacity(capacity + 1),
            distribution: HashMap::new(),
            seen_fractions: if use_dedup { Some(HashSet::with_capacity(100_000)) } else { None },
            total_analyzed: 0,
            unique_count: 0,
            capacity,
            use_dedup,
        }
    }

    #[inline]
    pub fn process_candidate(&mut self, candidate: ParticleCandidate) -> bool {
        if self.use_dedup {
            if let Some(ref mut seen) = self.seen_fractions {
                let key = (candidate.nf.numerator, candidate.nf.denominator);
                if !seen.insert(key) { return false; }
                self.unique_count += 1;
                if seen.len() > 500_000 {
                    self.use_dedup = false;
                    println!("  ⚠️ Deduplication disabled ({} entries)", seen.len());
                    self.seen_fractions = None;
                }
            }
        }
        self.total_analyzed += 1;
        *self.distribution.entry(candidate.classify()).or_insert(0) += 1;

        self.top_peaks.push(candidate.clone());
        if self.top_peaks.len() > self.capacity { self.top_peaks.pop(); }
        if candidate.mass_gev >= 1.0 && candidate.mass_gev <= 1000.0 && candidate.stability_metric > 0.35 {
            self.top_wimps.push(candidate.clone());
            if self.top_wimps.len() > self.capacity { self.top_wimps.pop(); }
        }
        if candidate.mass_ev < 1.0 && candidate.stability_metric > 0.25 {
            self.top_axions.push(candidate.clone());
            if self.top_axions.len() > self.capacity { self.top_axions.pop(); }
        }
        if candidate.mass_ev <= 2.2 && candidate.mass_ev >= 0.0001 {
            self.top_neutrinos.push(candidate.clone());
            if self.top_neutrinos.len() > self.capacity { self.top_neutrinos.pop(); }
        }
        true
    }

    pub fn clear_dedup(&mut self) {
        if let Some(ref mut seen) = self.seen_fractions {
            let size = seen.len();
            seen.clear(); seen.shrink_to_fit();
            println!("  🧹 Deduplication freed: {} entries", size);
        }
        self.seen_fractions = None;
        self.use_dedup = false;
        self.distribution.shrink_to_fit();
    }

    pub fn get_top_peaks(&self) -> Vec<ParticleCandidate> { self.top_peaks.clone().into_sorted_vec() }
    pub fn get_top_wimps(&self) -> Vec<ParticleCandidate> { self.top_wimps.clone().into_sorted_vec() }
    pub fn get_top_axions(&self) -> Vec<ParticleCandidate> { self.top_axions.clone().into_sorted_vec() }
    pub fn get_top_neutrinos(&self) -> Vec<ParticleCandidate> { self.top_neutrinos.clone().into_sorted_vec() }
    pub fn get_distribution(&self) -> &HashMap<&'static str, usize> { &self.distribution }
    pub fn total_analyzed(&self) -> usize { self.total_analyzed }
}

fn best_rational_approximations(target: f64, max_den: u64) -> Vec<FractionalNf> {
    let mut result = Vec::new();
    if target <= 0.0 || !target.is_finite() { return result; }
    let mut x = target;
    let (mut h2, mut k2): (i128, i128) = (0, 1);
    let (mut h1, mut k1): (i128, i128) = (1, 0);
    for _ in 0..64 {
        if !x.is_finite() || x < 0.0 { break; }
        let a = x.floor() as i128;
        let h = a * h1 + h2; let k = a * k1 + k2;
        if k > max_den as i128 { break; }
        if h > 0 && k > 0 && h <= u64::MAX as i128 && k <= u64::MAX as i128 {
            result.push(FractionalNf::new(h as u64, k as u64));
        }
        let frac = x - x.floor();
        if frac < 1e-15 { break; }
        x = 1.0 / frac;
        h2 = h1; k2 = k1; h1 = h; k1 = k;
    }
    result
}

fn run_phase2_memory_safe(target_nf: f64, min_mass: f64, max_mass: f64, radius: u64) -> Vec<ParticleCandidate> {
    let oh_bases: [u64; 5] = [48, 288, 1728, 10368, 62208];
    let min_q = (1.0 / (max_mass / M_E_EV)) as u64;
    let max_q = (1.0 / (min_mass / M_E_EV)) as u64;

    oh_bases.par_iter().flat_map(|&base| {
        let tk = ((1.0 / target_nf) / base as f64) as u64;
        let sk = tk.saturating_sub(radius);
        let ek = tk.saturating_add(radius);
        (sk..=ek).into_par_iter().filter_map(move |k| {
            let q = base * k;
            if q < min_q || q > max_q { return None; }
            let p = (target_nf * (q as f64)).round() as u64;
            if p >= 1 { Some(ParticleCandidate::analyze(FractionalNf::new(p, q))) } else { None }
        })
    }).fold(
        || BinaryHeap::with_capacity(101),
        |mut h, c| { h.push(c); if h.len() > 100 { h.pop(); } h }
    ).reduce(
        || BinaryHeap::with_capacity(101),
        |mut a, b| { for c in b.into_vec() { a.push(c); if a.len() > 100 { a.pop(); } } a }
    ).into_vec()
}

#[pyfunction]
fn search_qcd_axions() -> PyResult<Py<PyDict>> {
    let min_m: f64 = 1e-6; let max_m: f64 = 1e-3; let max_den: u64 = 1_000_000_000_000;
    let mut eng = StreamSearchEngine::new(100, true);
    println!("╔══════════════════════════════════════════════════════════╗\n║  QCD AXION SEARCH (1 μeV – 1 meV)\n╚══════════════════════════════════════════════════════════╝");
    
    for i in 0..=4000 {
        let t = (min_m.ln() + (max_m.ln() - min_m.ln()) * (i as f64) / 4000.0).exp() / M_E_EV;
        for nf in best_rational_approximations(t, max_den) {
            let m = nf.mass_ev();
            if m >= min_m && m <= max_m { eng.process_candidate(ParticleCandidate::analyze(nf)); }
        }
    }
    println!("  Phase 1: {} candidates ({} unique)", eng.total_analyzed(), eng.unique_count);
    eng.clear_dedup();

    println!("  Starting Phase 2 (Parallel, 2 threads, radius 2M, O(1) memory)...");
    let seeds = eng.get_top_axions();
    let before = eng.total_analyzed();
    
    if let Some(seed) = seeds.iter().rev().next() {
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
            let res = run_phase2_memory_safe(seed.nf.to_f64(), min_m, max_m, 2_000_000);
            for c in res { eng.process_candidate(c); }
        });
    }
    println!("  Phase 2: +{} candidates", eng.total_analyzed() - before);
    eng.clear_dedup();

    let sorted = eng.get_top_axions();
    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("total_candidates", eng.total_analyzed())?;
        let ad = PyDict::new(py);
        for (i, a) in sorted.iter().rev().enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", a.nf.to_f64())?; x.set_item("numerator", a.nf.numerator)?;
            x.set_item("denominator", a.nf.denominator)?; x.set_item("mass_ev", a.mass_ev)?;
            x.set_item("mass_uev", a.mass_ev * 1e6)?; x.set_item("stability", a.stability_metric)?;
            x.set_item("symmetry", a.symmetry_score)?; x.set_item("charge", a.topological_charge)?;
            ad.set_item(i, x)?;
        }
        d.set_item("axion_candidates", ad)?;
        if let Some(b) = sorted.iter().rev().next() {
            d.set_item("best_mass_ev", b.mass_ev)?; d.set_item("best_num", b.nf.numerator)?;
            d.set_item("best_den", b.nf.denominator)?; d.set_item("best_stability", b.stability_metric)?;
            println!("\n  ✅ BEST AXION: {:.6e} eV | Nf={}/{} | est={:.4}", b.mass_ev, b.nf.numerator, b.nf.denominator, b.stability_metric);
        }
        Ok(d.into())
    })
}

#[pyfunction]
fn search_atmospheric_neutrino() -> PyResult<Py<PyDict>> {
    let lo: f64 = DELTA_M32_SQ.sqrt(); let hi: f64 = 0.055; let max_den: u64 = 500_000_000;
    let mut eng = StreamSearchEngine::new(100, true);
    println!("╔══════════════════════════════════════════════════════════╗\n║  ATMOSPHERIC NEUTRINO ν₃ (m ≈ {:.6} eV)\n╚══════════════════════════════════════════════════════════╝", lo);
    
    for i in 0..=2000 {
        let t = (lo.ln() + (hi.ln() - lo.ln()) * (i as f64) / 2000.0).exp() / M_E_EV;
        for nf in best_rational_approximations(t, max_den) {
            let m = nf.mass_ev();
            if m >= lo && m <= hi { eng.process_candidate(ParticleCandidate::analyze(nf)); }
        }
    }
    println!("  Phase 1: {} candidates ({} unique)", eng.total_analyzed(), eng.unique_count);
    eng.clear_dedup();

    println!("  Starting Phase 2 (Parallel, 2 threads, radius 100M)...");
    let seeds = eng.get_top_neutrinos();
    let before = eng.total_analyzed();
    
    if let Some(seed) = seeds.iter().rev().next() {
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
            let res = run_phase2_memory_safe(seed.nf.to_f64(), lo, hi, 100_000_000);
            for c in res { eng.process_candidate(c); }
        });
    }
    println!("  Phase 2: +{} candidates", eng.total_analyzed() - before);
    eng.clear_dedup();

    let sorted = eng.get_top_neutrinos();
    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("total_candidates", eng.total_analyzed())?;
        d.set_item("target_mass_ev", lo)?;
        let nd = PyDict::new(py);
        for (i, n) in sorted.iter().rev().take(100).enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", n.nf.to_f64())?; x.set_item("numerator", n.nf.numerator)?;
            x.set_item("denominator", n.nf.denominator)?; x.set_item("mass_ev", n.mass_ev)?;
            x.set_item("mass_meV", n.mass_ev * 1000.0)?; x.set_item("stability", n.stability_metric)?;
            x.set_item("symmetry", n.symmetry_score)?; x.set_item("charge", n.topological_charge)?;
            x.set_item("rel_error", ((n.mass_ev - lo) / lo).abs())?;
            nd.set_item(i, x)?;
        }
        d.set_item("neutrino_candidates", nd)?;
        if let Some(b) = sorted.iter().rev().next() {
            let e = ((b.mass_ev - lo) / lo).abs();
            d.set_item("best_mass_ev", b.mass_ev)?; d.set_item("best_num", b.nf.numerator)?;
            d.set_item("best_den", b.nf.denominator)?; d.set_item("best_stability", b.stability_metric)?;
            d.set_item("best_rel_error", e)?;
            println!("\n  ✅ BEST ν₃: {:.6e} eV | Nf={}/{} | err={:.4e}", b.mass_ev, b.nf.numerator, b.nf.denominator, e);
        }
        Ok(d.into())
    })
}

#[pyfunction]
fn search_solar_neutrinos() -> PyResult<Py<PyDict>> {
    let lo: f64 = DELTA_M21_SQ.sqrt(); let hi: f64 = 0.012; let max_den: u64 = 1_000_000_000;
    let mut eng = StreamSearchEngine::new(100, true);
    println!("╔══════════════════════════════════════════════════════════╗\n║  SOLAR NEUTRINOS ν₂ (m ≈ {:.6} eV)\n╚══════════════════════════════════════════════════════════╝", lo);
    
    for i in 0..=2000 {
        let t = (lo.ln() + (hi.ln() - lo.ln()) * (i as f64) / 2000.0).exp() / M_E_EV;
        for nf in best_rational_approximations(t, max_den) {
            let m = nf.mass_ev();
            if m >= lo && m <= hi { eng.process_candidate(ParticleCandidate::analyze(nf)); }
        }
    }
    println!("  Phase 1: {} candidates ({} unique)", eng.total_analyzed(), eng.unique_count);
    eng.clear_dedup();

    println!("  Starting Phase 2 (Parallel, 2 threads, radius 100M)...");
    let seeds = eng.get_top_neutrinos();
    let before = eng.total_analyzed();
    
    if let Some(seed) = seeds.iter().rev().next() {
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
            let res = run_phase2_memory_safe(seed.nf.to_f64(), lo, hi, 100_000_000);
            for c in res { eng.process_candidate(c); }
        });
    }
    println!("  Phase 2: +{} candidates", eng.total_analyzed() - before);
    eng.clear_dedup();

    let sorted = eng.get_top_neutrinos();
    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("total_candidates", eng.total_analyzed())?;
        d.set_item("target_mass_ev", lo)?;
        let nd = PyDict::new(py);
        for (i, n) in sorted.iter().rev().take(100).enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", n.nf.to_f64())?; x.set_item("numerator", n.nf.numerator)?;
            x.set_item("denominator", n.nf.denominator)?; x.set_item("mass_ev", n.mass_ev)?;
            x.set_item("mass_meV", n.mass_ev * 1000.0)?; x.set_item("stability", n.stability_metric)?;
            x.set_item("symmetry", n.symmetry_score)?; x.set_item("charge", n.topological_charge)?;
            x.set_item("rel_error", ((n.mass_ev - lo) / lo).abs())?;
            nd.set_item(i, x)?;
        }
        d.set_item("neutrino_candidates", nd)?;
        if let Some(b) = sorted.iter().rev().next() {
            let e = ((b.mass_ev - lo) / lo).abs();
            d.set_item("best_mass_ev", b.mass_ev)?; d.set_item("best_num", b.nf.numerator)?;
            d.set_item("best_den", b.nf.denominator)?; d.set_item("best_stability", b.stability_metric)?;
            d.set_item("best_rel_error", e)?;
            println!("\n  ✅ BEST ν₂: {:.6e} eV | Nf={}/{} | err={:.4e}", b.mass_ev, b.nf.numerator, b.nf.denominator, e);
        }
        Ok(d.into())
    })
}

#[pyfunction]
fn search_nu1() -> PyResult<Py<PyDict>> {
    let lo: f64 = 0.001; // 1 meV
    let hi: f64 = 0.010; // 10 meV
    let max_den: u64 = 1_000_000_000;
    let mut eng = StreamSearchEngine::new(100, true);
    
    println!("╔══════════════════════════════════════════════════════════╗\n║  SEARCH: LIGHT NEUTRINO ν₁ (1 - 10 meV)\n╚══════════════════════════════════════════════════════════╝");
    
    for i in 0..=2000 {
        let t = (lo.ln() + (hi.ln() - lo.ln()) * (i as f64) / 2000.0).exp() / M_E_EV;
        for nf in best_rational_approximations(t, max_den) {
            let m = nf.mass_ev();
            if m >= lo && m <= hi { eng.process_candidate(ParticleCandidate::analyze(nf)); }
        }
    }
    println!("  Phase 1: {} candidates ({} unique)", eng.total_analyzed(), eng.unique_count);
    eng.clear_dedup();

    println!("  Starting Phase 2 (Parallel, 2 threads, radius 100M)...");
    let seeds = eng.get_top_neutrinos();
    let before = eng.total_analyzed();
    
    if let Some(seed) = seeds.iter().rev().next() {
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap().install(|| {
            let res = run_phase2_memory_safe(seed.nf.to_f64(), lo, hi, 100_000_000);
            for c in res { eng.process_candidate(c); }
        });
    }
    println!("  Phase 2: +{} candidates", eng.total_analyzed() - before);
    eng.clear_dedup();

    let sorted = eng.get_top_neutrinos();
    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("total_candidates", eng.total_analyzed())?;
        let nd = PyDict::new(py);
        for (i, n) in sorted.iter().rev().take(100).enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", n.nf.to_f64())?; x.set_item("numerator", n.nf.numerator)?;
            x.set_item("denominator", n.nf.denominator)?; x.set_item("mass_ev", n.mass_ev)?;
            x.set_item("mass_meV", n.mass_ev * 1000.0)?; x.set_item("stability", n.stability_metric)?;
            x.set_item("symmetry", n.symmetry_score)?; x.set_item("charge", n.topological_charge)?;
            nd.set_item(i, x)?;
        }
        d.set_item("neutrino_candidates", nd)?;
        if let Some(b) = sorted.iter().rev().next() {
            d.set_item("best_mass_ev", b.mass_ev)?; d.set_item("best_num", b.nf.numerator)?;
            d.set_item("best_den", b.nf.denominator)?; d.set_item("best_stability", b.stability_metric)?;
            println!("\n  ✅ BEST ν₁: {:.6e} eV | Nf={}/{} | est={:.4}", b.mass_ev, b.nf.numerator, b.nf.denominator, b.stability_metric);
        }
        Ok(d.into())
    })
}

#[pyfunction]
fn search_mass_range(min_mass_ev: f64, max_mass_ev: f64, max_denominator: u64) -> PyResult<Py<PyDict>> {
    let mut eng = StreamSearchEngine::new(50, false);
    let min_nf = min_mass_ev / M_E_EV;
    let max_nf = max_mass_ev / M_E_EV;
    let safe_max_den = max_denominator.min(200_000);
    
    if safe_max_den < max_denominator {
        println!("  ⚠️ Denominator limited to {} to avoid OOM.", safe_max_den);
    }

    for den in 1..=safe_max_den {
        let a = (min_nf * den as f64).ceil() as u64;
        let b = (max_nf * den as f64).floor() as u64;
        if a <= b && a >= 1 {
            let step = if b - a > 100_000 { (b - a) / 100_000 } else { 1 };
            let mut num = a;
            while num <= b {
                let nf = FractionalNf::new(num, den);
                if nf.denominator == den {
                    eng.process_candidate(ParticleCandidate::analyze(nf));
                }
                num += step;
            }
        }
    }
    eng.clear_dedup();

    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("total_candidates", eng.total_analyzed())?;
        
        let pd = PyDict::new(py);
        for (i, p) in eng.get_top_peaks().iter().rev().enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", p.nf.to_f64())?; x.set_item("numerator", p.nf.numerator)?;
            x.set_item("denominator", p.nf.denominator)?; x.set_item("mass_ev", p.mass_ev)?;
            x.set_item("mass_gev", p.mass_gev)?; x.set_item("stability", p.stability_metric)?;
            x.set_item("charge", p.topological_charge)?; x.set_item("class", p.classify())?;
            pd.set_item(i, x)?;
        }
        d.set_item("stability_peaks", pd)?;

        let dd = PyDict::new(py);
        for (k, v) in eng.get_distribution() { dd.set_item(k, v)?; }
        d.set_item("mass_distribution", dd)?;

        let wd = PyDict::new(py);
        for (i, w) in eng.get_top_wimps().iter().rev().enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", w.nf.to_f64())?; x.set_item("mass_gev", w.mass_gev)?;
            x.set_item("stability", w.stability_metric)?; x.set_item("charge", w.topological_charge)?;
            wd.set_item(i, x)?;
        }
        d.set_item("wimp_candidates", wd)?;

        let ad = PyDict::new(py);
        for (i, a) in eng.get_top_axions().iter().rev().enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", a.nf.to_f64())?; x.set_item("mass_ev", a.mass_ev)?;
            x.set_item("stability", a.stability_metric)?; x.set_item("charge", a.topological_charge)?;
            ad.set_item(i, x)?;
        }
        d.set_item("axion_candidates", ad)?;

        let nd = PyDict::new(py);
        for (i, n) in eng.get_top_neutrinos().iter().rev().enumerate() {
            let x = PyDict::new(py);
            x.set_item("nf", n.nf.to_f64())?; x.set_item("numerator", n.nf.numerator)?;
            x.set_item("denominator", n.nf.denominator)?; x.set_item("mass_ev", n.mass_ev)?;
            x.set_item("stability", n.stability_metric)?; x.set_item("charge", n.topological_charge)?;
            nd.set_item(i, x)?;
        }
        d.set_item("neutrino_candidates", nd)?;

        Ok(d.into())
    })
}

#[pymodule]
fn neutrino_search(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search_mass_range, m)?)?;
    m.add_function(wrap_pyfunction!(search_qcd_axions, m)?)?;
    m.add_function(wrap_pyfunction!(search_atmospheric_neutrino, m)?)?;
    m.add_function(wrap_pyfunction!(search_solar_neutrinos, m)?)?;
    m.add_function(wrap_pyfunction!(search_nu1, m)?)?; // NEW
    Ok(())
}