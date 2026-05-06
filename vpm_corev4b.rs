//! vpm_core_v10 — Kernel de Coherencia de Fase para VCV48                       
//! ============================================================================ 
//! Calcula el espectro de potencias P(k) Y las fases de los modos de Fourier.
//!
//! Métodos expuestos a Python:
//!   - inicializar(ra, dec, z, pesos)
//!   - power_spectrum(grid_size, n_bins) -> (k, P(k), box_size)
//!   - power_spectrum_with_phases(grid_size, n_bins) -> (k, P(k), Re(φ), Im(φ), N_modes, shot, box)
//!   - stacking_fase_vcv(grid_size, n_bins) -> (k_obs, R, φ, Rayleigh, p-value, σ)
//!   - test_alineacion_fase(grid_size, n_bins) -> (k_obs, Δφ, Δφ_bg_mean, Δφ_bg_std, Nσ, p-value)
//!
//! Constantes expuestas:
//!   - A0 = 14.075 Mpc
//!   - H0 = 70.0 km/s/Mpc
//!   - K_VCV = 0.4464 h/Mpc
//!   - PHASE_REF = 0.6465 rad

use rayon::prelude::*;
use std::f64::consts::PI;

// ============================================================================
// CONSTANTES COSMOLÓGICAS
// ============================================================================
const H0: f64 = 70.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const C: f64 = 299792.458;

// ============================================================================
// CONSTANTES DEL MODELO VCV48 (EXPUESTAS A PYTHON)
// ============================================================================
pub const A0: f64 = 14.075;
pub const K_VCV: f64 = 2.0 * PI / A0;   // ≈ 0.4464 h/Mpc
pub const PHASE_REF: f64 = 0.646466;     // rad

// ============================================================================
// COSMOLOGÍA
// ============================================================================

#[inline(always)]
fn hubble_factor(z: f64) -> f64 {
    (OMEGA_M * (1.0 + z).powi(3) + OMEGA_L).sqrt()
}

pub fn distancia_comovil(z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let n_steps = 100;
    let dz = z / (n_steps as f64);
    let mut integral = 0.0;
    for i in 0..n_steps {
        let z_mid = (i as f64 + 0.5) * dz;
        integral += dz / hubble_factor(z_mid);
    }
    (C / H0) * integral
}

fn ra_dec_z_to_xyz(ra: f64, dec: f64, z: f64) -> (f64, f64, f64) {
    let d = distancia_comovil(z);
    let ra_rad = ra * PI / 180.0;
    let dec_rad = dec * PI / 180.0;
    let cos_dec = dec_rad.cos();
    (
        d * cos_dec * ra_rad.cos(),
        d * cos_dec * ra_rad.sin(),
        d * dec_rad.sin(),
    )
}

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

#[derive(Debug, Clone)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub phase: f32,
}

#[derive(Debug, Clone)]
pub struct PowerSpectrumResult {
    pub k_vals: Vec<f64>,
    pub p_k: Vec<f64>,
    pub phases: Vec<(f64, f64)>,
    pub n_modes: Vec<u64>,
    pub shot_noise: f64,
    pub box_size: f64,
}

// ============================================================================
// CACHE FOURIER
// ============================================================================

#[derive(Debug, Clone)]
pub struct FourierCache {
    pub points: Vec<Point3D>,
    pub weights: Vec<f32>,
}

impl FourierCache {
    pub fn new(ra: &[f64], dec: &[f64], z: &[f64], weights: &[f64]) -> Self {
        let n = ra.len();

        let points: Vec<Point3D> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (x, y, z_cart) = ra_dec_z_to_xyz(ra[i], dec[i], z[i]);
                let d_comovil = distancia_comovil(z[i]);
                let phase = 2.0 * PI * (d_comovil % A0) / A0;

                Point3D {
                    x: x as f32,
                    y: y as f32,
                    z: z_cart as f32,
                    phase: phase as f32,
                }
            })
            .collect();

        let w: Vec<f32> = weights.iter().map(|&w| w as f32).collect();

        Self { points, weights: w }
    }

    fn asignar_rejilla_cic(&self, grid_size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let n = self.points.len();

        let x_min = self.points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min) as f64;
        let x_max = self.points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max) as f64;
        let y_min = self.points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) as f64;
        let y_max = self.points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max) as f64;
        let z_min = self.points.iter().map(|p| p.z).fold(f32::INFINITY, f32::min) as f64;
        let z_max = self.points.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max) as f64;

        let margin = 0.05;
        let bx = (x_max - x_min) * margin;
        let by = (y_max - y_min) * margin;
        let bz = (z_max - z_min) * margin;

        let x0 = x_min - bx;
        let y0 = y_min - by;
        let z0 = z_min - bz;
        let box_size = ((x_max + bx) - x0)
            .max((y_max + by) - y0)
            .max((z_max + bz) - z0);
        let cell_size = box_size / (grid_size as f64);

        let n_cells = grid_size * grid_size * grid_size;
        let mut grid_real = vec![0.0f64; n_cells];
        let mut grid_phase_real = vec![0.0f64; n_cells];
        let mut grid_phase_imag = vec![0.0f64; n_cells];

        for i in 0..n {
            let p = &self.points[i];
            let w = self.weights[i] as f64;

            let fx = (p.x as f64 - x0) / cell_size - 0.5;
            let fy = (p.y as f64 - y0) / cell_size - 0.5;
            let fz = (p.z as f64 - z0) / cell_size - 0.5;

            let ix = fx.floor() as isize;
            let iy = fy.floor() as isize;
            let iz = fz.floor() as isize;

            let dx = fx - ix as f64;
            let dy = fy - iy as f64;
            let dz = fz - iz as f64;

            let phase_rad = p.phase as f64;
            let phase_cos = phase_rad.cos();
            let phase_sin = phase_rad.sin();

            for ox in 0..2 {
                let wx = if ox == 0 { 1.0 - dx } else { dx };
                let jx = (ix + ox) as usize;
                if jx >= grid_size { continue; }

                for oy in 0..2 {
                    let wy = if oy == 0 { 1.0 - dy } else { dy };
                    let jy = (iy + oy) as usize;
                    if jy >= grid_size { continue; }

                    for oz in 0..2 {
                        let wz = if oz == 0 { 1.0 - dz } else { dz };
                        let jz = (iz + oz) as usize;
                        if jz >= grid_size { continue; }

                        let weight = w * wx * wy * wz;
                        let idx = jx + jy * grid_size + jz * grid_size * grid_size;

                        grid_real[idx] += weight;
                        grid_phase_real[idx] += weight * phase_cos;
                        grid_phase_imag[idx] += weight * phase_sin;
                    }
                }
            }
        }

        (grid_real, grid_phase_real, grid_phase_imag, box_size)
    }

    pub fn power_spectrum_with_phases(
        &self,
        grid_size: usize,
        n_bins: usize,
    ) -> PowerSpectrumResult {
        let (grid_real, grid_phase_real, grid_phase_imag, box_size) =
            self.asignar_rejilla_cic(grid_size);

        let n_total_w: f64 = self.weights.iter().map(|&x| x as f64).sum();
        let sum_w2: f64 = self.weights.iter().map(|&x| (x as f64).powi(2)).sum();
        let n_cells = (grid_size * grid_size * grid_size) as f64;

        let shot_noise = (sum_w2 / (n_total_w * n_total_w)) * box_size.powi(3);

        let rho_mean = n_total_w / n_cells;
        let n = grid_size;
        let n_f64 = n as f64;

        let n_cells_total = n * n * n;
        let mut delta_real = vec![0.0f64; n_cells_total];
        let mut delta_phase_real = vec![0.0f64; n_cells_total];
        let mut delta_phase_imag = vec![0.0f64; n_cells_total];

        // Ventana de Blackman-Harris 3D
        for i in 0..n {
            let wx = blackman_harris_window(i as f64, n_f64);
            for j in 0..n {
                let wy = blackman_harris_window(j as f64, n_f64);
                for k in 0..n {
                    let wz = blackman_harris_window(k as f64, n_f64);
                    let idx = i + j * n + k * n * n;

                    let rho = grid_real[idx];
                    let phase_r = grid_phase_real[idx];
                    let phase_i = grid_phase_imag[idx];

                    if rho_mean > 0.0 {
                        delta_real[idx] = ((rho - rho_mean) / rho_mean) * wx * wy * wz;
                        delta_phase_real[idx] = if rho > 0.0 {
                            (phase_r / rho) * wx * wy * wz
                        } else {
                            0.0
                        };
                        delta_phase_imag[idx] = if rho > 0.0 {
                            (phase_i / rho) * wx * wy * wz
                        } else {
                            0.0
                        };
                    }
                }
            }
        }

        // FFT 3D
        let mut fft_real = delta_real.clone();
        let mut fft_phase_r = delta_phase_real.clone();
        let mut fft_phase_i = delta_phase_imag.clone();

        fft3d_naive(&mut fft_real, n, true);
        fft3d_naive(&mut fft_phase_r, n, true);
        fft3d_naive(&mut fft_phase_i, n, true);

        // Frecuencias
        let k_fund = 2.0 * PI / box_size;
        let k_nyq = PI * n_f64 / box_size;

        // Bins logarítmicos
        let k_edges: Vec<f64> = (0..=n_bins)
            .map(|i| {
                let log_k_min = (k_fund * 0.5).ln();
                let log_k_max = (k_nyq * 1.1).ln();
                let log_k = log_k_min + (log_k_max - log_k_min) * (i as f64 / n_bins as f64);
                log_k.exp()
            })
            .collect();

        let k_centers: Vec<f64> = (0..n_bins)
            .map(|i| (k_edges[i] * k_edges[i + 1]).sqrt())
            .collect();

        let mut p_k = vec![0.0f64; n_bins];
        let mut phase_sum_real = vec![0.0f64; n_bins];
        let mut phase_sum_imag = vec![0.0f64; n_bins];
        let mut n_modes = vec![0u64; n_bins];

        let n_half = (n / 2) as isize;

        for ix in 0..n {
            let kx = if (ix as isize) <= n_half { ix as f64 } else { ix as f64 - n_f64 };
            let kx_val = kx * k_fund;

            for iy in 0..n {
                let ky = if (iy as isize) <= n_half { iy as f64 } else { iy as f64 - n_f64 };
                let ky_val = ky * k_fund;

                for iz in 0..n {
                    let kz = if (iz as isize) <= n_half { iz as f64 } else { iz as f64 - n_f64 };
                    let kz_val = kz * k_fund;

                    let k_mag = (kx_val * kx_val + ky_val * ky_val + kz_val * kz_val).sqrt();

                    if k_mag < k_edges[0] || k_mag >= k_edges[n_bins] {
                        continue;
                    }

                    let mut bin_idx = 0;
                    for b in 0..n_bins {
                        if k_mag >= k_edges[b] && k_mag < k_edges[b + 1] {
                            bin_idx = b;
                            break;
                        }
                    }

                    let idx = ix + iy * n + iz * n * n;

                    let norm = box_size.powi(3) / (n_f64.powi(6));
                    let power = (fft_real[idx].powi(2)) * norm;

                    p_k[bin_idx] += power;
                    phase_sum_real[bin_idx] += fft_phase_r[idx];
                    phase_sum_imag[bin_idx] += fft_phase_i[idx];
                    n_modes[bin_idx] += 1;
                }
            }
        }

        // Promediar por bin
        let mut phases = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            if n_modes[b] > 0 {
                p_k[b] /= n_modes[b] as f64;
                let avg_real = phase_sum_real[b] / n_modes[b] as f64;
                let avg_imag = phase_sum_imag[b] / n_modes[b] as f64;
                phases.push((avg_real, avg_imag));
            } else {
                phases.push((0.0, 0.0));
            }
        }

        // Sustraer shot noise
        for b in 0..n_bins {
            p_k[b] -= shot_noise;
            if p_k[b] < 0.0 {
                p_k[b] = 0.0;
            }
        }

        PowerSpectrumResult {
            k_vals: k_centers,
            p_k,
            phases,
            n_modes,
            shot_noise,
            box_size,
        }
    }
}

// ============================================================================
// PROCESAMIENTO DE SEÑALES
// ============================================================================

#[inline]
fn blackman_harris_window(i: f64, n: f64) -> f64 {
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    let x = 2.0 * PI * i / (n - 1.0);
    a0 - a1 * x.cos() + a2 * (2.0 * x).cos() - a3 * (3.0 * x).cos()
}

fn fft1d(data: &mut [f64], n: usize, forward: bool) {
    let mut bits = 0;
    let mut temp = n;
    while temp > 1 {
        bits += 1;
        temp >>= 1;
    }

    let mut j: usize;
    for i in 0..n {
        j = 0;
        let mut bit = bits;
        let mut k = i;
        while bit > 0 {
            bit -= 1;
            j = (j << 1) | (k & 1);
            k >>= 1;
        }
        if j > i {
            data.swap(i, j);
        }
    }

    let mut step = 1;
    while step < n {
        let jump = step << 1;
        let angle = PI / step as f64 * if forward { -1.0 } else { 1.0 };
        for m in (0..n).step_by(jump) {
            for k in 0..step {
                let idx1 = m + k;
                let idx2 = idx1 + step;
                let theta = angle * k as f64;
                let wr = theta.cos();
                let wi = theta.sin();
                let tr = wr * data[idx2];
                let _ti = wi * data[idx2];
                data[idx2] = data[idx1] - tr;
                data[idx1] += tr;
            }
        }
        step <<= 1;
    }

    if forward {
        let scale = 1.0 / (n as f64).sqrt();
        for v in data.iter_mut() {
            *v *= scale;
        }
    }
}

fn fft3d_naive(data: &mut [f64], n: usize, forward: bool) {
    // Eje Z
    for i in 0..n {
        for j in 0..n {
            let start = i + j * n;
            let mut slice: Vec<f64> = (0..n).map(|k| data[start + k * n * n]).collect();
            fft1d(&mut slice, n, forward);
            for k in 0..n {
                data[start + k * n * n] = slice[k];
            }
        }
    }
    // Eje Y
    for i in 0..n {
        for k in 0..n {
            let start = i + k * n * n;
            let mut slice: Vec<f64> = (0..n).map(|j| data[start + j * n]).collect();
            fft1d(&mut slice, n, forward);
            for j in 0..n {
                data[start + j * n] = slice[j];
            }
        }
    }
    // Eje X
    for j in 0..n {
        for k in 0..n {
            let start = j * n + k * n * n;
            let mut slice: Vec<f64> = (0..n).map(|i| data[i + start]).collect();
            fft1d(&mut slice, n, forward);
            for i in 0..n {
                data[i + start] = slice[i];
            }
        }
    }
}

/// Distancia angular mínima entre dos fases (considera wrap-around en 2π)
fn distancia_angular(phi1: f64, phi2: f64) -> f64 {
    let mut delta = (phi1 - phi2).abs();
    if delta > PI {
        delta = 2.0 * PI - delta;
    }
    delta
}

// ============================================================================
// INTERFAZ PYTHON (PyO3)
// ============================================================================

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
#[pyclass]
pub struct VPMEngine {
    cache: Option<FourierCache>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl VPMEngine {
    #[new]
    pub fn new() -> Self {
        Self { cache: None }
    }

    /// Inicializa el motor con datos de galaxias.
    #[pyo3(signature = (ra, dec, z, pesos))]
    pub fn inicializar(
        &mut self,
        ra: Vec<f64>,
        dec: Vec<f64>,
        z: Vec<f64>,
        pesos: Vec<f64>,
    ) {
        self.cache = Some(FourierCache::new(&ra, &dec, &z, &pesos));
    }

    /// Espectro de potencias con fases.
    #[pyo3(signature = (grid_size, n_bins))]
    pub fn power_spectrum_with_phases(
        &self,
        grid_size: usize,
        n_bins: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u64>, f64, f64)> {
        let cache = self.cache.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Motor no inicializado. Usá inicializar().")
        })?;

        let result = cache.power_spectrum_with_phases(grid_size, n_bins);

        let phase_real: Vec<f64> = result.phases.iter().map(|(r, _)| *r).collect();
        let phase_imag: Vec<f64> = result.phases.iter().map(|(_, i)| *i).collect();

        Ok((
            result.k_vals,
            result.p_k,
            phase_real,
            phase_imag,
            result.n_modes,
            result.shot_noise,
            result.box_size,
        ))
    }

    /// Versión simple: solo P(k)
    #[pyo3(signature = (grid_size, n_bins))]
    pub fn power_spectrum(
        &self,
        grid_size: usize,
        n_bins: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, f64)> {
        let cache = self.cache.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Motor no inicializado. Usá inicializar().")
        })?;

        let result = cache.power_spectrum_with_phases(grid_size, n_bins);
        Ok((result.k_vals, result.p_k, result.box_size))
    }

    /// Stacking de fase en k_vcv.
    #[pyo3(signature = (grid_size, n_bins))]
    pub fn stacking_fase_vcv(
        &self,
        grid_size: usize,
        n_bins: usize,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let cache = self.cache.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Motor no inicializado. Usá inicializar().")
        })?;

        let result = cache.power_spectrum_with_phases(grid_size, n_bins);

        let k_target = K_VCV;
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, &k) in result.k_vals.iter().enumerate() {
            let dist = (k - k_target).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        let k_obs = result.k_vals[best_idx];
        let (phase_real, phase_imag) = result.phases[best_idx];

        let r_coherence = (phase_real.powi(2) + phase_imag.powi(2)).sqrt();

        let phase_rad = phase_imag.atan2(phase_real);
        let phase_rad = if phase_rad < 0.0 {
            phase_rad + 2.0 * PI
        } else {
            phase_rad
        };

        let n_modes_bin = result.n_modes[best_idx] as f64;
        let rayleigh_stat = n_modes_bin * r_coherence * r_coherence;

        let p_value = (-rayleigh_stat).exp();

        let sigma = if p_value > 0.0 && p_value < 1.0 {
            (-2.0 * p_value.ln()).sqrt()
        } else {
            0.0
        };

        Ok((k_obs, r_coherence, phase_rad, rayleigh_stat, p_value, sigma))
    }

    /// Test de alineación de fase.
    /// Compara la fase en k_vcv con la fase de referencia,
    /// usando bins de fondo como grupo de control.
    ///
    /// Retorna: (k_obs, delta_phi, delta_phi_bg_mean, delta_phi_bg_std, n_sigma, p_value)
    #[pyo3(signature = (grid_size, n_bins))]
    pub fn test_alineacion_fase(
        &self,
        grid_size: usize,
        n_bins: usize,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        let cache = self.cache.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Motor no inicializado. Usá inicializar().")
        })?;

        let result = cache.power_spectrum_with_phases(grid_size, n_bins);

        // Encontrar bin de k_vcv
        let k_target = K_VCV;
        let mut idx_vcv = 0;
        let mut best_dist = f64::INFINITY;
        for (i, &k) in result.k_vals.iter().enumerate() {
            let dist = (k - k_target).abs();
            if dist < best_dist {
                best_dist = dist;
                idx_vcv = i;
            }
        }

        let k_obs = result.k_vals[idx_vcv];
        let (phase_real, phase_imag) = result.phases[idx_vcv];
        let phase_obs = phase_imag.atan2(phase_real);
        let phase_obs = if phase_obs < 0.0 { phase_obs + 2.0 * PI } else { phase_obs };

        // Distancia angular a la fase de referencia
        let delta_phi = distancia_angular(phase_obs, PHASE_REF);

        // Bins de fondo: todos los bins con al menos 1 modo,
        // excluyendo el bin de k_vcv ± 2 bins
        let mut bg_deltas = Vec::new();
        for i in 0..result.k_vals.len() {
            if (i as isize - idx_vcv as isize).abs() <= 2 {
                continue;
            }
            if result.n_modes[i] == 0 {
                continue;
            }
            let (pr, pi) = result.phases[i];
            let phase_bg = pi.atan2(pr);
            let phase_bg = if phase_bg < 0.0 { phase_bg + 2.0 * PI } else { phase_bg };
            let dphi_bg = distancia_angular(phase_bg, PHASE_REF);
            bg_deltas.push(dphi_bg);
        }

        if bg_deltas.len() < 5 {
            return Ok((k_obs, delta_phi, 0.0, 0.0, 0.0, 1.0));
        }

        // Estadísticas del fondo
        let n_bg = bg_deltas.len() as f64;
        let bg_mean: f64 = bg_deltas.iter().sum::<f64>() / n_bg;
        let bg_var: f64 = bg_deltas.iter()
            .map(|&d| (d - bg_mean).powi(2))
            .sum::<f64>() / (n_bg - 1.0);
        let bg_std = bg_var.sqrt();

        // ¿Cuántas desviaciones estándar por debajo de la media del fondo?
        let n_sigma = if bg_std > 0.0 {
            (bg_mean - delta_phi) / bg_std
        } else {
            0.0
        };

        // p-value: fracción de bins de fondo con Δφ ≤ delta_phi
        let n_below = bg_deltas.iter().filter(|&&d| d <= delta_phi).count() as f64;
        let p_value = n_below / n_bg;

        Ok((k_obs, delta_phi, bg_mean, bg_std, n_sigma, p_value))
    }
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn vpm_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VPMEngine>()?;
    m.add("A0", A0)?;
    m.add("H0", H0)?;
    m.add("K_VCV", K_VCV)?;
    m.add("PHASE_REF", PHASE_REF)?;
    Ok(())
}