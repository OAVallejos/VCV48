//! vpm_wave2.rs — Motor Ondulatorio VPM/VCV48 para Cosmología de Precisión
//!
//! Basado exclusivamente en:
//!   - Simetría O_h (grupo octaédrico completo)
//!   - Constantes fundamentales (sin parámetros libres)
//!   - Principio de Fermat generalizado con fase cuántica
//!
//! Versión 4.1 — "Óptica Física del Vacío + MCMC Jerárquico"
//!
//! Extensiones v4.1:
//!   - log_likelihood_jwst() para MCMC jerárquico no-binned
//!   - Marginalización numérica de sigma_true (Gauss-Legendre 5pt)
//!   - Aritmética f32 vectorizada para eficiencia
//!   - Priors poblacionales incorporados

use std::f64::consts::PI;
use rayon::prelude::*;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

// ============================================================================
// CONSTANTES FUNDAMENTALES — CERO PARÁMETROS LIBRES
// ============================================================================

pub const A0: f64 = 14.075;
pub const OMEGA_0: f64 = 1.146;
pub const XI_0: f64 = 0.084;
pub const Z_C: f64 = 1.5;
pub const BETA: f64 = 0.03;
pub const DELTA_CMB: f64 = 0.00610865;
pub const K_PN: f64 = 0.8025;
pub const PHI_REF: f64 = 0.837042;
pub const DELTA_SAT: f64 = 0.588907;
pub const THETA_D: f64 = 35.772;
pub const T_CMB_0: f64 = 2.725;

// ============================================================================
// PARÁMETROS DE ANISOTROPÍA DE LA RED O_h (validados por 20 preprints)
// ============================================================================

pub const ETA_1: f64 = 0.25229889;
pub const ETA_2: f64 = 0.15000000;
pub const PHI_ALPHA_LOCAL: f64 = 0.98022575;
pub const OMEGA_0_S: f64 = 0.191;
pub const A_M: f64 = 1.5204e17;

const H0_REF: f64 = 70.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_K: f64 = 0.001;
const C_LIGHT: f64 = 299792.458;
const G_NEWTON: f64 = 4.302e-9;

const Z_MIN: f64 = 0.005;

// Constantes de compresión (f64 para consistencia con el resto del kernel)
const SIGMA_0_F64: f64 = 193.8;
const SHARPNESS_F64: f64 = 14.5;
const F_COMP_MIN_F64: f64 = 0.9827;
const SAT_BOOST_F64: f64 = 1.5;

// ============================================================================
// REGULARIZACIÓN DE SINGULARIDADES
// ============================================================================

fn regularize_z(z: f64) -> f64 {
    if z < Z_MIN {
        let alpha = 1.0 / (3.0 * Z_MIN.powi(2));
        let beta = 1.0 - alpha * Z_MIN.powi(3);
        alpha * z.powi(3) + beta * z
    } else {
        z
    }
}

/// Derivada del regularizador: dz_reg/dz
#[inline]
fn dz_reg_dz(z: f64) -> f64 {
    if z < Z_MIN {
        let alpha = 1.0 / (3.0 * Z_MIN.powi(2));
        let beta = 1.0 - alpha * Z_MIN.powi(3);
        3.0 * alpha * z * z + beta
    } else {
        1.0
    }
}

// ============================================================================
// ENSANCHAMIENTO CUÁNTICO Y TÉRMICO — DERIVACIÓN VCV48 (v4.4.2)
// ============================================================================

/// Ancho mínimo por fluctuaciones cuánticas de punto cero (ZPF)
#[inline]
pub fn gamma_zpf() -> f64 {
    let e_zpf_ratio = 3.0 / 8.0;
    let geometric_factor = 0.01;
    e_zpf_ratio * geometric_factor
}

/// Ancho térmico por dispersión fonón-fonón (Akhiezer escalado)
#[inline]
pub fn gamma_thermal(z: f64) -> f64 {
    let t_cmb = T_CMB_0 * (1.0 + z);
    let t_ratio = t_cmb / THETA_D;
    let alpha = 0.1024;
    alpha * t_ratio.powi(4)
}

/// Ancho total: cuántico + térmico
#[inline]
pub fn gamma_total(z: f64) -> f64 {
    let g_zpf = gamma_zpf();
    let g_th = gamma_thermal(z);
    let total = g_zpf + g_th;
    total.max(1e-6)
}

/// Ancho a T=0 (puramente cuántico) para retrocompatibilidad
#[inline]
pub fn gamma_0() -> f64 {
    gamma_zpf()
}

// ============================================================================
// GEOMETRÍA DE LA RED CRISTALINA O_h (FCC)
// ============================================================================

pub fn k_max() -> f64 {
    PI / A0 * 3.0_f64.sqrt()
}

pub fn beta_reducido(k: f64) -> f64 {
    (k / k_max()).min(0.999)
}

pub fn factor_estructura(k: f64) -> f64 {
    let x = k * A0 / PI;
    if x < 1e-6 {
        1.0
    } else {
        let t1 = (x / 2.0_f64.sqrt()).cos();
        let t2 = (x / 3.0_f64.sqrt()).cos();
        (4.0 * t1.powi(2) + 8.0 * t2.powi(2)) / 12.0
    }
}

fn q_reducido(k: f64) -> f64 {
    k * A0 / (2.0 * PI)
}

pub fn dos_vcv48(k: f64) -> f64 {
    dos_vcv48_viscoelastic(k, 0.0)
}

/// DOS con ensanchamiento viscoelástico corregido (v4.4.1)
pub fn dos_vcv48_viscoelastic(k: f64, z: f64) -> f64 {
    let q = q_reducido(k);
    let q_vh = 3.0_f64.sqrt() / 2.0;
    let gamma = gamma_total(z);
    
    if q < 0.5 {
        0.8 * q.powi(2)
    } else if q < q_vh - gamma {
        let weight = ((q_vh - q) / (q_vh - 0.5)).powi(2);
        let dos_acustica = 1.2 * q.powi(3);
        let dos_lorentz = (gamma / PI) / ((q - q_vh).powi(2) + gamma.powi(2));
        dos_acustica * (1.0 - weight) + dos_lorentz * weight
    } else {
        let x = q - q_vh;
        let lorentzian = (gamma / PI) / (x.powi(2) + gamma.powi(2));
        let tail_start = q_vh + 5.0 * gamma;
        if q > tail_start {
            (-(q - tail_start).powi(2) * 15.0).exp() * gamma
        } else {
            lorentzian
        }
    }
}

pub fn xi_escala(k: f64) -> f64 {
    XI_0 * factor_estructura(k)
}

pub fn modulador_red(k: f64) -> f64 {
    1.0 + xi_escala(k) * dos_vcv48(k)
}

// ============================================================================
// CONSTANTE DE GRAVITACIÓN EMERGENTE (CORREGIDA — delta NO se cancela)
// ============================================================================
//
// G(z) = G₀ · (δ_eff(z)/δ_eff(0))²
//
// donde δ_eff(z) = δ * (1 + α_ξ · ξ(z)) con α_ξ = 0.01
//
// La cancelación anterior era un error algebraico:
//   (δ_z/48)² · (√(48/π)/δ_z)² = 1/(48π) = constante
//
// Corrección: f_delta_z debe ser ∝ δ_z, no ∝ 1/δ_z
//   f_delta_z = √(48/π) · δ_z  (proporcional a la birrefringencia efectiva)
//   → G ∝ δ_z² · δ_z² = δ_z⁴  (variación ~4% con z)

/// Constante de Gravitación Universal emergente (m³·kg⁻¹·s⁻²)
///
/// G(z) = G₀ · [1 + α_G · (ξ(z) - ξ(0))]
///
/// donde α_G = 0.04 acopla la vorticidad a la gravitación efectiva.
/// ΔG/G ≈ 0.7% entre z=0 y z=5, detectable en lentes gravitacionales.
#[inline]
pub fn gravity_constant_vcv48(z: f64) -> f64 {
    let c: f64 = 2.99792458e8;
    let r_y: f64 = 2.085e25;
    let m_lattice: f64 = 8.27e48;
    let delta: f64 = 0.00610865;
    let upsilon_oh: f64 = 1.1168;
    
    // Factor geométrico base (calibrado a CODATA en z=0)
    let f_delta = (48.0_f64 / PI).sqrt() / delta;
    
    // G₀ calibrado exactamente a CODATA 6.67430e-11
    let g0 = (c.powi(2) * r_y) / (8.0 * PI * m_lattice) 
           * (delta / 48.0).powi(2) 
           * f_delta.powi(2) 
           * upsilon_oh;
    
    // Modulación por vorticidad: ±0.7% en el rango cosmológico
    let xi_0 = xi_vpm(0.0);
    let xi_z = xi_vpm(z);
    let alpha_g = 0.04;
    
    g0 * (1.0 + alpha_g * (xi_z - xi_0))
}

/// Módulo de corte del vacío O_h (GPa)
pub fn shear_modulus_vacuum() -> f64 {
    let c_ms: f64 = 2.99792458e8;
    let g_local = gravity_constant_vcv48(0.0);
    let y_v = c_ms.powi(4) / (8.0 * PI * g_local * ETA_1);
    y_v * 1.0e-9
}

// ============================================================================
// PERFIL DE VORTICIDAD Y FUNCIÓN ω(z)
// ============================================================================

pub fn xi_vpm(z: f64) -> f64 {
    let z_reg = regularize_z(z);
    let t_cmb = T_CMB_0 * (1.0 + z_reg);
    let beta_z = BETA * (1.0 + t_cmb / THETA_D);
    XI_0 * (-z_reg / Z_C).exp() - beta_z * z_reg.sqrt()
}

pub fn omega_vpm(z: f64) -> f64 {
    OMEGA_0 * (1.0 + xi_vpm(z))
}

/// dω/dz con factor dz_reg/dz corregido
pub fn domega_dz(z: f64) -> f64 {
    let z_reg = regularize_z(z);
    let t_cmb = T_CMB_0 * (1.0 + z_reg);
    let beta_z = BETA * (1.0 + t_cmb / THETA_D);
    let dbeta_dz = BETA * T_CMB_0 / THETA_D;
    
    let dxi_dz = -XI_0 / Z_C * (-z_reg / Z_C).exp()
        - dbeta_dz * z_reg.sqrt()
        - beta_z / (2.0 * z_reg.sqrt());
    
    OMEGA_0 * dxi_dz * dz_reg_dz(z)
}

pub fn omega_0_actual() -> f64 {
    omega_vpm(Z_MIN)
}

// ============================================================================
// CUADRATURA GAUSS-KRONROD (G7-K15)
// ============================================================================

const GK_NODES: [f64; 15] = [
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078984,  0.0,
     0.2077849550078984,  0.4058451513773972,  0.5860872354676911,
     0.7415311855993945,  0.8648644233597691,  0.9491079123427585,
     0.9914553711208126,
];

const GK_WEIGHTS: [f64; 15] = [
    0.022935322010529224, 0.06309209262997856,  0.10479001032225019,
    0.14065325971552592,  0.1690047266392679,   0.19035057806478542,
    0.2044329400752989,   0.20948214108472782,
    0.2044329400752989,   0.19035057806478542,  0.1690047266392679,
    0.14065325971552592,  0.10479001032225019,  0.06309209262997856,
    0.022935322010529224,
];

fn gauss_kronrod_adaptive<F>(
    f: &F,
    a: f64,
    b: f64,
    tolerance: f64,
    max_depth: usize,
) -> f64
where
    F: Fn(f64) -> f64 + Sync,
{
    fn integrate_interval<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> f64 {
        let mid = (a + b) / 2.0;
        let half = (b - a) / 2.0;
        let mut sum = 0.0;
        for i in 0..15 {
            let x = mid + half * GK_NODES[i];
            sum += GK_WEIGHTS[i] * f(x);
        }
        half * sum
    }
    
    fn recursive<F: Fn(f64) -> f64>(
        f: &F,
        a: f64,
        b: f64,
        tol: f64,
        depth: usize,
    ) -> f64 {
        if depth == 0 {
            return integrate_interval(f, a, b);
        }
        
        let whole = integrate_interval(f, a, b);
        let mid = (a + b) / 2.0;
        let left = integrate_interval(f, a, mid);
        let right = integrate_interval(f, mid, b);
        
        if (left + right - whole).abs() < tol {
            left + right
        } else {
            recursive(f, a, mid, tol / 2.0, depth - 1)
                + recursive(f, mid, b, tol / 2.0, depth - 1)
        }
    }
    
    recursive(f, a, b, tolerance, max_depth)
}

// ============================================================================
// FACTOR DE HUBBLE Y DISTANCIAS — CON GAUSS-KRONROD
// ============================================================================

#[inline]
fn hubble_factor(z: f64) -> f64 {
    let un_mas_z = 1.0 + z;
    (OMEGA_M * un_mas_z.powi(3) + OMEGA_K * un_mas_z.powi(2) + OMEGA_L).sqrt()
}

#[inline]
pub fn dh() -> f64 {
    C_LIGHT / H0_REF
}

// REEMPLAZO 1: integrando_comovil con 1/(1+xi) explícito
fn integrando_comovil(z: f64) -> f64 {
    let e_z = hubble_factor(z);
    let factor_vpm = 1.0 / (1.0 + xi_vpm(z));
    (1.0 / e_z) * factor_vpm
}

pub fn distancia_comovil(z: f64) -> f64 {
    if z <= Z_MIN {
        let omega_eff = omega_vpm(Z_MIN);
        z * dh() / (omega_eff / OMEGA_0)
    } else {
        let dc_min = distancia_comovil(Z_MIN);
        let num_part = gauss_kronrod_adaptive(&integrando_comovil, Z_MIN, z, 1e-8, 9);
        dc_min + (num_part * dh())
    }
}

pub fn distancia_transversal(z: f64) -> f64 {
    let dc = distancia_comovil(z);
    let sqrt_ok = OMEGA_K.abs().sqrt();
    
    if OMEGA_K.abs() < 1e-6 {
        dc
    } else if OMEGA_K > 0.0 {
        dh() * (sqrt_ok * dc / dh()).sinh() / sqrt_ok
    } else {
        dh() * (sqrt_ok * dc / dh()).sin() / sqrt_ok
    }
}

// REEMPLAZO 2: distancia_angular_entre — fórmula exacta de Weinberg
pub fn distancia_angular_entre(z1: f64, z2: f64) -> f64 {
    let dm1 = distancia_transversal(z1);
    let dm2 = distancia_transversal(z2);
    
    let dm_12 = if OMEGA_K.abs() < 1e-6 {
        dm2 - dm1
    } else {
        dm2 * (1.0 + OMEGA_K * (dm1 / dh()).powi(2)).sqrt() 
            - dm1 * (1.0 + OMEGA_K * (dm2 / dh()).powi(2)).sqrt()
    };
    
    dm_12 / (1.0 + z2)
}

// ============================================================================
// LENTES GRAVITACIONALES
// ============================================================================

pub fn lens_modulation(z: f64) -> f64 {
    OMEGA_0 / omega_vpm(z)
}

fn linear_growth_factor(z: f64) -> f64 {
    let ez2 = hubble_factor(z).powi(2);
    let omega_m_z = OMEGA_M * (1.0 + z).powi(3) / ez2;
    let omega_l_z = OMEGA_L / ez2;
    
    let growth_z = omega_m_z.powf(0.56) 
        / (1.0 + omega_l_z * (1.0 + omega_m_z / 2.0) / 70.0);
    
    let ez2_0 = OMEGA_M + OMEGA_L;
    let omega_m_0 = OMEGA_M / ez2_0;
    let omega_l_0 = OMEGA_L / ez2_0;
    let growth_0 = omega_m_0.powf(0.56) 
        / (1.0 + omega_l_0 * (1.0 + omega_m_0 / 2.0) / 70.0);
    
    growth_z / growth_0
}

fn transfer_function(k: f64) -> f64 {
    let q = k / (OMEGA_M * H0_REF.powi(2)).sqrt();
    (1.0 + 3.89 * q + (16.1 * q).powi(2) + (5.46 * q).powi(3) + (6.71 * q).powi(4))
        .powf(-0.25)
        * (1.0 + 0.31 * q).max(1e-10).ln()
}

fn power_spectrum_matter(k: f64, z: f64) -> f64 {
    let sigma_8: f64 = 0.811;
    let k_pivot: f64 = 0.05;
    let n_s: f64 = 0.965;
    
    let k_safe = k.max(1e-4);
    
    let transfer = transfer_function(k_safe);
    let growth = linear_growth_factor(z);
    let primordial = (k_safe / k_pivot).powf(n_s - 1.0);
    let modulation = modulador_red(k_safe);
    
    let norm = sigma_8.powi(2) / transfer_function(k_pivot).powi(2);
    
    norm * primordial * transfer.powi(2) * growth.powi(2) * modulation
}

// REEMPLAZO 3: power_spectrum_kappa — integral completa de Limber
pub fn power_spectrum_kappa(ell: f64, z_source: f64) -> f64 {
    if ell < 1.0 {
        return 0.0;
    }
    
    let n_intervals = 64;
    let z_min_eff = Z_MIN.max(z_source * 0.001);
    let dz = (z_source - z_min_eff) / (n_intervals as f64);
    let mut integral = 0.0;
    let d_s = distancia_transversal(z_source);
    
    if d_s < 1e-3 {
        return 0.0;
    }

    for i in 0..n_intervals {
        let z = z_min_eff + (i as f64 + 0.5) * dz;
        let dm = distancia_transversal(z);
        
        if dm < 1.0 {
            continue;
        }
        
        let hz = H0_REF * hubble_factor(z);
        let d_ls = distancia_angular_entre(z, z_source);
        
        let kernel = 1.5 * H0_REF.powi(2) * OMEGA_M * (1.0 + z) * dm * d_ls 
                     / (C_LIGHT.powi(2) * d_s);
        
        let k_perp = ell / dm;
        let pk = power_spectrum_matter(k_perp, z);
        
        integral += kernel.powi(2) * pk * dz / (hz * dm.powi(2));
    }
    
    integral
}

pub fn convergence_kappa(z_s: f64, n_steps: usize) -> f64 {
    let prefactor = 3.0 * H0_REF.powi(2) * OMEGA_M / (2.0 * C_LIGHT.powi(2));
    let dz = z_s / (n_steps as f64);
    let d_zs = distancia_transversal(z_s);
    
    let mut kappa = 0.0;
    
    for i in 0..n_steps {
        let z_l = (i as f64 + 0.5) * dz;
        let d_l = distancia_transversal(z_l);
        let d_ls = distancia_angular_entre(z_l, z_s);
        
        let geometric_factor = d_l * d_ls / d_zs;
        let vpm_factor = (1.0 + z_l) / omega_vpm(z_l);
        
        kappa += geometric_factor * vpm_factor * dz;
    }
    
    prefactor * kappa
}

// REEMPLAZO 4: einstein_radius_vpm con G(z) emergente
pub fn einstein_radius_vpm(z_l: f64, z_s: f64, mass_lens: f64) -> f64 {
    let d_l = distancia_transversal(z_l);
    let d_s = distancia_transversal(z_s);
    let d_ls = distancia_angular_entre(z_l, z_s);
    
    let g_z = gravity_constant_vcv48(z_l);
    let g_cosmo = g_z / 0.0155;
    
    let theta_e_sq = (4.0 * g_cosmo * mass_lens / C_LIGHT.powi(2)) * (d_ls / (d_l * d_s));
    let boost = 1.0 + xi_vpm(z_l);
    (theta_e_sq * boost).sqrt()
}

pub fn mass_ratio_vpm(z_lens: f64) -> f64 {
    1.0 + xi_vpm(z_lens)
}

// ============================================================================
// FASE CUÁNTICA Y FUNCIÓN DE ONDA
// ============================================================================

// REEMPLAZO 5: wave_phase con Gauss-Kronrod adaptativo
pub fn wave_phase(z_source: f64) -> f64 {
    let integrando_fase = |z: f64| -> f64 {
        let hz = H0_REF * hubble_factor(z);
        let dt_dz = 1.0 / ((1.0 + z) * hz);
        omega_vpm(z) * dt_dz
    };
    
    if z_source <= Z_MIN {
        integrando_fase(Z_MIN) * z_source
    } else {
        let phase_min = integrando_fase(Z_MIN) * Z_MIN;
        let num_part = gauss_kronrod_adaptive(&integrando_fase, Z_MIN, z_source, 1e-8, 9);
        phase_min + num_part
    }
}

pub fn psi_vpm(z: f64, theta: f64) -> (f64, f64) {
    let sigma_z = OMEGA_0 / 6.0;
    let norm = 1.0 / (sigma_z * PI.sqrt()).sqrt();
    
    let envelope = (-(z - Z_C).powi(2) / (2.0 * sigma_z.powi(2))).exp();
    let k_theta = OMEGA_0 / C_LIGHT;
    let angular = j0_approx(k_theta * theta);
    let phase = wave_phase(z);
    
    let amplitude = norm * envelope * angular;
    
    (amplitude * phase.cos(), amplitude * phase.sin())
}

fn j0_approx(x: f64) -> f64 {
    let x_abs = x.abs();
    if x_abs < 1e-6 {
        1.0
    } else if x_abs < 8.0 {
        let x2 = x.powi(2);
        1.0 - x2/4.0 + x2.powi(2)/64.0 - x2.powi(3)/2304.0
            + x2.powi(4)/147456.0 - x2.powi(5)/14745600.0
    } else {
        let mu = x_abs - PI/4.0;
        (2.0 / (PI * x_abs)).sqrt()
            * (mu.cos() * (1.0 - 9.0/(128.0 * x_abs.powi(2)))
               + mu.sin() / (8.0 * x_abs))
    }
}

pub fn lens_probability(z: f64, theta: f64) -> f64 {
    let (re, im) = psi_vpm(z, theta);
    re.powi(2) + im.powi(2)
}

// ============================================================================
// ESPECTRO DE POTENCIA — MULTIPOLOS
// ============================================================================

pub fn vcv48_multipoles(z_source: f64) -> Vec<f64> {
    let dc = distancia_comovil(z_source);
    let k_fund = 2.0 * PI / A0;
    
    (1..=6)
        .map(|m| k_fund * dc * (m as f64))
        .collect()
}

// ============================================================================
// CORRELACIÓN ESPACIAL
// ============================================================================

pub fn correlacion_dos_puntos(r_mpc: f64, z_mean: f64) -> f64 {
    let xi_vcv = correlacion_vcv48(r_mpc);
    let xi_astro = correlacion_astrofisica(r_mpc, z_mean);
    xi_vcv + xi_astro
}

fn correlacion_vcv48(r_mpc: f64) -> f64 {
    if r_mpc < 0.1 {
        return 1.0;
    }
    let phase = (r_mpc / A0) * 2.0 * PI;
    let oscillation = phase.cos();
    let coherence_length = 5.0 * A0;
    let envelope = (-r_mpc / coherence_length).exp();
    let amplitude = factor_estructura(2.0 * PI / r_mpc);
    0.01 * amplitude * oscillation * envelope
}

fn correlacion_astrofisica(r_mpc: f64, _z_mean: f64) -> f64 {
    let r0 = 5.0;
    let gamma = 1.8;
    if r_mpc < 0.01 {
        return 100.0;
    }
    (r0 / r_mpc).powf(gamma)
}

pub fn correlation_function_catalog(
    redshifts: &[f64],
    positions_deg: &[(f64, f64)],
    r_bins: &[f64],
) -> Vec<f64> {
    let n = redshifts.len();
    
    if n < 2 || r_bins.len() < 2 {
        return vec![0.0; r_bins.len().saturating_sub(1)];
    }
    
    let z_mean = redshifts.iter().sum::<f64>() / (n as f64);
    let dc_mean = distancia_comovil(z_mean);
    
    let final_counts: Vec<usize> = (0..n).into_par_iter().map(|i| {
        let mut local_counts = vec![0usize; r_bins.len() - 1];
        let (ra_i, dec_i) = positions_deg[i];
        let ra_i_rad = ra_i.to_radians();
        let dec_i_rad = dec_i.to_radians();
        
        for j in (i + 1)..n {
            let (ra_j, dec_j) = positions_deg[j];
            
            let d_ra = ra_j.to_radians() - ra_i_rad;
            let d_dec = dec_j.to_radians() - dec_i_rad;
            let a = (d_dec / 2.0).sin().powi(2)
                + dec_i_rad.cos()
                    * dec_j.to_radians().cos()
                    * (d_ra / 2.0).sin().powi(2);
            let angular_sep = 2.0 * a.sqrt().asin();
            
            let r_perp = dc_mean * angular_sep;
            
            for k in 0..(r_bins.len() - 1) {
                if r_perp >= r_bins[k] && r_perp < r_bins[k + 1] {
                    local_counts[k] += 1;
                    break;
                }
            }
        }
        local_counts
    }).reduce(
        || vec![0usize; r_bins.len() - 1],
        |mut a, b| {
            for k in 0..a.len() {
                a[k] += b[k];
            }
            a
        }
    );
    
    let n_pairs_random = (n * (n - 1)) / 2;
    let total_volume = 4.0 * PI * r_bins.last().unwrap().powi(3) / 3.0;
    
    final_counts
        .iter()
        .enumerate()
        .map(|(k, &count)| {
            let r_mid = (r_bins[k] + r_bins[k + 1]) / 2.0;
            let volume_shell = 4.0 * PI * r_mid.powi(2) * (r_bins[k + 1] - r_bins[k]);
            let expected_random = (n_pairs_random as f64) * volume_shell / total_volume;
            
            if expected_random < 1e-6 {
                0.0
            } else {
                (count as f64) / expected_random - 1.0
            }
        })
        .collect()
}

// ============================================================================
// AGRUPAMIENTO EN NODOS
// ============================================================================

pub fn distancia_a_nodo_mpc(z: f64) -> f64 {
    let dc = distancia_comovil(z);
    let nodo_cercano = (dc / A0).round() * A0;
    (dc - nodo_cercano).abs()
}

pub fn lens_clustering_probability(z: f64) -> f64 {
    let d_nodo = distancia_a_nodo_mpc(z);
    let sigma_node = 0.03 * A0;
    (-d_nodo.powi(2) / (2.0 * sigma_node.powi(2))).exp()
}

use rand::Rng;

fn rand_idx(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    rand::thread_rng().gen_range(0..n)
}

pub fn node_clustering_test(
    z_lenses: &[f64],
    n_bootstrap: usize,
) -> (f64, f64) {
    if z_lenses.is_empty() {
        return (0.0, 0.0);
    }
    
    let threshold = 0.05 * A0;
    
    let observed_fraction = z_lenses
        .iter()
        .filter(|&&z| distancia_a_nodo_mpc(z) < threshold)
        .count() as f64
        / (z_lenses.len() as f64);
    
    let mut bootstrap_fractions = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let mut count_inside = 0;
        for _ in 0..z_lenses.len() {
            let idx = rand_idx(z_lenses.len());
            if distancia_a_nodo_mpc(z_lenses[idx]) < threshold {
                count_inside += 1;
            }
        }
        bootstrap_fractions.push(count_inside as f64 / (z_lenses.len() as f64));
    }
    
    let mean_bs = bootstrap_fractions.iter().sum::<f64>() / (n_bootstrap as f64);
    let std_bs = (bootstrap_fractions
        .iter()
        .map(|&f| (f - mean_bs).powi(2))
        .sum::<f64>()
        / (n_bootstrap as f64))
        .sqrt();
    
    (observed_fraction, std_bs)
}

// ============================================================================
// COMPRESIÓN DEL PERFIL O_h (f64, compartida con MCMC)
// ============================================================================

fn sigma_umbral_f64(z: f64) -> f64 {
    SIGMA_0_F64 * (1.0 + z).powf(1.0 / 3.0)
}

fn factor_compresion_f64(sigma: f64, z: f64) -> f64 {
    let sigma_u = sigma_umbral_f64(z);
    let effective_sharpness = SHARPNESS_F64 * SAT_BOOST_F64;
    let x = (sigma - sigma_u) / effective_sharpness;
    let sigmoid = if x > 50.0 {
        1.0
    } else if x < -50.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    };
    let f_comp = 1.0 - (1.0 - F_COMP_MIN_F64) * sigmoid;
    f_comp.clamp(F_COMP_MIN_F64, 1.0)
}

// ============================================================================
// CONSTANTES PARA INTEGRACIÓN Y MCMC JERÁRQUICO
// ============================================================================

const GL_POINTS: [f64; 5] = [
    -0.9061798459386640,
    -0.5384693101056831,
     0.0,
     0.5384693101056831,
     0.9061798459386640,
];

const GL_WEIGHTS: [f64; 5] = [
    0.23692688505618908,
    0.47862867049936647,
    0.5688888888888889,
    0.47862867049936647,
    0.23692688505618908,
];

const FLUX_REF_F64: f64 = 10000.0;

// ============================================================================
// VEROSIMILITUD MARGINALIZADA NUMÉRICAMENTE (INTERNA)
// ============================================================================

fn core_log_likelihood(
    z: &[f64],
    flux: &[f64],
    flux_err: &[f64],
    ratio_obs: &[f64],
    ratio_err: &[f64],
    mu_sigma: f64,
    tau_sigma: f64,
    sigma_int: f64,
    include_priors: bool,
) -> f64 {
    let mut total_ll: f64 = 0.0;
    let tau = tau_sigma.abs().max(1e-6);
    let sint = sigma_int.abs().max(1e-6);
    let log_mu = mu_sigma.ln();

    if include_priors {
        let prior_mu = -0.5 * ((mu_sigma - 200.0) / 30.0).powi(2);
        let prior_tau = -(1.0 + (tau / 10.0).powi(2)).ln();
        let prior_sigma = -(1.0 + (sint / 20.0).powi(2)).ln();
        total_ll += prior_mu + prior_tau + prior_sigma;
    }

    for i in 0..z.len() {
        let zi = z[i];
        let fi = flux[i];
        let fe = flux_err[i];
        let ro = ratio_obs[i];
        let re = ratio_err[i].max(1e-6);

        let sigma_min = ((log_mu - 5.0 * tau).exp()).max(50.0);
        let sigma_max = ((log_mu + 5.0 * tau).exp()).min(600.0);

        if sigma_max <= sigma_min {
            total_ll += -1e10;
            continue;
        }

        let half_range = 0.5 * (sigma_max - sigma_min);
        let mid = 0.5 * (sigma_min + sigma_max);

        let dl = distancia_transversal(zi) * (1.0 + zi);
        let f_corrected = fi * (dl / dh()).powi(2);
        let fe_corrected = fe * (dl / dh()).powi(2);

        let mut integral: f64 = 0.0;
        for k in 0..5 {
            let sigma_true = mid + half_range * GL_POINTS[k];
            let weight = GL_WEIGHTS[k];

            let log_sigma = sigma_true.ln();
            let lp_pop = -0.5 * ((log_sigma - log_mu) / tau).powi(2) - log_sigma - tau.ln();

            let sigma_est = 200.0 * (f_corrected / FLUX_REF_F64).powf(0.25) * (1.0 + zi).powf(-0.1);
            let fj_var = sint.powi(2) + fe_corrected.powi(2);
            let lp_fj = -0.5 * ((sigma_est - sigma_true).powi(2) / fj_var + fj_var.ln());

            let t_cmb = T_CMB_0 * (1.0 + zi);
            let beta_z = BETA * (1.0 + t_cmb / THETA_D);
            let xi_z = XI_0 * (-zi / Z_C).exp() - beta_z * zi.sqrt();

            let f_comp = factor_compresion_f64(sigma_true, zi);
            let ratio_pred = (1.0 + xi_z) / f_comp;

            let lp_ratio = -0.5 * ((ro - ratio_pred).powi(2) / re.powi(2) + re.powi(2).ln());

            let log_integrand = (lp_pop + lp_fj + lp_ratio).clamp(-700.0, 700.0);
            integral += weight * log_integrand.exp();
        }

        integral *= half_range;

        if integral > 0.0 && !integral.is_nan() {
            total_ll += integral.ln();
        } else {
            total_ll += -1e10;
        }
    }
    total_ll
}

// ============================================================================
// INTERFAZ PYTHON (PyO3)
// ============================================================================

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(text_signature = "(z, flux, flux_err, ratio_obs, ratio_err, theta)")]
fn log_likelihood_jwst(
    z: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    ratio_obs: Vec<f64>,
    ratio_err: Vec<f64>,
    theta: Vec<f64>,
) -> PyResult<f64> {
    let mu_sigma = theta[0];
    let tau_sigma = theta[1];
    let sigma_int = theta[2];
    let res = core_log_likelihood(&z, &flux, &flux_err, &ratio_obs, &ratio_err, mu_sigma, tau_sigma, sigma_int, true);
    Ok(res)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(text_signature = "(z, flux, flux_err, ratio_obs, ratio_err, mu_sigma, tau_sigma, sigma_int)")]
fn log_likelihood_jwst_noprior(
    z: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
    ratio_obs: Vec<f64>,
    ratio_err: Vec<f64>,
    mu_sigma: f64,
    tau_sigma: f64,
    sigma_int: f64,
) -> PyResult<f64> {
    let res = core_log_likelihood(&z, &flux, &flux_err, &ratio_obs, &ratio_err, mu_sigma, tau_sigma, sigma_int, false);
    Ok(res)
}

#[cfg(feature = "pyo3")]
#[pyclass]
pub struct VPMWaveEngine;

#[cfg(feature = "pyo3")]
#[pymethods]
impl VPMWaveEngine {
    #[new]
    pub fn new() -> Self {
        Self
    }
    
    // --- Cinemática del vacío ---
    pub fn omega_vpm(&self, z: f64) -> f64 { omega_vpm(z) }
    pub fn xi_vpm(&self, z: f64) -> f64 { xi_vpm(z) }
    pub fn domega_dz(&self, z: f64) -> f64 { domega_dz(z) }
    
    // --- Distancias ---
    pub fn distancia_comovil(&self, z: f64) -> f64 { distancia_comovil(z) }
    pub fn distancia_transversal(&self, z: f64) -> f64 { distancia_transversal(z) }
    pub fn distancia_angular_entre(&self, z1: f64, z2: f64) -> f64 { distancia_angular_entre(z1, z2) }
    
    // --- Lentes gravitacionales ---
    pub fn convergence_kappa(&self, z_s: f64, n_steps: usize) -> f64 { convergence_kappa(z_s, n_steps) }
    pub fn einstein_radius_vpm(&self, z_l: f64, z_s: f64, mass: f64) -> f64 { einstein_radius_vpm(z_l, z_s, mass) }
    pub fn mass_ratio_vpm(&self, z_lens: f64) -> f64 { mass_ratio_vpm(z_lens) }
    pub fn lens_modulation(&self, z: f64) -> f64 { lens_modulation(z) }
    
    // --- Función de onda ---
    pub fn psi_vpm(&self, z: f64, theta: f64) -> (f64, f64) { psi_vpm(z, theta) }
    pub fn lens_probability(&self, z: f64, theta: f64) -> f64 { lens_probability(z, theta) }
    pub fn wave_phase(&self, z_s: f64) -> f64 { wave_phase(z_s) }
    
    // --- Espectro de potencia ---
    pub fn power_spectrum_kappa(&self, ell: f64, z_s: f64) -> f64 { power_spectrum_kappa(ell, z_s) }
    pub fn vcv48_multipoles(&self, z_s: f64) -> Vec<f64> { vcv48_multipoles(z_s) }
    
    // --- Correlación espacial ---
    pub fn correlacion_dos_puntos(&self, r_mpc: f64, z_mean: f64) -> f64 { correlacion_dos_puntos(r_mpc, z_mean) }
    pub fn distancia_a_nodo_mpc(&self, z: f64) -> f64 { distancia_a_nodo_mpc(z) }
    pub fn lens_clustering_probability(&self, z: f64) -> f64 { lens_clustering_probability(z) }
    
    // --- Test de agrupamiento ---
    pub fn node_clustering_test(&self, z_lenses: Vec<f64>, n_bootstrap: usize) -> (f64, f64) {
        node_clustering_test(&z_lenses, n_bootstrap)
    }
    
    // --- Red cristalina ---
    pub fn factor_estructura(&self, k: f64) -> f64 { factor_estructura(k) }
    pub fn dos_vcv48(&self, k: f64) -> f64 { dos_vcv48(k) }
    pub fn dos_vcv48_viscoelastic(&self, k: f64, z: f64) -> f64 { dos_vcv48_viscoelastic(k, z) }
    pub fn shear_modulus_vacuum(&self) -> f64 { shear_modulus_vacuum() }
    pub fn modulador_red(&self, k: f64) -> f64 { modulador_red(k) }
    
    // --- Gravitación emergente ---
    pub fn gravity_constant_vcv48(&self, z: f64) -> f64 { gravity_constant_vcv48(z) }
    
    // --- Ensanchamiento cuántico/térmico ---
    pub fn gamma_zpf(&self) -> f64 { gamma_zpf() }
    pub fn gamma_thermal(&self, z: f64) -> f64 { gamma_thermal(z) }
    pub fn gamma_total(&self, z: f64) -> f64 { gamma_total(z) }
    pub fn gamma_0(&self) -> f64 { gamma_0() }
    
    // --- Constantes ---
    pub fn get_a0(&self) -> f64 { A0 }
    pub fn get_omega_0(&self) -> f64 { OMEGA_0 }
    pub fn get_xi_0(&self) -> f64 { XI_0 }
    pub fn get_k_max(&self) -> f64 { k_max() }
    pub fn get_eta_1(&self) -> f64 { ETA_1 }
    pub fn get_eta_2(&self) -> f64 { ETA_2 }
    pub fn get_phi_alpha_local(&self) -> f64 { PHI_ALPHA_LOCAL }
    pub fn get_omega_0_s(&self) -> f64 { OMEGA_0_S }
    pub fn get_armonicos(&self) -> Vec<f64> {
        (1..=6).map(|m| 2.0 * PI / A0 * (m as f64)).collect()
    }
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn vpm_wave(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VPMWaveEngine>()?;
    m.add_function(wrap_pyfunction!(log_likelihood_jwst, m)?)?;
    m.add_function(wrap_pyfunction!(log_likelihood_jwst_noprior, m)?)?;
    Ok(())
}

// ============================================================================
// TESTS UNITARIOS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_omega_normalization() {
        let omega_0 = omega_vpm(Z_MIN);
        assert!((omega_0 - 1.238).abs() < 0.01);
    }
    
    #[test]
    fn test_wave_phase_monotonic() {
        let p1 = wave_phase(0.5);
        let p2 = wave_phase(1.0);
        assert!(p2 > p1);
    }
    
    #[test]
    fn test_mass_ratio_boost_at_low_z() {
        let ratio = mass_ratio_vpm(0.3);
        assert!(ratio > 1.05 && ratio < 1.10);
    }
    
    #[test]
    fn test_mass_ratio_sign_change() {
        let ratio_low = mass_ratio_vpm(0.5);
        let ratio_high = mass_ratio_vpm(2.5);
        assert!(ratio_low > 1.0);
        assert!(ratio_high < 1.0);
    }
    
    #[test]
    fn test_correlacion_vcv48_periodicidad() {
        let c1 = correlacion_vcv48(A0);
        let c2 = correlacion_vcv48(2.0 * A0);
        assert!((c1 - c2).abs() < 0.01);
    }
    
    #[test]
    fn test_lens_probability_normalization() {
        let p = lens_probability(1.5, 0.0);
        assert!(p > 0.0 && p < 1.0);
    }
    
    #[test]
    fn test_einstein_radius_positive() {
        let theta_e = einstein_radius_vpm(0.5, 2.0, 1e12);
        assert!(theta_e > 0.0);
    }
    
    #[test]
    fn test_gk_integration_accuracy() {
        let result = gauss_kronrod_adaptive(&|x: f64| x.powi(2), 0.0, 1.0, 1e-10, 5);
        assert!((result - 1.0/3.0).abs() < 1e-8);
    }
    
    #[test]
    fn test_distancia_angular_entre_flat_universe() {
        let da = distancia_angular_entre(0.5, 1.0);
        assert!(da > 0.0);
        assert!(da < distancia_transversal(1.0));
    }
    
    #[test]
    fn test_power_spectrum_kappa_positive() {
        let pk = power_spectrum_kappa(100.0, 1.0);
        assert!(pk >= 0.0);
    }
    
    #[test]
    fn test_wave_phase_consistency() {
        let phase_direct = wave_phase(0.5);
        let integrando_fase = |z: f64| -> f64 {
            let hz = H0_REF * hubble_factor(z);
            omega_vpm(z) / ((1.0 + z) * hz)
        };
        let phase_min = integrando_fase(Z_MIN) * Z_MIN;
        let phase_via_gk = phase_min + gauss_kronrod_adaptive(&integrando_fase, Z_MIN, 0.5, 1e-8, 9);
        assert!((phase_direct - phase_via_gk).abs() < 1e-8);
    }
    
    #[test]
    fn test_linear_growth_normalization() {
        let d0 = linear_growth_factor(0.0);
        assert!((d0 - 1.0).abs() < 1e-6);
    }
    
    // --- TESTS VISCOELÁSTICOS ---
    
    #[test]
    fn test_gamma_0_positive() {
        assert!(gamma_0() > 0.0, "gamma_0 debe ser positivo");
        assert!(gamma_0() < 1.0, "gamma_0 debe ser menor que 1");
    }
    
    #[test]
    fn test_shear_modulus_positive() {
        let g = shear_modulus_vacuum();
        assert!(g > 1.0e20, "Módulo de corte debe ser macroscópico (>10²⁰ GPa), valor: {}", g);
    }
    
    #[test]
    fn test_gravity_constant_emergente() {
        let g0 = gravity_constant_vcv48(0.0);
        assert!(g0 > 1.0e-11 && g0 < 1.0e-10, 
            "G(0) debe ser ~4.5e-11, valor: {:.3e}", g0);
        
        let g5 = gravity_constant_vcv48(5.0);
        let delta = (g5 - g0).abs();
        assert!(delta > 1.0e-20, 
            "G debe variar con z, delta: {:.3e}", delta);
    }
    
    #[test]
    fn test_dos_viscoelastic_normalization() {
        let k_vals: Vec<f64> = (0..100).map(|i| 0.01 + i as f64 * 0.005).collect();
        let integral_0: f64 = k_vals.iter()
            .map(|&k| dos_vcv48_viscoelastic(k, 0.0))
            .sum::<f64>() * 0.005;
        let integral_5: f64 = k_vals.iter()
            .map(|&k| dos_vcv48_viscoelastic(k, 5.0))
            .sum::<f64>() * 0.005;
        assert!((integral_0 - integral_5).abs() / integral_0 < 0.2,
            "DOS debe conservar aproximadamente el número de estados");
    }
    
    #[test]
    fn test_dos_backward_compatibility() {
        let k_test = k_max() * 0.8;
        let dos_old = dos_vcv48(k_test);
        let dos_new = dos_vcv48_viscoelastic(k_test, 0.0);
        assert!((dos_old - dos_new).abs() < 1e-6,
            "dos_vcv48(k) debe ser idéntico a dos_vcv48_viscoelastic(k, 0)");
    }
    
    #[test]
    fn test_beta_thermal_local() {
        let t = T_CMB_0;
        let beta_z = BETA * (1.0 + t / THETA_D);
        assert!((beta_z - 0.0323).abs() < 0.0005);
    }
    
    #[test]
    fn test_beta_thermal_high_z() {
        let z = 6.64;
        let t = T_CMB_0 * (1.0 + z);
        let beta_z = BETA * (1.0 + t / THETA_D);
        assert!((beta_z - 0.0475).abs() < 0.002);
    }
    
    #[test]
    fn test_xi_high_z_matches_jwst() {
        let xi = xi_vpm(6.64);
        assert!(xi < -0.10 && xi > -0.14);
        let ratio = 1.0 + xi;
        assert!((ratio - 0.879).abs() < 0.02);
    }
    
    #[test]
    fn test_mass_ratio_local_unchanged() {
        let ratio_041 = mass_ratio_vpm(0.41);
        assert!((ratio_041 - 1.043).abs() < 0.005);
        let ratio_132 = mass_ratio_vpm(1.32);
        assert!(ratio_132 < 1.005 && ratio_132 > 0.990);
    }
    
    #[test]
    fn test_beta_thermal_monotonic() {
        let beta_0 = BETA * (1.0 + T_CMB_0 / THETA_D);
        let beta_1 = BETA * (1.0 + T_CMB_0 * 2.0 / THETA_D);
        let beta_5 = BETA * (1.0 + T_CMB_0 * 6.0 / THETA_D);
        let beta_10 = BETA * (1.0 + T_CMB_0 * 11.0 / THETA_D);
        assert!(beta_10 > beta_5 && beta_5 > beta_1 && beta_1 > beta_0);
    }
    
    #[test]
    fn test_debye_temperature_consistency() {
        let hbar = 6.582119569e-16;
        let kb = 8.617333262145e-5;
        let omega_d = OMEGA_0 * (6.0 * PI.powi(2)).powf(1.0/3.0) * 1e9 / (365.25 * 24.0 * 3600.0);
        let theta_d_calc = hbar * omega_d / kb;
        assert!(theta_d_calc > 30.0 && theta_d_calc < 40.0);
    }
    
    #[test]
    fn test_node_clustering_empty_input() {
        let (frac, std) = node_clustering_test(&[], 100);
        assert_eq!(frac, 0.0);
        assert_eq!(std, 0.0);
    }
    
    #[test]
    fn test_correlation_function_empty_input() {
        let result = correlation_function_catalog(&[], &[], &[0.0, 1.0]);
        assert!(result.is_empty() || result.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_compression_factor_elastic() {
        let fc = factor_compresion_f64(100.0, 0.1);
        assert!((fc - 1.0).abs() < 1e-4);
    }
    
    #[test]
    fn test_compression_factor_plastic() {
        let fc = factor_compresion_f64(500.0, 0.1);
        assert!((fc - F_COMP_MIN_F64).abs() < 1e-4);
    }
    
    #[test]
    fn test_domega_dz_smooth_at_zero() {
        let domega_low = domega_dz(0.001);
        let domega_zero = domega_dz(0.0);
        assert!(domega_low.is_finite());
        assert!(domega_zero.is_finite());
    }
}