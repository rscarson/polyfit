//! Generate optimized sRGB rational polynomial coefficients for linear-srgb.
//!
//! Output format: ready to paste into src/rational_poly.rs
//!
//! For each polynomial, tries multiple strategies and prints the best:
//!   - Unconstrained fit (best accuracy, ignores boundary)
//!   - SnapOnly (unconstrained fit + post-hoc boundary adjustment on p[0])
//!   - Weighted constraint (moderate weight, balances both)
//!   - Warm-start LM from current coefficients
//!
//! Run: cargo run --release --example generate_linear_srgb

use polyfit::rational::{
    Constraint, ConstraintMode, ErrorWeighting, F32SearchConfig, F32SearchResult, RationalFit,
    RationalFitOptions, RationalPolynomial,
};

const A: f64 = 0.055_010_718_947_586_6;
const A1: f64 = 1.0 + A;
const THRESHOLD_LINEAR: f64 = 0.003_041_282_560_127_521;
const THRESHOLD_GAMMA: f64 = 12.92 * THRESHOLD_LINEAR;

fn ref_s2l(v: f64) -> f64 {
    if v <= THRESHOLD_GAMMA {
        v / 12.92
    } else {
        ((v + A) / A1).powf(2.4)
    }
}
fn ref_l2s(v: f64) -> f64 {
    if v <= THRESHOLD_LINEAR {
        v * 12.92
    } else {
        A1 * v.powf(1.0 / 2.4) - A
    }
}

fn chebyshev_data(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|k| {
            let theta = pi * (2.0 * k as f64 + 1.0) / (2.0 * n as f64);
            let x = (lo + hi) / 2.0 + (hi - lo) / 2.0 * theta.cos();
            (x, f(x))
        })
        .collect()
}

fn find_best(
    label: &str,
    fit_fn: impl Fn(f64) -> f64 + Copy,
    domain_lo: f64,
    domain_hi: f64,
    p_deg: usize,
    q_deg: usize,
    reference: impl Fn(f64) -> f64 + Copy,
    sweep_lo: f32,
    sweep_hi: f32,
    transform: impl Fn(f32) -> f32 + Copy,
    transform_f64: impl Fn(f64) -> f64 + Copy,
    boundary: (f64, f64),
    current_p: &[f32],
    current_q: &[f32],
) -> F32SearchResult {
    println!("\n{}", "=".repeat(70));
    println!("  {label}");
    println!("{}\n", "=".repeat(70));

    // Use sampled mode (500K uniform samples) — much faster than exhaustive f32 sweep,
    // and we validate with a real exhaustive sweep at the end.
    let cfg = F32SearchConfig {
        samples: Some(500_000),
        ..F32SearchConfig::default()
    };

    let mut candidates: Vec<(String, F32SearchResult)> = Vec::new();

    // Current baseline
    let cur_poly = RationalPolynomial::new(
        current_p.iter().map(|&c| f64::from(c)).collect(),
        current_q.iter().map(|&c| f64::from(c)).collect(),
    );
    let cur = cur_poly.optimize_f32(reference, sweep_lo, sweep_hi, transform, cfg.clone());
    println!(
        "Current (no search): max={} bnd={} avg={:.3} mono={}",
        cur.max_ulp, cur.boundary_ulp, cur.avg_ulp, cur.mono_violations
    );
    candidates.push(("current       ".to_string(), cur));

    // Strategy 1: Unconstrained
    {
        let opt = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            n_samples: 10_000,
            restarts: 4,
            ..RationalFitOptions::default()
        };
        let fit =
            RationalFit::from_function(fit_fn, domain_lo..=domain_hi, p_deg, q_deg, opt).unwrap();
        let r =
            fit.as_polynomial()
                .optimize_f32(reference, sweep_lo, sweep_hi, transform, cfg.clone());
        println!("Unconstrained:       {r}");
        candidates.push(("unconstrained ".to_string(), r));
    }

    // Strategy 2: SnapOnly (nails boundary=0 or 1 via p[0] adjustment)
    {
        let opt = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            n_samples: 10_000,
            restarts: 4,
            constraints: vec![Constraint::new(boundary.0, boundary.1)],
            constraint_mode: ConstraintMode::SnapOnly,
            ..RationalFitOptions::default()
        };
        let fit =
            RationalFit::from_function(fit_fn, domain_lo..=domain_hi, p_deg, q_deg, opt).unwrap();
        let r =
            fit.as_polynomial()
                .optimize_f32(reference, sweep_lo, sweep_hi, transform, cfg.clone());
        println!("SnapOnly:            {r}");
        candidates.push(("snap_only     ".to_string(), r));
    }

    // Strategy 3: Weighted constraint (moderate)
    for &weight in &[100.0, 1000.0, 10000.0] {
        let opt = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            n_samples: 10_000,
            restarts: 4,
            constraints: vec![Constraint::with_weight(boundary.0, boundary.1, weight)],
            constraint_mode: ConstraintMode::Both,
            ..RationalFitOptions::default()
        };
        let fit =
            RationalFit::from_function(fit_fn, domain_lo..=domain_hi, p_deg, q_deg, opt).unwrap();
        let r =
            fit.as_polynomial()
                .optimize_f32(reference, sweep_lo, sweep_hi, transform, cfg.clone());
        println!("Weighted w={weight:>6.0}:    {r}");
        candidates.push((format!("weighted_{weight:.0}    "), r));
    }

    // Strategy 4: Warm-start LM from current
    {
        let data = chebyshev_data(fit_fn, domain_lo, domain_hi, 10_000);
        let r = RationalPolynomial::<f64>::refine_from_f32(
            current_p,
            current_q,
            reference,
            &data,
            sweep_lo,
            sweep_hi,
            transform,
            cfg.clone(),
            ErrorWeighting::Relative,
            300,
        );
        println!("Warm-start LM:       {r}");
        candidates.push(("warm_start    ".to_string(), r));
    }

    // Suppress unused warning
    let _ = transform_f64;

    // Pick best by (max_ulp ascending, then boundary_ulp ascending, then avg_ulp)
    let best_idx = candidates
        .iter()
        .enumerate()
        .filter(|(_, (_, r))| r.mono_violations == 0)
        .min_by(|a, b| {
            let ar = &a.1 .1;
            let br = &b.1 .1;
            (ar.max_ulp, ar.boundary_ulp)
                .cmp(&(br.max_ulp, br.boundary_ulp))
                .then(ar.avg_ulp.partial_cmp(&br.avg_ulp).unwrap())
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let (best_label, best) = &candidates[best_idx];
    println!("\n>>> Best strategy: {}", best_label.trim());
    println!(">>> {}", best);
    best.clone()
}

fn print_coeffs(name: &str, coeffs: &[f32], comment: &str) {
    if !comment.is_empty() {
        println!("/// {comment}");
    }
    println!("pub(crate) const {name}: [f32; {}] = [", coeffs.len());
    for (i, &c) in coeffs.iter().enumerate() {
        let comma = if i + 1 < coeffs.len() { "," } else { "" };
        println!("    {c:.10e}{comma}");
    }
    println!("];");
}

fn main() {
    // S2L 4/4
    let s2l = find_best(
        "S2L 4/4 (base clamped path, [0, 1])",
        |v| ((v + A) / A1).powf(2.4),
        THRESHOLD_GAMMA,
        1.0,
        4,
        4,
        ref_s2l,
        THRESHOLD_GAMMA as f32,
        1.0,
        |v| v,
        |v| v,
        (THRESHOLD_GAMMA, THRESHOLD_LINEAR),
        &[
            1.724_942_4e-2,
            8.335_514_7e-1,
            1.326_215_8e1,
            7.033_073_4e1,
            8.387_046e1,
        ],
        &[2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0],
    );

    // L2S 4/4 (on √x)
    let l2s = find_best(
        "L2S 4/4 (base clamped path, on √x, [0, 1])",
        |s| A1 * (s * s).powf(1.0 / 2.4) - A,
        THRESHOLD_LINEAR.sqrt(),
        1.0,
        4,
        4,
        ref_l2s,
        THRESHOLD_LINEAR as f32,
        1.0,
        |v| v.sqrt(),
        |v| v.sqrt(),
        (THRESHOLD_LINEAR.sqrt(), THRESHOLD_GAMMA),
        &[
            -1.513_885e-2,
            1.167_372_8e-1,
            1.257_921_2e1,
            5.259_309_8e1,
            2.852_907_6e1,
        ],
        &[2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0],
    );

    // EXT_S2L 6/6 (extended [0, 8])
    let ext_s2l = find_best(
        "EXT_S2L 6/6 (extended SIMD path, [0, 8])",
        |v| ((v + A) / A1).powf(2.4),
        THRESHOLD_GAMMA,
        8.0,
        6,
        6,
        ref_s2l,
        THRESHOLD_GAMMA as f32,
        1.0,
        |v| v,
        |v| v,
        (THRESHOLD_GAMMA, THRESHOLD_LINEAR),
        &[
            1.802_136_5e1,
            9.110_411_4e2,
            1.570_602_1e4,
            1.020_638_2e5,
            2.199_931_2e5,
            1.338_269_2e5,
            1.706_519_4e4,
        ],
        &[
            2.159_401_7e4,
            1.508_555_1e5,
            2.303_299_0e5,
            8.239_410_8e4,
            4.473_249_1e3,
            -6.359_000_1e1,
            1.0,
        ],
    );

    // EXT_L2S 6/6 (extended √x, [0, 64])
    let ext_l2s = find_best(
        "EXT_L2S 6/6 (extended SIMD path, on √x, [0, 64])",
        |s| A1 * (s * s).powf(1.0 / 2.4) - A,
        THRESHOLD_LINEAR.sqrt(),
        8.0,
        6,
        6,
        ref_l2s,
        THRESHOLD_LINEAR as f32,
        1.0,
        |v| v.sqrt(),
        |v| v.sqrt(),
        (THRESHOLD_LINEAR.sqrt(), THRESHOLD_GAMMA),
        &[
            -1.025_467_4,
            -3.075_361_5e-1,
            1.027_286e3,
            7.093_665e3,
            1.006_868_9e4,
            3.230_716e3,
            1.769_130_4e2,
        ],
        &[
            1.977_460_5e1,
            8.308_271e2,
            6.024_792_5e3,
            1.024_407_5e4,
            4.157_534e3,
            3.179_324_6e2,
            1.0,
        ],
    );

    // Final summary
    println!("\n\n{}", "=".repeat(70));
    println!("  COEFFICIENTS FOR linear-srgb/src/rational_poly.rs");
    println!("{}\n", "=".repeat(70));
    print_coeffs(
        "S2L_P",
        &s2l.numerator,
        &format!(
            "sRGB EOTF (encoded → linear) numerator coefficients. Max {} ULP, boundary {} ULP.",
            s2l.max_ulp, s2l.boundary_ulp
        ),
    );
    print_coeffs(
        "S2L_Q",
        &s2l.denominator,
        "sRGB EOTF (encoded → linear) denominator coefficients.",
    );
    println!();
    print_coeffs("L2S_P", &l2s.numerator,
        &format!("sRGB inverse EOTF (linear → encoded) numerator coefficients. Max {} ULP, boundary {} ULP. Evaluated on sqrt(linear).",
            l2s.max_ulp, l2s.boundary_ulp));
    print_coeffs(
        "L2S_Q",
        &l2s.denominator,
        "sRGB inverse EOTF (linear → encoded) denominator coefficients. Evaluated on sqrt(linear).",
    );
    println!();
    print_coeffs("EXT_S2L_P", &ext_s2l.numerator,
        &format!("Extended sRGB EOTF numerator (degree 6, fitted to [threshold, 8]). Max {} ULP in [0,1], boundary {} ULP.",
            ext_s2l.max_ulp, ext_s2l.boundary_ulp));
    print_coeffs(
        "EXT_S2L_Q",
        &ext_s2l.denominator,
        "Extended sRGB EOTF denominator (degree 6, fitted to [threshold, 8]).",
    );
    println!();
    print_coeffs("EXT_L2S_P", &ext_l2s.numerator,
        &format!("Extended sRGB inverse EOTF numerator (degree 6, on sqrt to [threshold, 64]). Max {} ULP in [0,1], boundary {} ULP.",
            ext_l2s.max_ulp, ext_l2s.boundary_ulp));
    print_coeffs(
        "EXT_L2S_Q",
        &ext_l2s.denominator,
        "Extended sRGB inverse EOTF denominator (degree 6, on sqrt to [threshold, 64]).",
    );
}
