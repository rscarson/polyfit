//! Optimize sRGB rational polynomial coefficients for the linear-srgb crate.
//!
//! Evaluates with f64-intermediate Horner (matching linear-srgb's scalar path).
//! Exhaustive sweep of every f32 in the power segment [threshold, 1.0].
//!
//! Covers:
//!   S2L 4/4 — base clamped path, ((v + offset) / scale)^2.4
//!   L2S 4/4 — base clamped path, evaluated on √(linear)
//!   S2L 6/6 — extended SIMD path [threshold, 8]
//!   L2S 6/6 — extended SIMD path on √x [threshold, 64]
//!
//! Run: cargo run --release --example optimize_srgb

use polyfit::rational::{
    Constraint, ConstraintMode, ErrorWeighting, F32SearchConfig, RationalFit, RationalFitOptions,
};

// =============================================================================
// sRGB constants (C0-continuous, moxcms-derived)
// =============================================================================

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

// =============================================================================
// f32 measurement utilities (sign-correct ULP, f64-intermediate Horner)
// =============================================================================

/// Evaluate rational polynomial with f32 coefficients via f64 Horner.
/// Matches eval_rational_poly_5 in linear-srgb/src/rational_poly.rs.
fn eval_f64mid(x: f32, p: &[f32], q: &[f32]) -> f32 {
    let x = x as f64;
    let mut yp = *p.last().unwrap() as f64;
    for &c in p[..p.len() - 1].iter().rev() {
        yp = yp.mul_add(x, c as f64);
    }
    let mut yq = *q.last().unwrap() as f64;
    for &c in q[..q.len() - 1].iter().rev() {
        yq = yq.mul_add(x, c as f64);
    }
    (yp / yq) as f32
}

/// ULP distance handling sign crossings correctly.
fn ulp_dist(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let map = |v: f32| -> i64 {
        let bits = v.to_bits() as i32;
        i64::from(if bits < 0 { i32::MIN - bits } else { bits })
    };
    (map(a) - map(b)).unsigned_abs() as u32
}

fn f32_next(v: f32) -> f32 {
    if v.is_nan() || v >= f32::MAX {
        return v;
    }
    if v == 0.0 {
        return f32::from_bits(1);
    }
    if v > 0.0 {
        f32::from_bits(v.to_bits() + 1)
    } else {
        f32::from_bits(v.to_bits() - 1)
    }
}

struct Metrics {
    max_ulp: u32,
    avg_ulp: f64,
    bnd: u32,
    mono: u64,
    count: u64,
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max={:>3} bnd={} avg={:.3} mono={} ({:.1}M values)",
            self.max_ulp,
            self.bnd,
            self.avg_ulp,
            self.mono,
            self.count as f64 / 1e6
        )
    }
}

/// Exhaustive sweep of every f32 in [lo, hi], with optional input transform.
fn measure(
    p: &[f32],
    q: &[f32],
    reference: impl Fn(f64) -> f64,
    lo: f32,
    hi: f32,
    xform: impl Fn(f32) -> f32,
) -> Metrics {
    let mut max_ulp = 0u32;
    let mut sum = 0u64;
    let mut count = 0u64;
    let mut mono = 0u64;
    let mut prev = eval_f64mid(xform(lo), p, q);
    let mut v = lo;
    while v <= hi {
        let got = eval_f64mid(xform(v), p, q);
        let expected = reference(v as f64) as f32;
        let ulp = ulp_dist(got, expected);
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        sum += ulp as u64;
        count += 1;
        if v > lo && got < prev {
            mono += 1;
        }
        prev = got;
        v = f32_next(v);
    }
    let bnd = ulp_dist(eval_f64mid(xform(lo), p, q), reference(lo as f64) as f32);
    Metrics {
        max_ulp,
        avg_ulp: sum as f64 / count.max(1) as f64,
        bnd,
        mono,
        count,
    }
}

/// Measure u16 and u8 safe boundaries via dense sampling.
fn safe_bounds(
    p: &[f32],
    q: &[f32],
    reference: impl Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    xform: impl Fn(f64) -> f64,
) -> (f64, f64, f64) {
    let u16_h = 0.5 / 65535.0;
    let u16_1 = 1.0 / 65535.0;
    let u8_h = 0.5 / 255.0;
    let (mut u16_exact, mut u16_pm1, mut u8_safe) = (hi, hi, hi);
    let n = 10_000_000;
    for i in 0..=n {
        let x = lo + (hi - lo) * i as f64 / n as f64;
        let got = eval_f64mid(xform(x) as f32, p, q) as f64;
        let err = (got - reference(x)).abs();
        if err >= u16_h && x < u16_exact {
            u16_exact = x;
        }
        if err >= u16_1 && x < u16_pm1 {
            u16_pm1 = x;
        }
        if err >= u8_h && x < u8_safe {
            u8_safe = x;
        }
    }
    (u16_exact, u16_pm1, u8_safe)
}

fn fit_and_report(
    label: &str,
    fit_fn: impl Fn(f64) -> f64,
    domain_lo: f64,
    domain_hi: f64,
    p_deg: usize,
    q_deg: usize,
    reference: impl Fn(f64) -> f64 + Copy,
    sweep_lo: f32,
    sweep_hi: f32,
    xform: impl Fn(f32) -> f32 + Copy,
    xform_f64: impl Fn(f64) -> f64 + Copy,
    constraint: Option<(f64, f64)>,
    current_p: &[f32],
    current_q: &[f32],
    safe_hi: f64,
) {
    println!("\n{}", "=".repeat(65));
    println!("  {label}");
    println!("{}\n", "=".repeat(65));

    // --- Current coefficients ---
    let cur = measure(current_p, current_q, reference, sweep_lo, sweep_hi, xform);
    println!("Current:    {cur}");
    if safe_hi > 1.0 {
        let (u16e, u16p, u8s) = safe_bounds(
            current_p,
            current_q,
            reference,
            sweep_lo as f64,
            safe_hi,
            xform_f64,
        );
        println!("            u16\u{00b1}0 \u{2264}{u16e:.2}  u16\u{00b1}1 \u{2264}{u16p:.2}  u8 \u{2264}{u8s:.2}");
    }

    // --- Fit ---
    let mut options = RationalFitOptions {
        weighting: ErrorWeighting::Relative,
        n_samples: 10_000,
        restarts: 8,
        ..RationalFitOptions::default()
    };
    if let Some((cx, cy)) = constraint {
        options.constraints = vec![Constraint::new(cx, cy)];
        options.constraint_mode = ConstraintMode::SnapOnly;
    }

    let fit =
        RationalFit::from_function(fit_fn, domain_lo..=domain_hi, p_deg, q_deg, options).unwrap();

    // --- Boundary report ---
    let report = fit.boundary_error(reference);
    println!("\n{report}");

    // --- f32 optimization (uses f64-intermediate Horner internally) ---
    let result = fit.as_polynomial().optimize_f32(
        reference,
        sweep_lo,
        sweep_hi,
        xform,
        F32SearchConfig::default(),
    );
    println!("Optimized:  {result}");

    if safe_hi > 1.0 {
        let (u16e, u16p, u8s) = safe_bounds(
            &result.numerator,
            &result.denominator,
            reference,
            sweep_lo as f64,
            safe_hi,
            xform_f64,
        );
        println!("            u16\u{00b1}0 \u{2264}{u16e:.2}  u16\u{00b1}1 \u{2264}{u16p:.2}  u8 \u{2264}{u8s:.2}");
    }

    // --- Comparison ---
    let better_max = result.max_ulp < cur.max_ulp;
    let better_avg = result.max_ulp == cur.max_ulp && result.avg_ulp < cur.avg_ulp;
    if better_max || better_avg {
        println!(
            "\n  >>> IMPROVED: max {} \u{2192} {}, avg {:.2} \u{2192} {:.2}",
            cur.max_ulp, result.max_ulp, cur.avg_ulp, result.avg_ulp
        );
    } else if result.max_ulp == cur.max_ulp {
        println!("\n  --- MATCHED max ULP");
    } else {
        println!(
            "\n  !!! WORSE: max {} \u{2192} {}",
            cur.max_ulp, result.max_ulp
        );
    }

    println!("\nP: {:?}", result.numerator);
    println!("Q: {:?}", result.denominator);

    // Rust const format
    println!("\n// Rust const:");
    print_const("P", &result.numerator);
    print_const("Q", &result.denominator);
}

fn print_const(name: &str, coeffs: &[f32]) {
    println!("const {name}: [f32; {}] = [", coeffs.len());
    for (i, &c) in coeffs.iter().enumerate() {
        let comma = if i + 1 < coeffs.len() { "," } else { "" };
        println!("    {c:>16.6e}{comma}");
    }
    println!("];");
}

fn main() {
    fit_and_report(
        "S2L 4/4 \u{2014} sRGB encoded \u{2192} linear [0, 1]",
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
        Some((THRESHOLD_GAMMA, THRESHOLD_LINEAR)),
        &[
            1.724_942_4e-2,
            8.335_514_7e-1,
            1.326_215_8e1,
            7.033_073_4e1,
            8.387_046e1,
        ],
        &[2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0],
        1.0,
    );

    fit_and_report(
        "L2S 4/4 \u{2014} linear \u{2192} sRGB encoded [0, 1] (on \u{221a}x)",
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
        Some((THRESHOLD_LINEAR.sqrt(), THRESHOLD_GAMMA)),
        &[
            -1.513_885e-2,
            1.167_372_8e-1,
            1.257_921_2e1,
            5.259_309_8e1,
            2.852_907_6e1,
        ],
        &[2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0],
        1.0,
    );

    fit_and_report(
        "S2L 6/6 \u{2014} sRGB encoded \u{2192} linear [0, 8] (extended)",
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
        Some((THRESHOLD_GAMMA, THRESHOLD_LINEAR)),
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
        8.0,
    );

    fit_and_report(
        "L2S 6/6 \u{2014} linear \u{2192} sRGB encoded [0, 64] on \u{221a}x (extended)",
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
        Some((THRESHOLD_LINEAR.sqrt(), THRESHOLD_GAMMA)),
        &[
            -1.780_184_6,
            5.030_732_0,
            1.656_664_9e3,
            1.017_330_6e4,
            1.298_072_5e4,
            3.771_270_8e3,
            1.888_817_8e2,
        ],
        &[
            3.446_206_7e1,
            1.327_327_9e3,
            8.730_898_7e3,
            1.340_767_0e4,
            4.928_503_9e3,
            3.442_401_8e2,
            1.0,
        ],
        64.0,
    );
}
