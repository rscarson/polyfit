//! Optimize rational polynomial coefficients for fast transcendentals.
//!
//! These polynomials are evaluated in pure f32 (SIMD), not f64-intermediate.
//! Uses sampled measurement because the polynomial input domain is the
//! transformed variable, not the original function domain.
//!
//! For each target, tries two strategies and keeps the best:
//!   A) Fresh fit (SK + LM + multi-restart from random)
//!   B) Warm-start LM refinement from the existing coefficients
//!
//! For log2 (output crosses zero), uses absolute-error metric.
//!
//! Run: cargo run --release --example optimize_transcendentals

use polyfit::rational::{
    ErrorWeighting, F32ScoreMetric, F32SearchConfig, F32SearchResult, RationalFit,
    RationalFitOptions, RationalPolynomial,
};

// =============================================================================
// f32 measurement (pure f32 Horner — matches SIMD evaluation)
// =============================================================================

fn eval_f32(x: f32, p: &[f32], q: &[f32]) -> f32 {
    let mut yp = *p.last().unwrap();
    for &c in p[..p.len() - 1].iter().rev() {
        yp = yp.mul_add(x, c);
    }
    let mut yq = *q.last().unwrap();
    for &c in q[..q.len() - 1].iter().rev() {
        yq = yq.mul_add(x, c);
    }
    yp / yq
}

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

#[derive(Clone, Copy)]
struct Metrics {
    max_ulp: u32,
    avg_ulp: f64,
    max_abs: f64,
    avg_abs: f64,
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max_ulp={:>4} avg_ulp={:>7.3} max_abs={:.3e} avg_abs={:.3e}",
            self.max_ulp, self.avg_ulp, self.max_abs, self.avg_abs,
        )
    }
}

fn measure(
    p: &[f32],
    q: &[f32],
    reference: impl Fn(f64) -> f64,
    lo: f32,
    hi: f32,
    n: usize,
) -> Metrics {
    let mut max_ulp = 0u32;
    let mut sum_ulp = 0u64;
    let mut max_abs: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    for i in 0..=n {
        let x = lo as f64 + (hi as f64 - lo as f64) * i as f64 / n as f64;
        let got = eval_f32(x as f32, p, q);
        let expected = reference(x) as f32;
        let ulp = ulp_dist(got, expected);
        let abs_err = (f64::from(got) - f64::from(expected)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        sum_ulp += u64::from(ulp);
        sum_abs += abs_err;
    }
    Metrics {
        max_ulp,
        avg_ulp: sum_ulp as f64 / (n + 1) as f64,
        max_abs,
        avg_abs: sum_abs / (n + 1) as f64,
    }
}

// =============================================================================
// PQ reference
// =============================================================================

fn pq_eotf_exact(v: f64) -> f64 {
    let (m1, m2) = (0.159_301_757_812_5_f64, 78.843_75_f64);
    let (c1, c2, c3) = (0.835_937_5_f64, 18.851_562_5_f64, 18.687_5_f64);
    let vp = v.powf(1.0 / m2);
    let num = (vp - c1).max(0.0);
    let den = c2 - c3 * vp;
    if den <= 0.0 {
        1.0
    } else {
        (num / den).powf(1.0 / m1)
    }
}

fn pq_inv_exact(v: f64) -> f64 {
    let (m1, m2) = (0.159_301_757_812_5_f64, 78.843_75_f64);
    let (c1, c2, c3) = (0.835_937_5_f64, 18.851_562_5_f64, 18.687_5_f64);
    let vp = v.powf(m1);
    ((c1 + c2 * vp) / (1.0 + c3 * vp)).powf(m2)
}

// =============================================================================
// Chebyshev data generator (for warm-start LM)
// =============================================================================

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

// =============================================================================
// Strategy runner
// =============================================================================

struct Strategy {
    label: &'static str,
    result: F32SearchResult,
}

fn run_both_strategies(
    fit_fn: impl Fn(f64) -> f64 + Copy,
    domain_lo: f64,
    domain_hi: f64,
    p_deg: usize,
    q_deg: usize,
    reference: impl Fn(f64) -> f64 + Copy,
    sweep_lo: f32,
    sweep_hi: f32,
    current_p: &[f32],
    current_q: &[f32],
    metric: F32ScoreMetric,
) -> Vec<Strategy> {
    let config = F32SearchConfig {
        metric,
        samples: Some(100_000),
        ..F32SearchConfig::default()
    };

    // Strategy A: fresh fit
    let fresh = {
        let options = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            n_samples: 10_000,
            restarts: 4,
            ..RationalFitOptions::default()
        };
        let fit = RationalFit::from_function(fit_fn, domain_lo..=domain_hi, p_deg, q_deg, options)
            .unwrap();
        fit.as_polynomial()
            .optimize_f32(reference, sweep_lo, sweep_hi, |v| v, config.clone())
    };

    // Strategy B: warm-start from existing coefficients
    let warm = {
        let data = chebyshev_data(fit_fn, domain_lo, domain_hi, 10_000);
        RationalPolynomial::<f64>::refine_from_f32(
            current_p,
            current_q,
            reference,
            &data,
            sweep_lo,
            sweep_hi,
            |v| v,
            config,
            ErrorWeighting::Relative,
            200,
        )
    };

    vec![
        Strategy {
            label: "fresh fit      ",
            result: fresh,
        },
        Strategy {
            label: "warm-start LM  ",
            result: warm,
        },
    ]
}

fn report_strategies(
    label: &str,
    current_p: &[f32],
    current_q: &[f32],
    reference: impl Fn(f64) -> f64 + Copy,
    sweep_lo: f32,
    sweep_hi: f32,
    strategies: Vec<Strategy>,
    metric: F32ScoreMetric,
) {
    println!("\n{}", "=".repeat(70));
    println!("  {label}");
    println!("{}\n", "=".repeat(70));

    let cur = measure(
        current_p, current_q, reference, sweep_lo, sweep_hi, 1_000_000,
    );
    println!("Current:           {cur}");

    let mut best: Option<&Strategy> = None;
    for s in &strategies {
        let m = measure(
            &s.result.numerator,
            &s.result.denominator,
            reference,
            sweep_lo,
            sweep_hi,
            1_000_000,
        );
        println!("{}:  {m}", s.label);
        let is_better = match metric {
            F32ScoreMetric::Ulp => best.is_none_or(|b| {
                let bm = measure(
                    &b.result.numerator,
                    &b.result.denominator,
                    reference,
                    sweep_lo,
                    sweep_hi,
                    100_000,
                );
                m.max_ulp < bm.max_ulp || (m.max_ulp == bm.max_ulp && m.avg_ulp < bm.avg_ulp)
            }),
            F32ScoreMetric::AbsoluteError => best.is_none_or(|b| {
                let bm = measure(
                    &b.result.numerator,
                    &b.result.denominator,
                    reference,
                    sweep_lo,
                    sweep_hi,
                    100_000,
                );
                m.max_abs < bm.max_abs
            }),
        };
        if is_better {
            best = Some(s);
        }
    }

    let best = best.unwrap();
    let bm = measure(
        &best.result.numerator,
        &best.result.denominator,
        reference,
        sweep_lo,
        sweep_hi,
        1_000_000,
    );

    match metric {
        F32ScoreMetric::Ulp => {
            if bm.max_ulp < cur.max_ulp {
                println!(
                    "\n  >>> IMPROVED ({}): max {} \u{2192} {}, avg {:.2} \u{2192} {:.2}",
                    best.label.trim(),
                    cur.max_ulp,
                    bm.max_ulp,
                    cur.avg_ulp,
                    bm.avg_ulp
                );
            } else if bm.max_ulp == cur.max_ulp && bm.avg_ulp < cur.avg_ulp {
                println!(
                    "\n  >>> IMPROVED avg ({}): {:.3} \u{2192} {:.3}",
                    best.label.trim(),
                    cur.avg_ulp,
                    bm.avg_ulp
                );
            } else if bm.max_ulp == cur.max_ulp {
                println!("\n  --- MATCHED");
            } else {
                println!("\n  !!! WORSE: max {} \u{2192} {}", cur.max_ulp, bm.max_ulp);
            }
        }
        F32ScoreMetric::AbsoluteError => {
            if bm.max_abs < cur.max_abs {
                println!(
                    "\n  >>> IMPROVED ({}): max_abs {:.3e} \u{2192} {:.3e}",
                    best.label.trim(),
                    cur.max_abs,
                    bm.max_abs
                );
            } else {
                println!("\n  --- no improvement");
            }
        }
    }

    print_const("P", &best.result.numerator);
    print_const("Q", &best.result.denominator);
}

fn print_const(name: &str, coeffs: &[f32]) {
    println!("const {name}: [f32; {}] = [", coeffs.len());
    for (i, &c) in coeffs.iter().enumerate() {
        let comma = if i + 1 < coeffs.len() { "," } else { "" };
        println!("    {c:>20.10e}{comma}");
    }
    println!("];");
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    // log2 mantissa: output crosses zero, use absolute error metric
    report_strategies(
        "log2 mantissa (2/2 rational on m-1, abs-error metric)",
        &[-1.850_383_3e-6, 1.428_716_1, 7.424_587_3e-1],
        &[9.903_281_4e-1, 1.009_671_8, 1.740_934_3e-1],
        |m| (1.0 + m).log2(),
        -0.34,
        0.34,
        run_both_strategies(
            |m| (1.0 + m).log2(),
            -0.34,
            0.34,
            2,
            2,
            |m| (1.0 + m).log2(),
            -0.34,
            0.34,
            &[-1.850_383_3e-6, 1.428_716_1, 7.424_587_3e-1],
            &[9.903_281_4e-1, 1.009_671_8, 1.740_934_3e-1],
            F32ScoreMetric::AbsoluteError,
        ),
        F32ScoreMetric::AbsoluteError,
    );

    // pow2 fractional: 3/3 standard form
    report_strategies(
        "pow2 frac 3/3 (fresh only — libjxl uses non-standard Horner)",
        &[1.0, 1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0],
        |f| 2.0_f64.powf(f),
        0.0,
        1.0,
        {
            let options = RationalFitOptions {
                weighting: ErrorWeighting::Relative,
                n_samples: 10_000,
                restarts: 4,
                ..RationalFitOptions::default()
            };
            let fit =
                RationalFit::from_function(|f| 2.0_f64.powf(f), 0.0..=1.0, 3, 3, options).unwrap();
            let result = fit.as_polynomial().optimize_f32(
                |f| 2.0_f64.powf(f),
                0.0_f32,
                1.0_f32,
                |v| v,
                F32SearchConfig {
                    samples: Some(100_000),
                    ..F32SearchConfig::default()
                },
            );
            vec![Strategy {
                label: "fresh fit      ",
                result,
            }]
        },
        F32ScoreMetric::Ulp,
    );

    // PQ EOTF small: v in [0.02, 0.25), polynomial input x = v + v²
    {
        let lo_v = 0.02_f64;
        let hi_v = 0.25_f64;
        let lo_x = lo_v + lo_v * lo_v;
        let hi_x = hi_v + hi_v * hi_v;
        let pq_on_x = |x: f64| pq_eotf_exact((-1.0 + (1.0 + 4.0 * x).sqrt()) / 2.0);
        let cur_p = [
            4.360_332_03e-10,
            -1.536_510_77e-07,
            2.068_641_96e-06,
            2.803_047_11e-03,
            1.478_099_48e-02,
        ];
        let cur_q = [
            -9.899_550_22e-03,
            1.899_117_67,
            -1.348_280_20,
            -7.105_475_02e-01,
            1.0,
        ];

        report_strategies(
            "PQ EOTF small (4/4 on v+v\u{00b2}, v \u{2208} [0.02, 0.25))",
            &cur_p,
            &cur_q,
            pq_on_x,
            lo_x as f32,
            hi_x as f32,
            run_both_strategies(
                pq_on_x,
                lo_x,
                hi_x,
                4,
                4,
                pq_on_x,
                lo_x as f32,
                hi_x as f32,
                &cur_p,
                &cur_q,
                F32ScoreMetric::Ulp,
            ),
            F32ScoreMetric::Ulp,
        );
    }

    // PQ EOTF large
    {
        let lo_v = 0.25_f64;
        let hi_v = 1.0_f64;
        let lo_x = lo_v + lo_v * lo_v;
        let hi_x = hi_v + hi_v * hi_v;
        let pq_on_x = |x: f64| pq_eotf_exact((-1.0 + (1.0 + 4.0 * x).sqrt()) / 2.0);
        let cur_p = [
            5.430_461_66e-04,
            -6.310_174_31e-03,
            2.904_807_34e-01,
            9.803_630_32e-01,
            2.228_194_83e-01,
        ];
        let cur_q = [
            1.586_885_61e2,
            -1.607_640_60e2,
            6.513_703_72e1,
            -1.264_383_91e1,
            1.0,
        ];

        report_strategies(
            "PQ EOTF large (4/4 on v+v\u{00b2}, v \u{2208} [0.25, 1.0])",
            &cur_p,
            &cur_q,
            pq_on_x,
            lo_x as f32,
            hi_x as f32,
            run_both_strategies(
                pq_on_x,
                lo_x,
                hi_x,
                4,
                4,
                pq_on_x,
                lo_x as f32,
                hi_x as f32,
                &cur_p,
                &cur_q,
                F32ScoreMetric::Ulp,
            ),
            F32ScoreMetric::Ulp,
        );
    }

    // PQ inverse large
    {
        let pq_inv_on_a = |a: f64| pq_inv_exact(a * a * a * a);
        let lo_a = 1e-4_f64.powf(0.25);
        let cur_p = [
            1.351_392e-2,
            -1.095_778,
            5.522_776e1,
            1.492_516e2,
            4.838_434e1,
        ];
        let cur_q = [1.012_416, 2.016_708e1, 9.263_71e1, 1.120_607e2, 2.590_418e1];

        report_strategies(
            "PQ inverse large (4/4 on v\u{00bc}, v \u{2208} [1e-4, 1.0])",
            &cur_p,
            &cur_q,
            pq_inv_on_a,
            lo_a as f32,
            1.0,
            run_both_strategies(
                pq_inv_on_a,
                lo_a,
                1.0,
                4,
                4,
                pq_inv_on_a,
                lo_a as f32,
                1.0,
                &cur_p,
                &cur_q,
                F32ScoreMetric::Ulp,
            ),
            F32ScoreMetric::Ulp,
        );
    }

    // PQ inverse small
    {
        let pq_inv_on_a = |a: f64| pq_inv_exact(a * a * a * a);
        let lo_a = 1e-7_f64.powf(0.25);
        let hi_a = 1e-4_f64.powf(0.25);
        let cur_p = [
            9.863_406e-6,
            3.881_234e-1,
            1.352_821e2,
            6.889_862e4,
            -2.864_824e5,
        ];
        let cur_q = [
            3.371_868e1,
            1.477_719e3,
            1.608_477e4,
            -4.389_884e4,
            -2.072_546e5,
        ];

        report_strategies(
            "PQ inverse small (4/4 on v\u{00bc}, v \u{2208} [1e-7, 1e-4))",
            &cur_p,
            &cur_q,
            pq_inv_on_a,
            lo_a as f32,
            hi_a as f32,
            run_both_strategies(
                pq_inv_on_a,
                lo_a,
                hi_a,
                4,
                4,
                pq_inv_on_a,
                lo_a as f32,
                hi_a as f32,
                &cur_p,
                &cur_q,
                F32ScoreMetric::Ulp,
            ),
            F32ScoreMetric::Ulp,
        );
    }
}
