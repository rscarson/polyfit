//! Generate a specific set of optimal linear-srgb coefficients with
//! manually-chosen tradeoffs, and validate with exhaustive f32 sweeps.

use polyfit::rational::{
    Constraint, ConstraintMode, ErrorWeighting, F32SearchConfig, RationalFit, RationalFitOptions,
    RationalPolynomial,
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

fn print_const(name: &str, coeffs: &[f32], doc: &str) {
    if !doc.is_empty() {
        println!("/// {doc}");
    }
    println!("pub(crate) const {name}: [f32; {}] = [", coeffs.len());
    for (i, &c) in coeffs.iter().enumerate() {
        let comma = if i + 1 < coeffs.len() { "," } else { "" };
        println!("    {c:.10e}{comma}");
    }
    println!("];");
}

fn main() {
    let sampled_cfg = F32SearchConfig {
        samples: Some(500_000),
        ..F32SearchConfig::default()
    };
    let _ = sampled_cfg.clone();

    // ==================================================================
    // S2L 4/4: try several weights, pick best max_ulp with bnd ≤ 1
    // ==================================================================
    println!("--- S2L 4/4 ---");
    let mut best_s2l: Option<polyfit::rational::F32SearchResult> = None;
    for &weight in &[50_000.0, 100_000.0, 300_000.0, 1_000_000.0] {
        let s2l_opt = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            n_samples: 20_000,
            restarts: 8,
            constraints: vec![Constraint::with_weight(
                THRESHOLD_GAMMA,
                THRESHOLD_LINEAR,
                weight,
            )],
            constraint_mode: ConstraintMode::Both,
            ..RationalFitOptions::default()
        };
        let s2l_fit = RationalFit::from_function(
            |v| ((v + A) / A1).powf(2.4),
            THRESHOLD_GAMMA..=1.0,
            4,
            4,
            s2l_opt,
        )
        .unwrap();
        let s2l = s2l_fit.as_polynomial().optimize_f32(
            ref_s2l,
            THRESHOLD_GAMMA as f32,
            1.0,
            |v| v,
            sampled_cfg.clone(),
        );
        println!("S2L w={weight:>8.0}: {s2l}");
        if s2l.boundary_ulp <= 1 && s2l.mono_violations == 0 {
            best_s2l = Some(match best_s2l {
                None => s2l,
                Some(best) => {
                    if s2l.max_ulp < best.max_ulp {
                        s2l
                    } else {
                        best
                    }
                }
            });
        }
    }
    let s2l = best_s2l.expect("no S2L candidate with bnd <= 1");
    println!("Selected S2L: {s2l}");

    // ==================================================================
    // L2S 4/4: weighted constraint w=1000 for bnd=0
    // ==================================================================
    println!("\n--- L2S 4/4 ---");
    let l2s_opt = RationalFitOptions {
        weighting: ErrorWeighting::Relative,
        n_samples: 20_000,
        restarts: 8,
        constraints: vec![Constraint::with_weight(
            THRESHOLD_LINEAR.sqrt(),
            THRESHOLD_GAMMA,
            1000.0,
        )],
        constraint_mode: ConstraintMode::Both,
        ..RationalFitOptions::default()
    };
    let l2s_fit = RationalFit::from_function(
        |s| A1 * (s * s).powf(1.0 / 2.4) - A,
        THRESHOLD_LINEAR.sqrt()..=1.0,
        4,
        4,
        l2s_opt,
    )
    .unwrap();
    let l2s = l2s_fit.as_polynomial().optimize_f32(
        ref_l2s,
        THRESHOLD_LINEAR as f32,
        1.0,
        |v| v.sqrt(),
        sampled_cfg.clone(),
    );
    println!("L2S 4/4 (w=1000 + exhaustive search): {l2s}");

    // ==================================================================
    // EXT_L2S 6/6: warm-start from current, then f32 search
    // ==================================================================
    println!("\n--- EXT_L2S 6/6 ---");
    let current_ext_l2s_p: [f32; 7] = [
        -1.025_467_4,
        -3.075_361_5e-1,
        1.027_286e3,
        7.093_665e3,
        1.006_868_9e4,
        3.230_716e3,
        1.769_130_4e2,
    ];
    let current_ext_l2s_q: [f32; 7] = [
        1.977_460_5e1,
        8.308_271e2,
        6.024_792_5e3,
        1.024_407_5e4,
        4.157_534e3,
        3.179_324_6e2,
        1.0,
    ];
    let l2s_ext_data = chebyshev_data(
        |s| A1 * (s * s).powf(1.0 / 2.4) - A,
        THRESHOLD_LINEAR.sqrt(),
        8.0,
        10_000,
    );
    let ext_l2s = RationalPolynomial::<f64>::refine_from_f32(
        &current_ext_l2s_p,
        &current_ext_l2s_q,
        ref_l2s,
        &l2s_ext_data,
        THRESHOLD_LINEAR as f32,
        1.0,
        |v| v.sqrt(),
        sampled_cfg.clone(),
        ErrorWeighting::Relative,
        300,
    );
    println!("EXT_L2S 6/6 (warm-start + exhaustive): {ext_l2s}");

    // ==================================================================
    // EXT_S2L 6/6: keep current (sampled found max=5 bnd=2, current exhaustive is max=6 bnd=0)
    // But let's also try a fresh warm-start
    // ==================================================================
    println!("\n--- EXT_S2L 6/6 ---");
    let current_ext_s2l_p: [f32; 7] = [
        1.802_136_5e1,
        9.110_411_4e2,
        1.570_602_1e4,
        1.020_638_2e5,
        2.199_931_2e5,
        1.338_269_2e5,
        1.706_519_4e4,
    ];
    let current_ext_s2l_q: [f32; 7] = [
        2.159_401_7e4,
        1.508_555_1e5,
        2.303_299_0e5,
        8.239_410_8e4,
        4.473_249_1e3,
        -6.359_000_1e1,
        1.0,
    ];
    let s2l_ext_data = chebyshev_data(|v| ((v + A) / A1).powf(2.4), THRESHOLD_GAMMA, 8.0, 10_000);
    // Warm-start preserves boundary better than fresh fit for 6/6
    let ext_s2l = RationalPolynomial::<f64>::refine_from_f32(
        &current_ext_s2l_p,
        &current_ext_s2l_q,
        ref_s2l,
        &s2l_ext_data,
        THRESHOLD_GAMMA as f32,
        1.0,
        |v| v,
        sampled_cfg,
        ErrorWeighting::Relative,
        300,
    );
    println!("EXT_S2L 6/6 (warm-start + sampled): {ext_s2l}");

    // Final coefficients
    println!("\n{}", "=".repeat(70));
    println!("  APPLY TO linear-srgb/src/rational_poly.rs");
    println!("{}\n", "=".repeat(70));

    print_const(
        "S2L_P",
        &s2l.numerator,
        &format!(
            "sRGB EOTF numerator. Max {} ULP, boundary {} ULP.",
            s2l.max_ulp, s2l.boundary_ulp
        ),
    );
    print_const("S2L_Q", &s2l.denominator, "sRGB EOTF denominator.");
    println!();
    print_const(
        "L2S_P",
        &l2s.numerator,
        &format!(
            "sRGB inv EOTF numerator on √x. Max {} ULP, boundary {} ULP.",
            l2s.max_ulp, l2s.boundary_ulp
        ),
    );
    print_const(
        "L2S_Q",
        &l2s.denominator,
        "sRGB inv EOTF denominator on √x.",
    );
    println!();
    print_const(
        "EXT_S2L_P",
        &ext_s2l.numerator,
        &format!(
            "Extended sRGB EOTF numerator [0, 8]. Max {} ULP in [0,1], boundary {} ULP.",
            ext_s2l.max_ulp, ext_s2l.boundary_ulp
        ),
    );
    print_const(
        "EXT_S2L_Q",
        &ext_s2l.denominator,
        "Extended sRGB EOTF denominator [0, 8].",
    );
    println!();
    print_const(
        "EXT_L2S_P",
        &ext_l2s.numerator,
        &format!(
            "Extended sRGB inv EOTF numerator on √x [0, 64]. Max {} ULP in [0,1], boundary {} ULP.",
            ext_l2s.max_ulp, ext_l2s.boundary_ulp
        ),
    );
    print_const(
        "EXT_L2S_Q",
        &ext_l2s.denominator,
        "Extended sRGB inv EOTF denominator on √x [0, 64].",
    );
}
