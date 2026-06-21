//! Jointly optimize S2L and L2S coefficients to minimize roundtrip error.
//!
//! Starts from the best individual-fit coefficients, then performs local f32
//! ULP search with a joint objective: lexicographic (max_ulp, roundtrip_max).
//! The search explores both polynomials simultaneously — a coefficient nudge
//! in S2L may correlate with a nudge in L2S that lowers both the direct ULP
//! AND the roundtrip error.
//!
//! Run: cargo run --release --example joint_roundtrip

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

const A: f64 = 0.055_010_718_947_586_6;
const A1: f64 = 1.0 + A;
const THRESHOLD_LINEAR: f64 = 0.003_041_282_560_127_521;
const THRESHOLD_GAMMA: f64 = 12.92 * THRESHOLD_LINEAR;
const LINEAR_SCALE: f32 = 1.0 / 12.92;
const TWELVE_92: f32 = 12.92;

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

fn f32_nudge(v: f32, delta: i32) -> f32 {
    f32::from_bits((v.to_bits() as i32 + delta) as u32)
}

// Full sRGB fast forward with piecewise threshold
fn srgb_to_linear(v: f32, s2l_p: &[f32], s2l_q: &[f32]) -> f32 {
    if v < 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    if v <= THRESHOLD_GAMMA as f32 {
        return v * LINEAR_SCALE;
    }
    eval_f64mid(v, s2l_p, s2l_q)
}

fn linear_to_srgb(v: f32, l2s_p: &[f32], l2s_q: &[f32]) -> f32 {
    if v < 0.0 {
        return 0.0;
    }
    if v >= 1.0 {
        return 1.0;
    }
    if v <= THRESHOLD_LINEAR as f32 {
        return v * TWELVE_92;
    }
    eval_f64mid(v.sqrt(), l2s_p, l2s_q)
}

struct Score {
    s2l_max_ulp: u32,
    s2l_avg_ulp: f64,
    s2l_bnd: u32,
    l2s_max_ulp: u32,
    l2s_avg_ulp: f64,
    l2s_bnd: u32,
    rt_fwd_max: u32,   // roundtrip s2l then l2s: max output ULP vs input
    rt_fwd_over1: u64, // count of inputs where roundtrip exceeds 1 u16 step
    rt_inv_max: u32,   // roundtrip l2s then s2l
    rt_inv_over1: u64,
    mono_viol: u64,
}

const U16_STEP: f64 = 1.0 / 65535.0;

fn score(s2l_p: &[f32], s2l_q: &[f32], l2s_p: &[f32], l2s_q: &[f32]) -> Score {
    // Sampled: 500K uniformly-spaced points. Fast enough to iterate.
    // Final result should be verified exhaustively.
    const N: usize = 500_000;
    let mut s2l_max = 0u32;
    let mut s2l_sum = 0u64;
    let mut l2s_max = 0u32;
    let mut l2s_sum = 0u64;
    let mut rt_fwd_max = 0u32;
    let mut rt_fwd_over = 0u64;
    let mut rt_inv_max = 0u32;
    let mut rt_inv_over = 0u64;
    let mut count = 0u64;
    let mut mono = 0u64;
    let mut prev_s2l = srgb_to_linear(0.0, s2l_p, s2l_q);
    let mut prev_l2s = linear_to_srgb(0.0, l2s_p, l2s_q);

    for i in 0..=N {
        let v = (i as f64 / N as f64) as f32;
        let s2l_got = srgb_to_linear(v, s2l_p, s2l_q);
        let s2l_ref = ref_s2l(v as f64) as f32;
        let u = ulp_dist(s2l_got, s2l_ref);
        if u > s2l_max {
            s2l_max = u;
        }
        s2l_sum += u as u64;

        let l2s_got = linear_to_srgb(v, l2s_p, l2s_q);
        let l2s_ref = ref_l2s(v as f64) as f32;
        let u = ulp_dist(l2s_got, l2s_ref);
        if u > l2s_max {
            l2s_max = u;
        }
        l2s_sum += u as u64;

        let rt_fwd = linear_to_srgb(s2l_got, l2s_p, l2s_q);
        let rt_fwd_err = (rt_fwd as f64 - v as f64).abs();
        if rt_fwd_err > U16_STEP {
            rt_fwd_over += 1;
        }
        let u = ulp_dist(rt_fwd, v);
        if u > rt_fwd_max {
            rt_fwd_max = u;
        }

        let rt_inv = srgb_to_linear(l2s_got, s2l_p, s2l_q);
        let rt_inv_err = (rt_inv as f64 - v as f64).abs();
        if rt_inv_err > U16_STEP {
            rt_inv_over += 1;
        }
        let u = ulp_dist(rt_inv, v);
        if u > rt_inv_max {
            rt_inv_max = u;
        }

        if count > 0 && s2l_got < prev_s2l {
            mono += 1;
        }
        if count > 0 && l2s_got < prev_l2s {
            mono += 1;
        }
        prev_s2l = s2l_got;
        prev_l2s = l2s_got;
        count += 1;
    }

    let s2l_bnd = ulp_dist(
        eval_f64mid(THRESHOLD_GAMMA as f32, s2l_p, s2l_q),
        ref_s2l(THRESHOLD_GAMMA) as f32,
    );
    let l2s_bnd = ulp_dist(
        eval_f64mid((THRESHOLD_LINEAR as f32).sqrt(), l2s_p, l2s_q),
        ref_l2s(THRESHOLD_LINEAR) as f32,
    );

    Score {
        s2l_max_ulp: s2l_max,
        s2l_avg_ulp: s2l_sum as f64 / count as f64,
        s2l_bnd,
        l2s_max_ulp: l2s_max,
        l2s_avg_ulp: l2s_sum as f64 / count as f64,
        l2s_bnd,
        rt_fwd_max,
        rt_fwd_over1: rt_fwd_over,
        rt_inv_max,
        rt_inv_over1: rt_inv_over,
        mono_viol: mono,
    }
}

fn print_score(label: &str, s: &Score) {
    println!(
        "{label}:\n  S2L max={} avg={:.3} bnd={}\n  L2S max={} avg={:.3} bnd={}\n  RT fwd max={} over1={}, inv max={} over1={}\n  mono={}",
        s.s2l_max_ulp, s.s2l_avg_ulp, s.s2l_bnd,
        s.l2s_max_ulp, s.l2s_avg_ulp, s.l2s_bnd,
        s.rt_fwd_max, s.rt_fwd_over1, s.rt_inv_max, s.rt_inv_over1,
        s.mono_viol,
    );
}

/// Scoring: lexicographic. First satisfy hard constraints, then optimize primary/secondary.
/// - mono = 0 (hard)
/// - S2L bnd <= 2, L2S bnd <= 2 (hard, for piecewise continuity)
/// - rt_over1 = 0 (hard: no roundtrip exceeds 1 u16 step)
/// - minimize max(s2l_max, l2s_max) + rt_max sum
fn score_cmp(a: &Score, b: &Score) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;
    let ok_a = a.mono_viol == 0
        && a.s2l_bnd <= 2
        && a.l2s_bnd <= 2
        && a.rt_fwd_over1 == 0
        && a.rt_inv_over1 == 0;
    let ok_b = b.mono_viol == 0
        && b.s2l_bnd <= 2
        && b.l2s_bnd <= 2
        && b.rt_fwd_over1 == 0
        && b.rt_inv_over1 == 0;
    match (ok_a, ok_b) {
        (true, false) => return Less,
        (false, true) => return Greater,
        _ => {}
    }
    // Primary: max of (S2L max_ulp, L2S max_ulp)
    let max_a = a.s2l_max_ulp.max(a.l2s_max_ulp);
    let max_b = b.s2l_max_ulp.max(b.l2s_max_ulp);
    match max_a.cmp(&max_b) {
        Equal => {}
        o => return o,
    }
    // Secondary: sum of roundtrip maxes
    let rt_a = a.rt_fwd_max + a.rt_inv_max;
    let rt_b = b.rt_fwd_max + b.rt_inv_max;
    match rt_a.cmp(&rt_b) {
        Equal => {}
        o => return o,
    }
    // Tertiary: sum of avg ULPs
    a.s2l_avg_ulp.partial_cmp(&b.s2l_avg_ulp).unwrap_or(Equal)
}

fn joint_search(
    s2l_p: &mut Vec<f32>,
    s2l_q: &mut Vec<f32>,
    l2s_p: &mut Vec<f32>,
    l2s_q: &mut Vec<f32>,
) {
    // Each polynomial has p.len() + q.len() - 1 free coeffs (last q is 1.0)
    let s2l_n = s2l_p.len() + s2l_q.len() - 1;
    let l2s_n = l2s_p.len() + l2s_q.len() - 1;
    let total = s2l_n + l2s_n;

    let mut best = score(s2l_p, s2l_q, l2s_p, l2s_q);
    println!("\nInitial score:");
    print_score("  init", &best);

    for round in 1..=5 {
        let mut improved = false;
        for ci in 0..total {
            let (is_s2l, local_ci) = if ci < s2l_n {
                (true, ci)
            } else {
                (false, ci - s2l_n)
            };
            let (vec_p, vec_q) = if is_s2l {
                (&mut *s2l_p, &mut *s2l_q)
            } else {
                (&mut *l2s_p, &mut *l2s_q)
            };
            let is_p = local_ci < vec_p.len();
            let idx = if is_p {
                local_ci
            } else {
                local_ci - vec_p.len()
            };
            let original = if is_p { vec_p[idx] } else { vec_q[idx] };

            let mut best_delta = 0i32;
            for delta in (-3..=3).filter(|&d| d != 0) {
                let nudged = f32_nudge(original, delta);
                if is_s2l {
                    if is_p {
                        s2l_p[idx] = nudged;
                    } else {
                        s2l_q[idx] = nudged;
                    }
                } else {
                    if is_p {
                        l2s_p[idx] = nudged;
                    } else {
                        l2s_q[idx] = nudged;
                    }
                }
                let s = score(s2l_p, s2l_q, l2s_p, l2s_q);
                if score_cmp(&s, &best) == std::cmp::Ordering::Less {
                    best = s;
                    best_delta = delta;
                }
                // Revert
                if is_s2l {
                    if is_p {
                        s2l_p[idx] = original;
                    } else {
                        s2l_q[idx] = original;
                    }
                } else {
                    if is_p {
                        l2s_p[idx] = original;
                    } else {
                        l2s_q[idx] = original;
                    }
                }
            }

            if best_delta != 0 {
                let nudged = f32_nudge(original, best_delta);
                if is_s2l {
                    if is_p {
                        s2l_p[idx] = nudged;
                    } else {
                        s2l_q[idx] = nudged;
                    }
                } else {
                    if is_p {
                        l2s_p[idx] = nudged;
                    } else {
                        l2s_q[idx] = nudged;
                    }
                }
                let which = if is_s2l { "S2L" } else { "L2S" };
                let kind = if is_p { "P" } else { "Q" };
                println!("  round {round} {which}.{kind}[{idx}] {best_delta:+}: S2L={} L2S={} rt_fwd={} rt_inv={}",
                    best.s2l_max_ulp, best.l2s_max_ulp, best.rt_fwd_max, best.rt_inv_max);
                improved = true;
            }
        }
        if !improved {
            println!("  round {round}: converged");
            break;
        }
    }

    println!("\nFinal score:");
    print_score("  final", &best);
}

fn print_const(name: &str, coeffs: &[f32]) {
    println!("pub(crate) const {name}: [f32; {}] = [", coeffs.len());
    for (i, &c) in coeffs.iter().enumerate() {
        let comma = if i + 1 < coeffs.len() { "," } else { "" };
        println!("    {c:.10e}{comma}");
    }
    println!("];");
}

fn main() {
    // Baseline: main branch (before PR #8)
    let main_s2l_p: Vec<f32> = vec![
        1.724_942_4e-2,
        8.335_514_7e-1,
        1.326_215_8e1,
        7.033_073_4e1,
        8.387_046e1,
    ];
    let main_s2l_q: Vec<f32> = vec![2.066_183e1, 9.917_607e1, 5.466_011e1, -7.183_806, 1.0];
    let main_l2s_p: Vec<f32> = vec![
        -1.513_885e-2,
        1.167_372_8e-1,
        1.257_921_2e1,
        5.259_309_8e1,
        2.852_907_6e1,
    ];
    let main_l2s_q: Vec<f32> = vec![2.943_901_4e-1, 9.779_103, 4.726_487_7e1, 3.546_463_8e1, 1.0];

    println!("=== MAIN BRANCH (current release) ===");
    let main_score = score(&main_s2l_p, &main_s2l_q, &main_l2s_p, &main_l2s_q);
    print_score("  main", &main_score);

    // PR branch coefficients (already polyfit-optimized individually)
    let mut s2l_p = vec![
        1.672_814_6e-2,
        8.089_536_4e-1,
        1.288_439_66e1,
        6.854_374_7e1,
        8.224_625_4e1,
    ];
    let mut s2l_q = vec![
        2.003_807_83e1,
        9.681_468_2e1,
        5.377_303_3e1,
        -7.125_638,
        1.0,
    ];
    let mut l2s_p = vec![
        -1.356_440_97e-2,
        9.336_896_24e-2,
        1.157_270_53e1,
        5.002_124_79e1,
        2.792_420_96e1,
    ];
    let mut l2s_q = vec![
        2.633_812_73e-1,
        8.998_917_58,
        4.477_739_72e1,
        3.455_830_76e1,
        1.0,
    ];

    println!("\n=== Joint S2L/L2S roundtrip optimization ===");
    println!("Starting from individually-optimized polyfit coefficients (PR #8)");

    joint_search(&mut s2l_p, &mut s2l_q, &mut l2s_p, &mut l2s_q);

    println!("\n{}", "=".repeat(60));
    println!("  Final coefficients");
    println!("{}\n", "=".repeat(60));
    print_const("S2L_P", &s2l_p);
    print_const("S2L_Q", &s2l_q);
    print_const("L2S_P", &l2s_p);
    print_const("L2S_Q", &l2s_q);
}
