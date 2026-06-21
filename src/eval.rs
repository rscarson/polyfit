//! Public Horner-form evaluation helpers for fitted polynomials.
//!
//! Every caller that ships a polyfit-derived polynomial into a runtime path
//! ends up writing the same Horner-form evaluator. This module exposes the
//! canonical implementations so callers can drop them in without
//! re-implementing the loop (and the off-by-one footguns that come with it).
//!
//! Coefficients are always lowest-degree-first (Horner order):
//! ```text
//! P(x) = c[0] + c[1]*x + c[2]*x² + ... + c[n]*x^n
//! ```
//! This matches the convention used by [`MonomialPolynomial`](crate::MonomialPolynomial),
//! [`rational::RationalPolynomial`](crate::rational::RationalPolynomial), and
//! every other coefficient-returning API in this crate.
//!
//! # Choosing an evaluator
//!
//! - [`horner_f32`] — pure-f32 evaluator. Use when the caller's hot loop is
//!   f32 end-to-end (e.g., SIMD-vectorized contexts that splat coefficients
//!   into f32 vector lanes).
//! - [`horner_f32_via_f64`] — f32 coefficients with f64 accumulator. Use when
//!   the f32 coefficients came from a polyfit f32-search routine (e.g.,
//!   [`rational::RationalPolynomial::optimize_f32`](crate::rational::RationalPolynomial::optimize_f32))
//!   so the evaluator matches the scorer the f32 search used.
//! - [`horner_f64`] — full f64 evaluator. Use for f64 callers, validation
//!   passes, or any context where the polynomial is evaluated at high
//!   precision.
//! - [`rational_horner_f32_via_f64`] — `P(x)/Q(x)` for f32 coefficients,
//!   evaluated through an f64 accumulator. Matches
//!   [`rational::RationalPolynomial::optimize_f32`](crate::rational::RationalPolynomial::optimize_f32)
//!   so the caller can ship the same evaluation path the f32 search scored
//!   against.
//!
//! # Empty-slice behavior
//!
//! All helpers return `0.0` when the coefficient slice is empty. They never
//! panic on coefficient input. (The `rational_*` helpers will return
//! `NaN`/`Inf` if the denominator evaluates to zero — same as any explicit
//! division.)
//!
//! # Example
//!
//! ```
//! use polyfit::eval::{horner_f32, horner_f64};
//!
//! // Evaluate 1 + 2x + 3x² at x = 2.0 → 1 + 4 + 12 = 17
//! let coeffs = [1.0_f32, 2.0, 3.0];
//! assert!((horner_f32(2.0, &coeffs) - 17.0).abs() < 1e-6);
//!
//! let coeffs64 = [1.0_f64, 2.0, 3.0];
//! assert!((horner_f64(2.0, &coeffs64) - 17.0).abs() < 1e-12);
//! ```

/// Evaluate a polynomial at `x` using Horner's method with f64 precision.
///
/// Coefficients are lowest-degree-first: `c[0] + c[1]*x + c[2]*x² + ...`.
/// Returns `0.0` if `coefficients` is empty.
///
/// Uses [`f64::mul_add`] for the accumulator step so the implementation
/// matches the standard fused-multiply-add Horner pattern (one rounding
/// per inner step on platforms with hardware FMA).
#[must_use]
#[inline]
pub fn horner_f64(x: f64, coefficients: &[f64]) -> f64 {
    let Some((&last, rest)) = coefficients.split_last() else {
        return 0.0;
    };
    let mut acc = last;
    for &c in rest.iter().rev() {
        acc = acc.mul_add(x, c);
    }
    acc
}

/// Evaluate a polynomial at `x` using Horner's method with pure f32 arithmetic.
///
/// Coefficients are lowest-degree-first: `c[0] + c[1]*x + c[2]*x² + ...`.
/// Returns `0.0` if `coefficients` is empty.
///
/// Uses [`f32::mul_add`] for the accumulator step. Use this evaluator when
/// the caller's hot loop is f32 end-to-end (e.g., a SIMD-vectorized
/// per-pixel kernel that splats each coefficient into vector lanes).
///
/// If the coefficients came from a polyfit f32-search routine that scored
/// against an f64-accumulator evaluator
/// (see [`horner_f32_via_f64`] / [`rational_horner_f32_via_f64`]), prefer
/// the matching evaluator — otherwise the runtime path's rounding will
/// differ from what the search scored against.
#[must_use]
#[inline]
pub fn horner_f32(x: f32, coefficients: &[f32]) -> f32 {
    let Some((&last, rest)) = coefficients.split_last() else {
        return 0.0;
    };
    let mut acc = last;
    for &c in rest.iter().rev() {
        acc = acc.mul_add(x, c);
    }
    acc
}

/// Evaluate a polynomial with f32 coefficients at `x` using an f64
/// accumulator, then truncate to f32.
///
/// Coefficients are lowest-degree-first: `c[0] + c[1]*x + c[2]*x² + ...`.
/// Returns `0.0` if `coefficients` is empty.
///
/// This matches the evaluation path used internally by
/// [`rational::RationalPolynomial::optimize_f32`](crate::rational::RationalPolynomial::optimize_f32)
/// when scoring f32 coefficients during the local search. Use it for the
/// runtime evaluation when the search produced the coefficients, so the
/// production path's rounding behavior matches the rounding the f32 search
/// scored against.
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn horner_f32_via_f64(x: f32, coefficients: &[f32]) -> f32 {
    let x = f64::from(x);
    let Some((&last, rest)) = coefficients.split_last() else {
        return 0.0;
    };
    let mut acc = f64::from(last);
    for &c in rest.iter().rev() {
        acc = acc.mul_add(x, f64::from(c));
    }
    acc as f32
}

/// Evaluate a rational polynomial `P(x)/Q(x)` with f32 coefficients via
/// an f64 accumulator, then truncate to f32.
///
/// Both `numerator` and `denominator` are lowest-degree-first. Returns
/// `0.0` if both slices are empty, `NaN`/`Inf` if the denominator
/// evaluates to zero.
///
/// This matches the evaluation path used internally by
/// [`rational::RationalPolynomial::optimize_f32`](crate::rational::RationalPolynomial::optimize_f32):
/// the f32 search scores candidates using this exact rounding pattern.
/// Ship this evaluator in production paths that consume f32-search-derived
/// rational coefficients, so the production path's f32 output matches the
/// search's scored output bit-for-bit.
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn rational_horner_f32_via_f64(x: f32, numerator: &[f32], denominator: &[f32]) -> f32 {
    let x = f64::from(x);
    let p = match numerator.split_last() {
        Some((&last, rest)) => {
            let mut acc = f64::from(last);
            for &c in rest.iter().rev() {
                acc = acc.mul_add(x, f64::from(c));
            }
            acc
        }
        None => 0.0,
    };
    let q = {
        let Some((&last, rest)) = denominator.split_last() else {
            return 0.0;
        };
        let mut acc = f64::from(last);
        for &c in rest.iter().rev() {
            acc = acc.mul_add(x, f64::from(c));
        }
        acc
    };
    (p / q) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horner_f64_matches_naive() {
        // 1 + 2x + 3x² at x = 2.0 → 1 + 4 + 12 = 17.0
        let coeffs = [1.0, 2.0, 3.0];
        assert!((horner_f64(2.0, &coeffs) - 17.0).abs() < 1e-12);
        // 5 at x = anything → 5
        assert!((horner_f64(1.5, &[5.0]) - 5.0).abs() < 1e-12);
        // empty → 0
        assert_eq!(horner_f64(2.0, &[]).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn horner_f32_matches_naive() {
        let coeffs = [1.0_f32, 2.0, 3.0];
        assert!((horner_f32(2.0, &coeffs) - 17.0).abs() < 1e-6);
        assert!((horner_f32(1.5, &[5.0]) - 5.0).abs() < 1e-6);
        assert_eq!(horner_f32(2.0, &[]).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn horner_f32_via_f64_matches_naive() {
        // For small benign inputs, results match
        let coeffs = [1.0_f32, 2.0, 3.0];
        assert!((horner_f32_via_f64(2.0, &coeffs) - 17.0).abs() < 1e-6);
        assert!((horner_f32_via_f64(1.5, &[5.0]) - 5.0).abs() < 1e-6);
        assert_eq!(horner_f32_via_f64(2.0, &[]).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn rational_horner_f32_via_f64_matches_naive() {
        // P(x) = 1 + 2x, Q(x) = 1 (constant) → identical to P
        let p = [1.0_f32, 2.0];
        let q = [1.0_f32];
        assert!((rational_horner_f32_via_f64(3.0, &p, &q) - 7.0).abs() < 1e-6);

        // P(x) = x, Q(x) = 1 + x → x / (1 + x)
        let p = [0.0_f32, 1.0];
        let q = [1.0_f32, 1.0];
        let got = rational_horner_f32_via_f64(2.0, &p, &q);
        assert!((got - (2.0 / 3.0)).abs() < 1e-6);

        // Empty denominator returns 0.0 (defined behavior, not panic)
        assert_eq!(
            rational_horner_f32_via_f64(1.0, &[1.0, 2.0], &[]).to_bits(),
            0.0_f32.to_bits()
        );

        // Zero denominator → NaN or Inf (defined behavior, not panic)
        let zero_q = rational_horner_f32_via_f64(0.0, &[1.0_f32], &[0.0_f32]);
        assert!(zero_q.is_nan() || zero_q.is_infinite());
    }

    /// `rational_horner_f32_via_f64` MUST match the internal `eval_f32_horner`
    /// used by `optimize_f32`'s scorer, so callers can ship the public helper
    /// in production paths and stay bit-identical to what the search scored.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn rational_horner_matches_internal_optimizer_path() {
        use crate::rational::{RationalFit, RationalFitOptions};

        let fit = RationalFit::from_function(
            |x: f64| x.powf(2.4),
            0.04..=1.0,
            4,
            4,
            RationalFitOptions::default(),
        )
        .unwrap();

        let result = fit.as_polynomial().optimize_f32(
            |x: f64| x.powf(2.4),
            0.04_f32,
            1.0_f32,
            |v| v,
            crate::rational::F32SearchConfig {
                samples: Some(1000),
                ..crate::rational::F32SearchConfig::default()
            },
        );

        // Sample a handful of points and verify the public helper produces
        // the same bits as the internal scorer's evaluator at every one.
        let lo = f64::from(0.04_f32);
        let hi = f64::from(1.0_f32);
        for i in 0..=200 {
            let x = lo + (hi - lo) * f64::from(i) / 200.0;
            let x32 = x as f32;
            let public = rational_horner_f32_via_f64(x32, &result.numerator, &result.denominator);

            // Re-implement the internal helper inline for the parity check
            let internal = {
                let xd = f64::from(x32);
                let mut yp = f64::from(*result.numerator.last().unwrap());
                for &c in result.numerator[..result.numerator.len() - 1].iter().rev() {
                    yp = yp.mul_add(xd, f64::from(c));
                }
                let mut yq = f64::from(*result.denominator.last().unwrap());
                for &c in result.denominator[..result.denominator.len() - 1]
                    .iter()
                    .rev()
                {
                    yq = yq.mul_add(xd, f64::from(c));
                }
                (yp / yq) as f32
            };

            assert_eq!(
                public.to_bits(),
                internal.to_bits(),
                "public eval differs from internal scorer at x={x32}: public={public} internal={internal}",
            );
        }
    }
}
