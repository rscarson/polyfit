//! Rational polynomial fitting: P(x)/Q(x) approximation.
//!
//! Fits a rational polynomial P(x)/Q(x) to data or a known function using
//! iteratively reweighted least squares (Sanathanan-Koerner iteration).
//!
//! Unlike standard polynomial fitting (which is a linear problem), rational
//! fitting is inherently nonlinear because the denominator coefficients appear
//! in the divisor. The Sanathanan-Koerner algorithm linearizes this by
//! iteratively reweighting residuals, converging to the true least-squares
//! solution.
//!
//! # Use cases
//!
//! Rational polynomials are particularly good at approximating:
//! - Transfer functions (sRGB, PQ, HLG)
//! - Transcendental functions (log₂, exp₂, powf)
//! - Functions with steep gradients or asymptotic behavior
//! - Any function where a standard polynomial needs high degree but P/Q needs low degree
//!
//! # Coefficient convention
//!
//! Coefficients are stored **lowest-degree-first** (Horner order):
//! ```text
//! P(x) = p[0] + p[1]*x + p[2]*x² + ... + p[m]*x^m
//! Q(x) = q[0] + q[1]*x + q[2]*x² + ... + x^n     (monic: leading coeff = 1)
//! ```
//!
//! # Example
//!
//! ```rust
//! use polyfit::rational::{RationalFit, RationalFitOptions};
//!
//! // Approximate x^2.4 on [0.04, 1.0] with a 4/4 rational polynomial
//! let fit = RationalFit::from_function(
//!     |x: f64| x.powf(2.4),
//!     0.04..=1.0,
//!     4, 4,
//!     RationalFitOptions::default(),
//! ).unwrap();
//!
//! // Evaluate
//! let y = fit.y(0.5);
//! assert!((y - 0.5_f64.powf(2.4)).abs() < 1e-6);
//!
//! // Get coefficients for use in Horner evaluation
//! let (p, q) = fit.coefficients();
//! println!("P: {:?}", p);
//! println!("Q: {:?}", q);
//! ```
//!
//! # Determinism
//!
//! The full pipeline is deterministic: identical input produces identical output.
//! Sample nodes are Chebyshev-spaced (no RNG), SK/LM iterations are fully
//! numerical, multi-restart perturbations use `sin`/`cos` of fixed-constant
//! phases (no RNG), and f32 local search is a deterministic coordinate descent.
//! No `rand` dependency, no hidden seeds. Coefficient changes must come from
//! changing the *inputs* (degree, domain, restart count, weight), never the
//! wall clock.
//!
//! # Measurement honesty
//!
//! [`F32SearchResult`] records `exhaustive: bool` and `values_scored: u64` so
//! callers can tell whether reported metrics come from a full f32 sweep or
//! from uniform sampling. [`piecewise_gap`] measures the ULP gap between a
//! linear segment and the rational polynomial at their join — different from
//! the reference-vs-polynomial ULP at the boundary. [`measure_roundtrip`]
//! measures `f⁻¹(f(v))` error including `u8` and `u16` step counts for
//! regression gating.

use std::{borrow::Cow, fmt, ops::RangeInclusive};

use nalgebra::{DMatrix, DVector, SVD};

use crate::{
    error::{Error, Result},
    value::Value,
};

// =============================================================================
// Types
// =============================================================================

/// A rational polynomial P(x)/Q(x).
///
/// Coefficients are lowest-degree-first. The denominator is monic (leading
/// coefficient = 1).
#[derive(Debug, Clone, PartialEq)]
pub struct RationalPolynomial<T: Value = f64> {
    /// Numerator coefficients: p[0] + p[1]*x + ... + p[m]*x^m
    numerator: Vec<T>,
    /// Denominator coefficients: q[0] + q[1]*x + ... + x^n (last element is 1)
    denominator: Vec<T>,
}

/// How to weight fitting errors.
#[derive(Debug, Clone, Copy, Default)]
pub enum ErrorWeighting {
    /// Minimize sum of |P(x)/Q(x) - y|² (absolute error).
    #[default]
    Absolute,
    /// Minimize sum of |P(x)/Q(x) - y|² / |y|² (relative error).
    ///
    /// Better for transfer functions where accuracy matters proportionally.
    Relative,
}

/// How constraints are applied across the SK and LM fitting phases.
#[derive(Debug, Clone, Copy, Default)]
pub enum ConstraintMode {
    /// Fit entirely unconstrained, then adjust `p[0]` to satisfy each
    /// constraint exactly via [`RationalPolynomial::snap_boundary`].
    ///
    /// Best for piecewise functions (sRGB, PQ) where the join point matters.
    /// Gives the best overall accuracy because neither SK nor LM is distorted
    /// by constraint weights. The snap only perturbs `p[0]` (a DC offset),
    /// which has minimal impact on accuracy — especially when the constraint
    /// x-value is near zero.
    #[default]
    SnapOnly,

    /// Apply constraints during SK initialization only. LM refines freely,
    /// then `snap_boundary` adjusts `p[0]` to satisfy each constraint exactly.
    SkThenSnap,

    /// Apply constraints during both SK and LM as weighted residuals.
    /// Use moderate weights (100-10000) to avoid LM instability.
    Both,

    /// Apply constraints during SK only, no snap. LM refines freely.
    /// Boundary accuracy depends on how well SK placed the initial guess.
    SkOnly,
}

/// A point constraint: the rational polynomial must pass through (x, y).
///
/// # All `Constraint` weights are *soft* penalties, not hard pins
///
/// Despite the name and the default weight of `1e6`, every `Constraint`
/// participates in the fit's least-squares objective as a
/// `weight * (P(x_i)/Q(x_i) - y_i)²` term — a *penalty*, not a *pin*.
/// The Sanathanan-Koerner iteration and the Levenberg-Marquardt
/// refinement will trade a small constraint violation for a small
/// bulk-domain accuracy gain whenever the overall objective improves.
/// Even with `weight = 1e6` the fitted polynomial can drift from the
/// constraint point — for `f(0) = 0` in particular, expect drift on the
/// order of the bulk-domain absolute error.
///
/// If you need an *exact* hit at the constraint point (e.g., to make a
/// piecewise function continuous at its threshold), use
/// [`ConstraintMode::SnapOnly`] (the default) or
/// [`ConstraintMode::SkThenSnap`]: after fitting, the constant term
/// `p[0]` is adjusted so `P(x)/Q(x) = y` exactly at the constraint
/// point. See [`RationalPolynomial::snap_boundary`] for the math.
///
/// For the special case `f(0) = 0` with the goal of saving a constant
/// load in a SIMD evaluator: a "substitution trick" is more reliable
/// than any constraint — fit `f(x)/x` (or `f(√x)/√x` etc.) and
/// multiply by `x` at evaluation time, which makes `c[0] = 0` exact by
/// construction.
#[derive(Debug, Clone, Copy)]
pub struct Constraint<T: Value = f64> {
    /// The x coordinate of the constraint.
    pub x: T,
    /// The y value the rational polynomial must match.
    pub y: T,
    /// How strongly to weight this constraint in the least-squares
    /// objective (default: `1e6`).
    ///
    /// **This is a soft penalty multiplier, not a hard equality** —
    /// arbitrarily large weights still let the optimizer trade a tiny
    /// constraint residual for a bulk-domain accuracy gain. See the
    /// [`Constraint`] type-level docs for the workaround.
    pub weight: T,
}

/// Configuration for rational polynomial fitting.
#[derive(Debug, Clone)]
pub struct RationalFitOptions<T: Value = f64> {
    /// Error weighting strategy.
    pub weighting: ErrorWeighting,
    /// Maximum Sanathanan-Koerner iterations (default: 50).
    pub max_iterations: usize,
    /// Convergence tolerance on coefficient change (default: 1e-12).
    pub tolerance: T,
    /// Point constraints the fit must satisfy.
    pub constraints: Vec<Constraint<T>>,
    /// Number of sample points for `from_function` (default: 2000).
    /// Uses Chebyshev-spaced nodes for stability.
    pub n_samples: usize,
    /// Maximum Levenberg-Marquardt refinement iterations after SK converges
    /// (default: 200). Set to 0 to disable LM refinement and use SK only.
    pub lm_iterations: usize,
    /// How constraints are applied across the fitting phases (default: `SnapOnly`).
    pub constraint_mode: ConstraintMode,
    /// Number of restarts with perturbed initial conditions (default: 1, no restarts).
    /// Each restart perturbs the SK result by ~5% deterministic noise, re-runs LM,
    /// and keeps the best result. Higher values explore more basins at linear cost.
    pub restarts: usize,
}

/// Error at a single evaluation point (at or past a domain boundary).
#[derive(Debug, Clone, Copy)]
pub struct BoundaryPoint<T: Value = f64> {
    /// The x coordinate.
    pub x: T,
    /// The rational polynomial's output at x.
    pub predicted: T,
    /// The reference function's output at x.
    pub expected: T,
    /// Absolute error: |predicted - expected|.
    pub abs_err: T,
    /// Relative error: |predicted - expected| / |expected|.
    pub rel_err: T,
}

/// Error report at and near the domain boundaries.
///
/// Contains error at the endpoints (`lo`, `hi`) and at points extrapolated
/// past the boundary at 0.1%, 1%, 5%, and 10% of the domain width.
#[derive(Debug, Clone)]
pub struct BoundaryReport<T: Value = f64> {
    /// Error at the low endpoint (domain start).
    pub lo: BoundaryPoint<T>,
    /// Error at the high endpoint (domain end).
    pub hi: BoundaryPoint<T>,
    /// Error at points past the low boundary (0.1%, 1%, 5%, 10% of width).
    pub past_lo: Vec<BoundaryPoint<T>>,
    /// Error at points past the high boundary (0.1%, 1%, 5%, 10% of width).
    pub past_hi: Vec<BoundaryPoint<T>>,
}

impl<T: Value> fmt::Display for BoundaryReport<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Boundary error:")?;
        writeln!(
            f,
            "  lo (x={:.6e}): abs={:.3e} rel={:.3e}",
            self.lo.x, self.lo.abs_err, self.lo.rel_err
        )?;
        writeln!(
            f,
            "  hi (x={:.6e}): abs={:.3e} rel={:.3e}",
            self.hi.x, self.hi.abs_err, self.hi.rel_err
        )?;

        if !self.past_lo.is_empty() {
            writeln!(f, "  Extrapolation past lo:")?;
            for (i, p) in self.past_lo.iter().enumerate() {
                let pct = [0.1, 1.0, 5.0, 10.0];
                let pct_label = pct.get(i).copied().unwrap_or(0.0);
                writeln!(
                    f,
                    "    -{pct_label:>5.1}% (x={:.6e}): abs={:.3e} rel={:.3e}",
                    p.x, p.abs_err, p.rel_err
                )?;
            }
        }
        if !self.past_hi.is_empty() {
            writeln!(f, "  Extrapolation past hi:")?;
            for (i, p) in self.past_hi.iter().enumerate() {
                let pct = [0.1, 1.0, 5.0, 10.0];
                let pct_label = pct.get(i).copied().unwrap_or(0.0);
                writeln!(
                    f,
                    "    +{pct_label:>5.1}% (x={:.6e}): abs={:.3e} rel={:.3e}",
                    p.x, p.abs_err, p.rel_err
                )?;
            }
        }
        Ok(())
    }
}

/// How the f32 search scores candidate coefficients.
///
/// # When NOT to trust [`F32ScoreMetric::Ulp`]
///
/// The ULP metric measures distance in units-in-the-last-place of the
/// reference function's f32 output. **ULP distance balloons whenever the
/// reference output approaches zero** because f32 has very fine resolution
/// near zero (a single bit-flip near subnormal magnitudes can correspond to
/// `~1e9` ULPs). For these reference functions, switch to
/// [`F32ScoreMetric::AbsoluteError`]:
///
/// - `x^p` for `0 < p < 1` (square root, cube root, the sRGB `1/2.4`
///   exponent) on any domain that includes `x ≈ 0`
/// - `log(x)`, `ln(1+x)`, `arcsinh(x)` near `x = 0`
/// - `1/x` and other functions whose output passes near zero
/// - Any function whose codomain spans many orders of magnitude
///
/// As a rule of thumb: if `min(|f(x)|) < 1e-2 * max(|f(x)|)` over the
/// sweep domain, the ULP metric will report misleading numbers and you
/// should use absolute error instead. A sub-`1e-4` absolute-error fit can
/// trivially show `max_ulp ≈ 1e9` for these functions while being entirely
/// usable.
#[derive(Debug, Clone, Copy, Default)]
pub enum F32ScoreMetric {
    /// Score by ULP distance.
    ///
    /// Best for transfer functions whose output stays bounded *away from
    /// zero* (sRGB EOTF on `[threshold, 1]`, PQ on `[threshold, 1]`, etc.)
    /// where the f32 ULP grid is uniform enough that ULPs track absolute
    /// error linearly. See the enum-level documentation for the cases
    /// where this metric is misleading.
    #[default]
    Ulp,
    /// Score by absolute error.
    ///
    /// Use when the reference output passes near zero (output crosses
    /// zero, or is dense near subnormal magnitudes), where ULP distance
    /// stops tracking absolute error linearly. Specifically: `x^p` for
    /// `0 < p < 1`, `log(x)` near zero, mantissa-of-`log2` style outputs,
    /// or any function whose codomain spans many decades.
    AbsoluteError,
}

/// Configuration for f32 coefficient optimization via local search.
#[derive(Debug, Clone)]
pub struct F32SearchConfig {
    /// Search radius per coefficient in f32 ULPs (default: 3).
    pub radius: i32,
    /// Maximum search rounds (default: 5).
    pub max_rounds: usize,
    /// Scoring metric (default: ULP).
    pub metric: F32ScoreMetric,
    /// Number of sample points. `None` = exhaustive sweep of every f32 in the
    /// domain (default, slow for wide domains). `Some(n)` = uniformly-spaced
    /// samples (use for domains > 10M f32 values, or transformed inputs).
    pub samples: Option<usize>,
}

impl Default for F32SearchConfig {
    fn default() -> Self {
        Self {
            radius: 3,
            max_rounds: 5,
            metric: F32ScoreMetric::default(),
            samples: None,
        }
    }
}

/// Result of f32 coefficient optimization.
///
/// # Picking the right quality metric
///
/// Both the ULP fields (`max_ulp`, `avg_ulp`, `boundary_ulp`) and the
/// absolute-error fields (`max_abs_err`, `avg_abs_err`) are reported on
/// every result regardless of which [`F32ScoreMetric`] the optimizer
/// scored against. The "right" number to read depends on the reference
/// function:
///
/// - If the reference output stays bounded *away from zero* over the
///   sweep domain (most piecewise transfer functions in their power
///   segment): trust `max_ulp` / `boundary_ulp`.
/// - If the reference output approaches zero anywhere in the sweep
///   domain (`x^p` for `0 < p < 1`, `log(x)`, `1/x`, anything whose
///   codomain spans many decades): the ULP fields will look
///   catastrophic (often `~1e9`) even for sub-`1e-4` fits — read
///   `max_abs_err` instead. See [`F32ScoreMetric`] for the full
///   discussion.
///
/// `mono_violations` and `values_scored` / `exhaustive` are unconditional
/// quality signals.
#[derive(Debug, Clone)]
pub struct F32SearchResult {
    /// Optimized f32 numerator coefficients.
    pub numerator: Vec<f32>,
    /// Optimized f32 denominator coefficients.
    pub denominator: Vec<f32>,
    /// Maximum ULP error across the sweep domain.
    ///
    /// **Unreliable when the reference output approaches zero** — read
    /// `max_abs_err` instead in that case. See [`F32SearchResult`]'s
    /// type-level docs for guidance.
    pub max_ulp: u32,
    /// Average ULP error across the sweep domain.
    ///
    /// See the caveat on `max_ulp`: ULP averaging is also distorted when
    /// the reference output approaches zero.
    pub avg_ulp: f64,
    /// ULP error at the domain start (boundary).
    ///
    /// See the caveat on `max_ulp`: if `sweep_lo` is near zero for a
    /// fractional-power-style reference function, this number will look
    /// huge even for a usable fit.
    pub boundary_ulp: u32,
    /// Number of monotonicity violations.
    pub mono_violations: u64,
    /// Maximum absolute error.
    ///
    /// The "ground truth" quality number when the reference output
    /// approaches zero — see the caveats on the ULP fields and the
    /// [`F32ScoreMetric`] docs.
    pub max_abs_err: f64,
    /// Average absolute error.
    pub avg_abs_err: f64,
    /// Number of f32 values scored (exhaustive) or sample count (sampled).
    pub values_scored: u64,
    /// `true` if every f32 in the sweep domain was evaluated,
    /// `false` if uniform sampling was used.
    pub exhaustive: bool,
}

impl fmt::Display for F32SearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = if self.exhaustive {
            "exhaustive"
        } else {
            "sampled"
        };
        write!(
            f,
            "max_ulp={:>4} avg_ulp={:>7.3} bnd={} mono={} max_abs={:.3e} [{mode}, n={}]",
            self.max_ulp,
            self.avg_ulp,
            self.boundary_ulp,
            self.mono_violations,
            self.max_abs_err,
            self.values_scored,
        )
    }
}

/// A fitted rational polynomial with data and diagnostics.
#[derive(Debug, Clone)]
pub struct RationalFit<'data, T: Value = f64> {
    data: Cow<'data, [(T, T)]>,
    x_range: RangeInclusive<T>,
    function: RationalPolynomial<T>,
    p_degree: usize,
    q_degree: usize,
    iterations: usize,
}

// =============================================================================
// Defaults
// =============================================================================

impl Default for RationalFitOptions<f64> {
    fn default() -> Self {
        Self {
            weighting: ErrorWeighting::Absolute,
            max_iterations: 50,
            tolerance: 1e-12,
            constraints: Vec::new(),
            n_samples: 2000,
            lm_iterations: 200,
            constraint_mode: ConstraintMode::default(),
            restarts: 1,
        }
    }
}

impl Default for RationalFitOptions<f32> {
    fn default() -> Self {
        Self {
            weighting: ErrorWeighting::Absolute,
            max_iterations: 50,
            tolerance: 1e-7,
            constraints: Vec::new(),
            n_samples: 2000,
            lm_iterations: 200,
            constraint_mode: ConstraintMode::default(),
            restarts: 1,
        }
    }
}

impl<T: Value> Constraint<T> {
    /// Create a soft point constraint at `(x, y)` with the default
    /// weight `1e6`.
    ///
    /// **Despite the high default weight, this is not a hard pin** —
    /// see the [`Constraint`] type-level docs. For an exact-match
    /// guarantee at the constraint point, pair this with
    /// [`ConstraintMode::SnapOnly`] (the default) or
    /// [`ConstraintMode::SkThenSnap`].
    pub fn new(x: T, y: T) -> Self {
        Self {
            x,
            y,
            weight: T::try_cast(1_000_000.0).unwrap_or(T::one()),
        }
    }

    /// Create a soft point constraint at `(x, y)` with a caller-chosen
    /// penalty weight.
    ///
    /// The `weight` is a multiplier on the `(P(x)/Q(x) - y)²` term in
    /// the least-squares objective — **not a hard equality**. Arbitrarily
    /// large weights are tolerated by SK/LM but never produce an exact
    /// pin. For exact-match behavior at the constraint point, see the
    /// [`Constraint`] type-level docs.
    pub fn soft(x: T, y: T, weight: T) -> Self {
        Self { x, y, weight }
    }

    /// Create a constraint with custom weight.
    ///
    /// This is the historical name for [`Constraint::soft`] — both
    /// produce identical soft (penalty-weighted) constraints. New code
    /// should prefer [`Constraint::soft`] because the name does not
    /// invite the misreading "high weight = hard pin." See the
    /// [`Constraint`] type-level docs.
    pub fn with_weight(x: T, y: T, weight: T) -> Self {
        Self::soft(x, y, weight)
    }
}

// =============================================================================
// RationalPolynomial evaluation
// =============================================================================

impl<T: Value> RationalPolynomial<T> {
    /// Create a rational polynomial from numerator and denominator coefficients.
    ///
    /// Both are lowest-degree-first. The denominator's last element should be 1
    /// (monic convention).
    #[must_use]
    pub fn new(numerator: Vec<T>, denominator: Vec<T>) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Evaluate `P(x)/Q(x)` using Horner's method.
    #[inline]
    #[must_use]
    pub fn y(&self, x: T) -> T {
        horner(&self.numerator, x) / horner(&self.denominator, x)
    }

    /// Numerator coefficients (lowest-degree-first).
    #[must_use]
    pub fn numerator(&self) -> &[T] {
        &self.numerator
    }

    /// Denominator coefficients (lowest-degree-first, last = 1.0).
    #[must_use]
    pub fn denominator(&self) -> &[T] {
        &self.denominator
    }

    /// Numerator and denominator as a pair.
    #[must_use]
    pub fn coefficients(&self) -> (&[T], &[T]) {
        (&self.numerator, &self.denominator)
    }

    /// Numerator degree.
    #[must_use]
    pub fn numerator_degree(&self) -> usize {
        self.numerator.len().saturating_sub(1)
    }

    /// Denominator degree.
    #[must_use]
    pub fn denominator_degree(&self) -> usize {
        self.denominator.len().saturating_sub(1)
    }

    /// Adjust the numerator constant term (`p[0]`) so that `P(x)/Q(x) = y` exactly.
    ///
    /// This is a post-hoc boundary snap: after fitting unconstrained for best
    /// overall accuracy, call this to nail the piecewise join point. Since `p[0]`
    /// is a DC offset, the perturbation has minimal impact on accuracy elsewhere —
    /// especially when `x` is small (as it is for sRGB/PQ thresholds near zero).
    ///
    /// For f32 coefficients evaluated via f64 Horner, truncate each coefficient
    /// to f32 **before** calling this, so the snap accounts for truncation error.
    pub fn snap_boundary(&mut self, x: T, y: T) {
        // P(x) = p[0] + x * (p[1] + x * (p[2] + ...))
        // We want P(x)/Q(x) = y  =>  p[0] = y * Q(x) - x * horner(p[1..], x)
        let q_at_x = horner(&self.denominator, x);
        let p_rest = if self.numerator.len() > 1 {
            x * horner(&self.numerator[1..], x)
        } else {
            T::zero()
        };
        self.numerator[0] = y * q_at_x - p_rest;
    }
}

impl RationalPolynomial<f64> {
    /// Truncate to f32 and locally search for the best f32 coefficients.
    ///
    /// This is the final optimization step for transfer function approximations:
    /// the f64 fit finds the right basin, truncation to f32 introduces rounding,
    /// and this search explores ±`radius` f32 ULP perturbations per coefficient
    /// to find the f32 combination with lowest error.
    ///
    /// The `transform` maps domain values to polynomial input (e.g., `|v| v.sqrt()`
    /// for L2S which evaluates the polynomial on √x). Pass `|v| v` for identity.
    ///
    /// Evaluation uses f64-intermediate Horner (matching linear-srgb's scalar path).
    /// For different evaluation semantics, use the coefficients from this result
    /// with your own search.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::needless_pass_by_value
    )]
    pub fn optimize_f32(
        &self,
        reference: impl Fn(f64) -> f64,
        sweep_lo: f32,
        sweep_hi: f32,
        transform: impl Fn(f32) -> f32,
        config: F32SearchConfig,
    ) -> F32SearchResult {
        let p: Vec<f32> = self.numerator.iter().map(|&c| c as f32).collect();
        let q: Vec<f32> = self.denominator.iter().map(|&c| c as f32).collect();
        f32_local_search(&p, &q, &reference, sweep_lo, sweep_hi, &transform, &config)
    }

    /// Warm-start f32 optimization from existing f32 coefficients.
    ///
    /// Promotes to f64, runs LM refinement, truncates back to f32, then runs
    /// the local f32 search. Use this when you already have good coefficients
    /// and want to explore nearby basins.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::needless_pass_by_value,
        clippy::too_many_arguments
    )]
    pub fn refine_from_f32(
        existing_p: &[f32],
        existing_q: &[f32],
        reference: impl Fn(f64) -> f64,
        data: &[(f64, f64)],
        sweep_lo: f32,
        sweep_hi: f32,
        transform: impl Fn(f32) -> f32,
        config: F32SearchConfig,
        weighting: ErrorWeighting,
        lm_iters: usize,
    ) -> F32SearchResult {
        // Promote to f64
        let mut poly = RationalPolynomial::new(
            existing_p.iter().map(|&c| f64::from(c)).collect(),
            existing_q.iter().map(|&c| f64::from(c)).collect(),
        );

        // Run LM refinement on f64
        refine_lm(&mut poly, data, &[], weighting, lm_iters, 1e-15);

        // Truncate to f32 and run local search
        let p: Vec<f32> = poly.numerator.iter().map(|&c| c as f32).collect();
        let q: Vec<f32> = poly.denominator.iter().map(|&c| c as f32).collect();
        f32_local_search(&p, &q, &reference, sweep_lo, sweep_hi, &transform, &config)
    }
}

/// Measure the f32-ULP gap at a piecewise join point.
///
/// Piecewise transfer functions (sRGB, PQ, HLG) have a threshold where the
/// polynomial meets a linear segment. The *practical* continuity metric is
/// `ulp_distance(linear_segment(threshold - 1 ULP), polynomial(threshold + 1 ULP))`
/// — the gap between the two branches at adjacent f32 values.
///
/// This differs from `F32SearchResult::boundary_ulp` (which measures the
/// polynomial's error against the reference function at the threshold). A fit
/// that's close to the reference at the threshold can still produce a visible
/// discontinuity if the linear segment rounds differently from the polynomial.
///
/// # Parameters
/// - `threshold`: the f32 input at which the two branches meet
/// - `poly_p`, `poly_q`: the rational polynomial's f32 coefficients
/// - `linear_branch`: the linear segment (e.g., `|v| v / 12.92`)
/// - `poly_input_transform`: maps the threshold value to polynomial input
///   (use `|v| v` for identity, `|v| v.sqrt()` for L2S)
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub fn piecewise_gap(
    threshold: f32,
    poly_p: &[f32],
    poly_q: &[f32],
    linear_branch: impl Fn(f32) -> f32,
    poly_input_transform: impl Fn(f32) -> f32,
) -> u32 {
    let below = f32::from_bits(threshold.to_bits() - 1);
    let above = f32::from_bits(threshold.to_bits() + 1);
    let left = linear_branch(below);
    let right = eval_f32_horner(poly_input_transform(above), poly_p, poly_q);
    f32_ulp_distance(left, right)
}

/// Report of composed-function error (roundtrip).
///
/// When two polynomials approximate inverse functions `f` and `f⁻¹`,
/// the composed roundtrip `f⁻¹(f(v))` should return `v`. This measures
/// how well that holds across a sweep domain.
#[derive(Debug, Clone)]
pub struct RoundtripReport {
    /// Maximum absolute error: `|f⁻¹(f(v)) - v|` across the sweep domain.
    pub max_abs_err: f64,
    /// Average absolute error.
    pub avg_abs_err: f64,
    /// Maximum ULP distance between roundtrip output and input.
    pub max_ulp: u32,
    /// Number of inputs where roundtrip error exceeds 1/65535 (one u16 step).
    pub u16_over_1: u64,
    /// Number of inputs where roundtrip error exceeds 0.5/255 (half u8 step).
    pub u8_over_half: u64,
    /// Number of f32 values scored.
    pub values_scored: u64,
    /// True if every f32 in the sweep domain was evaluated.
    pub exhaustive: bool,
}

impl fmt::Display for RoundtripReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mode = if self.exhaustive {
            "exhaustive"
        } else {
            "sampled"
        };
        write!(
            f,
            "max_ulp={} max_abs={:.3e} avg_abs={:.3e} u16>1={} u8>0.5={} [{mode}, n={}]",
            self.max_ulp,
            self.max_abs_err,
            self.avg_abs_err,
            self.u16_over_1,
            self.u8_over_half,
            self.values_scored,
        )
    }
}

/// Measure roundtrip error of `f⁻¹(f(v))` across a sweep domain.
///
/// Each of `forward` and `inverse` is a closure that maps f32→f32, typically
/// a piecewise function combining a polynomial with linear segments and clamps.
/// The implementation is opaque to `measure_roundtrip` — pass whatever composition
/// matches your production evaluation path.
///
/// # Parameters
/// - `forward`: the f(v) function (e.g., `|v| srgb_to_linear_fast(v)`)
/// - `inverse`: the f⁻¹(u) function (e.g., `|u| linear_to_srgb_fast(u)`)
/// - `sweep_lo`, `sweep_hi`: f32 range to sweep
/// - `samples`: `None` = exhaustive f32 sweep, `Some(n)` = n uniform samples
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
pub fn measure_roundtrip(
    forward: impl Fn(f32) -> f32,
    inverse: impl Fn(f32) -> f32,
    sweep_lo: f32,
    sweep_hi: f32,
    samples: Option<usize>,
) -> RoundtripReport {
    const U16_STEP: f64 = 1.0 / 65535.0;
    const U8_HALF: f64 = 0.5 / 255.0;

    let mut max_ulp = 0u32;
    let mut max_abs: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut u16_over = 0u64;
    let mut u8_over = 0u64;
    let mut count = 0u64;

    let mut eval_at = |v: f32| {
        let rt = inverse(forward(v));
        let abs_err = (f64::from(rt) - f64::from(v)).abs();
        let ulp = f32_ulp_distance(rt, v);
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        sum_abs += abs_err;
        if abs_err > U16_STEP {
            u16_over += 1;
        }
        if abs_err > U8_HALF {
            u8_over += 1;
        }
        count += 1;
    };

    if let Some(n) = samples {
        let lo = f64::from(sweep_lo);
        let hi = f64::from(sweep_hi);
        for i in 0..=n {
            let v = lo + (hi - lo) * i as f64 / n as f64;
            eval_at(v as f32);
        }
    } else {
        let mut v = sweep_lo;
        while v <= sweep_hi {
            eval_at(v);
            v = f32_next(v);
        }
    }

    RoundtripReport {
        max_abs_err: max_abs,
        avg_abs_err: sum_abs / count.max(1) as f64,
        max_ulp,
        u16_over_1: u16_over,
        u8_over_half: u8_over,
        values_scored: count,
        exhaustive: samples.is_none(),
    }
}

/// Core f32 local search — shared by `optimize_f32` and `refine_from_f32`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::too_many_lines
)]
fn f32_local_search(
    initial_p: &[f32],
    initial_q: &[f32],
    reference: &dyn Fn(f64) -> f64,
    sweep_lo: f32,
    sweep_hi: f32,
    transform: &dyn Fn(f32) -> f32,
    config: &F32SearchConfig,
) -> F32SearchResult {
    let mut p = initial_p.to_vec();
    let mut q = initial_q.to_vec();

    // Score: (primary, secondary, avg, mono)
    // For ULP: primary = max_ulp, secondary = boundary_ulp
    // For AbsError: primary = max_abs as u64 bits, secondary = boundary_ulp
    let score = |p: &[f32], q: &[f32]| -> (u64, u64, f64, u64) {
        let mut max_ulp = 0u32;
        let mut sum_ulp = 0u64;
        let mut max_abs: f64 = 0.0;
        let mut sum_abs: f64 = 0.0;
        let mut count = 0u64;
        let mut mono = 0u64;
        let mut prev = eval_f32_horner(transform(sweep_lo), p, q);

        let mut eval_at = |x_f32: f32, x_f64: f64| {
            let got = eval_f32_horner(transform(x_f32), p, q);
            let expected = reference(x_f64) as f32;
            let ulp = f32_ulp_distance(got, expected);
            let abs_err = (f64::from(got) - f64::from(expected)).abs();
            if ulp > max_ulp {
                max_ulp = ulp;
            }
            if abs_err > max_abs {
                max_abs = abs_err;
            }
            sum_ulp += u64::from(ulp);
            sum_abs += abs_err;
            count += 1;
            if count > 1 && got < prev {
                mono += 1;
            }
            prev = got;
        };

        if let Some(n) = config.samples {
            let lo = f64::from(sweep_lo);
            let hi = f64::from(sweep_hi);
            for i in 0..=n {
                let x = lo + (hi - lo) * i as f64 / n as f64;
                eval_at(x as f32, x);
            }
        } else {
            let mut v = sweep_lo;
            while v <= sweep_hi {
                eval_at(v, f64::from(v));
                v = f32_next(v);
            }
        }

        let bnd_ulp = f32_ulp_distance(
            eval_f32_horner(transform(sweep_lo), p, q),
            reference(f64::from(sweep_lo)) as f32,
        );
        match config.metric {
            F32ScoreMetric::Ulp => (
                u64::from(max_ulp),
                u64::from(bnd_ulp),
                sum_ulp as f64 / count.max(1) as f64,
                mono,
            ),
            F32ScoreMetric::AbsoluteError => (
                max_abs.to_bits(),
                u64::from(bnd_ulp),
                sum_abs / count.max(1) as f64,
                mono,
            ),
        }
    };

    let is_better = |a: (u64, u64, f64, u64), b: (u64, u64, f64, u64)| -> bool {
        if a.3 > 0 && b.3 == 0 {
            return false;
        }
        if a.3 == 0 && b.3 > 0 {
            return true;
        }
        (a.0, a.1) < (b.0, b.1) || ((a.0, a.1) == (b.0, b.1) && a.2 < b.2)
    };

    let mut best = score(&p, &q);
    let n_coeffs = p.len() + q.len() - 1;

    for _round in 0..config.max_rounds {
        let mut improved = false;
        for ci in 0..n_coeffs {
            let is_p = ci < p.len();
            let idx = if is_p { ci } else { ci - p.len() };
            let original = if is_p { p[idx] } else { q[idx] };

            let mut best_delta = 0i32;
            let mut best_delta_score: Option<(u64, u64, f64, u64)> = None;

            for delta in (-config.radius..=config.radius).filter(|&d| d != 0) {
                let nudged = f32_nudge(original, delta);
                if is_p {
                    p[idx] = nudged;
                } else {
                    q[idx] = nudged;
                }
                let s = score(&p, &q);
                let dominated = best_delta_score
                    .as_ref()
                    .is_some_and(|bs| !is_better(s, *bs));
                if is_better(s, best) && !dominated {
                    best_delta = delta;
                    best_delta_score = Some(s);
                }
                if is_p {
                    p[idx] = original;
                } else {
                    q[idx] = original;
                }
            }

            if let Some(s) = best_delta_score {
                let nudged = f32_nudge(original, best_delta);
                if is_p {
                    p[idx] = nudged;
                } else {
                    q[idx] = nudged;
                }
                best = s;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    // Final full measurement for reporting (always compute both metrics)
    let mut max_ulp = 0u32;
    let mut sum_ulp = 0u64;
    let mut max_abs: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut count = 0u64;
    let mut mono = 0u64;
    let mut prev = eval_f32_horner(transform(sweep_lo), &p, &q);
    let mut final_eval = |x_f32: f32, x_f64: f64| {
        let got = eval_f32_horner(transform(x_f32), &p, &q);
        let expected = reference(x_f64) as f32;
        let ulp = f32_ulp_distance(got, expected);
        let abs_err = (f64::from(got) - f64::from(expected)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        sum_ulp += u64::from(ulp);
        sum_abs += abs_err;
        count += 1;
        if count > 1 && got < prev {
            mono += 1;
        }
        prev = got;
    };
    if let Some(n) = config.samples {
        let lo = f64::from(sweep_lo);
        let hi = f64::from(sweep_hi);
        for i in 0..=n {
            let x = lo + (hi - lo) * i as f64 / n as f64;
            final_eval(x as f32, x);
        }
    } else {
        let mut v = sweep_lo;
        while v <= sweep_hi {
            final_eval(v, f64::from(v));
            v = f32_next(v);
        }
    }
    let bnd = f32_ulp_distance(
        eval_f32_horner(transform(sweep_lo), &p, &q),
        reference(f64::from(sweep_lo)) as f32,
    );

    F32SearchResult {
        numerator: p,
        denominator: q,
        max_ulp,
        avg_ulp: sum_ulp as f64 / count.max(1) as f64,
        boundary_ulp: bnd,
        mono_violations: mono,
        max_abs_err: max_abs,
        avg_abs_err: sum_abs / count.max(1) as f64,
        values_scored: count,
        exhaustive: config.samples.is_none(),
    }
}

// =============================================================================
// RationalFit — public API
// =============================================================================

impl<'data, T: Value> RationalFit<'data, T> {
    /// Fit a rational polynomial P(x)/Q(x) to data.
    ///
    /// # Parameters
    /// - `data`: (x, y) pairs
    /// - `p_degree`: degree of numerator P
    /// - `q_degree`: degree of denominator Q
    /// - `options`: fitting configuration (weighting, constraints, iterations)
    ///
    /// # Errors
    /// Returns an error if the data is empty, degrees are too high for the data,
    /// or the SVD solver fails to converge.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        data: impl Into<Cow<'data, [(T, T)]>>,
        p_degree: usize,
        q_degree: usize,
        options: RationalFitOptions<T>,
    ) -> Result<Self> {
        let data = data.into();
        if data.is_empty() {
            return Err(Error::NoData);
        }

        let p_terms = p_degree + 1;
        let q_terms = q_degree + 1; // includes the fixed leading 1
        let total_free = p_terms + q_degree; // q_degree free params (leading 1 is fixed)

        if data.len() < total_free {
            return Err(Error::DegreeTooHigh(p_degree.max(q_degree)));
        }

        let x_min = data.iter().map(|(x, _)| *x).fold(T::infinity(), Value::min);
        let x_max = data
            .iter()
            .map(|(x, _)| *x)
            .fold(T::neg_infinity(), Value::max);
        let x_range = x_min..=x_max;

        let (poly, iterations) = fit_rational(
            &data,
            p_terms,
            q_terms,
            options.max_iterations,
            options.lm_iterations,
            options.tolerance,
            &options.constraints,
            options.weighting,
            options.constraint_mode,
            options.restarts,
        )?;

        Ok(Self {
            data,
            x_range,
            function: poly,
            p_degree,
            q_degree,
            iterations,
        })
    }

    /// Fit a rational polynomial to a known function sampled on a domain.
    ///
    /// Generates Chebyshev-spaced sample points for near-minimax conditioning.
    ///
    /// # Parameters
    /// - `f`: the target function
    /// - `domain`: closed interval \[a, b\] to fit over
    /// - `p_degree`: degree of numerator P
    /// - `q_degree`: degree of denominator Q
    /// - `options`: fitting configuration
    ///
    /// # Errors
    /// Returns an error if degrees are too high for the sample count,
    /// or the SVD solver fails to converge.
    #[allow(clippy::needless_pass_by_value)]
    pub fn from_function(
        f: impl Fn(T) -> T,
        domain: RangeInclusive<T>,
        p_degree: usize,
        q_degree: usize,
        options: RationalFitOptions<T>,
    ) -> Result<RationalFit<'static, T>> {
        let a = *domain.start();
        let b = *domain.end();
        let n = options.n_samples;

        let data: Vec<(T, T)> = chebyshev_nodes(n, a, b)
            .into_iter()
            .map(|x| (x, f(x)))
            .collect();

        let p_terms = p_degree + 1;
        let q_terms = q_degree + 1;

        if data.len() < p_terms + q_degree {
            return Err(Error::DegreeTooHigh(p_degree.max(q_degree)));
        }

        let (poly, iterations) = fit_rational(
            &data,
            p_terms,
            q_terms,
            options.max_iterations,
            options.lm_iterations,
            options.tolerance,
            &options.constraints,
            options.weighting,
            options.constraint_mode,
            options.restarts,
        )?;

        Ok(RationalFit {
            data: Cow::Owned(data),
            x_range: a..=b,
            function: poly,
            p_degree,
            q_degree,
            iterations,
        })
    }

    /// Evaluate the fitted rational polynomial at x.
    #[inline]
    #[must_use]
    pub fn y(&self, x: T) -> T {
        self.function.y(x)
    }

    /// The underlying rational polynomial.
    #[must_use]
    pub fn as_polynomial(&self) -> &RationalPolynomial<T> {
        &self.function
    }

    /// Numerator and denominator coefficients (lowest-degree-first).
    #[must_use]
    pub fn coefficients(&self) -> (&[T], &[T]) {
        self.function.coefficients()
    }

    /// The degree of the numerator.
    #[must_use]
    pub fn p_degree(&self) -> usize {
        self.p_degree
    }

    /// The degree of the denominator.
    #[must_use]
    pub fn q_degree(&self) -> usize {
        self.q_degree
    }

    /// How many Sanathanan-Koerner iterations were used.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The x-range of the fitted data.
    #[must_use]
    pub fn x_range(&self) -> &RangeInclusive<T> {
        &self.x_range
    }

    /// The data used for fitting.
    #[must_use]
    pub fn data(&self) -> &[(T, T)] {
        &self.data
    }

    /// Coefficient of determination R².
    ///
    /// Values near 1.0 indicate a good fit.
    #[must_use]
    pub fn r_squared(&self) -> T {
        let n = self.data.len();
        if n == 0 {
            return T::zero();
        }

        let y_mean = self
            .data
            .iter()
            .map(|(_, y)| *y)
            .fold(T::zero(), |a, b| a + b)
            / T::try_cast(n).unwrap_or(T::one());

        let ss_tot = self
            .data
            .iter()
            .map(|(_, y)| {
                let d = *y - y_mean;
                d * d
            })
            .fold(T::zero(), |a, b| a + b);

        let ss_res = self
            .data
            .iter()
            .map(|(x, y)| {
                let d = self.function.y(*x) - *y;
                d * d
            })
            .fold(T::zero(), |a, b| a + b);

        if ss_tot.is_near_zero() {
            T::one()
        } else {
            T::one() - ss_res / ss_tot
        }
    }

    /// Maximum absolute error across all data points.
    #[must_use]
    pub fn max_abs_error(&self) -> T {
        self.data
            .iter()
            .map(|(x, y)| Value::abs(self.function.y(*x) - *y))
            .fold(T::zero(), Value::max)
    }

    /// Maximum relative error across all data points.
    ///
    /// Relative error is `|predicted - actual| / |actual|`.
    /// Points where `|actual|` < epsilon are skipped.
    #[must_use]
    pub fn max_rel_error(&self) -> T {
        self.data
            .iter()
            .filter_map(|(x, y)| {
                let ay = Value::abs(*y);
                if ay.is_near_zero() {
                    return None;
                }
                Some(Value::abs(self.function.y(*x) - *y) / ay)
            })
            .fold(T::zero(), Value::max)
    }

    /// Residuals: predicted - actual for each data point.
    #[must_use]
    pub fn residuals(&self) -> Vec<T> {
        self.data
            .iter()
            .map(|(x, y)| self.function.y(*x) - *y)
            .collect()
    }

    /// Error at and near the domain boundaries.
    ///
    /// For piecewise functions (sRGB, PQ), the domain start is where the
    /// rational polynomial joins another segment. This method reports
    /// absolute and relative error at both endpoints, plus extrapolation
    /// error at small offsets past the boundary.
    ///
    /// The `reference` function provides ground-truth values for evaluation
    /// points at and beyond the fitted domain.
    #[must_use]
    pub fn boundary_error(&self, reference: impl Fn(T) -> T) -> BoundaryReport<T> {
        let lo = *self.x_range.start();
        let hi = *self.x_range.end();
        let width = hi - lo;

        let eval_point = |x: T| -> BoundaryPoint<T> {
            let predicted = self.function.y(x);
            let expected = reference(x);
            let abs_err = Value::abs(predicted - expected);
            let rel_err = if Value::abs(expected).is_near_zero() {
                abs_err
            } else {
                abs_err / Value::abs(expected)
            };
            BoundaryPoint {
                x,
                predicted,
                expected,
                abs_err,
                rel_err,
            }
        };

        // Sample at the endpoints
        let at_lo = eval_point(lo);
        let at_hi = eval_point(hi);

        // Extrapolation past boundaries: 0.1%, 1%, 5%, 10% of domain width
        let fractions = [
            T::try_cast(0.001).unwrap_or(T::zero()),
            T::try_cast(0.01).unwrap_or(T::zero()),
            T::try_cast(0.05).unwrap_or(T::zero()),
            T::try_cast(0.1).unwrap_or(T::zero()),
        ];
        let past_lo: Vec<BoundaryPoint<T>> = fractions
            .iter()
            .map(|&f| eval_point(lo - width * f))
            .collect();
        let past_hi: Vec<BoundaryPoint<T>> = fractions
            .iter()
            .map(|&f| eval_point(hi + width * f))
            .collect();

        BoundaryReport {
            lo: at_lo,
            hi: at_hi,
            past_lo,
            past_hi,
        }
    }

    /// Returns an owned version of this fit.
    #[must_use]
    pub fn to_owned(&self) -> RationalFit<'static, T> {
        RationalFit {
            data: Cow::Owned(self.data.to_vec()),
            x_range: self.x_range.clone(),
            function: self.function.clone(),
            p_degree: self.p_degree,
            q_degree: self.q_degree,
            iterations: self.iterations,
        }
    }
}

// =============================================================================
// Display
// =============================================================================

impl<T: Value> fmt::Display for RationalPolynomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        write_poly_terms(f, &self.numerator)?;
        write!(f, ") / (")?;
        write_poly_terms(f, &self.denominator)?;
        write!(f, ")")
    }
}

impl<T: Value> fmt::Display for RationalFit<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RationalFit {}/{} (R²={:.6}, {} iters): {}",
            self.p_degree,
            self.q_degree,
            self.r_squared(),
            self.iterations,
            self.function,
        )
    }
}

fn write_poly_terms<T: Value>(f: &mut fmt::Formatter<'_>, coeffs: &[T]) -> fmt::Result {
    let mut first = true;
    for (i, c) in coeffs.iter().enumerate() {
        if !first {
            write!(f, " + ")?;
        }
        first = false;
        match i {
            0 => write!(f, "{c:.6e}")?,
            1 => write!(f, "{c:.6e}*x")?,
            _ => write!(f, "{c:.6e}*x^{i}")?,
        }
    }
    Ok(())
}

// =============================================================================
// Horner evaluation
// =============================================================================

/// Evaluate a polynomial using Horner's method.
/// Coefficients are lowest-degree-first: c[0] + c[1]*x + c[2]*x² + ...
#[inline(always)]
fn horner<T: Value>(coeffs: &[T], x: T) -> T {
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result * x + coeffs[i];
    }
    result
}

// =============================================================================
// f32 helpers for optimize_f32
// =============================================================================

/// Evaluate rational polynomial with f32 coefficients via f64 Horner.
/// Matches the evaluation path in linear-srgb's scalar implementation.
#[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
fn eval_f32_horner(x: f32, p: &[f32], q: &[f32]) -> f32 {
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

/// ULP distance between two f32 values, handling sign correctly.
///
/// Maps negative floats through zero so the integer representation
/// is monotonic across the entire f32 range (including sign crossings).
#[allow(clippy::float_cmp, clippy::cast_possible_wrap)]
fn f32_ulp_distance(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    // Map to linear integer space: negate and flip negative floats
    // so that the integer representation is monotonic across zero.
    let ai = {
        let bits = a.to_bits() as i32;
        if bits < 0 {
            i32::MIN - bits
        } else {
            bits
        }
    };
    let bi = {
        let bits = b.to_bits() as i32;
        if bits < 0 {
            i32::MIN - bits
        } else {
            bits
        }
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    {
        (i64::from(ai) - i64::from(bi)).unsigned_abs() as u32
    }
}

/// Next representable f32 toward +infinity, handling negative values correctly.
fn f32_next(v: f32) -> f32 {
    if v.is_nan() || v >= f32::MAX {
        return v;
    }
    if v == 0.0 {
        return f32::from_bits(1);
    } // +0 → smallest positive
    if v > 0.0 {
        f32::from_bits(v.to_bits() + 1)
    } else {
        // Negative: decrement bit pattern to move toward zero
        f32::from_bits(v.to_bits() - 1)
    }
}

/// Nudge an f32 by `delta` ULPs.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn f32_nudge(v: f32, delta: i32) -> f32 {
    f32::from_bits((v.to_bits() as i32 + delta) as u32)
}

// =============================================================================
// Chebyshev nodes
// =============================================================================

/// Generate n Chebyshev-spaced nodes on \[a, b\].
///
/// Chebyshev nodes cluster near endpoints, reducing Runge's phenomenon and
/// improving conditioning for polynomial fitting.
fn chebyshev_nodes<T: Value>(n: usize, a: T, b: T) -> Vec<T> {
    let pi = T::try_cast(std::f64::consts::PI).unwrap_or(T::zero());
    let two_n = T::try_cast(2 * n).unwrap_or(T::one());
    let half = T::half();
    let center = (a + b) * half;
    let half_width = (b - a) * half;

    (0..n)
        .map(|k| {
            let k_t = T::try_cast(2 * k + 1).unwrap_or(T::zero());
            let theta = pi * k_t / two_n;
            center + half_width * theta.cos()
        })
        .collect()
}

// =============================================================================
// SVD solver (standalone, mirrors polyfit's CurveFit::solve_matrix)
// =============================================================================

/// Solve Ax = b via SVD with polyfit-compatible epsilon and iteration tuning.
fn solve_svd<T: Value>(a: DMatrix<T>, b: &DVector<T>) -> Result<Vec<T>> {
    const SVD_ITER_LIMIT: (usize, usize) = (1000, 10_000);

    let size = a.shape();
    let svd_eps = T::RealField::default_epsilon() * nalgebra::convert(5.0);

    let max_dim = size.0.max(size.1);
    let iters = max_dim
        .saturating_mul(max_dim)
        .clamp(SVD_ITER_LIMIT.0, SVD_ITER_LIMIT.1);

    let decomp =
        SVD::try_new_unordered(a, true, true, svd_eps, iters).ok_or(Error::DidNotConverge)?;

    let machine_epsilon = T::epsilon();
    let max_size = size.0.max(size.1);
    let sigma_max = decomp.singular_values.max();
    let epsilon = machine_epsilon * T::try_cast(max_size)? * sigma_max;

    let big_x = decomp.solve(b, epsilon).map_err(Error::Algebra)?;
    let coefficients: Vec<_> = big_x.data.into();

    if coefficients.iter().any(|c| c.is_nan()) {
        return Err(Error::Algebra("NaN in coefficients"));
    }

    Ok(coefficients)
}

// =============================================================================
// Sanathanan-Koerner fitting
// =============================================================================

/// Precompute x^0, x^1, ..., x^{max_power-1} for each x value.
fn precompute_powers<T: Value>(x_values: &[T], max_power: usize) -> Vec<Vec<T>> {
    x_values
        .iter()
        .map(|&x| {
            let mut powers = Vec::with_capacity(max_power);
            let mut xj = T::one();
            for _ in 0..max_power {
                powers.push(xj);
                xj *= x;
            }
            powers
        })
        .collect()
}

/// Build the weighted linear system for one SK iteration.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn build_sk_system<T: Value>(
    data: &[(T, T)],
    x_powers: &[Vec<T>],
    c_powers: &[Vec<T>],
    constraints: &[Constraint<T>],
    sk_weights: &[T],
    error_weights: &[T],
    p_terms: usize,
    q_free: usize,
) -> (DMatrix<T>, DVector<T>) {
    let n = data.len();
    let total_params = p_terms + q_free;
    let n_total = n + constraints.len();
    let mut mat = DMatrix::zeros(n_total, total_params);
    let mut rhs = DVector::zeros(n_total);

    for i in 0..n {
        let y = data[i].1;
        let w = sk_weights[i] * error_weights[i];

        for j in 0..p_terms {
            mat[(i, j)] = w * x_powers[i][j];
        }
        for j in 0..q_free {
            mat[(i, p_terms + j)] = -w * y * x_powers[i][j];
        }
        rhs[i] = w * y * x_powers[i][q_free];
    }

    for (ci, constraint) in constraints.iter().enumerate() {
        let row = n + ci;
        let cw = constraint.weight;
        for j in 0..p_terms {
            mat[(row, j)] = cw * c_powers[ci][j];
        }
        for j in 0..q_free {
            mat[(row, p_terms + j)] = -cw * constraint.y * c_powers[ci][j];
        }
        rhs[row] = cw * constraint.y * c_powers[ci][q_free];
    }

    (mat, rhs)
}

/// Fit P(x)/Q(x) using iteratively reweighted least squares.
///
/// The SK algorithm linearizes the rational fitting problem:
///   `P(x_i) - y_i * Q(x_i) = 0`
///
/// With Q monic (`q_n` = 1), this becomes a linear system in the free
/// parameters `[p_0, ..., p_m, q_0, ..., q_{n-1}]`. Each iteration
/// reweights by `1/Q(x_i)` from the previous solution, converging to
/// the true nonlinear least-squares solution.
fn fit_sk<T: Value>(
    data: &[(T, T)],
    p_terms: usize,
    q_terms: usize,
    max_iters: usize,
    tolerance: T,
    constraints: &[Constraint<T>],
    weighting: ErrorWeighting,
) -> Result<(RationalPolynomial<T>, usize)> {
    let n = data.len();
    let q_free = q_terms - 1;
    let total_params = p_terms + q_free;

    let max_power = p_terms.max(q_terms);
    let data_x: Vec<T> = data.iter().map(|&(x, _)| x).collect();
    let x_powers = precompute_powers(&data_x, max_power);

    let constraint_x: Vec<T> = constraints.iter().map(|c| c.x).collect();
    let c_powers = precompute_powers(&constraint_x, max_power);

    let error_weights: Vec<T> = match weighting {
        ErrorWeighting::Absolute => vec![T::one(); n],
        ErrorWeighting::Relative => data
            .iter()
            .map(|(_, y)| {
                let ay = Value::abs(*y);
                if ay.is_near_zero() {
                    T::one()
                } else {
                    T::one() / ay
                }
            })
            .collect(),
    };

    let mut sk_weights = vec![T::one(); n];
    let mut prev_params: Vec<T> = vec![T::zero(); total_params];
    let mut final_p = vec![T::zero(); p_terms];
    let mut final_q = vec![T::zero(); q_terms];
    let mut converged_iter = max_iters;

    for iter in 0..max_iters {
        let (mat, rhs) = build_sk_system(
            data,
            &x_powers,
            &c_powers,
            constraints,
            &sk_weights,
            &error_weights,
            p_terms,
            q_free,
        );

        let params = solve_svd(mat, &rhs)?;

        let p: Vec<T> = params[..p_terms].to_vec();
        let mut q: Vec<T> = params[p_terms..].to_vec();
        q.push(T::one());

        // Update SK weights: w_i = 1 / |Q(x_i)|
        for (i, sw) in sk_weights.iter_mut().enumerate() {
            let aqx = Value::abs(horner(&q, data[i].0));
            *sw = if aqx.is_near_zero() {
                T::try_cast(1e10).unwrap_or(T::one())
            } else {
                T::one() / aqx
            };
        }

        let max_change = params
            .iter()
            .zip(prev_params.iter())
            .map(|(&new, &old)| {
                let denom = Value::max(Value::abs(new), Value::abs(old));
                if denom.is_near_zero() {
                    T::zero()
                } else {
                    Value::abs(new - old) / denom
                }
            })
            .fold(T::zero(), Value::max);

        prev_params.clone_from(&params);
        final_p = p;
        final_q = q;

        if max_change < tolerance && iter > 0 {
            converged_iter = iter + 1;
            break;
        }
    }

    Ok((RationalPolynomial::new(final_p, final_q), converged_iter))
}

// =============================================================================
// Top-level fit: SK initialization → LM refinement
// =============================================================================

/// Fit P(x)/Q(x) using SK initialization, LM refinement, and multi-restart.
#[allow(clippy::too_many_arguments, clippy::assign_op_pattern)]
fn fit_rational<T: Value>(
    data: &[(T, T)],
    p_terms: usize,
    q_terms: usize,
    sk_iters: usize,
    lm_iters: usize,
    tolerance: T,
    constraints: &[Constraint<T>],
    weighting: ErrorWeighting,
    constraint_mode: ConstraintMode,
    restarts: usize,
) -> Result<(RationalPolynomial<T>, usize)> {
    let sk_constraints = match constraint_mode {
        ConstraintMode::SnapOnly => &[] as &[Constraint<T>],
        ConstraintMode::SkThenSnap | ConstraintMode::SkOnly | ConstraintMode::Both => constraints,
    };

    let lm_constraints: &[Constraint<T>] = match constraint_mode {
        ConstraintMode::Both => constraints,
        _ => &[],
    };

    let (base_poly, sk_used) = fit_sk(
        data,
        p_terms,
        q_terms,
        sk_iters,
        tolerance,
        sk_constraints,
        weighting,
    )?;

    if lm_iters == 0 {
        let mut poly = base_poly;
        if matches!(
            constraint_mode,
            ConstraintMode::SnapOnly | ConstraintMode::SkThenSnap
        ) {
            for c in constraints {
                poly.snap_boundary(c.x, c.y);
            }
        }
        return Ok((poly, sk_used));
    }

    // Run LM from SK result (restart 0)
    let mut best_poly = base_poly.clone();
    let mut best_lm = refine_lm(
        &mut best_poly,
        data,
        lm_constraints,
        weighting,
        lm_iters,
        tolerance,
    );
    let mut best_cost = compute_cost(
        data,
        lm_constraints,
        &pack_params(&best_poly),
        p_terms,
        weighting,
    );

    // Multi-restart: perturb SK result, re-run LM, keep best
    for r in 1..restarts {
        let mut poly = base_poly.clone();
        // Deterministic perturbation using sin/cos with irrational-ish constants
        let r_f64 = T::try_cast(r).unwrap_or(T::one());
        for (i, c) in poly.numerator.iter_mut().enumerate() {
            let phase = r_f64 * T::try_cast(7.319).unwrap_or(T::one())
                + T::try_cast(i).unwrap_or(T::zero()) * T::try_cast(2.903).unwrap_or(T::one());
            *c = *c * (T::one() + T::try_cast(0.05).unwrap_or(T::zero()) * phase.sin());
        }
        let q_free_len = poly.denominator.len() - 1;
        for (i, c) in poly.denominator[..q_free_len].iter_mut().enumerate() {
            let phase = r_f64 * T::try_cast(5.147).unwrap_or(T::one())
                + T::try_cast(i).unwrap_or(T::zero()) * T::try_cast(3.571).unwrap_or(T::one());
            *c = *c * (T::one() + T::try_cast(0.05).unwrap_or(T::zero()) * phase.sin());
        }

        let lm_used = refine_lm(
            &mut poly,
            data,
            lm_constraints,
            weighting,
            lm_iters,
            tolerance,
        );
        let cost = compute_cost(
            data,
            lm_constraints,
            &pack_params(&poly),
            p_terms,
            weighting,
        );

        if cost < best_cost {
            best_poly = poly;
            best_lm = lm_used;
            best_cost = cost;
        }
    }

    // Post-hoc snap
    if matches!(
        constraint_mode,
        ConstraintMode::SnapOnly | ConstraintMode::SkThenSnap
    ) {
        for c in constraints {
            best_poly.snap_boundary(c.x, c.y);
        }
    }

    Ok((best_poly, sk_used + best_lm))
}

/// Pack polynomial coefficients into a `DVector` for cost computation.
fn pack_params<T: Value>(poly: &RationalPolynomial<T>) -> DVector<T> {
    let q_free = poly.denominator.len() - 1;
    let total = poly.numerator.len() + q_free;
    let mut params = DVector::zeros(total);
    for (j, &c) in poly.numerator.iter().enumerate() {
        params[j] = c;
    }
    for (j, &c) in poly.denominator[..q_free].iter().enumerate() {
        params[poly.numerator.len() + j] = c;
    }
    params
}

// =============================================================================
// Levenberg-Marquardt refinement
// =============================================================================

/// Refine a rational polynomial using Levenberg-Marquardt with Nielsen damping.
///
/// Uses the analytical Jacobian of P(x)/Q(x):
///   `∂(P/Q)/∂p_j = x^j / Q(x)`
///   `∂(P/Q)/∂q_j = -(P/Q) · x^j / Q(x)`
#[allow(clippy::assign_op_pattern)]
fn refine_lm<T: Value>(
    poly: &mut RationalPolynomial<T>,
    data: &[(T, T)],
    constraints: &[Constraint<T>],
    weighting: ErrorWeighting,
    max_iters: usize,
    tolerance: T,
) -> usize {
    let p_terms = poly.numerator.len();
    let q_free = poly.denominator.len() - 1; // last is fixed at 1
    let total_params = p_terms + q_free;

    // Pack parameters into a single vector
    let mut params = DVector::zeros(total_params);
    for (j, &c) in poly.numerator.iter().enumerate() {
        params[j] = c;
    }
    for (j, &c) in poly.denominator[..q_free].iter().enumerate() {
        params[p_terms + j] = c;
    }

    let mut cost = compute_cost(data, constraints, &params, p_terms, weighting);

    // Initial λ: τ * max(diag(J^T J)) where J is data-only (no constraints)
    // to avoid constraint weights inflating the initial damping.
    let (_, j_init) = compute_residuals_jacobian(data, &[], &params, p_terms, weighting);
    let jtj_init = j_init.transpose() * &j_init;
    let tau = T::try_cast(1e-3).unwrap_or(T::one());
    let mut lambda = tau
        * (0..total_params)
            .map(|i| Value::abs(jtj_init[(i, i)]))
            .fold(T::zero(), Value::max);
    if lambda.is_near_zero() {
        lambda = T::try_cast(1e-6).unwrap_or(T::one());
    }
    drop(j_init);
    drop(jtj_init);

    let two = T::two();
    let lm_ftol = T::try_cast(1e-15).unwrap_or(tolerance);
    let lm_gtol = T::try_cast(1e-15).unwrap_or(tolerance);
    let mut nu = two; // damping scale factor (Nielsen update)
    let mut consecutive_rejects = 0_usize;
    let mut iters_used = 0;

    for _iter in 0..max_iters {
        iters_used += 1;

        let (residuals, jacobian) =
            compute_residuals_jacobian(data, constraints, &params, p_terms, weighting);
        let jtj = &jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        // Gradient convergence
        let grad_norm = jtr
            .iter()
            .map(|&g| Value::abs(g))
            .fold(T::zero(), Value::max);
        if grad_norm < lm_gtol {
            break;
        }

        // Solve (J^T J + λI) Δ = -J^T r
        let mut damped = jtj.clone();
        for i in 0..total_params {
            damped[(i, i)] += lambda;
        }

        let neg_jtr = -&jtr;
        let step = match solve_svd(damped, &neg_jtr) {
            Ok(s) => DVector::from_vec(s),
            Err(_) => break,
        };

        // Gain ratio: actual reduction / predicted reduction
        let trial = &params + &step;
        let trial_cost = compute_cost(data, constraints, &trial, p_terms, weighting);
        let actual_reduction = cost - trial_cost;

        // Predicted reduction: step^T (λ * step - J^T r) / 2
        // (from linearized model: L(Δ) ≈ r^T r + 2 Δ^T J^T r + Δ^T J^T J Δ)
        let pred_reduction = {
            let mut ls = &step * lambda;
            for i in 0..total_params {
                ls[i] = ls[i] - jtr[i];
            }
            step.dot(&ls) * T::half()
        };

        let gain = if pred_reduction.is_near_zero() {
            T::zero()
        } else {
            actual_reduction / pred_reduction
        };

        let three_quarter = T::try_cast(0.75).unwrap_or(T::one());

        if gain > T::zero() && actual_reduction > T::zero() {
            // Accept step
            params = trial;
            cost = trial_cost;
            consecutive_rejects = 0;

            // Nielsen update: reduce λ if gain is good
            if gain > three_quarter {
                let third = T::try_cast(1.0 / 3.0).unwrap_or(T::one());
                let g2 = two * gain - T::one();
                lambda = lambda * Value::max(third, T::one() - g2 * g2 * g2);
                nu = two;
            }

            // Convergence checks
            let rel_reduction =
                actual_reduction / Value::max(cost, T::try_cast(1e-30).unwrap_or(T::one()));
            if rel_reduction < lm_ftol {
                break;
            }
            let step_norm = step
                .iter()
                .map(|&s| Value::abs(s))
                .fold(T::zero(), Value::max);
            let param_norm = params
                .iter()
                .map(|&p| Value::abs(p))
                .fold(T::one(), Value::max);
            if step_norm / param_norm < lm_ftol {
                break;
            }
        } else {
            // Reject: increase λ, widen trust region
            lambda = lambda * nu;
            nu = nu * two;
            consecutive_rejects += 1;
            if consecutive_rejects > 30 {
                break;
            }
        }
    }

    // Unpack parameters back into polynomial
    for j in 0..p_terms {
        poly.numerator[j] = params[j];
    }
    for j in 0..q_free {
        poly.denominator[j] = params[p_terms + j];
    }
    // poly.denominator[q_free] stays 1.0

    iters_used
}

/// Compute total cost (sum of squared residuals).
fn compute_cost<T: Value>(
    data: &[(T, T)],
    constraints: &[Constraint<T>],
    params: &DVector<T>,
    p_terms: usize,
    weighting: ErrorWeighting,
) -> T {
    let q_free = params.len() - p_terms;
    let p: Vec<T> = (0..p_terms).map(|j| params[j]).collect();
    let mut q: Vec<T> = (0..q_free).map(|j| params[p_terms + j]).collect();
    q.push(T::one());

    let mut cost = T::zero();
    for &(x, y) in data {
        let pred = horner(&p, x) / horner(&q, x);
        let r = match weighting {
            ErrorWeighting::Absolute => pred - y,
            ErrorWeighting::Relative => {
                let ay = Value::abs(y);
                if ay.is_near_zero() {
                    pred - y
                } else {
                    (pred - y) / ay
                }
            }
        };
        cost += r * r;
    }
    for c in constraints {
        let pred = horner(&p, c.x) / horner(&q, c.x);
        let r = c.weight * (pred - c.y);
        cost += r * r;
    }
    cost
}

/// Compute residual vector and analytical Jacobian.
fn compute_residuals_jacobian<T: Value>(
    data: &[(T, T)],
    constraints: &[Constraint<T>],
    params: &DVector<T>,
    p_terms: usize,
    weighting: ErrorWeighting,
) -> (DVector<T>, DMatrix<T>) {
    let q_free = params.len() - p_terms;
    let total_params = params.len();
    let n_total = data.len() + constraints.len();

    let p: Vec<T> = (0..p_terms).map(|j| params[j]).collect();
    let mut q: Vec<T> = (0..q_free).map(|j| params[p_terms + j]).collect();
    q.push(T::one());

    let mut residuals = DVector::zeros(n_total);
    let mut jacobian = DMatrix::zeros(n_total, total_params);

    for (i, &(x, y)) in data.iter().enumerate() {
        let qx = horner(&q, x);
        let px = horner(&p, x);
        let pred = px / qx;
        let inv_q = T::one() / qx;

        let (r, w) = match weighting {
            ErrorWeighting::Absolute => (pred - y, T::one()),
            ErrorWeighting::Relative => {
                let ay = Value::abs(y);
                if ay.is_near_zero() {
                    (pred - y, T::one())
                } else {
                    ((pred - y) / ay, T::one() / ay)
                }
            }
        };

        residuals[i] = r;

        // ∂r/∂p_j = w * x^j / Q(x)
        let mut xj = T::one();
        for j in 0..p_terms {
            jacobian[(i, j)] = w * xj * inv_q;
            xj *= x;
        }

        // ∂r/∂q_j = -w * pred * x^j / Q(x)
        let mut xj = T::one();
        for j in 0..q_free {
            jacobian[(i, p_terms + j)] = -w * pred * xj * inv_q;
            xj *= x;
        }
    }

    for (ci, constraint) in constraints.iter().enumerate() {
        let row = data.len() + ci;
        let x = constraint.x;
        let cw = constraint.weight;
        let qx = horner(&q, x);
        let px = horner(&p, x);
        let pred = px / qx;
        let inv_q = T::one() / qx;

        residuals[row] = cw * (pred - constraint.y);

        let mut xj = T::one();
        for j in 0..p_terms {
            jacobian[(row, j)] = cw * xj * inv_q;
            xj *= x;
        }
        let mut xj = T::one();
        for j in 0..q_free {
            jacobian[(row, p_terms + j)] = -cw * pred * xj * inv_q;
            xj *= x;
        }
    }

    (residuals, jacobian)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Fit a simple polynomial (no denominator needed): y = x²
    #[test]
    fn fit_polynomial_trivial() {
        let data: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = f64::from(i) / 100.0;
                (x, x * x)
            })
            .collect();

        let fit = RationalFit::new(&data[..], 2, 0, RationalFitOptions::default()).unwrap();

        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());

        // P should be ~[0, 0, 1], Q should be ~[1]
        let (p, q) = fit.coefficients();
        assert!(p[0].abs() < 1e-6, "p[0] = {}", p[0]);
        assert!(p[1].abs() < 1e-6, "p[1] = {}", p[1]);
        assert!((p[2] - 1.0).abs() < 1e-6, "p[2] = {}", p[2]);
        assert!((q[0] - 1.0).abs() < 1e-6, "q[0] = {}", q[0]);
    }

    /// Fit x^2.4 on [0.04, 1.0] — the sRGB EOTF power segment
    #[test]
    fn fit_srgb_power() {
        let fit = RationalFit::from_function(
            |x: f64| x.powf(2.4),
            0.04..=1.0,
            4,
            4,
            RationalFitOptions::default(),
        )
        .unwrap();

        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());

        // Check max error is small
        let max_err = fit.max_abs_error();
        assert!(max_err < 1e-5, "max abs error = {max_err:.2e}");
    }

    /// Fit with relative error weighting
    #[test]
    fn fit_relative_error() {
        let options = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            ..RationalFitOptions::default()
        };

        let fit =
            RationalFit::from_function(|x: f64| x.powf(2.4), 0.04..=1.0, 4, 4, options).unwrap();

        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());
        assert!(
            fit.max_rel_error() < 1e-4,
            "max rel error = {:.2e}",
            fit.max_rel_error()
        );
    }

    /// Fit with a boundary constraint
    #[test]
    fn fit_with_constraint() {
        let target_x: f64 = 0.04;
        let target_y = target_x.powf(2.4);

        let options = RationalFitOptions {
            constraints: vec![Constraint::new(target_x, target_y)],
            ..RationalFitOptions::default()
        };

        let fit =
            RationalFit::from_function(|x: f64| x.powf(2.4), 0.04..=1.0, 4, 4, options).unwrap();

        // The constraint point should be very accurate
        let err_at_constraint = (fit.y(target_x) - target_y).abs();
        assert!(
            err_at_constraint < 1e-10,
            "error at constraint = {err_at_constraint:.2e}"
        );
    }

    /// Asymmetric degrees: 3/4
    #[test]
    fn fit_asymmetric_degrees() {
        let fit = RationalFit::from_function(
            |x: f64| x.powf(2.4),
            0.04..=1.0,
            3,
            4,
            RationalFitOptions::default(),
        )
        .unwrap();

        let (p, q) = fit.coefficients();
        assert_eq!(p.len(), 4); // 3 + 1
        assert_eq!(q.len(), 5); // 4 + 1
        assert!((q[4] - 1.0).abs() < 1e-15, "Q must be monic");
        assert!(fit.r_squared() > 0.9999, "R² = {}", fit.r_squared());
    }

    /// Fit log2(x) on [0.5, 2.0] — transcendental approximation
    #[test]
    fn fit_log2() {
        let fit = RationalFit::from_function(
            |x: f64| x.log2(),
            0.5..=2.0,
            3,
            3,
            RationalFitOptions::default(),
        )
        .unwrap();

        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());
        assert!(
            fit.max_abs_error() < 1e-6,
            "max abs error = {:.2e}",
            fit.max_abs_error()
        );
    }

    /// Fit 2^x on [0, 1] — transcendental approximation
    #[test]
    fn fit_exp2() {
        let fit = RationalFit::from_function(
            |x: f64| (2.0_f64).powf(x),
            0.0..=1.0,
            3,
            3,
            RationalFitOptions::default(),
        )
        .unwrap();

        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());
        assert!(
            fit.max_abs_error() < 1e-6,
            "max abs error = {:.2e}",
            fit.max_abs_error()
        );
    }

    /// Display formatting
    #[test]
    fn display_rational() {
        let poly = RationalPolynomial::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 1.0]);
        let s = format!("{poly}");
        assert!(s.contains("1.000000e0"), "display: {s}");
        assert!(s.contains('/'), "display: {s}");
    }

    /// R² of a perfect fit
    #[test]
    fn r_squared_perfect() {
        let data: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = f64::from(i) / 100.0 + 0.01;
                (x, 1.0 / x)
            })
            .collect();

        let fit = RationalFit::new(&data[..], 0, 1, RationalFitOptions::default()).unwrap();
        assert!(
            fit.r_squared() > 0.99999,
            "R² for 1/x with 0/1 rational = {}",
            fit.r_squared()
        );
    }

    /// Convergence tracking
    #[test]
    fn iterations_tracked() {
        let fit = RationalFit::from_function(
            |x: f64| x * x,
            0.0..=1.0,
            2,
            0,
            RationalFitOptions::default(),
        )
        .unwrap();

        // Should converge quickly for a trivial case
        assert!(fit.iterations() <= 50, "iterations = {}", fit.iterations());
    }

    /// f32 support
    #[test]
    fn fit_f32() {
        let data: Vec<(f32, f32)> = (1..=100)
            .map(|i| {
                let x = i as f32 / 100.0;
                (x, x.powf(2.4))
            })
            .collect();

        let options = RationalFitOptions::<f32>::default();
        let fit = RationalFit::new(&data[..], 4, 4, options).unwrap();
        assert!(fit.r_squared() > 0.9999, "R² = {}", fit.r_squared());
    }

    /// sRGB EOTF with boundary constraint — matches the Python fitting scripts
    #[test]
    fn fit_srgb_with_constraint() {
        let a: f64 = 0.055_010_718_947_586_6;
        let a1: f64 = 1.0 + a;
        let threshold_gamma: f64 = 12.92 * 0.003_041_282_560_127_521;
        let threshold_linear: f64 = 0.003_041_282_560_127_521;

        let s2l = |x: f64| ((x + a) / a1).powf(2.4);

        let options = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            constraints: vec![Constraint::new(threshold_gamma, threshold_linear)],
            n_samples: 5000,
            ..RationalFitOptions::default()
        };

        let fit = RationalFit::from_function(s2l, threshold_gamma..=1.0, 4, 4, options).unwrap();

        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());
        assert!(
            fit.max_rel_error() < 1e-4,
            "max rel = {:.2e}",
            fit.max_rel_error()
        );

        // Boundary constraint should be very tight
        let err = (fit.y(threshold_gamma) - threshold_linear).abs();
        assert!(err < 1e-10, "boundary err = {err:.2e}");
    }

    /// LM refinement improves constrained sRGB fit over SK alone
    #[test]
    fn lm_improves_constrained_srgb() {
        let a: f64 = 0.055_010_718_947_586_6;
        let a1: f64 = 1.0 + a;
        let threshold_gamma: f64 = 12.92 * 0.003_041_282_560_127_521;
        let threshold_linear: f64 = 0.003_041_282_560_127_521;
        let s2l = |x: f64| ((x + a) / a1).powf(2.4);

        let base_options = RationalFitOptions {
            weighting: ErrorWeighting::Relative,
            constraints: vec![Constraint::new(threshold_gamma, threshold_linear)],
            constraint_mode: ConstraintMode::Both,
            n_samples: 5000,
            ..RationalFitOptions::default()
        };

        // SK only (with constraints applied during SK)
        let sk_options = RationalFitOptions {
            lm_iterations: 0,
            constraint_mode: ConstraintMode::SkOnly,
            ..base_options.clone()
        };
        let sk_fit =
            RationalFit::from_function(s2l, threshold_gamma..=1.0, 4, 4, sk_options).unwrap();

        // SK + LM (both with constraints)
        let lm_fit =
            RationalFit::from_function(s2l, threshold_gamma..=1.0, 4, 4, base_options).unwrap();

        let sk_err = sk_fit.max_rel_error();
        let lm_err = lm_fit.max_rel_error();

        // LM should significantly improve the constrained case
        assert!(
            lm_err < sk_err,
            "LM should improve: SK={sk_err:.2e}, LM={lm_err:.2e}"
        );
        assert!(
            lm_err < 1e-4,
            "LM should reach <1e-4 relative error, got {lm_err:.2e}"
        );
    }

    /// Boundary error report
    #[test]
    fn boundary_report() {
        let fit = RationalFit::from_function(
            |x: f64| x.powf(2.4),
            0.04..=1.0,
            4,
            4,
            RationalFitOptions::default(),
        )
        .unwrap();

        let report = fit.boundary_error(|x| x.powf(2.4));

        // At domain endpoints, error should be small
        assert!(
            report.lo.rel_err < 1e-3,
            "lo rel err = {:.2e}",
            report.lo.rel_err
        );
        assert!(
            report.hi.rel_err < 1e-4,
            "hi rel err = {:.2e}",
            report.hi.rel_err
        );

        // Extrapolation points are populated
        assert_eq!(report.past_lo.len(), 4);
        assert_eq!(report.past_hi.len(), 4);

        // Extrapolation points are at the expected offsets
        let width = 1.0 - 0.04;
        assert!(
            (report.past_lo[0].x - (0.04 - width * 0.001)).abs() < 1e-10,
            "past_lo[0] at wrong x"
        );
        assert!(
            (report.past_hi[0].x - (1.0 + width * 0.001)).abs() < 1e-10,
            "past_hi[0] at wrong x"
        );

        // Display works without panic
        let s = format!("{report}");
        assert!(s.contains("Boundary error"), "display: {s}");
        assert!(s.contains("Extrapolation past lo"), "display: {s}");
    }

    /// Self-consistency: `optimize_f32`'s reported metrics must match an
    /// independent recomputation over the same sweep. This guards against
    /// regressions where the scorer and the final measurement disagree.
    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_lossless,
        clippy::cast_precision_loss
    )]
    fn f32_search_result_is_self_consistent() {
        let fit = RationalFit::from_function(
            |x: f64| x.powf(2.4),
            0.04..=1.0,
            4,
            4,
            RationalFitOptions {
                weighting: ErrorWeighting::Relative,
                n_samples: 5000,
                restarts: 4,
                ..RationalFitOptions::default()
            },
        )
        .unwrap();

        let result = fit.as_polynomial().optimize_f32(
            |x: f64| x.powf(2.4),
            0.04_f32,
            1.0_f32,
            |v| v,
            F32SearchConfig {
                samples: Some(50_000),
                ..F32SearchConfig::default()
            },
        );

        // Independently recompute the metrics from the returned coefficients.
        // MUST use the same sampling convention as the library:
        // lo/hi are promoted via f64::from(f32), not the f64 literal.
        let mut max_ulp = 0_u32;
        let mut max_abs: f64 = 0.0;
        let n = 50_000;
        let lo = f64::from(0.04_f32);
        let hi = f64::from(1.0_f32);
        for i in 0..=n {
            let v = lo + (hi - lo) * i as f64 / n as f64;
            let got = eval_f32_horner(v as f32, &result.numerator, &result.denominator);
            let expected = v.powf(2.4) as f32;
            let ulp = f32_ulp_distance(got, expected);
            if ulp > max_ulp {
                max_ulp = ulp;
            }
            let a = (f64::from(got) - f64::from(expected)).abs();
            if a > max_abs {
                max_abs = a;
            }
        }

        assert_eq!(
            result.max_ulp, max_ulp,
            "reported max_ulp {} does not match independent recomputation {}",
            result.max_ulp, max_ulp
        );
        assert!(
            (result.max_abs_err - max_abs).abs() < 1e-15 * max_abs.max(1e-30),
            "reported max_abs {} differs from recomputation {}",
            result.max_abs_err,
            max_abs
        );
        assert_eq!(result.values_scored, u64::try_from(n + 1).unwrap());
        assert!(!result.exhaustive);
    }

    /// Piecewise gap detects the discontinuity between linear and polynomial
    /// branches, which `boundary_ulp` alone misses.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn piecewise_gap_matching_branches_is_small() {
        // Build a contrived piecewise: f(v) = v on both sides of threshold.
        // The polynomial fit to f(v)=v on [0.5, 1] should produce a small gap
        // to the identity linear branch v.
        let fit = RationalFit::from_function(
            |x: f64| x,
            0.5..=1.0,
            2,
            2,
            RationalFitOptions {
                weighting: ErrorWeighting::Absolute,
                n_samples: 1000,
                restarts: 2,
                ..RationalFitOptions::default()
            },
        )
        .unwrap();

        let (p64, q64) = fit.coefficients();
        let p: Vec<f32> = p64.iter().map(|&c| c as f32).collect();
        let q: Vec<f32> = q64.iter().map(|&c| c as f32).collect();

        // Gap between identity branch and polynomial approximation of identity.
        // Should be small (≤ a few ULPs) since both branches compute the same thing.
        let gap = piecewise_gap(0.5_f32, &p, &q, |v| v, |v| v);
        assert!(
            gap < 100,
            "piecewise gap for matching identity branches: {gap}"
        );
    }

    /// Piecewise gap correctly identifies a deliberately mismatched pair
    /// (documents the diagnostic value of the metric).
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn piecewise_gap_detects_mismatched_branches() {
        // Fit f(v) = 2v on [0.5, 1] but pretend the linear branch is v (wrong).
        let fit = RationalFit::from_function(
            |x: f64| 2.0 * x,
            0.5..=1.0,
            2,
            2,
            RationalFitOptions {
                weighting: ErrorWeighting::Absolute,
                n_samples: 1000,
                restarts: 2,
                ..RationalFitOptions::default()
            },
        )
        .unwrap();

        let (p64, q64) = fit.coefficients();
        let p: Vec<f32> = p64.iter().map(|&c| c as f32).collect();
        let q: Vec<f32> = q64.iter().map(|&c| c as f32).collect();

        // Gap between identity branch (wrong) and polynomial (2v)
        // should be large — that's the whole point of the metric.
        let gap = piecewise_gap(0.5_f32, &p, &q, |v| v, |v| v);
        assert!(
            gap > 1_000_000,
            "piecewise gap should flag mismatched branches, got {gap}"
        );
    }

    /// Roundtrip measurement returns consistent metrics.
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
    fn roundtrip_measurement_consistent() {
        // Trivial identity roundtrip: f(v)=v, inverse(u)=u → 0 error everywhere.
        let report = measure_roundtrip(|v| v, |u| u, 0.0, 1.0, Some(1000));
        assert_eq!(report.max_ulp, 0);
        assert!(report.max_abs_err == 0.0);
        assert_eq!(report.u16_over_1, 0);
        assert_eq!(report.u8_over_half, 0);
        assert_eq!(report.values_scored, 1001);
        assert!(!report.exhaustive);

        // Forward then inverse with f32 rounding introduces some error.
        let report2 = measure_roundtrip(
            |v: f32| (f64::from(v) * 2.0) as f32,
            |u: f32| (f64::from(u) * 0.5) as f32,
            0.01_f32,
            1.0_f32,
            Some(1000),
        );
        // The composition is exact for these doubling/halving operations when
        // f32 stays in [0.01, 1.0]. Max error should be tiny.
        assert!(report2.max_abs_err < 1e-6);

        // Display doesn't panic.
        let _ = format!("{report}");
    }

    /// `Constraint::soft` is a direct alias for `with_weight` — both
    /// produce identical constraints with identical fit behavior. New
    /// callers should prefer `soft` because the name doesn't invite the
    /// misreading "high weight = hard pin."
    #[test]
    fn constraint_soft_is_with_weight_alias() {
        let a = Constraint::<f64>::soft(0.5, 0.25, 1e3);
        let b = Constraint::<f64>::with_weight(0.5, 0.25, 1e3);
        assert_eq!(a.x.to_bits(), b.x.to_bits());
        assert_eq!(a.y.to_bits(), b.y.to_bits());
        assert_eq!(a.weight.to_bits(), b.weight.to_bits());

        // And the new constructor produces a usable constraint that the
        // fitter accepts end-to-end (smoke check — does not assert exact
        // boundary match because soft constraints don't pin).
        let options = RationalFitOptions {
            constraints: vec![Constraint::soft(0.04, (0.04_f64).powf(2.4), 1e6)],
            ..RationalFitOptions::default()
        };
        let fit =
            RationalFit::from_function(|x: f64| x.powf(2.4), 0.04..=1.0, 4, 4, options).unwrap();
        assert!(fit.r_squared() > 0.999_999, "R² = {}", fit.r_squared());
    }
}
