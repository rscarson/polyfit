//! A scoring method that adds a penalty for smoothness and monotonicity, in addition to the fit quality measured by RMSE.
//! - Use this if you want to select a model that not only fits the data well, but also has a smoother curve and/or is monotonic.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use num_traits::Zero;

use crate::{
    basis::{Basis, DifferentialBasis},
    display::PolynomialDisplay,
    score::{ModelScoreProvider, RMSE},
    value::Value,
    CurveFit,
};

/// Strength of the penalty to apply for curvature and non-monotonicity when using `ShapeConstraint`.
///
/// This is a simple enum to make it easier to choose a penalty strength without having to guess at specific lambda values.
/// The `Custom` variant allows you to specify an exact lambda value if you want more control.
///
/// - None: I don't care about this at all, just give me the best fit according to RMSE (See [`RMSE`])
/// - Small: A small penalty that will slightly favor smoother/monotonic models, but won't override a significantly better fit.
/// - Medium: A moderate penalty that will favor smoother/monotonic models, but will still allow a more complex model if it provides a noticeably better fit.
/// - Large: A strong penalty that will heavily favor smoother/monotonic models, and will only allow a more complex model if it provides a much better fit.
/// - Hard: Basically a hard constraint - the model must be perfectly smooth/monotonic to avoid an extremely large penalty.
/// - Custom: These didn't work and I want to find a better value by trial and error, so let me specify an exact lambda value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyWeight {
    /// No penalty - just use RMSE to select the best fit.
    None,

    /// A small penalty (1e-6) that will slightly favor smoother/monotonic models, but won't override a significantly better fit.
    Small,

    /// A moderate penalty (1e-4) that will favor smoother/monotonic models, but will still allow a more complex model if it provides a noticeably better fit.
    Medium,

    /// A strong penalty (1e-2) that will heavily favor smoother/monotonic models, and will only allow a more complex model if it provides a much better fit.
    Large,

    /// Basically a hard constraint (1e6) - the model must be perfectly smooth/monotonic to avoid an extremely large penalty.
    Hard,

    /// These didn't work and I want to find a better value by trial and error, so let me specify an exact lambda value.
    Custom(f64),
}
impl PenaltyWeight {
    fn to_lambda(self) -> f64 {
        match self {
            PenaltyWeight::None => 0.0,
            PenaltyWeight::Small => 1e-6,
            PenaltyWeight::Medium => 1e-4,
            PenaltyWeight::Large => 1e-2,
            PenaltyWeight::Hard => 1e6,
            PenaltyWeight::Custom(value) => value,
        }
    }
}
impl From<f64> for PenaltyWeight {
    fn from(value: f64) -> Self {
        PenaltyWeight::Custom(value)
    }
}

/// Strategy for sampling points along the curve when calculating curvature and monotonicity penalties in `ShapeConstraint`.
///
/// This determines how many points along the curve are evaluated to estimate the curvature and monotonicity, which in turn affects the smoothness penalty applied to the model score
///
/// - Percentage: Sample a percentage of the total number of data points (e.g., 10% of the data points).
/// - Count: Sample a fixed number of points (e.g., 100 points evenly spaced along the x-range).
/// - Total: Sample points at every data point (i.e., the same number of samples as the original data points).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    /// Sample a percentage of the total number of data points (e.g., 10% of the data points).
    Percentage(f64),

    /// Sample a fixed number of points (e.g., 100 points evenly spaced along the x-range).
    Count(usize),

    /// Sample points at every data point (i.e., the same number of samples as the original data points).
    Total,
}
impl SamplingStrategy {
    /// Creates a new `SamplingStrategy` that samples a percentage of the total number of data points (0 - 1)
    #[must_use]
    pub fn new_percentage(percent: f64) -> Self {
        SamplingStrategy::Percentage(percent.clamp(0.0, 1.0))
    }

    /// Creates a new `SamplingStrategy` that samples a fixed number of points (e.g., 100 points evenly spaced along the x-range).
    #[must_use]
    pub fn new_count(count: usize) -> Self {
        SamplingStrategy::Count(count)
    }

    /// Creates a new `SamplingStrategy` that samples points at every data point (i.e., the same number of samples as the original data points).
    #[must_use]
    pub fn new_total() -> Self {
        SamplingStrategy::Total
    }

    /// Calculates the number of samples to use based on the strategy and the length of the data.
    #[must_use]
    pub fn count(&self, data_len: usize) -> usize {
        match self {
            SamplingStrategy::Percentage(percent) => (percent * (data_len as f64)).round() as usize,
            SamplingStrategy::Count(count) => (*count).clamp(1, data_len),
            SamplingStrategy::Total => data_len,
        }
    }
}

/// Direction of monotonicity to apply when calculating the monotonicity penalty in `ShapeConstraint`.
///
/// Infer will automatically determine the direction based on the overall trend of the data.
///
/// Increasing will apply a penalty for any negative slopes, while Decreasing will apply a penalty for any positive slopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicityDirection {
    /// Infer will automatically determine the direction based on the overall trend of the data.
    Infer,

    /// Increasing will apply a penalty for any negative slopes, while Decreasing will apply a penalty for any positive slopes.
    Increasing,

    /// Decreasing will apply a penalty for any positive slopes, while Increasing will apply a penalty for any negative slopes.
    Decreasing,
}

/// Scoring method that adds a penalty for smoothness and monotonicity, in addition to the fit quality measured by RMSE.
/// - Use this if you want to select a model that not only fits the data well, but also has a smoother curve and/or is monotonic.
///
/// This is useful for specific data sets with known properties, such as growth curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeConstraint {
    lambda_curvature: f64,
    lambda_monotonic: f64,
    monotonic_direction: MonotonicityDirection,
    samples: SamplingStrategy,
}
impl ShapeConstraint {
    /// Creates a new `ShapeConstraint` scoring method with the specified sampling strategy and no penalties for curvature or monotonicity.
    ///
    /// Until you call `with_curvature_penalty` or `with_monotonic_penalty`, this scoring method is equivalent to just using RMSE
    #[must_use]
    pub fn new(sampling_strategy: SamplingStrategy) -> Self {
        Self {
            lambda_curvature: 0.0,
            lambda_monotonic: 0.0,
            monotonic_direction: MonotonicityDirection::Infer,
            samples: sampling_strategy,
        }
    }

    /// Add a curvature penalty to this `ShapeConstraint` scoring method, with the specified strength, to favor smoother curves.
    ///
    /// # Returns
    /// A new `ShapeConstraint` scoring method with the specified curvature penalty
    ///
    /// # Example
    /// ```
    /// # use polyfit::{score::shape_constraint::*, value::Value, ChebyshevFit, statistics::DegreeBound};
    /// let data = &[(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)];
    /// let score = ShapeConstraint::new(SamplingStrategy::new_total()).with_curvature_penalty(PenaltyWeight::Medium);
    /// let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, &score).unwrap();
    /// ```
    #[must_use]
    pub fn with_curvature_penalty(mut self, curvature_penalty: impl Into<PenaltyWeight>) -> Self {
        self.lambda_curvature = curvature_penalty.into().to_lambda();
        self
    }

    /// Add a monotonicity penalty to this `ShapeConstraint` scoring method, with the specified strength and direction, to favor monotonic curves.
    /// The direction of monotonicity can be set to `Infer` to automatically determine the direction based on the overall trend of the data.
    /// # Returns
    /// A new `ShapeConstraint` scoring method with the specified monotonicity penalty
    ///
    /// # Example
    /// ```
    /// # use polyfit::{score::shape_constraint::*, value::Value, ChebyshevFit, statistics::DegreeBound};
    /// let data = &[(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)];
    /// let score = ShapeConstraint::new(SamplingStrategy::new_total()).with_monotonic_penalty(PenaltyWeight::Medium, MonotonicityDirection::Infer);
    /// let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, &score).unwrap();
    /// ```
    #[must_use]
    pub fn with_monotonic_penalty(
        mut self,
        monotonic_penalty: impl Into<PenaltyWeight>,
        direction: MonotonicityDirection,
    ) -> Self {
        self.lambda_monotonic = monotonic_penalty.into().to_lambda();
        self.monotonic_direction = direction;
        self
    }
}
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T> for ShapeConstraint
where
    B: DifferentialBasis<T>,
    B::B2: DifferentialBasis<T>,
{
    fn minimum_significant_distance(&self) -> Option<usize> {
        None
    }

    fn score(
        &self,
        model: &CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> T {
        let base_score = RMSE.score(model, y, y_fit, k);

        let range = model.x_range();
        let min = *range.start();
        let max = *range.end();

        let x_range = max - min;
        let y_range = model.y_range();
        let y_range = *y_range.end() - *y_range.start();

        let mut curvature = T::zero();
        let mut monotonic = T::zero();

        let d1 = model.as_polynomial().derivative().unwrap();
        let d2 = d1.derivative().unwrap();

        let samples = self.samples.count(model.data().len());
        if samples < 2 {
            // Not enough samples to calculate curvature/monotonicity, so just return the base score
            return base_score;
        }

        let mono_epsilon = Value::abs(max - min) * T::from_f64(1e-8).unwrap_or(T::epsilon());
        let stepsize = (max - min) / T::from_positive_int(samples - 1);

        let mut monotonic_direction = self.monotonic_direction;
        if !self.lambda_monotonic.is_zero() && monotonic_direction == MonotonicityDirection::Infer {
            let (mut pos, mut neg) = (0, 0);
            for i in 0..samples {
                let x = min + T::from_positive_int(i) * stepsize;
                if x > max {
                    // Just in case
                    break;
                }

                let v1 = d1.y(x);
                if v1 > mono_epsilon {
                    pos += 1;
                } else if v1 < -mono_epsilon {
                    neg += 1;
                }
            }

            monotonic_direction = if pos >= neg {
                MonotonicityDirection::Increasing
            } else {
                MonotonicityDirection::Decreasing
            };
        }

        for i in 0..samples {
            let x = min + T::from_positive_int(i) * stepsize;
            if x > max {
                // Just in case
                break;
            }

            let mut v1 = d1.y(x);
            let mut v2 = d2.y(x);

            //
            // Ok So RMSE is in y units
            // curvature is in y^2 / x^2
            // monotonic violation is in y^2 / x units
            // We need to make everything in y
            v1 *= x_range / y_range;
            v2 *= x_range * x_range / y_range;

            // curvature penalty
            curvature += v2 * v2;

            // monotonic penalty
            let violation = match monotonic_direction {
                MonotonicityDirection::Increasing => Value::min(v1, T::zero()),
                MonotonicityDirection::Decreasing => Value::max(v1, T::zero()),
                MonotonicityDirection::Infer => unreachable!(),
            };
            if Value::abs(violation) > mono_epsilon {
                monotonic += violation * violation;
            }
        }

        // * stepsize here approximates a riemann sum, so the penalty is roughly proportional to the integral of the curvature/monotonicity violation across the curve
        // RMSE is already normalized by the number of data points, so we dont need to worry about that here
        let curvature_penalty = T::from_f64(self.lambda_curvature).unwrap_or(T::zero()) * stepsize;
        let monotonic_penalty = T::from_f64(self.lambda_monotonic).unwrap_or(T::zero()) * stepsize;
        base_score + curvature_penalty * curvature + monotonic_penalty * monotonic
    }
}
