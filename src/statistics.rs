//! Functions and tools for evaluating polynomial fits and scoring models
//!
//! This module provides functions and types to evaluate how well a polynomial model
//! fits a dataset, and to score models when automatically selecting polynomial degrees.
//!
//! # Model Fit / Regression Diagnostics
//! - [`r_squared`]: Proportion of variance explained by the model. Higher is better (0 to 1).
//! - [`adjusted_r_squared`]: R² adjusted for number of predictors. Use to compare models of different degrees.
//! - [`residual_variance`]: Unbiased estimate of variance of errors after fitting. Used for confidence intervals.
//! - [`residual_normality`]: Likelihood that the residuals are normally distributed. Results near 0 or 1 indicate non-normality, higher results do not guarantee normality.
//!
//! # Confidence Intervals
//! - [`ConfidenceBand`]: Represents a confidence interval with lower and upper bounds, determined by a given probability.
//! - [`Confidence`]: Enum for common confidence levels (68%, 95%, 99%).
//!
//! # Model Selection
//! - [`DegreeBound`]: Enum to specify constraints on polynomial degree when automatically selecting it.
//!
//! # Error Metrics
//! - [`mean_absolute_error`]: Average absolute difference between observed and predicted values. Lower is better.
//! - [`mean_squared_error`]: Average squared difference between observed and predicted values. Lower is better.
//! - [`root_mean_squared_error`]: Square root of MSE, giving error in same units as observed values. Lower is better.
//! - [`huber_log_likelihood`]: Robust error metric less sensitive to outliers. Higher is better.
//!
//! # Descriptive Statistics
//! - [`stddev_and_mean`]: Standard deviation and mean of a dataset.
//! - [`median_absolute_deviation`]: Average absolute deviation from the mean.
//! - [`spread`]: Difference between maximum and minimum values in a dataset.
//! - [`skewness_and_kurtosis`]: Measures of asymmetry and "tailedness" of the distribution.
//!
//! # Model Fit vs Model Selection
//!
//! - **Model Fit**: How well does the model explain the data? Use [`r_squared`] or [`residual_variance`].
//!   - Returns a value between 0 and 1.
//!   - 0 = model explains none of the variance.
//!   - 1 = model perfectly fits the data.
//!
//! - **Model selection**: Choosing the best polynomial degree to avoid overfitting.
//!   - Use [`crate::score`].
//!   - Options:
//!     - `AIC`: Akaike Information Criterion, more lenient penalty for complexity.
//!     - `BIC`: Bayesian Information Criterion, stricter penalty for complexity.
//!   - Lower scores are better; but not a measure of goodness-of-fit outside the context of model selection.
//!
//! # Examples
//!
//! ```rust
//! use polyfit::statistics::{DegreeBound, r_squared};
//! use polyfit::score::Aic;
//! use polyfit::ChebyshevFit;
//!
//! let data = &[(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)];
//! let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, &Aic).unwrap();
//!
//! // Goodness-of-fit
//! // R² is between 0 and 1, with higher values indicating a better fit.
//! // Not to be confused with AIC/BIC scores, which are only meaningful for comparing models of different degrees.
//! let r2 = fit.r_squared(None);
//! println!("R² = {r2}");
//!
//! // Model scoring
//! // AIC/BIC scores are only meaningful for comparing models of different degrees, and lower is better.
//! // Not an objective measure of fit quality outside the context of model selection.
//! let score = fit.model_score(&Aic);
//! println!("AIC score = {score}");
//! ```
//!

pub(crate) mod accumulator;

mod degree_bound;
pub use degree_bound::*;

mod metrics;
pub use metrics::*;

mod is_derivative;
pub use is_derivative::*;

mod confidence;
pub use confidence::*;

mod domain_normalizer;
pub use domain_normalizer::*;

mod k_fold;
pub use k_fold::*;

mod uncertain_value;
pub use uncertain_value::*;

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn residual_variance_zero_error() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 1.0).unwrap();
        assert_eq!(var, 0.0);
    }

    #[test]
    fn residual_variance_simple_case() {
        // y = [1, 2], y_fit = [0, 0], k = 1
        // errors: [1, 2], squared = [1, 4], sum = 5
        // n=2, n-k=1, variance = 5
        let y = vec![1.0, 2.0];
        let y_fit = vec![0.0, 0.0];
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 1.0).unwrap();
        assert_eq!(var, 5.0);
    }

    #[test]
    fn residual_variance_mse_equivalence() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![0.0, 0.0, 0.0];
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 0.0).unwrap();

        // manual MSE: (1² + 2² + 3²) / 3 = 14/3
        assert!((var - 14.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn residual_variance_invalid_degrees_of_freedom() {
        let y = vec![1.0, 2.0];
        let y_fit = vec![1.0, 2.0];
        // n = 2, k = 2 → division by zero
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 2.0);
        assert!(var.is_none());
    }

    #[test]
    fn r_squared_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(r2, 1.0);
    }

    #[test]
    fn r_squared_bad_fit() {
        // y = [1, 2, 3], y_fit = [2, 2, 2]
        // mean(y) = 2
        // SST = (1-2)² + (2-2)² + (3-2)² = 2
        // SSE = same, 2
        // R² = 1 - SSE/SST = 0
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn r_squared_partial_fit() {
        // y = [1, 2, 3], y_fit = [0, 2, 4]
        // mean(y) = 2
        // SST = (1-2)² + (2-2)² + (3-2)² = 2
        // SSE = (1-0)² + (2-2)² + (3-4)² = 2
        // R² = 0
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![0.0, 2.0, 4.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn r_squared_negative_case() {
        // y = [1, 2, 3], y_fit = [10, 10, 10]
        // mean(y) = 2, SST = 2
        // SSE = (1-10)² + (2-10)² + (3-10)² = 9² + 8² + 7² = 194
        // R² = 1 - 194/2 = -96
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![10.0, 10.0, 10.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(r2, -96.0);
    }

    #[test]
    fn r_squared_constant_y() {
        let y = vec![2.0, 2.0, 2.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(r2, 1.0);

        let y = vec![2.0, 2.0, 2.0];
        let y_fit = vec![20.0, 20.0, 20.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(r2, f64::NEG_INFINITY);
    }

    #[test]
    fn adjusted_r_squared_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = y.clone(); // perfect fit
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 2.0).unwrap();
        assert_eq!(r2_adj, 1.0);
    }

    #[test]
    fn adjusted_r_squared_equals_r2_when_k1() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y_fit = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let r2 = r_squared(y.clone().into_iter(), y_fit.clone().into_iter()).unwrap();
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 1.0).unwrap();
        assert!((r2 - r2_adj).abs() < 1e-12);
    }

    #[test]
    fn adjusted_r_squared_penalizes_complexity() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = vec![1.1, 1.9, 3.2, 3.8]; // decent fit
        let r2 = r_squared(y.clone().into_iter(), y_fit.clone().into_iter()).unwrap();
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 3.0).unwrap();
        assert!(r2_adj < r2);
    }

    #[test]
    fn adjusted_r_squared_negative_case() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = vec![10.0, 10.0, 10.0, 10.0]; // terrible fit
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 2.0).unwrap();
        assert!(r2_adj < 0.0);
    }

    #[test]
    fn adjusted_r_squared_invalid_degrees_of_freedom() {
        let y = vec![1.0, 2.0];
        let y_fit = vec![1.0, 2.0];
        // n = 2, k = 2 → division by zero
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 2.0);
        assert!(r2_adj.is_none());
    }

    #[test]
    fn mae_zero_error() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(mae, 0.0);
    }

    #[test]
    fn mae_simple_case() {
        // y = [1, 2, 3], y_fit = [2, 2, 2]
        // abs diffs = [1, 0, 1], sum = 2, mean = 2/3
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert!((mae - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mae_symmetric() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mae1 =
            mean_absolute_error::<f64>(y.clone().into_iter(), y_fit.clone().into_iter()).unwrap();
        let mae2 = mean_absolute_error::<f64>(y_fit.into_iter(), y.into_iter()).unwrap();
        assert_eq!(mae1, mae2);
    }

    #[test]
    fn mae_with_negatives() {
        // y = [-1, -2], y_fit = [1, 2]
        // diffs = [2, 4], mean = 3
        let y = vec![-1.0, -2.0];
        let y_fit = vec![1.0, 2.0];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(mae, 3.0);
    }

    #[test]
    fn mae_empty_input() {
        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert!(mae.is_none());
    }

    #[test]
    fn mse_zero_error() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn mse_simple_case() {
        // y = [1, 2, 3], y_fit = [2, 2, 2]
        // squared diffs = [1, 0, 1], sum = 2, mean = 2/3
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert!((mse - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mse_with_negatives() {
        // y = [-1, -2], y_fit = [1, 2]
        // diffs = [-2, -4], squared = [4, 16], mean = 10
        let y = vec![-1.0, -2.0];
        let y_fit = vec![1.0, 2.0];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_eq!(mse, 10.0);
    }

    #[test]
    fn mse_symmetric() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mse1 =
            mean_squared_error::<f64>(y.clone().into_iter(), y_fit.clone().into_iter()).unwrap();
        let mse2 = mean_squared_error::<f64>(y_fit.into_iter(), y.into_iter()).unwrap();
        assert_eq!(mse1, mse2);
    }

    #[test]
    fn mse_empty_input_returns_nan() {
        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert!(mse.is_none());
    }
}
