//! Scoring methods for model selection.
//!
//! These methods help choose the best polynomial degree by balancing fit quality and model complexity.
//! They are not measures of fit quality themselves; for that, use metrics like R².
//!
//! # Overview of Available Scoring Methods
//! - **Akaike Information Criterion (AIC)**: A commonly used method that balances fit quality and complexity. It tends to favor slightly more complex models if they provide a better fit.
//! - **Bayesian Information Criterion (BIC)**: Similar to AIC but applies a stricter penalty for model complexity. It often prefers simpler models, even if the fit is slightly worse.
//!
//! The [`ModelScoreProvider`] trait defines the interface for implementing custom scoring methods.
use crate::{statistics, value::Value};

/// Trait for implementing scoring methods for model selection.
pub trait ModelScoreProvider {
    /// Calculate the model's score using this scoring method.
    ///
    /// # Notes
    /// - Lower scores indicate a "better" choice for automatically selecting the polynomial degree.
    /// - This is **not** a measure of how well the model fits your data. For that, use `r_squared`.
    ///
    /// # Type Parameters
    /// - `T`: A numeric type implementing the `Value` trait.
    ///
    /// # Parameters
    /// - `y`: Iterator over the observed (actual) values.
    /// - `y_fit`: Iterator over the predicted values from the model.
    /// - `k`: Number of model parameters (degrees of freedom used by the fit).
    ///
    /// # Returns
    /// The computed score as a `T`.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{score::{Aic, ModelScoreProvider}, value::Value};
    /// # let y = vec![1.0, 2.0, 3.0];
    /// # let y_fit = vec![1.1, 1.9, 3.05];
    /// let score = Aic.score(y.into_iter(), y_fit.into_iter(), 3.0);
    /// ```
    fn score<T: Value>(
        &self,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> T;
}

/// Bayesian Information Criterion. Uses a stricter penalty for model complexity.
/// - Prefers simpler models, even if the fit is slightly worse.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// BIC is calculated as:
/// ```math
/// BIC = n * ln(L) + k * ln(n)
/// where
///   L = likelihood of the model (using Huber loss)
///   n = number of observations, k = number of model parameters
/// ```
/// </div>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bic;
impl ModelScoreProvider for Bic {
    fn score<T: Value>(
        &self,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> T {
        let (log_likelihood, n) = statistics::robust_mse_with_n(y, y_fit);
        let log_likelihood = nalgebra::RealField::max(log_likelihood, T::epsilon());
        if n == T::zero() {
            return T::nan();
        }

        n * log_likelihood.ln() + k * n.ln()
    }
}

/// Akaike Information Criterion. Uses a more lenient penalty for model complexity
/// - Picks a slightly more complex model if it fits better.
///
/// This is the default choice for most applications.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// AIC is calculated as:
/// ```math
/// AIC = { n * ln(L) + 2k
///       { n * ln(L) + 2k + (2k(k+1)) / (n - k - 1)  if (n / k < 2 + 2) AND (n > k + 1)
/// where
///   L = likelihood of the model (using Huber loss)
///   n = number of observations, k = number of model parameters
/// ```
/// </div>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aic;
impl ModelScoreProvider for Aic {
    fn score<T: Value>(
        &self,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> T {
        let (log_likelihood, n) = statistics::robust_mse_with_n(y, y_fit);
        let log_likelihood = nalgebra::RealField::max(log_likelihood, T::epsilon());
        if n == T::zero() {
            return T::nan();
        }

        let mut aic = n * log_likelihood.ln() + T::two() * k;
        if n / k < T::two() + T::two() && n > k + T::one() {
            // Apply AICc correction
            aic += T::two() * k * (k + T::one()) / (n - k - T::one());
        }

        aic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = y.clone();
        let k = 2.0;
        let aic: f64 = Aic.score(y.clone().into_iter(), y_fit.clone().into_iter(), k);
        let bic: f64 = Bic.score(y.into_iter(), y_fit.into_iter(), k);
        assert!(aic.is_finite());
        assert!(bic.is_finite());
    }

    #[test]
    fn scoring_aicc_correction() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.8, 2.7, 3.6];
        let k = 2.0; // n/k = 1.5 < 4 → triggers correction
        let score = Aic.score::<f64>(y.into_iter(), y_fit.into_iter(), k);
        assert!(score.is_finite());
    }

    #[test]
    fn scoring_higher_error_higher_score() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit_good = vec![1.0, 2.0, 3.0];
        let y_fit_bad = vec![0.0, 0.0, 0.0];
        let k = 2.0;
        let score_good = Bic.score::<f64>(y.clone().into_iter(), y_fit_good.into_iter(), k);
        let score_bad = Bic.score::<f64>(y.into_iter(), y_fit_bad.into_iter(), k);
        assert!(score_bad > score_good);
    }
    #[test]
    fn scoring_empty_input_returns_nan() {
        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let k = 2.0;
        let aic = Aic.score::<f64>(y.clone().into_iter(), y_fit.clone().into_iter(), k);
        let bic = Bic.score::<f64>(y.into_iter(), y_fit.into_iter(), k);
        assert!(aic.is_nan());
        assert!(bic.is_nan());
    }

    #[test]
    fn scoring_bic_stricter_than_aic() {
        let y = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
        ];
        let y_fit = vec![
            1.1, 2.1, 2.9, 4.0, 5.05, 1.1, 2.1, 2.9, 4.0, 5.05, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
        ];
        let k = 3.0;
        let aic = Aic.score::<f64>(y.clone().into_iter(), y_fit.clone().into_iter(), k);
        let bic = Bic.score::<f64>(y.into_iter(), y_fit.into_iter(), k);

        assert!(bic >= aic);
    }
}
