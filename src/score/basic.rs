//! Basic model scoring methods, including AIC, BIC, and RMSE.
//! These are used for model selection and evaluation, and can be customized or extended as needed.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::{
    basis::Basis, display::PolynomialDisplay, score::ModelScoreProvider, statistics, value::Value,
    CurveFit,
};

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
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T> for Aic {
    fn minimum_significant_distance(&self) -> Option<usize> {
        Some(2)
    }

    fn score(
        &self,
        _: &CurveFit<B, T>,
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
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T> for Bic {
    fn minimum_significant_distance(&self) -> Option<usize> {
        Some(2)
    }

    fn score(
        &self,
        _: &CurveFit<B, T>,
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

/// Root Mean Squared Error. A simple measure of fit quality, without any penalty for model complexity.
/// - Use this if you want to select the model that fits the data best, regardless of complexity. Be cautious of overfitting, especially with small datasets.
///
/// See [`statistics::root_mean_squared_error`] for more details on how the error is calculated.
pub struct RMSE;
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T> for RMSE {
    fn minimum_significant_distance(&self) -> Option<usize> {
        None
    }

    fn score(
        &self,
        _: &CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        _: T,
    ) -> T {
        statistics::root_mean_squared_error(y, y_fit)
    }
}
