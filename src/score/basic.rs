//! Basic model scoring methods, including AIC, BIC, and RMSE.
//! These are used for model selection and evaluation, and can be customized or extended as needed.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    score::ModelScoreProvider,
    statistics::{self, accumulator::HuberLogLikelihoodAccumulator, median_absolute_deviation},
    value::Value,
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
        _: &crate::CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> Option<T> {
        let y: Vec<_> = y.collect();
        let y_fit: Vec<_> = y_fit.collect();

        let mad = median_absolute_deviation(y.iter().copied(), y_fit.iter().copied())?;
        let mut acc = HuberLogLikelihoodAccumulator::new(mad, None);
        acc.add_iter(y.into_iter(), y_fit.into_iter());

        let n = acc.count();
        let log_likelihood = acc.log_likelihood()?;
        let log_likelihood = nalgebra::RealField::max(log_likelihood, T::epsilon());
        let mut aic = n * log_likelihood.ln() + T::two() * k;
        if n / k < T::two() + T::two() && n > k + T::one() {
            // Apply AICc correction
            aic += T::two() * k * (k + T::one()) / (n - k - T::one());
        }

        Some(aic)
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
        _: &crate::CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> Option<T> {
        let y: Vec<_> = y.collect();
        let y_fit: Vec<_> = y_fit.collect();

        let mad = median_absolute_deviation(y.iter().copied(), y_fit.iter().copied())?;
        let mut acc = HuberLogLikelihoodAccumulator::new(mad, None);
        acc.add_iter(y.into_iter(), y_fit.into_iter());

        let n = acc.count();
        let log_likelihood = acc.log_likelihood()?;
        let log_likelihood = nalgebra::RealField::max(log_likelihood, T::epsilon());

        Some(n * log_likelihood.ln() + k * n.ln())
    }
}

/// Root Mean Squared Error. A simple measure of fit quality, without any penalty for model complexity.
/// - Use this if you want to select the model that fits the data best, regardless of complexity. Be cautious of overfitting, especially with small datasets.
///
/// See [`statistics::root_mean_squared_error`] for more details on how the error is calculated.
pub struct RootMeanSquaredError;
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T>
    for RootMeanSquaredError
{
    fn minimum_significant_distance(&self) -> Option<usize> {
        None
    }

    fn score(
        &self,
        _: &crate::CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        _: T,
    ) -> Option<T> {
        statistics::root_mean_squared_error(y, y_fit)
    }
}

/// Mean Absolute Error. Similar to RMSE but uses absolute values instead of squares, making it less sensitive to outliers.
/// - Use this if you want a more robust measure of fit quality that is less influenced by outliers. Like RMSE, it does not penalize model complexity, so be cautious of overfitting.
///
/// See [`statistics::mean_absolute_error`] for more details on how the error is calculated.
pub struct MeanAbsoluteError;
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T> for MeanAbsoluteError {
    fn minimum_significant_distance(&self) -> Option<usize> {
        None
    }

    fn score(
        &self,
        _: &crate::CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        _: T,
    ) -> Option<T> {
        statistics::mean_absolute_error(y, y_fit)
    }
}

/// A convenient enum to allow users to easily select from common scoring methods without needing to import each one individually.
/// - Use this if you want a simple way to specify the scoring method without needing to import each one individually.
///   You can still use the individual structs if you need more control or want to implement custom scoring methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringMethod {
    /// Akaike Information Criterion. Uses a more lenient penalty for model complexity
    /// - Picks a slightly more complex model if it fits better.
    ///
    /// See [`Aic`] for more details on how the score is calculated and when to use it.
    Aic,

    /// Bayesian Information Criterion. Uses a stricter penalty for model complexity.
    /// - Prefers simpler models, even if the fit is slightly worse.
    ///
    /// See [`Bic`] for more details on how the score is calculated and when to use it.
    Bic,

    /// Root Mean Squared Error. A simple measure of fit quality, without any penalty for model complexity.
    /// - Use this if you want to select the model that fits the data best, regardless of complexity. Be cautious of overfitting, especially with small datasets.
    ///
    /// See [`RootMeanSquaredError`] for more details on how the error is calculated and when to use it.
    RootMeanSquaredError,

    /// Mean Absolute Error. Similar to RMSE but uses absolute values instead of squares, making it less sensitive to outliers.
    /// - Use this if you want a more robust measure of fit quality that is less influenced by outliers. Like RMSE, it does not penalize model complexity, so be cautious of overfitting.
    ///
    /// See [`MeanAbsoluteError`] for more details on how the error is calculated and when to use it.
    MeanAbsoluteError,
}
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> ModelScoreProvider<B, T> for ScoringMethod {
    fn minimum_significant_distance(&self) -> Option<usize> {
        match self {
            ScoringMethod::Aic => {
                <Aic as ModelScoreProvider<B, T>>::minimum_significant_distance(&Aic)
            }
            ScoringMethod::Bic => {
                <Bic as ModelScoreProvider<B, T>>::minimum_significant_distance(&Bic)
            }
            ScoringMethod::RootMeanSquaredError => {
                <RootMeanSquaredError as ModelScoreProvider<B, T>>::minimum_significant_distance(
                    &RootMeanSquaredError,
                )
            }
            ScoringMethod::MeanAbsoluteError => {
                <MeanAbsoluteError as ModelScoreProvider<B, T>>::minimum_significant_distance(
                    &MeanAbsoluteError,
                )
            }
        }
    }

    fn score(
        &self,
        model: &crate::CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> Option<T> {
        match self {
            ScoringMethod::Aic => Aic.score(model, y, y_fit, k),
            ScoringMethod::Bic => Bic.score(model, y, y_fit, k),
            ScoringMethod::RootMeanSquaredError => RootMeanSquaredError.score(model, y, y_fit, k),
            ScoringMethod::MeanAbsoluteError => MeanAbsoluteError.score(model, y, y_fit, k),
        }
    }
}

impl From<MeanAbsoluteError> for ScoringMethod {
    fn from(_: MeanAbsoluteError) -> Self {
        ScoringMethod::MeanAbsoluteError
    }
}
impl From<RootMeanSquaredError> for ScoringMethod {
    fn from(_: RootMeanSquaredError) -> Self {
        ScoringMethod::RootMeanSquaredError
    }
}
impl From<Aic> for ScoringMethod {
    fn from(_: Aic) -> Self {
        ScoringMethod::Aic
    }
}
impl From<Bic> for ScoringMethod {
    fn from(_: Bic) -> Self {
        ScoringMethod::Bic
    }
}
