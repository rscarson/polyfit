//! Scoring methods for model selection.
//!
//! These methods help choose the best polynomial degree by balancing fit quality and model complexity.
//! They are not measures of fit quality themselves; for that, use metrics like R².
//!
//! # Overview of Available Scoring Methods
//! - **Akaike Information Criterion (AIC)**: A commonly used method that balances fit quality and complexity. It tends to favor slightly more complex models if they provide a better fit.
//! - **Bayesian Information Criterion (BIC)**: Similar to AIC but applies a stricter penalty for model complexity. It often prefers simpler models, even if the fit is slightly worse.
//! - **Root Mean Squared Error (RMSE)**: A straightforward measure of fit quality that does not penalize complexity. Use this if you want to select the model that fits the data best, regardless of how complex it is. Be cautious of overfitting, especially with small datasets.
//! - **`ShapeConstraint`**: A custom scoring method that adds a penalty for curvature and non-monotonicity, in addition to the fit quality measured by RMSE. Use this if you want to select a model that not only fits the data well, but also has a smoother curve and/or is monotonic.
//!
//! The [`ModelScoreProvider`] trait defines the interface for implementing custom scoring methods.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::{basis::Basis, display::PolynomialDisplay, value::Value, CurveFit};

mod basic;
pub use basic::*;

pub mod shape_constraint;

/// Trait for implementing scoring methods for model selection.
///
/// It is generic over basis so that a given scoring method can be made useable in only a subset of bases if desired.
/// For example, `ShapeConstraint` is only implemented for bases that implement `DifferentialBasis`, since it needs to calculate the first and second derivatives of the curve.
pub trait ModelScoreProvider<B: Basis<T> + PolynomialDisplay<T>, T: Value>: Send + Sync {
    /// The minimum distance between scores for them to be considered significantly different.
    /// As per burnham-anderson guidelines, a difference of less than 2 is not considered significant for AIC/BIC scores.
    fn minimum_significant_distance(&self) -> Option<usize>;

    /// Calculate the model's score using this scoring method.
    ///
    /// # Notes
    /// - Lower scores indicate a "better" choice for automatically selecting the polynomial degree.
    /// - This is **not** a measure of how well the model fits your data. For that, use `r_squared`.
    ///
    /// # Type Parameters
    /// - `B`: The type of basis used in the curve fit, which must implement both `Basis<T>` and `PolynomialDisplay<T>`.
    /// - `T`: A numeric type implementing the `Value` trait.
    ///
    /// # Parameters
    /// - `model`: The fitted curve model for which the score is being calculated.
    /// - `y`: Iterator over the observed (actual) values.
    /// - `y_fit`: Iterator over the predicted values from the model.
    /// - `k`: Number of model parameters (degrees of freedom used by the fit).
    ///
    /// # Returns
    /// The computed score as a `T`.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{score::{Aic, ModelScoreProvider}, value::Value, ChebyshevFit, statistics::DegreeBound};
    /// let data = &[(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)];
    /// let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, &Aic).unwrap();
    /// let score = fit.model_score(&Aic);
    /// ```
    fn score(
        &self,
        model: &CurveFit<B, T>,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> T;
}

#[cfg(test)]
mod tests {
    use crate::MonomialFit;

    use super::*;

    #[test]
    fn scoring_perfect_fit() {
        let model = MonomialFit::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], 2).unwrap();

        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = y.clone();
        let k = 2.0;
        let aic: f64 = Aic.score(&model, y.clone().into_iter(), y_fit.clone().into_iter(), k);
        let bic: f64 = Bic.score(&model, y.into_iter(), y_fit.into_iter(), k);
        assert!(aic.is_finite());
        assert!(bic.is_finite());
    }

    #[test]
    fn scoring_aicc_correction() {
        let model = MonomialFit::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], 2).unwrap();

        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.8, 2.7, 3.6];
        let k = 2.0; // n/k = 1.5 < 4 → triggers correction
        let score: f64 = Aic.score(&model, y.into_iter(), y_fit.into_iter(), k);
        assert!(score.is_finite());
    }

    #[test]
    fn scoring_higher_error_higher_score() {
        let model = MonomialFit::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], 2).unwrap();

        let y = vec![1.0, 2.0, 3.0];
        let y_fit_good = vec![1.0, 2.0, 3.0];
        let y_fit_bad = vec![0.0, 0.0, 0.0];
        let k = 2.0;
        let score_good = Bic.score(&model, y.clone().into_iter(), y_fit_good.into_iter(), k);
        let score_bad = Bic.score(&model, y.into_iter(), y_fit_bad.into_iter(), k);
        assert!(score_bad > score_good);
    }
    #[test]
    fn scoring_empty_input_returns_nan() {
        let model = MonomialFit::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], 2).unwrap();

        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let k = 2.0;
        let aic: f64 = Aic.score(&model, y.clone().into_iter(), y_fit.clone().into_iter(), k);
        let bic: f64 = Bic.score(&model, y.into_iter(), y_fit.into_iter(), k);
        assert!(aic.is_nan());
        assert!(bic.is_nan());
    }

    #[test]
    fn scoring_bic_stricter_than_aic() {
        let model = MonomialFit::new(&[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)], 2).unwrap();

        let y = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
        ];
        let y_fit = vec![
            1.1, 2.1, 2.9, 4.0, 5.05, 1.1, 2.1, 2.9, 4.0, 5.05, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
        ];
        let k = 3.0;
        let aic: f64 = Aic.score(&model, y.clone().into_iter(), y_fit.clone().into_iter(), k);
        let bic: f64 = Bic.score(&model, y.into_iter(), y_fit.into_iter(), k);

        assert!(bic >= aic);
    }
}
