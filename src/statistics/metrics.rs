use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    error::Error,
    score::{Bic, ModelScoreProvider},
    value::{CoordExt, Value},
};

mod basic;
pub use basic::*;

mod deviation;
pub use deviation::*;

mod r_squared;
pub use r_squared::*;

mod normality;
pub use normality::*;

mod huber;
pub use huber::*;

//
// Model aware metrics
//

/// Computes the Bayes factor between two models.
///
/// The result indicates how much more likely the data is under one model compared to the other:
/// - < 1.0 → Model 2 is favored
/// - 1.0 → Both models are equally likely
/// - 1.0 to 3.0 → Weak evidence for Model 1
/// - 3.0 to 10.0 → Moderate evidence for Model 1
/// - > 10.0 → Strong evidence for Model 1
///
/// Essentially, the Bayes factor quantifies how well each model explains the data, while penalizing for model complexity (number of parameters).
///
/// # Errors
/// Returns `Err` if either model fails to solve for the given data, indicating the supplied data is out of the model's domain
pub fn bayes_factor<T: Value, B1, B2>(
    m1: &crate::CurveFit<B1, T>,
    m2: &crate::CurveFit<B2, T>,
    data: &[(T, T)],
) -> Result<T, Error>
where
    B1: Basis<T> + PolynomialDisplay<T>,
    B2: Basis<T> + PolynomialDisplay<T>,
{
    let y1 = m1.solve(data.x_iter())?.y();
    let y2 = m2.solve(data.x_iter())?.y();

    let k1 = T::from_positive_int(m1.coefficients().len());
    let bic1 = Bic
        .score(m1, y1.into_iter(), data.y_iter(), k1)
        .ok_or(Error::NoData)?;

    let k2 = T::from_positive_int(m2.coefficients().len());
    let bic2 = Bic
        .score(m2, y2.into_iter(), data.y_iter(), k2)
        .ok_or(Error::NoData)?;

    Ok(((bic2 - bic1) / T::two()).exp())
}
