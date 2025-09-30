use std::ops::Range;

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    statistics::{Confidence, ConfidenceBand, Tolerance},
    value::Value,
    CurveFit, Polynomial,
};

/// Elements that can be plotted
#[derive(Debug, Clone)]
pub enum PlottingElement<T: Value> {
    /// Has error bars, data, residuals etc
    Fit(Vec<(T, T)>, Vec<(T, ConfidenceBand<T>)>, String),

    /// The perfect golden child, flawless and pristine
    ///
    /// (x, function)
    Canonical(Vec<(T, T)>, String),

    /// Raw data points
    Data(Vec<(T, T)>),
}
impl<T: Value> PlottingElement<T> {
    /// Creates a new plotting element from raw data
    pub fn from_data(data: &[(T, T)]) -> Self {
        Self::Data(data.to_vec())
    }

    /// Creates a new plotting element from a polynomial
    pub fn from_polynomial<B: Basis<T> + PolynomialDisplay<T>>(
        poly: &Polynomial<'_, B, T>,
        xs: &[T],
    ) -> Self {
        let data = poly.solve(xs.iter().copied());
        let equation = poly.equation();
        Self::Canonical(data, equation)
    }

    /// Creates a new plotting element from a curve fit
    pub fn from_curve_fit<B: Basis<T> + PolynomialDisplay<T>>(
        fit: &CurveFit<B, T>,
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> Self {
        let equation = fit.equation();
        let raw_data = fit.data().to_vec();

        let bands = fit
            .covariance()
            .and_then(|covariance| covariance.solution_confidence(confidence, noise_tolerance))
            .unwrap_or_else(|_| {
                fit.solution()
                    .iter()
                    .map(|(x, y)| {
                        (
                            *x,
                            ConfidenceBand {
                                level: Confidence::Custom(1.0),
                                tolerance: None,
                                value: *y,
                                lower: *y,
                                upper: *y,
                            },
                        )
                    })
                    .collect()
            });
        Self::Fit(raw_data, bands, equation)
    }

    /// Returns the equation for this element
    #[must_use]
    pub fn equation(&self) -> Option<String> {
        match self {
            PlottingElement::Fit(_, _, equation) | PlottingElement::Canonical(_, equation) => {
                Some(equation.clone())
            }
            PlottingElement::Data(_) => None,
        }
    }

    /// Returns the x-values for this element
    #[must_use]
    pub fn x_values(&self) -> Vec<T> {
        match self {
            PlottingElement::Fit(data, _, _) => data.iter().map(|(x, _)| *x).collect(),
            PlottingElement::Canonical(points, _) => points.iter().map(|(x, _)| *x).collect(),
            PlottingElement::Data(points) => points.iter().map(|(x, _)| *x).collect(),
        }
    }

    /// Returns the x-axis range for this element
    #[must_use]
    pub fn x_range(&self) -> Range<T> {
        match self {
            PlottingElement::Fit(data, _, _)
            | PlottingElement::Canonical(data, _)
            | PlottingElement::Data(data) => match data.last() {
                Some((x_max, _)) => data[0].0..*x_max,
                None => T::zero()..T::one(),
            },
        }
    }

    /// Solves the element for the given iterator
    #[must_use]
    pub fn y_range(&self) -> Range<T> {
        let padding = T::from_usize(10).unwrap_or(T::one());

        match self {
            PlottingElement::Fit(_, bands, _) => {
                let (mut min, mut max) = (None, None);
                for (_, band) in bands {
                    min = Some(match min {
                        Some(m) => nalgebra::RealField::min(m, band.min()),
                        None => band.min(),
                    });
                    max = Some(match max {
                        Some(m) => nalgebra::RealField::max(m, band.max()),
                        None => band.max(),
                    });
                }

                if let (Some(min), Some(max)) = (min, max) {
                    (min - padding)..(max + padding)
                } else {
                    T::zero()..T::one()
                }
            }

            PlottingElement::Canonical(data, _) | PlottingElement::Data(data) => {
                let (mut min, mut max) = (None, None);
                for (_, y) in data {
                    min = Some(match min {
                        Some(m) => nalgebra::RealField::min(m, *y),
                        None => *y,
                    });
                    max = Some(match max {
                        Some(m) => nalgebra::RealField::max(m, *y),
                        None => *y,
                    });
                }

                if let (Some(min), Some(max)) = (min, max) {
                    (min - padding)..(max + padding)
                } else {
                    T::zero()..T::one()
                }
            }
        }
    }
}

/// A trait for types that can be converted to a plotting element
pub trait AsPlottingElement<T: Value> {
    /// Converts this to a plotting element
    fn as_plotting_element(
        &self,
        xs: &[T],
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> PlottingElement<T>;
}

impl<T> AsPlottingElement<T> for PlottingElement<T>
where
    T: Value,
{
    fn as_plotting_element(
        &self,
        _: &[T],
        _: Confidence,
        _: Option<Tolerance<T>>,
    ) -> PlottingElement<T> {
        self.clone()
    }
}
impl<T> AsPlottingElement<T> for &PlottingElement<T>
where
    T: Value,
{
    fn as_plotting_element(
        &self,
        _: &[T],
        _: Confidence,
        _: Option<Tolerance<T>>,
    ) -> PlottingElement<T> {
        (*self).clone()
    }
}

impl<B, T> AsPlottingElement<T> for CurveFit<'_, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
    T: Value,
{
    fn as_plotting_element(
        &self,
        _: &[T],
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> PlottingElement<T> {
        PlottingElement::from_curve_fit(self, confidence, noise_tolerance)
    }
}

impl<B, T> AsPlottingElement<T> for Polynomial<'_, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
    T: Value,
{
    fn as_plotting_element(
        &self,
        xs: &[T],
        _: Confidence,
        _: Option<Tolerance<T>>,
    ) -> PlottingElement<T> {
        PlottingElement::from_polynomial(self, xs)
    }
}

impl<T> AsPlottingElement<T> for &[(T, T)]
where
    T: Value,
{
    fn as_plotting_element(
        &self,
        _: &[T],
        _: Confidence,
        _: Option<Tolerance<T>>,
    ) -> PlottingElement<T> {
        PlottingElement::from_data(self)
    }
}

impl<T> AsPlottingElement<T> for Vec<(T, T)>
where
    T: Value,
{
    fn as_plotting_element(
        &self,
        _: &[T],
        _: Confidence,
        _: Option<Tolerance<T>>,
    ) -> PlottingElement<T> {
        PlottingElement::Data(self.clone())
    }
}
