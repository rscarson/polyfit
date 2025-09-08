use std::ops::Range;

use crate::{basis::Basis, display::PolynomialDisplay, value::Value, CurveFit, Polynomial};

/// Elements that can be plotted
pub enum PlottingElement<'a, B, T: Value>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    /// Has error bars, data, residuals etc
    Fit(&'a CurveFit<B, T>),

    /// The perfect golden child, flawless and pristine
    Canonical(&'a Polynomial<'a, B, T>),

    /// Raw data points
    Data(&'a [(T, T)]),
}
impl<B, T: Value> PlottingElement<'_, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    /// Returns the equation for this element
    #[must_use]
    pub fn equation(&self) -> Option<String> {
        match self {
            PlottingElement::Fit(fit) => Some(fit.equation()),
            PlottingElement::Canonical(canonical) => Some(canonical.equation()),
            PlottingElement::Data(_) => None,
        }
    }

    /// Returns the data points for this element
    #[must_use]
    pub fn data(&self) -> Option<&[(T, T)]> {
        match self {
            PlottingElement::Fit(fit) => Some(fit.data()),
            PlottingElement::Canonical(_) => None,
            PlottingElement::Data(data) => Some(data),
        }
    }

    /// Returns the x-axis range for this element
    #[must_use]
    pub fn x_range(&self) -> Option<Range<T>> {
        match self {
            PlottingElement::Fit(fit) => {
                let range = fit.x_range();
                Some(*range.start()..*range.end())
            }
            PlottingElement::Canonical(_) => None,
            PlottingElement::Data(data) => {
                let x_values = data.iter().map(|(x, _)| *x).collect::<Vec<_>>();
                Some(*x_values.first()?..*x_values.last()?)
            }
        }
    }

    /// Solves the element for the given iterator
    pub fn solve(&self, iter: impl Iterator<Item = T>) -> Vec<(T, T)> {
        match self {
            PlottingElement::Fit(fit) => fit.as_polynomial().solve(iter),
            PlottingElement::Canonical(canonical) => canonical.solve(iter),
            PlottingElement::Data(data) => data.to_vec(),
        }
    }
}
impl<'a, B, T: Value> From<&'a CurveFit<B, T>> for PlottingElement<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn from(fit: &'a CurveFit<B, T>) -> Self {
        PlottingElement::Fit(fit)
    }
}
impl<'a, B, T: Value> From<&'a Polynomial<'a, B, T>> for PlottingElement<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn from(canonical: &'a Polynomial<B, T>) -> Self {
        PlottingElement::Canonical(canonical)
    }
}
impl<'a, T: Value> From<&'a Vec<(T, T)>>
    for PlottingElement<'a, crate::basis::MonomialBasis<T>, T>
{
    fn from(data: &'a Vec<(T, T)>) -> Self {
        PlottingElement::Data(data)
    }
}

impl<'a, B, T: Value> From<&&'a CurveFit<B, T>> for PlottingElement<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn from(fit: &&'a CurveFit<B, T>) -> Self {
        PlottingElement::Fit(*fit)
    }
}
impl<'a, B, T: Value> From<&&'a Polynomial<'a, B, T>> for PlottingElement<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn from(canonical: &&'a Polynomial<B, T>) -> Self {
        PlottingElement::Canonical(*canonical)
    }
}
impl<'a, T: Value> From<&&'a Vec<(T, T)>>
    for PlottingElement<'a, crate::basis::MonomialBasis<T>, T>
{
    fn from(data: &&'a Vec<(T, T)>) -> Self {
        PlottingElement::Data(data)
    }
}
