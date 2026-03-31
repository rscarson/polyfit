use crate::{
    basis::{AugmentedFourierBasis, DifferentialBasis, FourierBasis, IntegralBasis},
    error::Result,
    value::Value,
    Polynomial,
};

/// Non-standard Fourier basis for periodic functions which includes an additional linear term to capture trends in the data.
///
/// While this can be useful for fitting data with both periodic and linear components, it is not orthogonal due to the presence of the linear term.
///
/// This is the result of integrating the Fourier basis, which introduces a linear term.
///
/// The Fourier basis represents functions using sine and cosine functions:
/// ```math
/// 1, sin(2πx), cos(2πx), sin(4πx), cos(4πx), ..., sin(2nπx), cos(2nπx)
/// ```
///
/// # When to use
/// - Use for fitting periodic data or functions.
/// - Ideal for applications in signal processing, time series analysis, and any domain with inherent periodicity.
///
/// # Why Fourier?
/// - Provides a natural way to represent periodic functions.
/// - Efficiently captures oscillatory behavior with fewer terms.
pub type LinearAugmentedFourierBasis<T = f64> = AugmentedFourierBasis<1, T>;

impl<T: Value> DifferentialBasis<T> for LinearAugmentedFourierBasis<T> {
    type B2 = FourierBasis<T>;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)> {
        let coefs = self.derivative_coefs(coefficients)?;
        let basis = FourierBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}

/// Type alias for a Fourier polynomial (`Polynomial<FourierBasis, T>`).
pub type LinearAugmentedFourierPolynomial<'a, T> =
    crate::Polynomial<'a, LinearAugmentedFourierBasis<T>, T>;
impl<T: Value> LinearAugmentedFourierPolynomial<'_, T> {
    /// Create a new Fourier polynomial with the given constant and Fourier coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Fourier basis is defined
    /// - `constant`: The constant term of the polynomial
    /// - `terms`: A slice of (`a_n`, `b_n`) pairs representing the sine and cosine coefficients
    ///
    /// # Returns
    /// A polynomial defined in the Fourier basis.
    ///
    /// For example to create a Fourier polynomial:
    /// ```math
    /// f(x) = 3 + 2 sin(2πx) - 0.5 cos(2πx)
    /// ```
    ///
    /// ```rust
    /// use polyfit::FourierPolynomial;
    /// let poly = FourierPolynomial::new((-1.0, 1.0), 3.0, &[(2.0, -0.5)]);
    /// ```
    #[allow(
        clippy::missing_panics_doc,
        reason = "Always has valid coefficients for Fourier basis"
    )]
    pub fn new(x_range: (T, T), constant: T, terms: &[(T, T)]) -> Self {
        let mut coefficients = Vec::with_capacity(1 + terms.len() * 2);
        coefficients.push(constant);
        for (a_n, b_n) in terms {
            coefficients.push(*a_n); // sin term
            coefficients.push(*b_n); // cos term
        }

        let basis = LinearAugmentedFourierBasis::new(x_range.0, x_range.1);
        Polynomial::from_basis(basis, coefficients).expect("Failed to create Fourier polynomial")
    }
}

//
// Below here is the limited support for higher order integrals of Fourier
macro_rules! support_fourier_level {
    ($own_degree:literal, $next_degree:literal) => {
        impl<T: Value> IntegralBasis<T> for AugmentedFourierBasis<$own_degree, T> {
            type B2 = AugmentedFourierBasis<$next_degree, T>;

            fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self::B2, Vec<T>)> {
                let coefs = self.integral_coefs(coefficients, constant)?;

                let basis = Self::B2::from_normalizer(self.normalizer);
                Ok((basis, coefs))
            }
        }
        impl<T: Value> DifferentialBasis<T> for AugmentedFourierBasis<$own_degree, T> {
            type B2 = AugmentedFourierBasis<$own_degree, T>;

            fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)> {
                let coefs = self.derivative_coefs(coefficients)?;
                let basis = Self::B2::from_normalizer(self.normalizer);
                Ok((basis, coefs))
            }
        }
    };
}

support_fourier_level!(2, 3);
support_fourier_level!(3, 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_close, assert_fits, basis::Basis, score::Aic, statistics::DegreeBound,
        test::basis_assertions, LinearAugmentedFourierFit, Polynomial,
    };

    fn get_poly() -> Polynomial<'static, LinearAugmentedFourierBasis<f64>> {
        let basis = LinearAugmentedFourierBasis::new(0.0, 100.0);
        Polynomial::from_basis(basis, &[1.0, 0.6, 3.0, -0.5]).unwrap()
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_basis() {
        // Recover polynomial
        let poly = get_poly();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        crate::plot!(data);
        let fit = LinearAugmentedFourierFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit);

        // Solve known values
        let basis = LinearAugmentedFourierBasis::new(0.0, 2.0 * std::f64::consts::PI);
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), 0.5);
        assert_close!(basis.solve_function(2, 0.5), 0.479425538604203);
        assert_close!(basis.solve_function(3, 0.5), 0.8775825618903728);
        assert_close!(basis.solve_function(4, 0.5), 0.8414709848078965);

        // Integrate -> differentiate = Original
        let poly =
            LinearAugmentedFourierBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, 3.0, -1.5])
                .unwrap();
        let normalizer = poly.basis().normalizer;
        basis_assertions::test_reversible_derivation(&poly, &normalizer);
    }
}
