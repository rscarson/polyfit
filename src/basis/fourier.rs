use nalgebra::MatrixViewMut;

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, MonomialBasis, OrthogonalBasis},
    display::{self, format_coefficient, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::{IntClampedCast, Value},
    Polynomial,
};

/// Type alias for a Fourier polynomial (`Polynomial<FourierBasis, T>`).
pub type FourierPolynomial<'a, T> = crate::Polynomial<'a, FourierBasis<T>, T>;
impl<T: Value> FourierPolynomial<'_, T> {
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

        let basis = FourierBasis::new(x_range.0, x_range.1);
        Polynomial::from_basis(basis, coefficients).expect("Failed to create Fourier polynomial")
    }
}

/// Standard Fourier basis for periodic functions.
///
/// The Fourier basis represents functions using sine and cosine functions:
/// ```math
/// 1, sin(2πx), cos(2πx), sin(4πx), cos(4πx), ..., sin(2nπx), cos(2nπx)
/// ```
///
/// This basis is ideal for modeling periodic phenomena, such as waves or seasonal patterns.
/// It is numerically stable and can efficiently represent complex periodic behavior.
///
/// # When to use
/// - Use for fitting periodic data or functions.
/// - Ideal for applications in signal processing, time series analysis, and any domain with inherent periodicity.
///
/// # Why Fourier?
/// - Provides a natural way to represent periodic functions.
/// - Efficiently captures oscillatory behavior with fewer terms.
/// - Numerically stable for a wide range of applications.
#[derive(Debug, Clone)]
pub struct FourierBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
    polynomial_terms: usize,
}
impl<T: Value> FourierBasis<T> {
    /// Creates a new Fourier basis that normalizes inputs from the given range to [0, 2π].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (T::zero(), T::two_pi()));
        Self {
            normalizer,
            polynomial_terms: 1,
        }
    }

    /// Creates a new Fourier polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Fourier basis is defined.
    /// - `coefficients`: The coefficients for the Fourier basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Fourier basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::FourierBasis;
    /// let fourier_poly = FourierBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}
impl<T: Value> Basis<T> for FourierBasis<T> {
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self {
        let normalizer = DomainNormalizer::from_range(x_range, (T::zero(), T::two_pi()));
        Self {
            normalizer,
            polynomial_terms: 1,
        }
    }

    #[inline(always)]
    fn normalize_x(&self, x: T) -> T {
        self.normalizer.normalize(x)
    }

    #[inline(always)]
    fn denormalize_x(&self, x: T) -> T {
        self.normalizer.denormalize(x)
    }

    #[inline(always)]
    fn k(&self, degree: usize) -> usize {
        2 * degree + 1
    }

    #[inline(always)]
    fn degree(&self, k: usize) -> Option<usize> {
        if k % 2 == 0 {
            None
        } else {
            Some((k - 1) / 2)
        }
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        // Because we support calculus, there can be a polynomial series at the start
        // The first [0..polynomial_terms] are monomial terms
        if j < self.polynomial_terms {
            return Value::powi(x, j.clamped_cast());
        }

        let j = j - self.polynomial_terms;
        // Now we have the Fourier terms

        if j % 2 == 0 {
            // Sine terms (odd indices)
            let n = (j + 1).div_ceil(2);

            // Infallible multiplication for the *n term
            let mut angle = x;
            for _ in 1..n {
                angle += x;
            }

            angle.sin()
        } else {
            // Cosine terms (even indices)
            let n = j.div_ceil(2);

            // Infallible multiplication for the *n term
            let mut angle = x;
            for _ in 1..n {
                angle += x;
            }

            angle.cos()
        }
    }

    //
    // invariant: polynomial_terms == 1 here
    // We only kerfuffle that in calculus methods
    #[inline(always)]
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<'_, T, R, C, RS, CS>,
    ) {
        for j in start_index..row.ncols() {
            row[j] = match j {
                0 => T::one(),
                1 => x.sin(), // first sin
                2 => x.cos(), // first cos
                3 => T::two() * x.cos() * row[1],
                4 => T::two() * x.cos() * row[2] - T::one(),
                _ => T::two() * x.cos() * row[j - 2] - row[j - 4],
            }
        }
    }
}

impl<T: Value> display::PolynomialDisplay<T> for FourierBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        if degree < self.polynomial_terms.clamped_cast() {
            return MonomialBasis::format_term(&MonomialBasis::default(), degree, coef);
        }

        let sign = Sign::from_coef(coef);
        let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        // frequency index
        let n = (degree + 1) / 2;
        let n = if n == 1 { String::new() } else { n.to_string() };

        // even -> cos, odd -> sin
        let function = if degree % 2 == 0 { "cos" } else { "sin" };

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        let glue = if coef.is_empty() || function.is_empty() {
            ""
        } else {
            "·"
        };

        let body = format!("{coef}{glue}{function}({n}{x})");
        Some(Term { sign, body })
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

impl<T: Value> IntegralBasis<T> for FourierBasis<T> {
    type B2 = Self;

    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self, Vec<T>)> {
        if coefficients.is_empty() {
            return Ok((self.clone(), vec![constant]));
        }

        //
        // We do a monomial integral for the first [0..polynomial_terms]
        let polynomial_terms = &coefficients[..self.polynomial_terms.min(coefficients.len())];
        let (_, mut integral_coeffs) =
            MonomialBasis::default().integral(polynomial_terms, constant)?;

        //
        // The integral is actually not hard here
        // 2- an * sin(nx) -> -an/n * cos(nx)
        // 3- bn * cos(nx) -> bn/n * sin(nx)
        // We can do the last 2 by flipping pairs of coefficients and dividing by n
        let fourier_terms = &coefficients[self.polynomial_terms.min(coefficients.len())..];

        let mut n = T::one(); // frequency index
        let mut coef_iter = fourier_terms.iter();
        while let Some(a) = coef_iter.next().copied() {
            let b = coef_iter.next().copied().unwrap_or(T::zero());

            // Fourier expects pairs of (sin, cos), so originally we had (a, b)
            // Now under integration we need (b, -a) to get the right functions
            integral_coeffs.push(b / n);
            integral_coeffs.push(-a / n);

            n += T::one(); // increment frequency index
        }

        let basis = Self {
            normalizer: self.normalizer,
            polynomial_terms: self.polynomial_terms + 1,
        };
        Ok((basis, integral_coeffs))
    }
}

impl<T: Value> DifferentialBasis<T> for FourierBasis<T> {
    type B2 = Self;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self, Vec<T>)> {
        if coefficients.len() <= 1 {
            return Ok((self.clone(), vec![T::zero()]));
        }

        //
        // We do a monomial differential for the first [0..polynomial_terms]
        let polynomial_terms = &coefficients[..self.polynomial_terms.min(coefficients.len())];
        let (_, mut derivative_coeffs) = MonomialBasis::default().derivative(polynomial_terms)?;

        let fourier_terms = &coefficients[self.polynomial_terms.min(coefficients.len())..];

        //
        // Similar to integral, we can do this by flipping pairs of coefficients:
        // 2- an * sin(nx) -> n * an * cos(nx)
        // 3- bn * cos(nx) -> -n * bn * sin(nx)

        let mut n = T::one(); // frequency index
        let mut coef_iter = fourier_terms.iter();
        while let Some(a) = coef_iter.next().copied() {
            let b = coef_iter.next().copied().unwrap_or(T::zero());

            // Fourier expects pairs of (sin, cos), so originally we had (a, b)
            // Now under differentiation we need (b, -a) to get the right functions
            derivative_coeffs.push(n * -b);
            derivative_coeffs.push(n * a);

            n += T::one(); // increment frequency index
        }

        let basis = Self {
            normalizer: self.normalizer,
            polynomial_terms: (self.polynomial_terms - 1).max(1),
        };
        Ok((basis, derivative_coeffs))
    }
}

impl<T: Value> OrthogonalBasis<T> for FourierBasis<T> {
    fn is_orthogonal(&self) -> bool {
        self.polynomial_terms < 2
    }

    fn gauss_weight(&self, _: T) -> T {
        T::one()
    }

    fn gauss_nodes(&self, n: usize) -> Vec<(T, T)> {
        // Nodes: equispaced in [0, 2π)
        // Weights: uniform (trapezoid rule for periodic functions)
        if n == 0 {
            return vec![(T::zero(), T::one())];
        }

        let two_pi = T::two() * T::pi();
        let w = two_pi / T::from_positive_int(n);
        (0..n)
            .map(|k| {
                let x = T::from_positive_int(k) * w; // x in [0, 2π)
                (x, w)
            })
            .collect()
    }

    fn gauss_normalization(&self, n: usize) -> T {
        if n == 0 {
            // constant term
            T::two() * T::pi()
        } else {
            // all sin/cos terms
            T::pi()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_close, assert_fits, score::Aic, statistics::DegreeBound,
        test::basis_assertions::assert_basis_orthogonal, FourierFit, Polynomial,
    };

    fn get_poly() -> Polynomial<'static, FourierBasis<f64>> {
        let basis = FourierBasis::new(0.0, 100.0);
        Polynomial::from_basis(basis, &[1.0, 2.0, -0.5]).unwrap()
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_fourier_basis() {
        // Recover polynomial
        let poly = get_poly();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = FourierFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit);

        // Solve known values
        let basis = FourierBasis::new(0.0, 2.0 * std::f64::consts::PI);
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), 0.479425538604203);
        assert_close!(basis.solve_function(2, 0.5), 0.8775825618903728);
        assert_close!(basis.solve_function(3, 0.5), 0.8414709848078965);

        // Integrate -> differentiate = Original
        let poly = FourierBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, -1.5]).unwrap();
        test_derivation!(poly, &fit.basis().normalizer, with_reverse = true);
        test_integration!(poly, &fit.basis().normalizer, with_reverse = true);

        let org_coefs = fit.coefficients();
        let (bi, int_coefs) = fit.basis().integral(org_coefs, 0.0).unwrap();
        let (_, diff_coefs) = bi.derivative(&int_coefs).unwrap();
        assert_close!(org_coefs[0], diff_coefs[0]); // constant term
        for (a, b) in org_coefs.iter().skip(1).zip(diff_coefs.iter().skip(1)) {
            assert_close!(*a, *b); // Phase-shifted Fourier terms (negated)
        }

        // Orthogonality test points
        assert_basis_orthogonal(&basis, 7, 100, 1e-12);
    }
}
