use nalgebra::{Complex, ComplexField, DMatrix, Normed};
use num_traits::Zero;

use crate::{
    basis::{
        AugmentedFourierBasis, Basis, DifferentialBasis, IntegralBasis,
        LinearAugmentedFourierBasis, OrthogonalBasis, Root, RootFindingBasis, RootFindingMethod,
    },
    error::Result,
    value::{IntClampedCast, Value},
    Polynomial,
};

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
pub type FourierBasis<T = f64> = AugmentedFourierBasis<0, T>;

impl<T: Value> FourierBasis<T> {
    /// Convert Fourier coefficients to complex monomial coefficients.
    ///
    /// This is useful for certain mathematical operations, such as root finding
    pub(crate) fn as_complex_monomial(coefs: &[T]) -> Vec<Complex<T>> {
        let n = (coefs.len() - 1) / 2;

        let mut c = vec![Complex::new(T::zero(), T::zero()); 2 * n + 1];

        // c_0
        c[n] = Complex::from_real(coefs[0]);

        let half = T::one() / T::two();

        for k in 1..=n {
            let b_k = coefs[2 * k - 1]; // sin
            let a_k = coefs[2 * k]; // cos

            // +k
            c[n + k] = Complex::new(a_k * half, -b_k * half);

            // -k
            c[n - k] = Complex::new(a_k * half, b_k * half);
        }

        c
    }
}

impl<T: Value> IntegralBasis<T> for FourierBasis<T> {
    type B2 = LinearAugmentedFourierBasis<T>;

    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self::B2, Vec<T>)> {
        let coefs = self.integral_coefs(coefficients, constant)?;

        let basis = LinearAugmentedFourierBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}

impl<T: Value> DifferentialBasis<T> for FourierBasis<T> {
    type B2 = Self;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)> {
        let coefs = self.derivative_coefs(coefficients)?;
        Ok((self.clone(), coefs))
    }
}

impl<T: Value> OrthogonalBasis<T> for FourierBasis<T> {
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

impl<T: Value> RootFindingBasis<T> for FourierBasis<T> {
    fn root_finding_method(&self) -> RootFindingMethod {
        RootFindingMethod::Analytical
    }

    fn roots(&self, coefs: &[T], x_range: std::ops::RangeInclusive<T>) -> Result<Vec<Root<T>>> {
        let mut coefs = Self::as_complex_monomial(coefs);

        let n = coefs.len() - 1; // degree of polynomial
        if n == 0 {
            return Ok(vec![]);
        }

        let mut companion = DMatrix::zeros(n, n);

        // Find last non-zero (by magnitude)
        let leading_index = coefs.iter().rposition(|c| c.norm() > T::epsilon());

        if let Some(idx) = leading_index {
            let leading = coefs[idx];

            if leading.norm() > T::epsilon() && leading != Complex::from_real(T::one()) {
                for c in &mut coefs {
                    *c /= leading;
                }
            }
        } else {
            // All coefficients are zero
            return Ok(vec![]);
        }

        for i in 1..n {
            companion[(i, i - 1)] = Complex::from_real(T::one());
        }

        let leading = coefs[n];

        for i in 0..n {
            companion[(i, n - 1)] = -coefs[i] / leading;
        }

        let Some(eigs) = companion.eigenvalues() else {
            return Err(crate::error::Error::Algebra(
                "Failed to compute eigenvalues for root finding",
            ));
        };

        let mut eigs = eigs.as_slice().to_vec();
        for z in &mut eigs {
            let norm = z.norm();
            if norm > T::epsilon() {
                *z /= norm;
            }
        }

        // Filter and categorize roots
        let mut roots = eigs
            .iter()
            .map(|e| {
                let mut arg = e.argument();
                if arg < T::zero() {
                    arg += T::two_pi();
                }

                self.denormalize_x(arg)
            })
            .collect::<Vec<_>>();

        roots.sort_by(|a, b| Value::total_cmp(a, b));
        let roots = roots
            .into_iter()
            .filter_map(|r| {
                if r >= *x_range.start() && r <= *x_range.end() {
                    Some(Root::Real(r))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        Ok(roots)
    }

    fn complex_y(&self, z: Complex<T>, coefs: &[T]) -> Complex<T> {
        if coefs.is_empty() {
            return Complex::zero();
        }

        let complex_coefs = Self::as_complex_monomial(coefs);
        let n = (complex_coefs.len() - 1) / 2;

        let angle = z.re;
        // This is basically the polar constructor without the trait bound: Self::new(r * theta.cos(), r * theta.sin())
        let z = Complex::new(angle.cos(), angle.sin());

        let mut result = Complex::zero();
        for k in 0..=2 * n {
            let exp = k.clamped_cast::<isize>() - n.clamped_cast::<isize>(); // signed exponent
            result += complex_coefs[k] * z.powi(exp.clamped_cast());
        }
        result
    }
}

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

    /// Creates a copy of the fourier series with a specified DC offset (monomial constant term).
    /// This is useful for adjusting the baseline of the function without affecting its oscillatory behavior.
    ///
    /// Useful if you integrate a Fourier series and want the terms to line up with the original function, for example.
    #[must_use]
    pub fn with_dc_offset(&self, offset: T) -> Self {
        let mut coefficients = self.coefficients().to_vec();
        if coefficients.is_empty() {
            coefficients.push(offset);
        } else {
            coefficients[0] = offset;
        }

        unsafe { Polynomial::from_raw(self.basis().clone(), coefficients.into(), self.degree()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_close, assert_fits,
        score::Aic,
        statistics::DegreeBound,
        test::basis_assertions::{self, assert_basis_orthogonal},
        FourierFit, Polynomial,
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
        basis_assertions::test_reversible_derivation(&poly, &fit.basis().normalizer);
        basis_assertions::test_reversible_integration(&poly, &fit.basis().normalizer);

        let org_coefs = fit.coefficients();
        let (bi, int_coefs) = fit.basis().integral(org_coefs, 0.0).unwrap();
        let (_, diff_coefs) = bi.derivative(&int_coefs).unwrap();
        assert_close!(org_coefs[0], diff_coefs[0]); // constant term
        for (a, b) in org_coefs.iter().skip(1).zip(diff_coefs.iter().skip(1)) {
            assert_close!(*a, *b); // Phase-shifted Fourier terms (negated)
        }

        // Orthogonality test points
        assert_basis_orthogonal(&basis, 7, 100, 1e-12);

        // Test root finding
        let poly = FourierBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, 0.5, 2.0, -1.5]).unwrap();
        basis_assertions::test_root_finding(&poly, 0.0..=100.0);

        basis_assertions::test_complex_y(&poly, 0.0..=100.0);
    }
}
