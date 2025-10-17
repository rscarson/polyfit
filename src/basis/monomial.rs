use std::{borrow::Cow, fmt::Debug};

use nalgebra::{Complex, ComplexField, DMatrix, MatrixViewMut, Normed};

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, IntoMonomialBasis, Root, RootFindingBasis},
    display::{self, Sign, DEFAULT_PRECISION},
    error::Result,
    value::{IntClampedCast, Value},
    Polynomial,
};

/// Standard (non-normalized) monomial basis for polynomials.
///
/// The monomial basis represents polynomials using the familiar powers of `x`:
///
/// ```text
/// 1, x, x², …, xⁿ
/// ```
///
/// This is the simplest and most intuitive polynomial basis, and is often
/// used as the “default” basis. However, it is **not normalized**, which means
/// it can suffer from numerical instability when fitting or evaluating
/// high-degree polynomials.
///
/// # When to use
/// - Use for simple or low-degree polynomials where clarity matters.
/// - For higher degrees, consider more numerically stable bases
///   (e.g., Chebyshev).
#[derive(Debug, Clone)]
pub struct MonomialBasis<T: Value = f64>(pub std::marker::PhantomData<T>);
impl<T: Value> MonomialBasis<T> {
    /// Creates a new monomial basis.
    #[must_use]
    pub const fn default() -> Self {
        Self(std::marker::PhantomData)
    }

    /// Creates a new Monomial polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `coefficients`: The coefficients for the Monomial basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Monomial basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::MonomialBasis;
    /// let monomial_poly = MonomialBasis::new_polynomial(&[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(coefficients: &[T]) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::default();
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }

    /// Evaluates the polynomial at a given complex x-value using Horner's method.
    pub fn complex_y(&self, x: Complex<T>, coefficients: &[T]) -> Complex<T> {
        let mut y = Complex::new(T::zero(), T::zero());
        for &coef in coefficients.iter().rev() {
            y = y * x + Complex::from_real(coef);
        }
        y
    }
}
impl<T: Value> Basis<T> for MonomialBasis<T> {
    fn from_range(_x_range: std::ops::RangeInclusive<T>) -> Self {
        Self::default()
    }

    #[inline(always)]
    fn normalize_x(&self, x: T) -> T {
        x
    }

    #[inline(always)]
    fn denormalize_x(&self, x: T) -> T {
        x
    }

    #[inline(always)]
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<T, R, C, RS, CS>,
    ) {
        for j in start_index..row.ncols() {
            row[j] = match j {
                0 => T::one(),
                1 => x,
                _ => Value::powi(x, j.clamped_cast()),
            };
        }
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),
            1 => x,
            _ => Value::powi(x, j.clamped_cast()),
        }
    }
}
impl<T: Value> IntoMonomialBasis<T> for MonomialBasis<T> {
    fn as_monomial(&self, _: &mut [T]) -> Result<()> {
        // Monomial basis is already in monomial form
        Ok(())
    }
}
impl<T: Value> DifferentialBasis<T> for MonomialBasis<T> {
    type B2 = Self;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self, Vec<T>)> {
        if coefficients.len() <= 1 {
            return Ok((self.clone(), vec![T::zero()]));
        }

        let mut coefficients = coefficients[1..].to_vec();
        for (i, c) in coefficients.iter_mut().enumerate() {
            let degree = T::try_cast(i)? + T::one();
            *c *= degree;
        }

        Ok((self.clone(), coefficients))
    }
}

impl<T: Value> RootFindingBasis<T> for MonomialBasis<T> {
    fn roots(&self, coefs: &[T]) -> Result<Vec<Root<T>>> {
        let n = coefs.len() - 1; // degree of polynomial
        if n == 0 {
            return Ok(vec![]);
        }

        let mut companion = DMatrix::zeros(n, n);

        // Reduce to monic form
        // Get the last non-zero coefficient
        let leading_index = coefs.iter().rposition(|&c| Value::abs(c) > T::epsilon());
        let mut coefs = coefs.to_vec();
        if let Some(idx) = leading_index {
            let leading = coefs[idx];
            if Value::abs(leading) > T::epsilon() && leading != T::one() {
                for c in &mut coefs {
                    *c /= leading;
                }
            }
        } else {
            // All coefficients are zero
            return Ok(vec![]);
        }

        // Fill sub-diagonal with 1s
        for i in 1..n {
            companion[(i, i - 1)] = T::one();
        }

        // Fill last column
        let leading = coefs[n];
        for i in 0..n {
            companion[(i, n - 1)] = -coefs[i] / leading;
        }

        let eigs: Vec<Complex<T>> = companion
            .complex_eigenvalues()
            .into_iter()
            .copied()
            .collect();

        Ok(categorize_roots(&eigs, |z| self.complex_y(*z, &coefs)))
    }
}
impl<T: Value> IntegralBasis<T> for MonomialBasis<T> {
    type B2 = Self;

    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self, Vec<T>)> {
        let mut coefficients = coefficients.to_vec();
        for (i, c) in coefficients.iter_mut().enumerate() {
            let degree = T::try_cast(i)? + T::one();
            *c /= degree;
        }

        coefficients.insert(0, constant);
        Ok((self.clone(), coefficients))
    }
}
impl<T: Value> display::PolynomialDisplay<T> for MonomialBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<display::Term> {
        let sign = Sign::from_coef(coef);

        let base = display::format_variable("x", None, degree);
        let coef = display::format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        let body = format!("{coef}{base}");
        Some(display::Term::new(sign, body))
    }
}

/// A monomial polynomial of the form `y = a_n * x^n + ... + a_1 * x + a_0`.
///
/// This is the what most people imagine when they hear "polynomial".
///
/// # Type Parameters
/// - `'a`: Lifetime of borrowed coefficients (if used).
/// - `T`: Numeric type (default `f64`).
pub type MonomialPolynomial<'a, T = f64> = Polynomial<'a, MonomialBasis<T>, T>;

impl<'a, T: Value> MonomialPolynomial<'a, T> {
    /// Creates a new borrowed monomial polynomial from a slice of coefficients.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients, starting from the constant term.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// ```
    pub const fn borrowed(coefficients: &'a [T]) -> Self {
        let degree = coefficients.len().saturating_sub(1);
        unsafe {
            Self::from_raw(
                MonomialBasis::default(),
                Cow::Borrowed(coefficients),
                degree,
            )
        } // Safety: Monomials expect k+1 coefficients
    }

    /// Creates a new owned monomial polynomial from a vector of coefficients.
    ///
    /// # Parameters
    /// - `coefficients`: Vec of coefficients, starting from the constant term.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::owned(vec![1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// ```
    #[must_use]
    pub const fn owned(coefficients: Vec<T>) -> Self {
        let degree = coefficients.len().saturating_sub(1);
        unsafe { Self::from_raw(MonomialBasis::default(), Cow::Owned(coefficients), degree) }
        // Safety: Monomials expect k+1 coefficients
    }
}

impl<T: Value> std::ops::Add for MonomialPolynomial<'_, T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let (lhs_basis, mut lhs_coeffs, _) = self.into_inner();
        let (_, rhs_coeffs, _) = rhs.into_inner();

        let k = lhs_coeffs.len().max(rhs_coeffs.len());
        lhs_coeffs.to_mut().resize(k, T::zero());

        for &c in rhs_coeffs.iter() {
            lhs_coeffs.to_mut()[0] += c;
        }

        // Pop off trailing zeros
        while let Some(&last) = lhs_coeffs.to_mut().last() {
            if last < T::epsilon() && lhs_coeffs.len() > 1 {
                lhs_coeffs.to_mut().pop();
            } else {
                break;
            }
        }

        let degree = lhs_coeffs.len().saturating_sub(1);
        unsafe { Self::from_raw(lhs_basis, lhs_coeffs, degree) }
    }
}
impl<T: Value> std::ops::AddAssign for MonomialPolynomial<'_, T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<T: Value> std::ops::Sub for MonomialPolynomial<'_, T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let (lhs_basis, mut lhs_coeffs, _) = self.into_inner();
        let (_, rhs_coeffs, _) = rhs.into_inner();

        let k = lhs_coeffs.len().max(rhs_coeffs.len());
        lhs_coeffs.to_mut().resize(k, T::zero());

        for (i, &c) in rhs_coeffs.iter().enumerate() {
            lhs_coeffs.to_mut()[i] -= c;
        }

        // Pop off trailing zeros
        while let Some(&last) = lhs_coeffs.to_mut().last() {
            if last < T::epsilon() && lhs_coeffs.len() > 1 {
                lhs_coeffs.to_mut().pop();
            } else {
                break;
            }
        }

        let degree = lhs_coeffs.len().saturating_sub(1);
        unsafe { Self::from_raw(lhs_basis, lhs_coeffs, degree) }
    }
}
impl<T: Value> std::ops::SubAssign for MonomialPolynomial<'_, T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<T: Value> std::ops::Mul for MonomialPolynomial<'_, T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let (lhs_basis, lhs_coeffs, _) = self.into_inner();
        let (_, rhs_coeffs, _) = rhs.into_inner();

        let mut new_coefs = vec![T::zero(); lhs_coeffs.len() + rhs_coeffs.len() - 1];

        for (i, &a) in lhs_coeffs.iter().enumerate() {
            for (j, &b) in rhs_coeffs.iter().enumerate() {
                new_coefs[i + j] += a * b;
            }
        }

        // Pop off trailing zeros
        while let Some(&last) = new_coefs.last() {
            if last < T::epsilon() && new_coefs.len() > 1 {
                new_coefs.pop();
            } else {
                break;
            }
        }

        let degree = lhs_coeffs.len().saturating_sub(1);
        unsafe { Self::from_raw(lhs_basis, Cow::Owned(new_coefs), degree) }
    }
}
impl<T: Value> std::ops::MulAssign for MonomialPolynomial<'_, T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

/// Categorizes the roots of a polynomial into real and complex roots, while removing duplicates.
pub fn categorize_roots<T: Value, F: Fn(&Complex<T>) -> Complex<T>>(
    eigenvalues: &[Complex<T>],
    solver: F,
) -> Vec<Root<T>> {
    let mut roots = Vec::new();
    let mut skip = vec![false; eigenvalues.len()];
    for i in 0..eigenvalues.len() {
        if skip[i] {
            continue;
        }

        // Skip INF/NAN roots
        if !eigenvalues[i].imaginary().is_finite() || !eigenvalues[i].real().is_finite() {
            continue;
        }

        // Skip roots where P(x) != 0
        let zero_tol = (T::one() + eigenvalues[i].norm()) * T::epsilon().sqrt();
        if solver(&eigenvalues[i]).norm() > zero_tol {
            continue;
        }

        // Skip future duplicates
        let conj_tol = T::epsilon().sqrt() * (T::one() + eigenvalues[i].norm());
        for j in (i + 1)..eigenvalues.len() {
            if (eigenvalues[i] - eigenvalues[j]).norm() < conj_tol {
                skip[j] = true;
            }
        }

        // At this point for reals we can stop
        if Value::abs(eigenvalues[i].imaginary()) < zero_tol {
            roots.push(Root::Real(eigenvalues[i].real()));
            continue;
        }

        // Complex root - we check for conjugate pairs
        for j in (i + 1)..eigenvalues.len() {
            if Value::abs(eigenvalues[i].real() - eigenvalues[j].real()) < conj_tol
                && Value::abs(eigenvalues[i].imaginary() + eigenvalues[j].imaginary()) < conj_tol
            {
                skip[j] = true;
                roots.push(Root::ComplexPair(eigenvalues[i], eigenvalues[j]));
                break;
            }
        }

        // Singular complex root - should only happen for complex coefficients
        roots.push(Root::Complex(eigenvalues[i]));
    }

    roots
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::{
        statistics::DomainNormalizer,
        test::basis_assertions::{assert_basis_functions_close, assert_basis_matrix_row},
    };

    use super::*;

    #[test]
    fn test_monomial() {
        let basis = MonomialBasis::<f64>::default();

        // Basic evaluation tests
        assert_basis_matrix_row(&basis, 2.0, &[1.0, 2.0, 4.0, 8.0]);
        assert_basis_functions_close(&basis, 0.5, &[1.0, 0.5, 0.25, 0.125], f64::EPSILON);
        assert_basis_functions_close(&basis, 1.0, &[1.0, 1.0, 1.0, 1.0], f64::EPSILON);
        assert_basis_functions_close(&basis, 2.0, &[1.0, 2.0, 4.0, 8.0], f64::EPSILON);

        // Normalization and dimension checks
        assert_eq!(basis.normalize_x(1.0), 1.0);
        assert_eq!(basis.normalize_x(2.0), 2.0);
        assert_eq!(basis.k(3), 4);
        assert_eq!(basis.k(0), 1);

        // Derivative and integral
        let poly = MonomialPolynomial::owned(vec![1.0, 2.0, 3.0, 4.0]); // 1 + 2x + 3x^2 + 4x^3
        test_derivation!(
            poly,
            &DomainNormalizer::<f64>::default(),
            with_reverse = true
        );
        test_integration!(
            poly,
            &DomainNormalizer::<f64>::default(),
            with_reverse = true
        );

        let (_, derivative) = basis
            .derivative(&[1.0, 2.0, 3.0, 4.0])
            .expect("Derivative failed");
        assert_eq!(derivative, &[2.0, 6.0, 12.0], "Derivative was incorrect");

        let (_, integral) = basis
            .integral(&[1.0, 2.0, 3.0, 4.0], 5.0)
            .expect("Integral failed");
        assert_eq!(
            integral,
            &[5.0, 1.0, 1.0, 1.0, 1.0],
            "Integral was incorrect"
        );

        // Edge cases
        // Degree 0 polynomial
        let (_, derivative0) = basis
            .derivative(&[42.0])
            .expect("Derivative failed for degree 0");
        assert_eq!(derivative0, &[0.0]);

        let (_, integral0) = basis
            .integral(&[42.0], 7.0)
            .expect("Integral failed for degree 0");
        assert_eq!(integral0, &[7.0, 42.0]);

        // Empty coefficients (should still work)
        let (_, integral_empty) = basis
            .integral(&[], 3.0)
            .expect("Integral failed for empty coefficients");
        assert_eq!(integral_empty, &[3.0]);
    }
}
