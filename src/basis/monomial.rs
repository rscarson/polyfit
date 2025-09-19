use std::fmt::Debug;

use nalgebra::{DMatrix, MatrixViewMut};

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, IntoMonomialBasis},
    display::{self, Sign, DEFAULT_PRECISION},
    error::{Error, Result},
    value::{IntClampedCast, Value},
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
}
impl<T: Value> Basis<T> for MonomialBasis<T> {
    fn from_data(_: &[(T, T)]) -> Self {
        Self::default()
    }

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

    fn solve_function(&self, j: usize, x: T) -> T {
        Value::powi(x, j.clamped_cast())
    }
}
impl<T: Value> IntoMonomialBasis<T> for MonomialBasis<T> {
    fn as_monomial(&self, _: &mut [T]) -> Result<()> {
        // Monomial basis is already in monomial form
        Ok(())
    }
}
impl<T: Value> DifferentialBasis<T> for MonomialBasis<T> {
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

    fn critical_points(&self, dx_coefs: &[T]) -> Result<Vec<T>> {
        let n = dx_coefs.len() - 1; // degree of derivative
        if n == 0 {
            return Ok(vec![]);
        }

        let mut companion = DMatrix::zeros(n, n);

        // Fill sub-diagonal with 1s
        for i in 1..n {
            companion[(i, i - 1)] = T::one();
        }

        // Fill last column
        let leading = dx_coefs[n];
        for i in 0..n {
            companion[(i, n - 1)] = -dx_coefs[i] / leading;
        }

        let eigs = companion
            .eigenvalues()
            .ok_or(Error::Algebra("Failed to compute eigenvalues"))?;

        Ok(eigs
            .into_iter()
            .filter_map(|c| {
                if Value::abs(c.imaginary()) < T::epsilon() {
                    Some(c.real())
                } else {
                    None
                }
            })
            .collect())
    }
}
impl<T: Value> IntegralBasis<T> for MonomialBasis<T> {
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

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::{test_basis_build, test_basis_functions};

    use super::*;

    #[test]
    fn test_monomial() {
        let basis = MonomialBasis::<f64>::default();

        // Basic evaluation tests
        test_basis_build!(basis, 2.0, &[1.0, 2.0, 4.0, 8.0]);
        test_basis_functions!(basis, 0.5, &[1.0, 0.5, 0.25, 0.125]);
        test_basis_functions!(basis, 1.0, &[1.0, 1.0, 1.0, 1.0]);
        test_basis_functions!(basis, 2.0, &[1.0, 2.0, 4.0, 8.0]);

        // Normalization and dimension checks
        assert_eq!(basis.normalize_x(1.0), 1.0);
        assert_eq!(basis.normalize_x(2.0), 2.0);
        assert_eq!(basis.k(3), 4);
        assert_eq!(basis.k(0), 1);

        // Derivative and integral
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
