use nalgebra::MatrixViewMut;

use crate::{
    basis::Basis,
    display::{
        self, format_coefficient, format_variable, PolynomialDisplay, Sign, Term, DEFAULT_PRECISION,
    },
    error::Result,
    statistics::DomainNormalizer,
    value::{IntClampedCast, Value},
    Polynomial,
};

/// Normalized logarithmic basis for polynomial-like curves.
///
/// This basis uses powers of the natural logarithm, forming a series of the form
/// `1, ln(x), ln(x)^2, ...`. Inputs are normalized so that the domain [`x_min`, `x_max`]
/// is mapped onto a positive interval `[ε, ∞)`, ensuring that `ln(x)` is well-defined.
///
/// # When to use
/// - Use when the relationship between `x` and `y` is expected to be logarithmic or
///   multiplicative in nature.
/// - Suitable for fitting models where standard monomial bases poorly capture slow-growth
///   or exponential-scaling behavior.
///
/// # Why logarithmic?
/// - Captures multiplicative and slow-growing trends naturally.
/// - Helps linearize relationships for regression or least-squares fitting.
/// - Avoids numerical instability from taking logs of non-positive values.
#[derive(Debug, Clone)]
pub struct LogarithmicBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> LogarithmicBasis<T> {
    /// Creates a new Laguerre basis that normalizes inputs from the given range to [1, ∞).
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (T::one(), T::infinity()));
        Self { normalizer }
    }

    /// Creates a new Logarithmic polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Logarithmic basis is defined.
    /// - `coefficients`: The coefficients for the Logarithmic basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Logarithmic basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::LogarithmicBasis;
    /// let log_poly = LogarithmicBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(x_range: (T, T), coefficients: &[T]) -> Result<Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}

impl<T: Value> Basis<T> for LogarithmicBasis<T> {
    fn from_data(data: &[(T, T)]) -> Self {
        let normalizer =
            DomainNormalizer::from_data(data.iter().map(|(x, _)| *x), (T::one(), T::infinity()));
        Self { normalizer }
    }

    fn normalize_x(&self, x: T) -> T {
        self.normalizer.normalize(x)
    }

    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            _ if x <= T::zero() => {
                T::nan() // Logarithm undefined for non-positive values. This just makes the behavior explicit.
            }
            0 => T::one(),
            _ => Value::powi(x.ln(), j.clamped_cast()),
        }
    }

    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<T, R, C, RS, CS>,
    ) {
        for j in start_index..row.ncols() {
            row[j] = self.solve_function(j, x);
        }
    }
}

impl<T: Value> PolynomialDisplay<T> for LogarithmicBasis<T> {
    fn format_scaling_formula(&self) -> Option<String> {
        if self.normalizer.src_range().0 == T::zero() {
            return None;
        }
        Some(format!("x' = x + {}", self.normalizer.src_range().0))
    }

    fn format_term(&self, degree: i32, coef: T) -> Option<display::Term> {
        let x = if self.normalizer.src_range().0 == T::zero() {
            "x"
        } else {
            "x'"
        };

        let sign = Sign::from_coef(coef);
        let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        let func = format_variable(&format!("ln({x})"), None, degree);
        let glue = if coef.is_empty() || func.is_empty() {
            ""
        } else {
            "·"
        };

        // ln(x')^degree
        let body = format_variable(&format!("{coef}{glue}{func}"), None, degree);
        Some(Term { sign, body })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use crate::{
        assert_close, assert_fits, score::Aic, statistics::DegreeBound, test_basis_normalizes,
        LogarithmicFit, Polynomial,
    };

    use super::*;

    fn get_poly() -> Polynomial<'static, LogarithmicBasis<f64>> {
        let basis = LogarithmicBasis::new(0.0, 100.0);
        Polynomial::from_basis(basis, &[1.0, 2.0, -0.5]).unwrap()
    }

    #[test]
    #[allow(clippy::approx_constant, clippy::unreadable_literal)]
    fn test_logarithmic() {
        // Recover polynomial
        let poly = get_poly();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = LogarithmicFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit);

        // Solve known values
        let basis = LogarithmicBasis::new(0.0, 100.0);
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), -0.6931471805599453);
        assert_close!(basis.solve_function(2, 0.5), 0.4804530139182014);
        assert_close!(basis.solve_function(3, 0.5), -0.33302465198892944);

        // Normalization (should map x in [0, 100] to [ε, ∞))
        test_basis_normalizes!(basis, 0.0..100.0, 1.0..101.0);

        // k() checks
        assert_eq!(basis.k(3), 4);
        assert_eq!(basis.k(0), 1);
    }
}
