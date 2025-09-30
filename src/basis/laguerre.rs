use nalgebra::{Dim, MatrixViewMut};

use crate::{
    basis::{Basis, IntoMonomialBasis},
    display::{self, format_coefficient, PolynomialDisplay, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
};

/// Normalized Laguerre basis for polynomial curves.
///
/// This basis uses the (generalized) Laguerre polynomials, which form an
/// orthogonal family of polynomials on the interval [0, ∞) with weight `exp(-x)`.
/// Orthogonality improves numerical stability for higher-degree polynomials.
///
/// # When to use
/// - Useful for problems with positive-valued domains or exponentially decaying data.
/// - Preferred when standard monomial fits become unstable.
///
/// # Why Laguerre?
/// - Minimizes numerical issues for high-degree polynomial approximations on [0, ∞).
/// - Naturally models decaying or exponential-type behavior.
/// - Orthogonal under weight `exp(-x)`, making coefficient estimation more stable.
#[derive(Debug, Clone)]
pub struct LaguerreBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> LaguerreBasis<T> {
    /// Creates a new Laguerre basis that normalizes inputs from the given range to [0, ∞).
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (T::zero(), T::infinity()));
        Self { normalizer }
    }

    /// Creates a new Laguerre polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Laguerre basis is defined.
    /// - `coefficients`: The coefficients for the Laguerre basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Laguerre basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::LaguerreBasis;
    /// let laguerre_poly = LaguerreBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}
impl<T: Value> Basis<T> for LaguerreBasis<T> {
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self {
        let normalizer = DomainNormalizer::from_range(x_range, (T::zero(), T::infinity()));
        Self { normalizer }
    }

    #[inline(always)]
    fn normalize_x(&self, x: T) -> T {
        self.normalizer.normalize(x)
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),
            1 => T::one() - x,
            _ => {
                let mut l0 = T::one();
                let mut l1 = T::one() - x;
                for n in 1..j {
                    let n_t = T::from_positive_int(n);
                    let l2 = ((T::two() * n_t + T::one() - x) * l1 - n_t * l0) / (n_t + T::one());
                    l0 = l1;
                    l1 = l2;
                }
                l1
            }
        }
    }

    #[inline(always)]
    fn fill_matrix_row<R: Dim, C: Dim, RS: Dim, CS: Dim>(
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

impl<T: Value> PolynomialDisplay<T> for LaguerreBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        let sign = Sign::from_coef(coef);
        let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        if degree == 0 {
            return Some(Term { sign, body: coef });
        }

        // Lₙ(x) for Laguerre polynomial
        let rank = display::unicode::subscript(&degree.to_string());
        let func = format!("L{rank}(x)");

        let glue = if coef.is_empty() || func.is_empty() {
            ""
        } else {
            "·"
        };

        let body = format!("{coef}{glue}{func}");
        Some(Term { sign, body })
    }
}

impl<T: Value> IntoMonomialBasis<T> for LaguerreBasis<T> {
    fn as_monomial(&self, coefficients: &mut [T]) -> Result<()> {
        let n = coefficients.len();
        let mut result = vec![T::zero(); n];

        for j in 0..n {
            let c_j = coefficients[j];
            for k in 0..=j {
                let sign = if k % 2 == 0 { T::one() } else { -T::one() };
                let binom = T::factorial(j) / (T::factorial(k) * T::factorial(j - k));
                let factor = binom / T::factorial(k); // divide by k!
                result[k] += c_j * sign * factor;
            }
        }

        //
        // Phase 2 - Un-normalize over x
        result = self.normalizer.denormalize_coefs(&result);

        coefficients.copy_from_slice(&result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_close, assert_fits, score::Aic, statistics::DegreeBound, LaguerreFit, Polynomial,
    };

    use super::*;

    fn get_poly() -> Polynomial<'static, LaguerreBasis<f64>> {
        let basis = LaguerreBasis::new(0.0, 100.0);
        Polynomial::from_basis(basis, &[1.0, 2.0, -0.5]).unwrap()
    }

    #[test]
    fn test_laguerre() {
        // Recover polynomial
        let poly = get_poly();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = LaguerreFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit);

        // Monomial conversion
        let mono = fit.as_monomial().unwrap();
        assert_fits!(mono, fit);

        // Solve known functions
        let basis = LaguerreBasis::new(-1.0, 1.0);
        assert_close!(basis.solve_function(0, 1.0), 1.0);
        assert_close!(basis.solve_function(1, 1.0), 0.0);
        assert_close!(basis.solve_function(2, 1.0), -0.5);
    }
}
