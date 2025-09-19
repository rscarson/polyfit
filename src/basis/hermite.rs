use nalgebra::{Dim, MatrixViewMut};

use crate::{
    basis::{Basis, IntoMonomialBasis},
    display::{self, format_coefficient, PolynomialDisplay, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    value::{IntClampedCast, Value},
};

/// Normalized Physicists’ Hermite basis for polynomial curves.
///
/// This basis uses the Physicists’ Hermite polynomials `H_n(x)`, which form an
/// orthogonal family of polynomials with respect to the weight `exp(-x^2)`
/// over the interval (-∞, ∞). Orthogonality improves numerical stability
/// compared to monomial bases, especially for higher-degree polynomials.
///
/// # When to use
/// - Use for polynomial approximations in probability, physics, or Gaussian-weighted data.
/// - Prefer for higher-degree polynomials where stability is a concern.
///
/// # Why Physicists’ Hermite?
/// - Orthogonal with respect to `exp(-x^2)`
/// - Reduces numerical instability in high-degree fits
/// - Convenient for physics problems (quantum mechanics, oscillator basis, etc.)
#[derive(Debug, Clone, Copy)]
pub struct PhysicistsHermiteBasis<T: Value> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: Value> Default for PhysicistsHermiteBasis<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Value> PhysicistsHermiteBasis<T> {
    /// Create a new Physicists' Hermite basis
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Value> Basis<T> for PhysicistsHermiteBasis<T> {
    fn from_data(_: &[(T, T)]) -> Self {
        Self::new()
    }

    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),
            1 => T::two() * x,
            _ => {
                let mut h0 = T::one();
                let mut h1 = T::two() * x;
                for n in 1..j {
                    let n = T::from_positive_int(n);
                    let h2 = T::two() * x * h1 - T::two() * n * h0;
                    h0 = h1;
                    h1 = h2;
                }
                h1
            }
        }
    }

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
impl<T: Value> PolynomialDisplay<T> for PhysicistsHermiteBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        format_herm(degree, coef)
    }
}
impl<T: Value> IntoMonomialBasis<T> for PhysicistsHermiteBasis<T> {
    fn as_monomial(&self, coefficients: &mut [T]) -> Result<()> {
        let n = coefficients.len();
        let mut result = vec![T::zero(); n]; // max degree = n-1

        for j in 0..n {
            let c_j = coefficients[j];
            for k in 0..=(j / 2) {
                let sign = if k % 2 == 0 { T::one() } else { -T::one() };
                let two_pow = Value::powi(T::two(), (j - 2 * k).clamped_cast());
                let factor = T::factorial(j) / (T::factorial(k) * T::factorial(j - 2 * k));
                let monomial_degree = j - 2 * k;
                result[monomial_degree] += c_j * sign * two_pow * factor;
            }
        }

        coefficients.copy_from_slice(&result);
        Ok(())
    }
}

/// Normalized Probabilists’ Hermite basis for polynomial curves.
///
/// This basis uses the Probabilists’ Hermite polynomials `He_n(x)`, which form an
/// orthogonal family of polynomials with respect to the weight `exp(-x^2 / 2)`
/// over the interval (-∞, ∞). Orthogonality improves numerical stability
/// compared to monomial bases, especially for higher-degree polynomials.
///
/// # When to use
/// - Use for polynomial approximations in statistics, stochastic processes, or Gaussian-weighted data.
/// - Prefer for higher-degree polynomials where stability is a concern.
///
/// # Why Probabilists’ Hermite?
/// - Orthogonal with respect to `exp(-x^2 / 2)`
/// - Reduces numerical instability in high-degree fits
/// - Convenient for probabilistic modeling and cumulant expansions

#[derive(Debug, Clone, Copy)]
pub struct ProbabilistsHermiteBasis<T: Value> {
    _marker: std::marker::PhantomData<T>,
}
impl<T: Value> Default for ProbabilistsHermiteBasis<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Value> ProbabilistsHermiteBasis<T> {
    /// Create a new Probabalists' Hermite basis
    #[must_use]
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Value> Basis<T> for ProbabilistsHermiteBasis<T> {
    fn from_data(_: &[(T, T)]) -> Self {
        Self::new()
    }

    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),
            1 => x,
            _ => {
                let mut h0 = T::one();
                let mut h1 = x;
                for n in 1..j {
                    let n = T::from_positive_int(n);
                    let h2 = x * h1 - n * h0;
                    h0 = h1;
                    h1 = h2;
                }
                h1
            }
        }
    }

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
impl<T: Value> PolynomialDisplay<T> for ProbabilistsHermiteBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        format_herm(degree, coef)
    }
}
impl<T: Value> IntoMonomialBasis<T> for ProbabilistsHermiteBasis<T> {
    fn as_monomial(&self, coefficients: &mut [T]) -> Result<()> {
        let n = coefficients.len();
        let mut result = vec![T::zero(); n]; // degree n-1

        for j in 0..n {
            let c_j = coefficients[j];
            for k in 0..=(j / 2) {
                let sign = if k % 2 == 0 { T::one() } else { -T::one() };
                let factor = T::factorial(j) / (T::factorial(k) * T::factorial(j - 2 * k));
                let monomial_degree = j - 2 * k;
                result[monomial_degree] += c_j * sign * factor;
            }
        }

        coefficients.copy_from_slice(&result);
        Ok(())
    }
}

fn format_herm<T: Value>(degree: i32, coef: T) -> Option<Term> {
    let sign = Sign::from_coef(coef);
    let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

    if degree == 0 {
        return Some(Term { sign, body: coef });
    }

    // Heₙ(x) for Hermite polynomial
    let rank = display::unicode::subscript(&degree.to_string());
    let func = format!("He{rank}(x)");

    let body = format!("{coef}{func}");
    Some(Term { sign, body })
}

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;
    use crate::{
        assert_close, assert_fits,
        statistics::{DegreeBound, ScoringMethod},
        PhysicistsHermiteFit, Polynomial, ProbabilistsHermiteFit,
    };

    fn get_poly<B: Basis<f64> + PolynomialDisplay<f64> + Default>() -> Polynomial<'static, B> {
        Polynomial::from_basis(B::default(), &[1.0, 2.0, -0.5]).unwrap()
    }

    #[test]
    fn test_physicists_hermite() {
        // Polynomial recovery
        let poly = get_poly::<PhysicistsHermiteBasis<f64>>();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = PhysicistsHermiteFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC)
            .unwrap();
        assert_fits!(&poly, &fit);

        // Monomial conversion
        let mono = fit.as_monomial().unwrap();
        assert_fits!(mono, fit);

        // Solve known functions
        let basis = PhysicistsHermiteBasis::new();
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), 1.0);
        assert_close!(basis.solve_function(2, 0.5), -1.0);
        assert_close!(basis.solve_function(3, 0.5), -5.0);

        // Orthogonality test points
        // todo
    }

    #[test]
    fn test_probabilists_hermite() {
        // Polynomial recovery
        let poly = get_poly::<ProbabilistsHermiteBasis<f64>>();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = ProbabilistsHermiteFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC)
            .unwrap();
        assert_fits!(&poly, &fit);

        // Monomial conversion
        let mono = fit.as_monomial().unwrap();
        assert_fits!(mono, fit);

        // Solve known functions
        let basis = ProbabilistsHermiteBasis::new();
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), 0.5);
        assert_close!(basis.solve_function(2, 0.5), -0.75);
        assert_close!(basis.solve_function(3, 0.5), -1.375);

        // Orthogonality test points
        // todo
    }
}
