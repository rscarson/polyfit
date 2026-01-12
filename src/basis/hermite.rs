use nalgebra::{Dim, MatrixViewMut};

use crate::{
    basis::{Basis, DifferentialBasis, IntoMonomialBasis, OrthogonalBasis},
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
pub struct PhysicistsHermiteBasis<T: Value = f64> {
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

    /// Creates a new Hermite polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Hermite basis is defined.
    /// - `coefficients`: The coefficients for the Hermite basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Hermite basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::PhysicistsHermiteBasis;
    /// let hermite_poly = PhysicistsHermiteBasis::new_polynomial(&[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(coefficients: &[T]) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new();
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}

impl<T: Value> Basis<T> for PhysicistsHermiteBasis<T> {
    fn from_range(_x_range: std::ops::RangeInclusive<T>) -> Self {
        Self::new()
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

impl<T: Value> PolynomialDisplay<T> for PhysicistsHermiteBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        format_herm(degree, coef)
    }
}

impl<T: Value> DifferentialBasis<T> for PhysicistsHermiteBasis<T> {
    type B2 = PhysicistsHermiteBasis<T>;

    fn derivative(&self, a: &[T]) -> Result<(Self::B2, Vec<T>)> {
        let n = a.len();
        let mut b = Vec::with_capacity(n);

        for k in 0..n {
            // He'_k = 2*(k+1) * He_{k+1}
            let val = if k + 1 < n {
                a[k + 1] * T::from_positive_int(2 * (k + 1))
            } else {
                T::zero()
            };
            b.push(val);
        }

        Ok((*self, b))
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

impl<T: Value> OrthogonalBasis<T> for PhysicistsHermiteBasis<T> {
    fn gauss_nodes(&self, n: usize) -> Vec<(T, T)> {
        if n == 0 {
            // ∫ e^{-x²} dx = √π
            return vec![(T::zero(), T::pi().sqrt())];
        }

        let mut a = nalgebra::DMatrix::<T>::zeros(n, n);

        // Build symmetric tridiagonal Jacobi matrix for Hermite
        for i in 1..n {
            let b = (T::from_positive_int(i) / T::two()).sqrt();
            a[(i, i - 1)] = b;
            a[(i - 1, i)] = b;
        }

        // Eigen-decomposition
        let eig = a.symmetric_eigen();

        let mut nodes = Vec::with_capacity(n);
        let v0 = eig.eigenvectors.row(0);

        // weights from first row of eigenvector matrix
        for (xi, vi0) in eig.eigenvalues.iter().zip(v0.iter()) {
            let x = *xi;
            let w = *vi0 * *vi0 * T::pi().sqrt();
            nodes.push((x, w));
        }

        // sort ascending
        nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nodes
    }

    fn gauss_weight(&self, x: T) -> T {
        // w(x) = e^{-x²}
        (-x * x).exp()
    }

    fn gauss_normalization(&self, n: usize) -> T {
        // ∫ H_n(x)^2 e^{-x²} dx = 2^n n! √π
        (Value::powi(T::two(), n.clamped_cast::<i32>())) * T::factorial(n) * T::pi().sqrt()
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
pub struct ProbabilistsHermiteBasis<T: Value = f64> {
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

    /// Creates a new Hermite polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Hermite basis is defined.
    /// - `coefficients`: The coefficients for the Hermite basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Hermite basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::ProbabilistsHermiteBasis;
    /// let hermite_poly = ProbabilistsHermiteBasis::new_polynomial(&[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(coefficients: &[T]) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new();
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}

impl<T: Value> Basis<T> for ProbabilistsHermiteBasis<T> {
    fn from_range(_: std::ops::RangeInclusive<T>) -> Self {
        Self::new()
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

impl<T: Value> PolynomialDisplay<T> for ProbabilistsHermiteBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        format_herm(degree, coef)
    }
}

impl<T: Value> DifferentialBasis<T> for ProbabilistsHermiteBasis<T> {
    type B2 = ProbabilistsHermiteBasis<T>;

    fn derivative(&self, a: &[T]) -> Result<(Self::B2, Vec<T>)> {
        let n = a.len();
        let mut b = Vec::with_capacity(n);

        for k in 0..n {
            // B_k = (k+1) * A_{k+1}
            let val = if k + 1 < n {
                a[k + 1] * T::from_positive_int(k + 1)
            } else {
                T::zero()
            };
            b.push(val);
        }

        Ok((*self, b))
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

impl<T: Value> OrthogonalBasis<T> for ProbabilistsHermiteBasis<T> {
    fn gauss_nodes(&self, n: usize) -> Vec<(T, T)> {
        let phys = PhysicistsHermiteBasis::<T>::default().gauss_nodes(n);
        let sqrt2 = T::two().sqrt();

        phys.into_iter()
            .map(|(x, w)| (x * sqrt2, w * sqrt2))
            .collect()
    }

    fn gauss_weight(&self, x: T) -> T {
        // w(x) = e^{-x²/2}
        (-x * x / T::two()).exp()
    }

    fn gauss_normalization(&self, n: usize) -> T {
        // ∫ He_n(x)² e^{-x²/2} dx = √(2π) n!
        T::two_pi().sqrt() * T::factorial(n)
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

    let glue = if coef.is_empty() || func.is_empty() {
        ""
    } else {
        "·"
    };

    let body = format!("{coef}{glue}{func}");
    Some(Term { sign, body })
}

#[cfg(test)]
mod tests {
    use std::f64;

    use super::*;
    use crate::{
        assert_close, assert_fits,
        score::Aic,
        statistics::{DegreeBound, DomainNormalizer},
        test::basis_assertions::assert_basis_orthogonal,
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
        let fit = PhysicistsHermiteFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
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

        let poly = PhysicistsHermiteBasis::new_polynomial(&[3.0, 2.0, 1.5]).unwrap();
        test_derivation!(poly, &DomainNormalizer::default());

        // Orthogonality test points
        assert_basis_orthogonal(&basis, 4, 100, 1e-12);
    }

    #[test]
    fn test_probabilists_hermite() {
        // Polynomial recovery
        let poly = get_poly::<ProbabilistsHermiteBasis<f64>>();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = ProbabilistsHermiteFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
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

        // Calculus tests
        let poly = ProbabilistsHermiteBasis::new_polynomial(&[3.0, 2.0, 1.5]).unwrap();
        test_derivation!(poly, &DomainNormalizer::default());

        // Orthogonality test points
        assert_basis_orthogonal(&basis, 4, 100, 1e-12);
    }
}
