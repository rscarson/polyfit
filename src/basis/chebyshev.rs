use nalgebra::MatrixViewMut;

use crate::{
    basis::{Basis, DifferentialBasis, IntoMonomialBasis, OrthogonalBasis, Root, RootFindingBasis},
    display::{self, Sign, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
};

mod second_form;
pub use second_form::SecondFormChebyshevBasis;

mod third_form;
pub use third_form::ThirdFormChebyshevBasis;

/// Normalized Chebyshev basis for polynomial curves.
///
/// This basis uses the Chebyshev polynomials of the first kind, which form an
/// orthogonal family of polynomials on the interval [-1, 1]. Orthogonality makes
/// them more numerically stable than the standard monomial basis, especially
/// for higher-degree polynomials.
///
/// By default, inputs are normalized so that the evaluation domain [`x_min`, `x_max`]
/// is mapped onto [-1, 1]. This allows Chebyshev polynomials to be used naturally
/// with arbitrary input ranges while retaining their stability properties.
///
/// # When to use
/// - Use when fitting polynomials to data in arbitrary input ranges.
/// - Prefer for higher-degree polynomials where stability is a concern.
///
/// # Why Chebyshev?
/// - Minimizes **Runge’s phenomenon** in polynomial interpolation.
/// - Provides near-optimal polynomial approximations with lower error.
/// - Useful for fitting problems where monomials become unstable.
#[derive(Debug, Clone)]
pub struct ChebyshevBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> ChebyshevBasis<T> {
    /// Creates a new Chebyshev basis that normalizes inputs from the given range to [-1, 1].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (-T::one(), T::one()));
        Self { normalizer }
    }

    /// Creates a Chebyshev basis from an existing domain normalizer.
    pub fn from_normalizer(normalizer: DomainNormalizer<T>) -> Self {
        Self { normalizer }
    }

    /// Creates a new Chebyshev polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Chebyshev basis is defined.
    /// - `coefficients`: The coefficients for the Chebyshev basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Chebyshev basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::ChebyshevBasis;
    /// let chebyshev_poly = ChebyshevBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}
impl<T: Value> Basis<T> for ChebyshevBasis<T> {
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self {
        let normalizer = DomainNormalizer::from_range(x_range, (-T::one(), T::one()));
        Self { normalizer }
    }

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
                1 => x,
                _ => T::two() * x * row[j - 1] - row[j - 2],
            }
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
    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(), // T0(x) = 1
            1 => x,        // T1(x) = x
            _ => {
                // Tn(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
                let mut t0 = T::one();
                let mut t1 = x;
                let mut t = T::zero();

                for _ in 2..=j {
                    t = T::two() * x * t1 - t0;
                    t0 = t1;
                    t1 = t;
                }

                t
            }
        }
    }
}

impl<T: Value> OrthogonalBasis<T> for ChebyshevBasis<T> {
    fn gauss_nodes(&self, n: usize) -> Vec<(T, T)> {
        let mut nodes = Vec::with_capacity(n);
        let n2 = T::two() * T::from_positive_int(n);
        let w = T::pi() / T::from_positive_int(n);
        for k in 1..=n {
            let tk1 = T::two() * T::from_positive_int(k) - T::one();
            let x = (T::pi() * tk1 / n2).cos();
            nodes.push((x, w));
        }

        nodes
    }

    fn gauss_normalization(&self, n: usize) -> T {
        if n == 0 {
            T::pi()
        } else {
            T::pi() / T::two()
        }
    }
}

impl<T: Value> IntoMonomialBasis<T> for ChebyshevBasis<T> {
    fn as_monomial(&self, coefficients: &mut [T]) -> Result<()> {
        let n = coefficients.len() - 1;

        //
        // Phase 1 - Chebyshev -> Monomial in x'
        //

        let mut monomial_prime = vec![T::zero(); n + 1];
        let mut tkm1 = vec![T::one()]; //T0 = 1
        let mut tk = vec![T::zero(), T::one()]; // T1 = x

        monomial_prime[0] = coefficients[0];
        if n >= 1 {
            monomial_prime[1] = coefficients[1];
        }

        for (k, &c) in coefficients.iter().enumerate().skip(2) {
            // Tk+1 = 2x*Tk - T_{k-1}
            let mut tk1 = vec![T::zero(); k + 1]; // degree k+1

            // 2x * Tk
            for (i, &coef) in tk.iter().enumerate() {
                if coef != T::zero() {
                    tk1[i + 1] += coef * T::two();
                }
            }

            // subtract T_{k-1}
            for (i, &coef) in tkm1.iter().enumerate() {
                tk1[i] -= coef;
            }

            // accumulate into monomial_hat
            for (i, &coef) in tk1.iter().enumerate() {
                monomial_prime[i] += c * coef;
            }

            tkm1 = tk;
            tk = tk1;
        }

        //
        // Phase 2 - Un-normalize over x
        let monomial = self.normalizer.denormalize_coefs(&monomial_prime);

        // Phase 3 - Write back to coefficients
        coefficients.copy_from_slice(&monomial);
        Ok(())
    }
}

fn format_cheb_term<T: Value>(function: &str, degree: i32, coef: T) -> Option<display::Term> {
    let sign = Sign::from_coef(coef);

    let x = display::unicode::subscript("s");
    let x = format!("x{x}");

    let rank = display::unicode::subscript(&degree.to_string());
    let func = if degree > 0 {
        format!("{function}{rank}({x})")
    } else {
        String::new()
    };
    let coef = display::format_coefficient(coef, degree, DEFAULT_PRECISION)?;

    let glue = if coef.is_empty() || func.is_empty() {
        ""
    } else {
        "·"
    };

    let body = format!("{coef}{glue}{func}");
    Some(display::Term::new(sign, body))
}

impl<T: Value> display::PolynomialDisplay<T> for ChebyshevBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<display::Term> {
        format_cheb_term("T", degree, coef)
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

impl<T: Value> RootFindingBasis<T> for ChebyshevBasis<T> {
    fn roots(&self, coefs: &[T]) -> Result<Vec<Root<T>>> {
        let mut roots = Vec::with_capacity(coefs.len() - 1);
        // Xk = cos((2k+1)π/(2n)) for k=0..n-1
        // All roots are real in [-1, 1]
        let n = coefs.len() - 1;
        let two_n = T::from_positive_int(2 * n);
        for k in 0..coefs.len() - 1 {
            let k = n - 1 - k; // Reverse order to get ascending roots

            let tk1 = 2 * k + 1;
            let tk1 = T::from_positive_int(tk1);

            let x = (T::pi() * tk1 / two_n).cos();
            let x = self.denormalize_x(x);
            roots.push(Root::Real(x));
        }

        Ok(roots)
    }
}

impl<T: Value> DifferentialBasis<T> for ChebyshevBasis<T> {
    type B2 = SecondFormChebyshevBasis<T>;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)> {
        // Drop the constant term and multiply each coefficient by its degree
        let mut coefs = coefficients[1..].to_vec();
        for (i, c) in coefs.iter_mut().enumerate() {
            *c *= T::from_positive_int(i + 1);
        }

        let basis = SecondFormChebyshevBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use core::f64;

    use crate::{
        assert_fits, function,
        score::Aic,
        statistics::DegreeBound,
        test::basis_assertions::{
            assert_basis_functions_close, assert_basis_matrix_row, assert_basis_normalizes,
            assert_basis_orthogonal,
        },
        ChebyshevFit,
    };

    use super::*;

    #[test]
    fn test_chebyshev() {
        // Large polynomial recover test
        function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2 + 5.0 x^3 + 4.0 x^4 + 3.0 x^5 + 2.0 x^6);
        let data = test.solve_range(0.0..=1000.0, 10.0);
        let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        let basis = fit.basis().clone();

        // Orthogonality test points
        assert_basis_orthogonal(&basis, 7, 100, 1e-12);

        // Basic evaluations at x = 0.5
        let x_norm = basis.normalize_x(0.5);
        assert_basis_matrix_row(
            &basis,
            0.5,
            &[
                /*T0*/ 1.0,
                /*T1*/ x_norm,
                /*T2*/ 2.0 * x_norm * x_norm - 1.0,
                /*T3*/ 2.0 * x_norm * (2.0 * x_norm * x_norm - 1.0) - x_norm,
            ],
        );

        assert_basis_functions_close(&basis, 0.0, &[1.0, 0.0, -1.0, 0.0], f64::EPSILON);
        assert_basis_functions_close(&basis, 1.0, &[1.0, 1.0, 1.0, 1.0], f64::EPSILON);
        assert_basis_functions_close(&basis, -1.0, &[1.0, -1.0, 1.0, -1.0], f64::EPSILON);

        // Normalization (should map x in [-1,1])
        assert_basis_normalizes(&basis, (0.0, 1000.0), (-1.0, 1.0));

        // k() checks
        assert_eq!(basis.k(3), 4);
        assert_eq!(basis.k(0), 1);

        // Now we convert the fit to monomial and compare the solutions
        let monomial_fit = fit.as_monomial().expect("Failed to convert to monomial");
        assert_fits!(&monomial_fit, &fit, 1.0);

        // Calculus tests - go T(x) -> U(x) -> V(x) -> U(x) -> T(x)
        // First let's save the lowest 2 coefficients for comparison
        let poly = ChebyshevBasis::new_polynomial((0.0, 1000.0), &[3.0, 2.0, 1.5, 3.0]).unwrap();
        test_derivation!(poly, &poly.basis().normalizer, with_reverse = true);
    }
}
