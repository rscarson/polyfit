use nalgebra::MatrixViewMut;

use crate::{
    basis::{Basis, IntoMonomialBasis},
    display::{self, Sign, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::{IntClampedCast, Value},
};

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

    // Simple binomial coefficient using Pascal triangle
    fn binomial(n: usize, k: usize) -> Result<T> {
        if k > n {
            return Ok(T::zero());
        }

        let mut res: u64 = 1;
        for i in 0..k {
            res = res * (n - i) as u64 / (i + 1) as u64;
        }

        T::try_cast(res)
    }
}
impl<T: Value> Basis<T> for ChebyshevBasis<T> {
    fn from_data(data: &[(T, T)]) -> Self {
        let normalizer =
            DomainNormalizer::from_data(data.iter().map(|(x, _)| *x), (-T::one(), T::one()));
        Self { normalizer }
    }

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

    fn normalize_x(&self, x: T) -> T {
        self.normalizer.normalize(x)
    }

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
        //
        let (x_min, x_max) = self.normalizer.src_range();
        let alpha = T::two() / (x_max - x_min);
        let beta = -(x_max + x_min) / (x_max - x_min);

        let mut monomial = vec![T::zero(); n + 1];
        for j in 0..=n {
            let aj = monomial_prime[j];
            if aj == T::zero() {
                continue;
            }

            for m in 0..=j {
                let binom = Self::binomial(j, m)?;
                monomial[m] += aj
                    * binom
                    * Value::powi(alpha, m.clamped_cast())
                    * Value::powi(beta, (j - m).clamped_cast());
            }
        }

        // Phase 3 - Write back to coefficients
        coefficients.copy_from_slice(&monomial);
        Ok(())
    }
}

impl<T: Value> display::PolynomialDisplay<T> for ChebyshevBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<display::Term> {
        let sign = Sign::from_coef(coef);

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        let rank = display::unicode::subscript(&degree.to_string());
        let func = if degree > 0 {
            format!("T{rank}({x})")
        } else {
            String::new()
        };
        let coef = display::format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        let body = format!("{coef}{func}");
        Some(display::Term::new(sign, body))
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        assert_fits, function,
        statistics::{DegreeBound, ScoringMethod},
        test_basis_build, test_basis_functions, test_basis_normalizes, test_basis_orthogonal,
        transforms::ApplyNoise,
        ChebyshevFit,
    };

    use super::*;

    #[test]
    fn test_chebyshev() {
        function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2 + 5.0 x^3 + 4.0 x^4 + 3.0 x^5 + 2.0 x^6);
        let data = test
            .solve_range(0.0..1000.0, 10.0)
            .apply_normal_noise(0.1, None);
        let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        let basis = fit.basis().clone();

        // Chebyshev is orthogonal over:
        // x_k = cos((2k+1)/(2n) * π) for k = 0..n-1
        let mut points = Vec::with_capacity(fit.degree());
        let n = fit.degree() as f64;
        for i in 0..fit.degree() {
            let x = ((2.0 * i as f64 + 1.0) / (2.0 * n) * PI).cos();
            points.push(x);
        }
        test_basis_orthogonal!(basis, &points);

        // Basic evaluations at x = 0.5
        let x_norm = basis.normalize_x(0.5);
        test_basis_build!(
            basis,
            0.5,
            &[
                /*T0*/ 1.0,
                /*T1*/ x_norm,
                /*T2*/ 2.0 * x_norm * x_norm - 1.0,
                /*T3*/ 2.0 * x_norm * (2.0 * x_norm * x_norm - 1.0) - x_norm
            ]
        );
        test_basis_functions!(basis, 0.0, &[1.0, 0.0, -1.0, 0.0]);
        test_basis_functions!(basis, 1.0, &[1.0, 1.0, 1.0, 1.0]);
        test_basis_functions!(basis, -1.0, &[1.0, -1.0, 1.0, -1.0]);

        // Normalization (should map x in [-1,1])
        test_basis_normalizes!(basis, 0.0..1000.0, -1.0..1.0);

        // k() checks
        assert_eq!(basis.k(3), 4);
        assert_eq!(basis.k(0), 1);

        // Now we convert the fit to monomial and compare the solutions
        let monomial_fit = fit.as_monomial().expect("Failed to convert to monomial");
        assert_fits!(&monomial_fit, &fit, 1.0);
    }
}
