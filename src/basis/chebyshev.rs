use std::ops::RangeInclusive;

use nalgebra::{Complex, ComplexField, DMatrix, MatrixViewMut};

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
            };
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

    fn gauss_weight(&self, x: T) -> T {
        T::one() / (T::one() - x * x).sqrt()
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

impl<T: Value> DifferentialBasis<T> for ChebyshevBasis<T> {
    type B2 = SecondFormChebyshevBasis<T>;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)> {
        // Drop the constant term and multiply each coefficient by its degree
        let mut coefs = coefficients[1..].to_vec();
        let scale = self.normalizer.scale();
        for (i, c) in coefs.iter_mut().enumerate() {
            let n = T::from_positive_int(i) + T::one(); // degree starts at 1 for the first term
            *c = *c * n * scale;
        }

        let basis = SecondFormChebyshevBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}

impl<T: Value> RootFindingBasis<T> for ChebyshevBasis<T> {
    fn roots(&self, coefs: &[T], x_range: RangeInclusive<T>) -> Result<Vec<Root<T>>> {
        let n = coefs.len() - 1;
        if n == 0 {
            return Ok(vec![]);
        }

        let mut mat = DMatrix::<T>::zeros(n, n);
        let half = T::one() / T::two();

        for i in 1..n {
            mat[(i, i - 1)] = half;
        }

        for i in 0..n - 1 {
            mat[(i, i + 1)] = half;
        }

        mat[(0, 1)] = T::one();

        let a_n = coefs[n];
        for j in 0..n {
            mat[(n - 1, j)] -= coefs[j] / (a_n * T::two());
        }

        let eigs: Vec<Complex<T>> = mat.complex_eigenvalues().into_iter().copied().collect();

        // Filter and categorize roots
        let mut roots = Root::roots_from_complex(&eigs, |z| self.complex_y(*z, coefs));
        for root in &mut roots {
            match root {
                Root::Real(r) => {
                    // Normalize real root back to original x-range if normalizer is provided
                    let denorm = self.normalizer.denormalize(*r);
                    *root = Root::Real(denorm);
                }
                Root::Complex(c) => {
                    let denorm = Complex::new(
                        self.normalizer.denormalize(c.real()),
                        self.normalizer.denormalize(c.imaginary()),
                    );
                    *root = Root::Complex(denorm);
                }
                Root::ComplexPair(c1, c2) => {
                    let denorm1 = Complex::new(
                        self.normalizer.denormalize(c1.real()),
                        self.normalizer.denormalize(c1.imaginary()),
                    );
                    let denorm2 = Complex::new(
                        self.normalizer.denormalize(c2.real()),
                        self.normalizer.denormalize(c2.imaginary()),
                    );
                    *root = Root::ComplexPair(denorm1, denorm2);
                }
            }
        }

        // Remove roots outside the specified x_range
        let x_min = *x_range.start();
        let x_max = *x_range.end();
        let roots = roots
            .into_iter()
            .filter(|root| match root {
                Root::Real(r) => *r >= x_min && *r <= x_max,
                Root::Complex(_) | Root::ComplexPair(_, _) => true, // Keep complex roots
            })
            .collect();

        Ok(roots)
    }

    fn complex_y(&self, z: Complex<T>, coefs: &[T]) -> Complex<T> {
        let n = coefs.len();
        if n == 0 {
            return Complex::new(T::zero(), T::zero());
        }

        let two = Complex::from_real(T::two());
        let mut b_next = Complex::new(T::zero(), T::zero());
        let mut b_next2 = Complex::new(T::zero(), T::zero());

        for &a_k in coefs[1..].iter().rev() {
            let b = two * z * b_next - b_next2 + Complex::from_real(a_k);
            b_next2 = b_next;
            b_next = b;
        }

        // Now handle a0
        Complex::from_real(coefs[0]) + z * b_next - b_next2
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use core::f64;

    use crate::{
        assert_close, assert_fits,
        score::Aic,
        statistics::DegreeBound,
        test::basis_assertions::{
            self, assert_basis_functions_close, assert_basis_matrix_row, assert_basis_normalizes,
            assert_basis_orthogonal,
        },
        ChebyshevFit,
    };

    use super::*;

    #[test]
    fn test_regression_derivative() {
        let points: &[(f64, f64)] = &[
            (61.0, 110.2647),
            (62.0, 110.8006),
            (63.0, 111.3338),
            (64.0, 111.8636),
            (65.0, 112.3895),
            (66.0, 112.9110),
            (67.0, 113.4280),
        ];
        let fit = ChebyshevFit::new_auto(points, DegreeBound::Custom(3), &Aic).unwrap();
        println!("Fit: {fit}");
        let dx_1 = fit.as_polynomial().derivative().unwrap();
        println!("dx/dx: {dx_1}");

        let mono_form = fit.as_monomial().unwrap();
        let dx_2 = mono_form.derivative().unwrap();
        println!("monomial form: {mono_form}");

        // x=61 should have the same y in both derivatives
        assert_close!(
            dx_1.y(61.0),
            dx_2.y(61.0),
            epsilon = 1e-5,
            "Derivatives differ at x=61"
        );
    }

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
        let polyt = ChebyshevBasis::new_polynomial((0.0, 1000.0), &[3.0, 2.0, 1.5, 3.0]).unwrap();
        basis_assertions::test_reversible_derivation(&polyt, &polyt.basis().normalizer);

        // Test root finding
        basis_assertions::test_root_finding(&polyt, 0.0..=1000.0);
    }
}
