//! Polynomial basis functions for curve fitting
//!
//! This module defines the [`Basis`] trait, which abstracts polynomial basis functions
//! for use in curve fitting. Implementations are provided for common bases, and users
//! can implement their own custom bases as needed.
//!
//! Also contains [`IntoMonomialBasis`], for bases which can be converted to monomial form.
//!
//! # Provided Bases
//! - [`MonomialBasis`]: The standard monomial basis, i.e., 1, x, x², … xⁿ. Simple but can
//!   become numerically unstable for high-degree polynomials.
//! - [`ChebyshevBasis`]: Orthogonal Chebyshev polynomials defined on [-1, 1], which reduce
//!   Runge's phenomenon and are more stable for high-degree fits.
//!
//! # Selecting a Basis
//! - **For most users**, [`ChebyshevBasis`] is recommended because it produces more
//!   stable fits for typical datasets.
//! - Use [`MonomialBasis`] if you specifically need the standard xⁿ form or are dealing
//!   with low-degree polynomials.
//!
//! # Rolling Your Own
//! To implement a custom basis:
//! 1. Implement the `Basis<T>` trait for your type.
//! 2. Define how to populate a row of the Vandermonde-style matrix in `fill_matrix_row`.
//! 3. Implement `solve(&self, x: T, coefficients: &[T]) -> T` to evaluate the polynomial.
//!
//! This allows `CurveFit` and `Polynomial` to use your custom basis seamlessly.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use std::fmt::Debug;

use nalgebra::{DMatrix, MatrixViewMut};

use crate::{
    display::{self, default_fixed_range, Sign, DEFAULT_PRECISION},
    error::{Error, Result},
    statistics::DomainNormalizer,
    value::Value,
};

/// A trait representing a polynomial basis.
///
/// Assumes a Vandermonde structure for the basis functions.
///
/// Most of the time, you want to use a built-in basis type, such as [`MonomialBasis`] or [`ChebyshevBasis`].
///
/// A polynomial basis defines the set of functions used to represent a polynomial.
/// Common examples include the monomial basis (`1, x, x^2, ...`) and Chebyshev basis.
/// This trait abstracts over any such basis so that polynomials can be expressed,
/// evaluated, and manipulated generically.
///
/// While you can implement this for custom bases,
/// it is not meant to be used on it's own, but through `Polynomial` or `CurveFit`, which are generic over basis.
///
/// # Type Parameters
/// - `T`: The numeric type used for coefficients and evaluation (e.g., `f64`).
pub trait Basis<T: Value>: Sized + Clone + Debug {
    /// Create a new basis from the given data
    ///
    /// Initializes any needed metadata for normalization
    fn new(data: &[(T, T)]) -> Self;

    /// Returns the number of basis functions needed for a polynomial of a given degree.
    ///
    /// Most polynomial bases have one function per degree plus the constant term,
    /// so the default implementation returns `degree + 1`. For example:
    /// - Degree 0 → 1 function (constant)
    /// - Degree 1 → 2 functions (constant + x¹)
    /// - Degree 2 → 3 functions (constant + x¹ + x²)
    ///
    /// Some custom bases may override this if the number of functions differs from `degree + 1`.
    ///
    /// # Parameters
    /// - `degree`: The polynomial degree you want to represent.
    ///
    /// # Returns
    /// The number of basis functions required to represent a polynomial of the given degree.
    fn k(&self, degree: usize) -> usize {
        degree + 1
    }

    /// Populates a row of a Vandermonde matrix with this basis evaluated at `x`.
    ///
    /// All basis functions (degree + 1 values) are written into `vector` starting
    /// at column `start_index`. This is a low-level helper for efficiently
    /// constructing Vandermonde matrices during polynomial fitting.
    ///
    /// `x` will be normalized by the caller using the `normalize_x` method.
    ///
    /// # Parameters
    /// - `start_index`: Column index where writing begins.
    /// - `x`: The evaluation point.
    /// - `vector`: Mutable row buffer.
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        row: MatrixViewMut<T, R, C, RS, CS>,
    );

    /// Normalizes the input value `x` for this basis.
    ///
    /// This is a no-op for the monomial basis.
    fn normalize_x(&self, x: T) -> T {
        x
    }

    /// Evaluates the jth function of a polynomial expressed in this basis at a given point.
    ///
    /// This computes the value at function j for this polynomial, evaluated at `x`.
    ///
    /// `x` will be normalized by the caller using the `normalize_x` method.
    ///
    /// Formally, the basis provides functions φ₀, φ₁, …, φₙ, and solves `φᵢ(x)`
    ///
    /// # Parameters
    /// - `j`: The index of the basis function to evaluate.
    /// - `x`: The point to evaluate at.
    fn solve_function(&self, j: usize, x: T) -> T;
}

/// A trait for converting polynomial representations into monomial form.
///
/// Some polynomial bases (e.g., Chebyshev) are easier to work with in their
/// native form but can be converted into the standard monomial basis
/// (`1, x, x², …`) when needed. This trait provides that conversion.
///
/// # Behavior
/// - The given `coefficients` slice is mutated in place to represent the same
///   polynomial expressed in the monomial basis.
/// - Implementations must overwrite the entire slice.
/// - The length of `coefficients` should always equal `degree + 1`.
///
/// # Errors
/// Returns an error if the conversion is not supported for the given basis
/// or coefficients (e.g., invalid length).
pub trait IntoMonomialBasis<T: Value>: Basis<T> {
    /// Converts this polynomial representation into monomial form.
    ///
    /// Mutates `coefficients` in place so that they represent the same
    /// polynomial expressed in the monomial basis.
    ///
    /// # Errors
    /// Returns an error if the coefficients cannot be converted.
    fn as_monomial(&self, coefficients: &mut [T]) -> Result<()>;
}

/// Trait for bases that support differentiation of polynomials.
///
/// # Type Parameters
/// - `T`: Numeric type for coefficients.
/// - `B2`: Basis type returned by the derivative (defaults to `Self`).
pub trait DifferentialBasis<T: Value, B2: Basis<T> = Self>: Basis<T> {
    /// Computes the derivative of a polynomial in this basis.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients of the polynomial to differentiate.
    ///
    /// # Errors
    /// Returns an error if the differentiation is not supported for the given basis
    /// or coefficients (e.g., invalid length), or for casting errors
    ///
    /// # Returns
    /// - `(B2, Vec<T>)`: The derivative's basis and its coefficients.
    fn derivative(&self, coefficients: &[T]) -> Result<(B2, Vec<T>)>;

    /// Finds the critical points (where the derivative is zero) of a polynomial in this basis.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients of the polynomial to analyze.
    ///
    /// # Errors
    /// Returns an error if the critical points cannot be found.
    fn critical_points(&self, coefficients: &[T]) -> Result<Vec<T>>;
}

/// Trait for bases that support integration of polynomials.
///
/// # Type Parameters
/// - `T`: Numeric type for coefficients.
/// - `B2`: Basis type returned by the integral (defaults to `Self`).
pub trait IntegralBasis<T: Value, B2: Basis<T> = Self>: Basis<T> {
    /// Computes the integral of a polynomial in this basis.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients of the polynomial to integrate.
    /// - `constant`: Constant of integration (value at x=0).
    ///
    /// # Errors
    /// Returns an error if the differentiation is not supported for the given basis
    /// or coefficients (e.g., invalid length), or for casting errors
    ///
    /// # Returns
    /// - `(B2, Vec<T>)`: The integral's basis and its coefficients.
    fn integral(&self, coefficients: &[T], constant: T) -> Result<(B2, Vec<T>)>;
}

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
pub struct MonomialBasis<T>(pub std::marker::PhantomData<T>);
impl<T: Value> MonomialBasis<T> {
    /// Creates a new monomial basis.
    #[must_use]
    pub const fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}
impl<T: Value> Basis<T> for MonomialBasis<T> {
    fn new(_: &[(T, T)]) -> Self {
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
                _ => Value::powi(x, j as i32),
            };
        }
    }

    fn solve_function(&self, j: usize, x: T) -> T {
        Value::powi(x, j as i32)
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
            return Ok((Self::default(), vec![T::zero()]));
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
pub struct ChebyshevBasis<T: Value> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> ChebyshevBasis<T> {
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
    fn new(data: &[(T, T)]) -> Self {
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
                monomial[m] +=
                    aj * binom * Value::powi(alpha, m as i32) * Value::powi(beta, (j - m) as i32);
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
        let fixed_range = default_fixed_range::<T>();
        let (x_min, x_max) = self.normalizer.src_range();
        let min = display::unicode::float(x_min, fixed_range.clone(), DEFAULT_PRECISION);
        let max = display::unicode::float(x_max, fixed_range, DEFAULT_PRECISION);

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = 2(x - a) / (b - a) - 1, a={min}, b={max}"))
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
    fn test_monomial() {
        let basis = MonomialBasis::<f64>::new(&[]);

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
        let (_basis, derivative) = basis
            .derivative(&[1.0, 2.0, 3.0, 4.0])
            .expect("Derivative failed");
        assert_eq!(derivative, &[2.0, 6.0, 12.0], "Derivative was incorrect");

        let (_basis, integral) = basis
            .integral(&[1.0, 2.0, 3.0, 4.0], 5.0)
            .expect("Integral failed");
        assert_eq!(
            integral,
            &[5.0, 1.0, 1.0, 1.0, 1.0],
            "Integral was incorrect"
        );

        // Edge cases
        // Degree 0 polynomial
        let (_basis, derivative0) = basis
            .derivative(&[42.0])
            .expect("Derivative failed for degree 0");
        assert_eq!(derivative0, &[0.0]);

        let (_basis, integral0) = basis
            .integral(&[42.0], 7.0)
            .expect("Integral failed for degree 0");
        assert_eq!(integral0, &[7.0, 42.0]);

        // Empty coefficients (should still work)
        let (_basis, integral_empty) = basis
            .integral(&[], 3.0)
            .expect("Integral failed for empty coefficients");
        assert_eq!(integral_empty, &[3.0]);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
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
