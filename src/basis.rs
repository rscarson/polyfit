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
use std::fmt::Debug;

use nalgebra::MatrixViewMut;

use crate::{error::Result, value::Value};

mod monomial;
pub use monomial::MonomialBasis;

mod chebyshev;
pub use chebyshev::ChebyshevBasis;

mod fourier;
pub use fourier::FourierBasis;

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
    fn from_data(data: &[(T, T)]) -> Self;

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

    /// Returns the polynomial degree corresponding to a given number of basis functions.
    ///
    /// Returns `None` if the number of functions does not correspond to a valid degree.
    ///
    /// # Parameters
    /// - `k`: The number of basis functions.
    ///
    /// # Returns
    /// - `Some(degree)`: The polynomial degree if `k` is valid.
    /// - `None`: If `k` does not correspond to a valid degree.
    fn degree(&self, k: usize) -> Option<usize> {
        if k > 0 {
            Some(k - 1)
        } else {
            None
        }
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
pub trait DifferentialBasis<T: Value>: Basis<T> {
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
    /// - The derivative's coefficients.
    fn derivative(&self, coefficients: &[T]) -> Result<Vec<T>>;

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
pub trait IntegralBasis<T: Value>: Basis<T> {
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
    /// - The integral's coefficients.
    fn integral(&self, coefficients: &[T], constant: T) -> Result<Vec<T>>;
}
