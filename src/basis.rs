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

use nalgebra::Complex;
use nalgebra::MatrixViewMut;

use crate::{error::Result, value::Value};

pub(crate) mod monomial;
pub use monomial::MonomialBasis;

pub(crate) mod chebyshev;
pub use chebyshev::ChebyshevBasis;

pub(crate) mod fourier;
pub use fourier::FourierBasis;

pub(crate) mod legendre;
pub use legendre::LegendreBasis;

pub(crate) mod hermite;
pub use hermite::PhysicistsHermiteBasis;
pub use hermite::ProbabilistsHermiteBasis;

pub(crate) mod laguerre;
pub use laguerre::LaguerreBasis;

pub(crate) mod logarithmic;
pub use logarithmic::LogarithmicBasis;

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
pub trait Basis<T: Value>: Sized + Clone + std::fmt::Debug + Send + Sync {
    /// Create a new basis from the given data
    ///
    /// Initializes any needed metadata for normalization
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self;

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
    #[inline(always)]
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
    #[inline(always)]
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
    fn normalize_x(&self, x: T) -> T;

    /// Denormalizes the input value `x` for this basis.
    ///
    /// This is a no-op for the monomial basis.
    fn denormalize_x(&self, x: T) -> T;

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
    /// The basis type returned by the derivative operation.
    /// This allows the derivative to be expressed in a different basis if needed.
    type B2: Basis<T> + crate::display::PolynomialDisplay<T>;

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
    fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)>;

    /// Computes the second derivative of a polynomial in this basis.
    ///
    /// This is a convenience method that applies `derivative` twice.
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
    fn second_derivative(&self, coefficients: &[T]) -> Result<(SecondDerivative<Self, T>, Vec<T>)>
    where
        Self::B2: DifferentialBasis<T>,
    {
        let (basis1, first) = self.derivative(coefficients)?;
        let (basis2, second) = basis1.derivative(&first)?;
        Ok((basis2, second))
    }
}

/// Helper type alias for the second derivative basis type.
pub type SecondDerivative<B, T> = <<B as DifferentialBasis<T>>::B2 as DifferentialBasis<T>>::B2;

/// Trait for bases that support root finding of polynomials.
pub trait RootFindingBasis<T: Value>: Basis<T> {
    /// Finds the roots (where the function is zero) of a polynomial in this basis.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients of the polynomial to analyze.
    ///
    /// # Errors
    /// Returns an error if the roots cannot be found.
    fn roots(&self, coefs: &[T]) -> Result<Vec<Root<T>>>;
}

/// Trait for bases that support integration of polynomials.
///
/// # Type Parameters
/// - `T`: Numeric type for coefficients.
/// - `B2`: Basis type returned by the integral (defaults to `Self`).
pub trait IntegralBasis<T: Value>: Basis<T> {
    /// The basis type returned by the derivative operation.
    /// This allows the derivative to be expressed in a different basis if needed.
    type B2: Basis<T> + crate::display::PolynomialDisplay<T>;

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
    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self::B2, Vec<T>)>;
}

/// Represents a critical point of a polynomial; representing a point where the curve changes direction.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum CriticalPoint<T: Value> {
    /// A local minimum point where the curve changes from decreasing to increasing.
    /// - This represents a point where the first derivative is zero and the second derivative is positive.
    /// - It's a "valley" in the curve.
    Minima(T, T),

    /// A local maximum point where the curve changes from increasing to decreasing.
    /// - This represents a point where the first derivative is zero and the second derivative is negative.
    /// - It's a "peak" in the curve.
    Maxima(T, T),

    /// A point where the curve changes concavity (from concave up to concave down, or vice versa).
    /// - This represents a point where the second derivative is zero.
    /// - It's where the curve "bends" but does not necessarily have a local extremum.
    Inflection(T, T),
}
impl<T: Value> std::fmt::Display for CriticalPoint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CriticalPoint::Minima(x, y) => write!(f, "Minima({x:.2}, {y:.2})"),
            CriticalPoint::Maxima(x, y) => write!(f, "Maxima({x:.2}, {y:.2})"),
            CriticalPoint::Inflection(x, y) => write!(f, "Inflection({x:.2}, {y:.2})"),
        }
    }
}
impl<T: Value> CriticalPoint<T> {
    /// Converts a slice of critical points into a `PlottingElement` for visualization.
    pub fn as_plotting_element(points: &[Self]) -> crate::plotting::PlottingElement<T> {
        crate::plotting::PlottingElement::from_markers(points.iter().map(|p| {
            let (x, y) = p.coords();
            (x, y, Some(p.to_string()))
        }))
    }

    /// Returns the x-coordinate of the critical point.
    pub fn x(&self) -> T {
        match self {
            CriticalPoint::Minima(x, _)
            | CriticalPoint::Maxima(x, _)
            | CriticalPoint::Inflection(x, _) => *x,
        }
    }

    /// Returns the y-coordinate of the critical point.
    pub fn y(&self) -> T {
        match self {
            CriticalPoint::Minima(_, y)
            | CriticalPoint::Maxima(_, y)
            | CriticalPoint::Inflection(_, y) => *y,
        }
    }

    /// Returns the coordinates of the critical point as a tuple (x, y).
    pub fn coords(&self) -> (T, T) {
        match self {
            CriticalPoint::Minima(x, y)
            | CriticalPoint::Maxima(x, y)
            | CriticalPoint::Inflection(x, y) => (*x, *y),
        }
    }
}

/// Represents a root of a polynomial, which can be real or complex.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Root<T: Value> {
    /// A root that is a real number, where the polynomial crosses the x-axis.
    /// - This represents a solution to the equation P(x) = 0 where x is a real number.
    Real(T),

    /// A root that is a complex number, where the polynomial does not cross the x-axis.
    /// - This represents a solution to the equation P(x) = 0 where x has
    Complex(Complex<T>),

    /// A pair of complex conjugate roots, which often occur in polynomials with real coefficients.
    /// - This represents two solutions to the equation P(x) = 0 that are complex
    ComplexPair(Complex<T>, Complex<T>),
}

/// A trait for orthogonal polynomial bases.
///
/// Orthogonal bases have special properties that make them useful for numerical stability and
/// integration. This trait extends the `Basis` trait with methods specific to orthogonal bases.
pub trait OrthogonalBasis<T: Value>: Basis<T> {
    /// Returns true if the series this basis represents is orthogonal.
    ///
    /// Can be false, for example in integrated fourier series
    fn is_orthogonal(&self) -> bool {
        true
    }

    /// Returns the nodes and weights for Gauss quadrature.
    ///
    /// The basis is orthogonal against these nodes
    fn gauss_nodes(&self, n: usize) -> Vec<(T, T)>;

    /// Returns the theoretical exact value of the integral of the square of the nth basis function over the weight function.
    fn gauss_normalization(&self, n: usize) -> T;

    /// Computes the inner product of two basis functions using the provided nodes and weights.
    ///
    /// Get these from [`OrthogonalBasis::gauss_nodes`].
    fn inner_product(&self, i: usize, j: usize, nodes: &[(T, T)]) -> T {
        let mut sum = T::zero();
        for (x, w) in nodes {
            sum += self.solve_function(i, *x) * self.solve_function(j, *x) * *w;
        }
        sum
    }

    /// Constructs the Gram matrix for the first `n` basis functions using Gauss quadrature.
    ///
    /// Should have shape (n, n), and be ~zero outside the diagonal.
    fn gauss_matrix(&self, functions: usize, nodes: usize) -> nalgebra::DMatrix<T> {
        let nodes = self.gauss_nodes(nodes);
        let mut mat = nalgebra::DMatrix::<T>::zeros(functions, functions);
        for i in 0..functions {
            for j in i..functions {
                let val = self.inner_product(i, j, &nodes);
                mat[(i, j)] = val;
                if i != j {
                    mat[(j, i)] = val; // enforce symmetry explicitly
                }
            }
        }
        mat
    }
}
