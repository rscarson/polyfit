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

use std::ops::RangeInclusive;

use nalgebra::Complex;
use nalgebra::ComplexField;
use nalgebra::MatrixViewMut;
use nalgebra::Normed;

use crate::value::bisect;
use crate::value::FloatClampedCast;
use crate::value::SteppedValues;
use crate::{error::Result, value::Value};

pub(crate) mod monomial;
pub use monomial::MonomialBasis;

pub(crate) mod chebyshev;
pub use chebyshev::{ChebyshevBasis, SecondFormChebyshevBasis, ThirdFormChebyshevBasis};

pub(crate) mod augmented_fourier;
pub use augmented_fourier::{AugmentedFourierBasis, FourierBasis, LinearAugmentedFourierBasis};

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

    /// Evaluates a polynomial expressed in this basis at a given point `x` using the provided coefficients.
    ///
    /// `x` will be normalized by the caller using the `normalize_x` method.
    ///
    /// The `coefficients` slice should have length equal to the number of basis functions (i.e., `self.k(degree)`).
    ///
    /// The polynomial is evaluated as `y = Σ (coefᵢ * φᵢ(x))` where `φᵢ` are the basis functions.
    ///
    /// By default this uses the `solve_function` method to compute each basis function's contribution, but
    /// implementations can override this for efficiency if needed.
    fn solve(&self, x: T, coefficients: &[T]) -> T {
        let mut y = T::zero();
        for (i, &coef) in coefficients.iter().enumerate() {
            y += coef * self.solve_function(i, x);
        }

        y
    }
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

/// Enumeration of root finding methods for polynomials in a given basis.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum RootFindingMethod {
    /// Indicates that the basis supports an analytical method for finding roots, which is typically much faster and more accurate than iterative methods.
    ///
    /// May not be appropriate for very high degree polynomials, or polynomials with widely spaced roots, due to numerical instability.
    /// In those cases the `iterative_*` methods may be more reliable, albeit slower.
    Analytical,

    /// Indicates that the basis does not support an analytical method for finding roots, and that iterative numerical methods will be used instead.
    ///
    /// This is a fallback method that can be used for any basis, but is much slower and less accurate than analytical methods,
    /// but may actually outperform analytical methods for very high degree polynomials, or polynomials with widely spaced roots.
    Iterative,
}
impl std::fmt::Display for RootFindingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RootFindingMethod::Analytical => write!(f, "Analytical"),
            RootFindingMethod::Iterative => write!(f, "Iterative"),
        }
    }
}

/// Trait for bases that support root finding of polynomials.
///
/// This requires the basis to also support differentiation
///
/// The reason is for fallback - for bases without closed-form root finding, we can use the derivative to perform numerical root finding (e.g. Newton's method).
///
/// And since at the time of writing there is no basis without differentiation that currently implements root finding, this is a safe assumption for now.
pub trait RootFindingBasis<T: Value>: Basis<T> + DifferentialBasis<T> {
    /// Default number of samples to use for iterative root finding methods when no specific value is provided.
    const DEFAULT_ROOT_FINDING_SAMPLES: usize = 5000;

    /// Default maximum number of iterations to use for iterative root finding methods when no specific value is provided.
    const DEFAULT_ROOT_FINDING_MAX_ITERATIONS: usize = 20;

    /// Returns the root finding method supported by this basis.
    fn root_finding_method(&self) -> RootFindingMethod {
        RootFindingMethod::Iterative
    }

    /// Finds the roots (where the function is zero) of a polynomial in this basis.
    ///
    /// For select Bases, this is a fast, analytical operation. For others, it may require numerical methods.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients of the polynomial to analyze.
    ///
    /// # Errors
    /// Returns an error if the roots cannot be found.
    fn roots(&self, coefs: &[T], x_range: RangeInclusive<T>) -> Result<Vec<Root<T>>> {
        self.roots_iterative(
            coefs,
            x_range,
            Self::DEFAULT_ROOT_FINDING_SAMPLES,
            Self::DEFAULT_ROOT_FINDING_MAX_ITERATIONS,
        )
    }

    /// Use iterative numerical methods to find the roots of a polynomial in this basis.
    ///
    /// This is less precise than [`Self::roots`] and will not find complex roots, and is far slower (Up to 12,000 times for some cases!)
    /// But nearly all bases support it and for hilariously wide or badly behaved polynomials, it can be more reliable.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values to search for real roots.
    /// - `samples`: The number of sample points to evaluate within `x_range` to detect sign changes. More samples can improve accuracy but increase runtime.
    /// - `max_iterations`: The maximum number of iterations for refining each root using methods like bisection and Newton's method. More iterations can improve accuracy but increase runtime.
    ///
    /// # Returns
    /// A vector of `T` representing the real roots of the polynomial within the specified range
    ///
    /// # Errors
    /// Returns an error if the derivative cannot be computed.
    fn roots_iterative(
        &self,
        coefs: &[T],
        x_range: RangeInclusive<T>,
        samples: usize,
        max_iterations: usize,
    ) -> Result<Vec<Root<T>>> {
        let mut roots = vec![];
        let mut prev_x = *x_range.start();
        let mut prev_y = self.solve(prev_x, coefs);

        // The aim is ~`num_samples` samples, given a 64bit float precision and a width of domain of 100
        // We scale this up or down based on the actual domain width and type precision
        //
        // We will scale logarithmically with respect to width, and linearly with respect to precision
        let domain_width = *x_range.end() - *x_range.start();
        let domain_scalar = match (Value::abs(domain_width) + T::one()).log10() - T::one() {
            x if x > T::zero() => x,
            x if x < T::zero() => T::one() / -x,
            _ => T::one(),
        };
        let precision_scalar = T::epsilon() / f64::EPSILON.clamped_cast::<T>();
        let num_samples = T::from_positive_int(samples) * domain_scalar * precision_scalar;

        let (dx_basis, dx_coefs) = self.derivative(coefs)?;
        let sqrt_eps = T::epsilon().sqrt();
        for x in SteppedValues::new(x_range, domain_width / num_samples) {
            let y = self.solve(x, coefs);
            if (prev_y * y).is_sign_negative() || prev_y.is_near_zero() || y.is_near_zero() {
                let (x, _) = bisect(
                    &|x| self.solve(x, coefs),
                    prev_x,
                    x,
                    prev_y,
                    y,
                    4, // small fixed count
                );

                let mut a = prev_x;
                let mut b = x;
                let mut fa = prev_y;

                // shrink interval FIRST
                for _ in 0..max_iterations {
                    let m = (a + b) / T::two();
                    let fm = self.solve(m, coefs);

                    if (fa * fm).is_sign_negative() {
                        b = m;
                    } else {
                        a = m;
                        fa = fm;
                    }
                }

                let x = (a + b) / T::two();

                // Newton iterations to refine
                let mut newton_prev_x = x;
                let mut newton_x;
                for _ in 0..max_iterations {
                    let y = self.solve(newton_prev_x, coefs);
                    let dy = dx_basis.solve(newton_prev_x, &dx_coefs);
                    newton_x = newton_prev_x - y / dy;
                    newton_x = Value::clamp(newton_x, prev_x, x);

                    let rel_tol = sqrt_eps + (T::one() * Value::abs(newton_x));
                    if Value::abs(y) <= rel_tol || Value::abs(newton_x - newton_prev_x) <= rel_tol {
                        break;
                    }

                    newton_prev_x = newton_x;
                }

                roots.push(newton_prev_x);
            }

            prev_x = x;
            prev_y = y;
        }

        let roots = roots.into_iter().map(Root::Real).collect();
        Ok(roots)
    }

    /// Evaluates the polynomial at a complex point `z` using the given coefficients.
    ///
    /// # Parameters
    /// - `z`: The complex point at which to evaluate the polynomial.
    /// - `coefs`: The coefficients of the polynomial in this basis.
    ///
    /// # Returns
    /// The value of the polynomial at the given complex point.
    fn complex_y(&self, z: Complex<T>, coefs: &[T]) -> Complex<T>;
}

/// Trait for bases that support integration of polynomials.
///
/// # Type Parameters
/// - `T`: Numeric type for coefficients.
/// - `B2`: Basis type returned by the integral (defaults to `Self`).
pub trait IntegralBasis<T: Value>: Basis<T> {
    /// The basis type returned by the integral operation.
    /// This allows the integral to be expressed in a different basis if needed.
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
    #[cfg(feature = "plotting")]
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
impl<T: Value> Root<T> {
    /// Returns true if this root is a real root.
    pub fn is_real(&self) -> bool {
        matches!(self, Root::Real(_))
    }

    /// Returns true if this root is a complex root (either a single complex root or a conjugate pair).
    pub fn is_complex(&self) -> bool {
        matches!(self, Root::Complex(_) | Root::ComplexPair(_, _))
    }

    /// Returns the real value of this root if it is a real root, or `None` if it is complex.
    pub fn as_real(&self) -> Option<T> {
        match self {
            Root::Real(x) => Some(*x),
            _ => None,
        }
    }

    /// Returns the complex value(s) of this root if it is a complex root, or `None` if it is real.
    pub fn as_complex(&self) -> Option<Vec<Complex<T>>> {
        match self {
            Root::Complex(z) => Some(vec![*z]),
            Root::ComplexPair(z1, z2) => Some(vec![*z1, *z2]),
            Root::Real(_) => None,
        }
    }

    /// Categorizes the roots of a polynomial into real and complex roots, while removing duplicates.
    ///
    /// This is a helper function used for implementing root finding in various bases.
    ///
    /// It takes a list of eigenvalues (which may include duplicates and non-roots) and filters them based on:
    /// 1. Removing non-finite values (INF/NAN).
    /// 2. Removing values where the polynomial does not evaluate to zero (not actual roots).
    /// 3. Removing duplicates (values that are very close to each other).
    /// 4. Categorizing remaining roots as real or complex (including conjugate pairs).
    ///
    /// The `solver` function is used to evaluate the polynomial at complex points to verify if they are roots.
    /// - This is generally the `complex_y` method of the basis, which evaluates the polynomial at a complex point.
    #[allow(
        clippy::match_same_arms,
        reason = "This is more readable as a match statement, even if some arms are the same"
    )]
    pub fn roots_from_complex<F: Fn(&Complex<T>) -> Complex<T>>(
        eigenvalues: &[Complex<T>],
        solver: F,
    ) -> Vec<Root<T>> {
        let mut roots = Vec::new();
        let mut skip = vec![false; eigenvalues.len()];
        for i in 0..eigenvalues.len() {
            if skip[i] {
                continue;
            }

            // Skip INF/NAN roots
            if !eigenvalues[i].imaginary().is_finite() || !eigenvalues[i].real().is_finite() {
                continue;
            }

            // Skip roots where P(x) != 0
            let zero_tol = (T::one() + eigenvalues[i].norm()) * T::epsilon().sqrt();
            if solver(&eigenvalues[i]).norm() > zero_tol {
                continue;
            }

            // Skip future duplicates
            let conj_tol = T::epsilon().sqrt() * (T::one() + eigenvalues[i].norm());
            for j in (i + 1)..eigenvalues.len() {
                if (eigenvalues[i] - eigenvalues[j]).norm() < conj_tol {
                    skip[j] = true;
                }
            }

            // At this point for reals we can stop
            if Value::abs(eigenvalues[i].imaginary()) < zero_tol {
                roots.push(Root::Real(eigenvalues[i].real()));
                continue;
            }

            // Complex root - we check for conjugate pairs
            for j in (i + 1)..eigenvalues.len() {
                if Value::abs(eigenvalues[i].real() - eigenvalues[j].real()) < conj_tol
                    && Value::abs(eigenvalues[i].imaginary() + eigenvalues[j].imaginary())
                        < conj_tol
                {
                    let root_i = eigenvalues[i];
                    let root_j = eigenvalues[j];

                    skip[j] = true;
                    roots.push(Root::ComplexPair(root_i, root_j));
                    break;
                }
            }

            // Singular complex root - should only happen for complex coefficients
            roots.push(Root::Complex(eigenvalues[i]));
        }

        // Now we sort them - reals first, then complex pairs, then singular complex roots, and within each category we sort by magnitude
        roots.sort_by(|a, b| match (a, b) {
            (Root::Real(x), Root::Real(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
            (Root::Real(_), _) => std::cmp::Ordering::Less,
            (_, Root::Real(_)) => std::cmp::Ordering::Greater,
            (Root::ComplexPair(a1, a2), Root::ComplexPair(b1, b2)) => {
                let mag_a = a1.norm() + a2.norm();
                let mag_b = b1.norm() + b2.norm();
                mag_a
                    .partial_cmp(&mag_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            (Root::ComplexPair(_, _), Root::Complex(_)) => std::cmp::Ordering::Less,
            (Root::Complex(_), Root::ComplexPair(_, _)) => std::cmp::Ordering::Greater,
            (Root::Complex(a), Root::Complex(b)) => a
                .norm()
                .partial_cmp(&b.norm())
                .unwrap_or(std::cmp::Ordering::Equal),
        });

        roots
    }
}

/// A trait for orthogonal polynomial bases.
///
/// Orthogonal bases have special properties that make them useful for numerical stability and
/// integration. This trait extends the `Basis` trait with methods specific to orthogonal bases.
pub trait OrthogonalBasis<T: Value>: Basis<T> {
    /// Returns the weight function value at `x` for Gauss quadrature.
    ///
    /// The basis is orthogonal against this weight function.
    fn gauss_weight(&self, x: T) -> T;

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
