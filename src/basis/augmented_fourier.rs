use nalgebra::MatrixViewMut;

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, MonomialBasis},
    display::{self, format_coefficient, PolynomialDisplay, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::{IntClampedCast, Value},
};

mod fourier;
pub use fourier::FourierBasis;

mod linear_fourier;
pub use linear_fourier::LinearAugmentedFourierBasis;

/// Generalized basis for periodic functions with a monomial component.
///
/// The Fourier basis represents functions using sine and cosine functions:
/// ```math
/// 1, sin(2πx), cos(2πx), sin(4πx), cos(4πx), ..., sin(2nπx), cos(2nπx)
/// ```
///
/// This basis also includes a monomial component of degree `MONOMIAL_DEGREE`, which allows it to capture non-periodic trends in the data:
/// ```math
/// 1, x, x^2, ..., x^MONOMIAL_DEGREE, sin(2πx), cos(2πx), sin(4πx), cos(4πx), ..., sin(2nπx), cos(2nπx)
/// ```
///
/// This basis is ideal for modeling periodic phenomena, such as waves or seasonal patterns.
/// Its numerical stability degrades as the monomial degree increases, so it is recommended to keep `MONOMIAL_DEGREE` small (e.g., 0 or 1) for best results.
///
/// # When to use
/// - Use for fitting periodic data or functions.
/// - Ideal for applications in signal processing, time series analysis, and any domain with inherent periodicity.
///
/// # Why Fourier?
/// - Provides a natural way to represent periodic functions.
/// - Efficiently captures oscillatory behavior with fewer terms.
/// - Numerically stable for a wide range of applications (For small monomial degrees).
#[derive(Debug, Clone)]
pub struct AugmentedFourierBasis<const MONOMIAL_DEGREE: usize, T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}

impl<const MONOMIAL_DEGREE: usize, T: Value> AugmentedFourierBasis<MONOMIAL_DEGREE, T> {
    /// Creates a new Fourier basis that normalizes inputs from the given range to [0, 2π].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (T::zero(), T::two_pi()));
        Self { normalizer }
    }

    /// Creates a new Fourier polynomial with the given coefficients over the specified x-range.
    pub fn from_normalizer(normalizer: DomainNormalizer<T>) -> Self {
        Self { normalizer }
    }

    /// Creates a new Fourier polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Fourier basis is defined.
    /// - `coefficients`: The coefficients for the Fourier basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Fourier basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::FourierBasis;
    /// let fourier_poly = FourierBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }

    fn integral_coefs(&self, coefficients: &[T], constant: T) -> Result<Vec<T>> {
        if coefficients.is_empty() {
            return Ok(vec![constant]);
        }

        let monomial_terms = &coefficients[..(MONOMIAL_DEGREE + 1).min(coefficients.len())];
        let fourier_terms = &coefficients[monomial_terms.len()..];

        //
        // We do a monomial integral for the first [0..polynomial_terms]
        let (_, mut integral_coeffs) =
            MonomialBasis::default().integral(monomial_terms, constant)?;
        let new_monomial_terms = integral_coeffs.len();

        //
        // The integral is actually not hard here
        // 2- an * sin(nx) -> -an/n * cos(nx)
        // 3- bn * cos(nx) -> bn/n * sin(nx)
        // We can do the last 2 by flipping pairs of coefficients and dividing by n

        let mut n = T::one(); // frequency index
        let mut coef_iter = fourier_terms.iter();
        while let Some(a) = coef_iter.next().copied() {
            let b = coef_iter.next().copied().unwrap_or(T::zero());

            // Fourier expects pairs of (sin, cos), so originally we had (a, b)
            // Now under integration we need (b, -a) to get the right functions
            integral_coeffs.push(b / n);
            integral_coeffs.push(-a / n);

            n += T::one(); // increment frequency index
        }

        // Scale only the fourier coefficients to account for original domain
        let scale = self.normalizer.scale();
        for coeff in &mut integral_coeffs[new_monomial_terms..] {
            *coeff /= scale;
        }

        Ok(integral_coeffs)
    }

    fn derivative_coefs(&self, coefficients: &[T]) -> Result<Vec<T>> {
        if coefficients.len() <= 1 {
            return Ok(vec![T::zero()]);
        }

        let monomial_terms = &coefficients[..(MONOMIAL_DEGREE + 1).min(coefficients.len())];
        let fourier_terms = &coefficients[monomial_terms.len()..];

        //
        // We do a monomial differential for the first part of the coefficients, which are just powers of x (after denormalization)
        let (_, mut derivative_coeffs) = MonomialBasis::default().derivative(monomial_terms)?;
        let new_monomial_terms = derivative_coeffs.len();

        //
        // Now for the Fourier terms,
        // Similar to integral, we can do this by flipping pairs of coefficients:
        // 2- an * sin(nx) -> n * an * cos(nx)
        // 3- bn * cos(nx) -> -n * bn * sin(nx)

        let mut n = T::one(); // frequency index
        let mut coef_iter = fourier_terms.iter();
        while let Some(a) = coef_iter.next().copied() {
            let b = coef_iter.next().copied().unwrap_or(T::zero());

            // Fourier expects pairs of (sin, cos), so originally we had (a, b)
            // Now under differentiation we need (b, -a) to get the right functions
            derivative_coeffs.push(n * -b);
            derivative_coeffs.push(n * a);

            n += T::one(); // increment frequency index
        }

        // Scale only the fourier coefficients to account for original domain
        let scale = self.normalizer.scale();
        for coeff in &mut derivative_coeffs[new_monomial_terms..] {
            *coeff *= scale;
        }

        Ok(derivative_coeffs)
    }
}

impl<const MONOMIAL_DEGREE: usize, T: Value> Basis<T>
    for AugmentedFourierBasis<MONOMIAL_DEGREE, T>
{
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self {
        Self::new(*x_range.start(), *x_range.end())
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
    fn k(&self, degree: usize) -> usize {
        (MONOMIAL_DEGREE + 1) + 2 * degree
    }

    #[inline(always)]
    fn degree(&self, k: usize) -> Option<usize> {
        let base = MONOMIAL_DEGREE + 1;
        let fourier_terms = k.checked_sub(base)?;
        if fourier_terms % 2 != 0 {
            return None; // must be even number of Fourier terms
        }
        let max_harmonic = fourier_terms / 2;
        Some(max_harmonic)
    }

    #[inline(always)]
    fn max_degree(&self, n: usize) -> usize {
        n.saturating_sub(1) / 2
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        // Because we support calculus, there can be a polynomial series at the start
        // The first [0..polynomial_terms] are monomial terms
        if j <= MONOMIAL_DEGREE {
            // we must undo the normalization for the monomial terms to get the correct values, since the Fourier basis normalizes x to [0, 2π]
            let x = self.denormalize_x(x);
            return Value::powi(x, j.clamped_cast());
        }

        let j = j - (MONOMIAL_DEGREE + 1);
        // Now we have the Fourier terms

        if j % 2 == 0 {
            // Sine terms (odd indices)
            let n = (j + 1).div_ceil(2);

            let angle = x * T::from_positive_int(n);
            angle.sin()
        } else {
            // Cosine terms (even indices)
            let n = j.div_ceil(2);

            let angle = x * T::from_positive_int(n);
            angle.cos()
        }
    }

    #[inline(always)]
    fn solve(&self, x: T, coefficients: &[T]) -> T {
        let mut y = T::zero();
        let monomial_terms = (MONOMIAL_DEGREE + 1).min(coefficients.len());

        // First we fill in the monomial terms, which are just powers of x (after denormalization)
        let monomial_coefs = &coefficients[..monomial_terms];
        y += MonomialBasis::default().solve(self.denormalize_x(x), monomial_coefs);

        // Now we have the Fourier terms

        // For the recurrence
        // Since, for s_1 = sin(x) and c_1 = cos(x), we have:
        // s_{n+1} = s_n * c_1 + c_n * s_1
        // c_{n+1} = c_n * c_1 - s_n * s_1
        let s1 = x.sin();
        let c1 = x.cos();
        let mut s_n = s1;
        let mut c_n = c1;

        let fourier_coefs = &coefficients[monomial_terms..];
        // println!("Calculating solve for x={x} with fourier_coefs={fourier_coefs:?}");
        for j in 0..fourier_coefs.len() {
            let coef = fourier_coefs[j];
            if j % 2 == 0 {
                // Sine terms (odd indices)
                y += coef * s_n;
            } else {
                // Cosine terms (even indices)
                y += coef * c_n;

                let s_next = s_n * c1 + c_n * s1;
                let c_next = c_n * c1 - s_n * s1;
                s_n = s_next;
                c_n = c_next;
            }
        }

        y
    }

    //
    // invariant: polynomial_terms == 1 here
    // We only kerfuffle that in calculus methods
    #[inline(always)]
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<'_, T, R, C, RS, CS>,
    ) {
        let sx = x.sin();
        let cx = x.cos();

        // First we fill in the monomial terms, which are just powers of x (after denormalization)
        let mut j = start_index;
        while j <= MONOMIAL_DEGREE.min(row.ncols() - 1) {
            let x = self.denormalize_x(x);
            row[j] = Value::powi(x, j.clamped_cast());
            j += 1;
        }

        while j < row.ncols() {
            let fourier_index = j - (MONOMIAL_DEGREE + 1);
            row[j] = match fourier_index {
                0 => sx, // first sin
                1 => cx, // first cos
                2 => T::two() * cx * row[1],
                3 => T::two() * cx * row[2] - T::one(),
                _ => T::two() * cx * row[j - 2] - row[j - 4],
            };

            j += 1;
        }
    }
}

impl<const MONOMIAL_DEGREE: usize, T: Value> PolynomialDisplay<T>
    for AugmentedFourierBasis<MONOMIAL_DEGREE, T>
{
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        let polynomial_terms = MONOMIAL_DEGREE + 1;

        if degree < polynomial_terms.clamped_cast() {
            return MonomialBasis::format_term(&MonomialBasis::default(), degree, coef);
        }

        let degree = degree - polynomial_terms.clamped_cast::<i32>() + 1;

        let sign = Sign::from_coef(coef);
        let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        // frequency index
        let n = (degree + 1) / 2;
        let n = if n == 1 { String::new() } else { n.to_string() };

        // even -> cos, odd -> sin
        let function = if degree % 2 == 0 { "cos" } else { "sin" };

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        let glue = if coef.is_empty() || function.is_empty() {
            ""
        } else {
            "·"
        };

        let body = format!("{coef}{glue}{function}({n}{x})");
        Some(Term { sign, body })
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

#[cfg(test)]
mod tests {}
