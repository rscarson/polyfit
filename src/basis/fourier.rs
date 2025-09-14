use nalgebra::MatrixViewMut;

use crate::{
    basis::Basis,
    display::{self, format_coefficient, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
};

/// Standard Fourier basis for periodic functions.
///
/// The Fourier basis represents functions using sine and cosine functions:
/// ```math
/// 1, sin(2πx), cos(2πx), sin(4πx), cos(4πx), ..., sin(2nπx), cos(2nπx)
/// ```
///
/// This basis is ideal for modeling periodic phenomena, such as waves or seasonal patterns.
/// It is numerically stable and can efficiently represent complex periodic behavior.
///
/// # When to use
/// - Use for fitting periodic data or functions.
/// - Ideal for applications in signal processing, time series analysis, and any domain with inherent periodicity.
///
/// # Why Fourier?
/// - Provides a natural way to represent periodic functions.
/// - Efficiently captures oscillatory behavior with fewer terms.
/// - Numerically stable for a wide range of applications.
#[derive(Debug, Clone)]
pub struct FourierBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> FourierBasis<T> {
    /// Creates a new Fourier basis that normalizes inputs from the given range to [0, 2π].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (T::zero(), T::two_pi()));
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
}
impl<T: Value> Basis<T> for FourierBasis<T> {
    fn from_data(data: &[(T, T)]) -> Self {
        let normalizer =
            DomainNormalizer::from_data(data.iter().map(|(x, _)| *x), (T::zero(), T::two_pi()));
        Self { normalizer }
    }

    fn normalize_x(&self, x: T) -> T {
        self.normalizer.normalize(x)
    }

    fn k(&self, degree: usize) -> usize {
        2 * degree + 1
    }

    fn degree(&self, k: usize) -> Option<usize> {
        if k % 2 == 0 {
            None
        } else {
            Some((k - 1) / 2)
        }
    }

    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(), // a0 / 2 term

            _ if j % 2 == 1 => {
                // Sine terms (odd indices)
                let n = j.div_ceil(2);

                // Infallible multiplication for the *n term
                let mut angle = x;
                for _ in 1..n {
                    angle += x;
                }

                angle.sin()
            }

            _ => {
                // Cosine terms (even indices)
                let n = j / 2;

                // Infallible multiplication for the *n term
                let mut angle = x;
                for _ in 1..n {
                    angle += x;
                }

                angle.cos()
            }
        }
    }

    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<'_, T, R, C, RS, CS>,
    ) {
        row[start_index] = T::one(); // constant term
        if row.ncols() <= start_index + 1 {
            return;
        }

        let cos_x = x.cos();
        let sin_x = x.sin();

        row[start_index + 1] = sin_x; // first sin
        if row.ncols() <= start_index + 2 {
            return;
        }

        row[start_index + 2] = cos_x; // first cost

        // then angle-doubling recurrence
        let mut sin_prev2 = T::zero(); // sin(0x)
        let mut sin_prev = row[start_index + 1];
        let mut cos_prev2 = T::one(); // cos(0x)
        let mut cos_prev = row[start_index + 2];

        let mut idx = start_index + 3;
        while idx + 1 < row.ncols() {
            let cos_nx = T::two() * cos_x * cos_prev - cos_prev2;
            let sin_nx = T::two() * cos_x * sin_prev - sin_prev2;

            row[idx] = sin_nx;
            row[idx + 1] = cos_nx;

            (cos_prev2, cos_prev) = (cos_prev, cos_nx);
            (sin_prev2, sin_prev) = (sin_prev, sin_nx);

            idx += 2;
        }
    }
}

impl<T: Value> display::PolynomialDisplay<T> for FourierBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        let sign = Sign::from_coef(coef);
        let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        if degree == 0 {
            return Some(Term { sign, body: coef });
        }

        // frequency index
        let n = (degree + 1) / 2;
        let n = if n == 1 { String::new() } else { n.to_string() };

        // even -> cos, odd -> sin
        let function = if degree % 2 == 0 { "cos" } else { "sin" };

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        let body = format!("{coef}{function}(2π{n}{x})");
        Some(Term { sign, body })
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}
