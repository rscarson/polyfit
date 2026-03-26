use nalgebra::MatrixViewMut;

use crate::{
    basis::{chebyshev::SecondFormChebyshevBasis, Basis, IntegralBasis},
    display,
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
};

/// Normalized Chebyshev basis of the third kind for polynomial curves (`V_n`).
///
/// This is the result of differentiating the Chebyshev basis of the second kind (Un).
#[derive(Debug, Clone)]
pub struct ThirdFormChebyshevBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> ThirdFormChebyshevBasis<T> {
    /// Creates a Chebyshev basis from an existing domain normalizer.
    pub fn from_normalizer(normalizer: DomainNormalizer<T>) -> Self {
        Self { normalizer }
    }

    /// Creates a new Chebyshev basis that normalizes inputs from the given range to [-1, 1].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (-T::one(), T::one()));
        Self { normalizer }
    }

    /// Creates a new 3rd form Chebyshev polynomial with the given coefficients over the specified x-range.
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
    /// use polyfit::basis::ThirdFormChebyshevBasis;
    /// let chebyshev_poly = ThirdFormChebyshevBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}
impl<T: Value> Basis<T> for ThirdFormChebyshevBasis<T> {
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self {
        let normalizer = DomainNormalizer::from_range(x_range, (-T::one(), T::one()));
        Self { normalizer }
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
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<'_, T, R, C, RS, CS>,
    ) {
        for j in start_index..row.ncols() {
            row[j] = match j {
                0 => T::one(),                               // V0(x) = 1
                1 => T::two() * x - T::one(),                // V1(x) = 2x - 1
                _ => T::two() * x * row[j - 1] - row[j - 2], // recurrence
            }
        }
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),                // V0(x) = 1
            1 => T::two() * x - T::one(), // V1(x) = 2x - 1
            _ => {
                // Vn(x) = 2x*V_{n-1}(x) - V_{n-2}(x)
                let mut v0 = T::one();
                let mut v1 = T::two() * x - T::one();
                let mut v = T::zero();
                for _ in 2..=j {
                    v = T::two() * x * v1 - v0;
                    v0 = v1;
                    v1 = v;
                }
                v
            }
        }
    }
}

impl<T: Value> display::PolynomialDisplay<T> for ThirdFormChebyshevBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<display::Term> {
        super::format_cheb_term("V", degree, coef)
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

impl<T: Value> IntegralBasis<T> for ThirdFormChebyshevBasis<T> {
    type B2 = SecondFormChebyshevBasis<T>;

    fn integral(&self, b: &[T], constant: T) -> Result<(Self::B2, Vec<T>)> {
        let n = b.len();
        let mut a = vec![T::zero(); n + 1];
        a[0] = constant;

        if n == 0 {
            let basis = SecondFormChebyshevBasis::from_normalizer(self.normalizer);
            return Ok((basis, a));
        }

        let mut b_next = T::zero();
        for k in (0..n).rev() {
            let factor = T::from_positive_int(k + 1) * T::two();
            a[k + 1] = (b[k] - b_next) / factor;
            b_next = b[k];
        }

        // scale coefficients to account for original domain
        let scale = self.normalizer.scale();
        for coeff in &mut a {
            *coeff /= scale;
        }

        let basis = SecondFormChebyshevBasis::from_normalizer(self.normalizer);
        Ok((basis, a))
    }
}
