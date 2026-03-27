use std::ops::RangeInclusive;

use nalgebra::{Complex, MatrixViewMut};

use crate::{
    basis::{
        chebyshev::ThirdFormChebyshevBasis, Basis, ChebyshevBasis, DifferentialBasis,
        IntegralBasis, Root, RootFindingBasis,
    },
    display,
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
    Polynomial,
};

/// Normalized Chebyshev basis of the second kind for polynomial curves (`U_n`).
///
/// This is the result of differentiating the Chebyshev basis of the first kind (`T_n`).
#[derive(Debug, Clone)]
pub struct SecondFormChebyshevBasis<T: Value = f64> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> SecondFormChebyshevBasis<T> {
    /// Creates a Chebyshev basis from an existing domain normalizer.
    pub fn from_normalizer(normalizer: DomainNormalizer<T>) -> Self {
        Self { normalizer }
    }

    /// Creates a new Chebyshev basis that normalizes inputs from the given range to [-1, 1].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (-T::one(), T::one()));
        Self { normalizer }
    }

    /// Creates a new 2nd form Chebyshev polynomial with the given coefficients over the specified x-range.
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
    /// use polyfit::basis::SecondFormChebyshevBasis;
    /// let chebyshev_poly = SecondFormChebyshevBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }

    /// Converts coefficients from the second form Chebyshev basis (`U_n`) to the first form Chebyshev basis (`T_n`).
    fn first_form_coefficients(u: &[T]) -> Vec<T> {
        let n = u.len();
        let mut t = vec![T::zero(); n];

        for (n_idx, &a_n) in u.iter().enumerate() {
            if n_idx == 0 {
                // U0 = T0
                t[0] += a_n;
                continue;
            }

            let mut m = n_idx;

            // Add 2*a_n to T_m, T_{m-2}, ..., down to T_2 or T_1
            while m >= 2 {
                t[m] += a_n + a_n; // 2 * a_n
                m -= 2;
            }

            if m == 1 {
                // odd n: U_n contributes 2*T1
                t[1] += a_n + a_n;
            } else {
                // even n >= 2: U_n contributes 1*T0
                t[0] += a_n;
            }
        }

        t
    }
}
impl<T: Value> Polynomial<'_, SecondFormChebyshevBasis<T>, T> {
    /// Computes a polynomial in the first form Chebyshev basis (`T_n`) that is equivalent to this second form basis (`U_n`).
    ///
    /// # Returns
    /// A polynomial in the first form Chebyshev basis with coefficients derived from this second form
    ///
    /// # Errors
    /// Can only fail if called on a polynomial that is already invalid (e.g. k=0)
    pub fn as_first_form(&self) -> Result<Polynomial<'_, ChebyshevBasis<T>, T>> {
        let u = self.coefficients();
        let t = SecondFormChebyshevBasis::first_form_coefficients(u);

        ChebyshevBasis::new_polynomial(self.basis().normalizer.src_range(), t)
    }
}
impl<T: Value> Basis<T> for SecondFormChebyshevBasis<T> {
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
                0 => T::one(),                               // U0(x) = 1
                1 => T::two() * x,                           // U1(x) = 2x
                _ => T::two() * x * row[j - 1] - row[j - 2], // Un(x) = 2x*U_{n-1}(x) - U_{n-2}(x)
            }
        }
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),     // U0(x) = 1
            1 => T::two() * x, // U1(x) = 2x
            _ => {
                // Un(x) = 2x*U_{n-1}(x) - U_{n-2}(x)
                let mut u0 = T::one();
                let mut u1 = T::two() * x;
                let mut u = T::zero();

                for _ in 2..=j {
                    u = T::two() * x * u1 - u0;
                    u0 = u1;
                    u1 = u;
                }
                u
            }
        }
    }
}

impl<T: Value> display::PolynomialDisplay<T> for SecondFormChebyshevBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<display::Term> {
        super::format_cheb_term("U", degree, coef)
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

impl<T: Value> DifferentialBasis<T> for SecondFormChebyshevBasis<T> {
    type B2 = ThirdFormChebyshevBasis<T>;

    fn derivative(&self, a: &[T]) -> Result<(Self::B2, Vec<T>)> {
        let n = a.len();
        if n < 2 {
            return Ok((
                ThirdFormChebyshevBasis::from_normalizer(self.normalizer),
                vec![T::zero()],
            ));
        }

        let mut b = vec![T::zero(); n - 1];
        let mut b_kplus2 = T::zero();

        for k in (0..(n - 1)).rev() {
            let factor = T::from_positive_int(k + 1) * T::two(); // 2*(k+1)
            b[k] = factor * a[k + 1] + b_kplus2;
            b_kplus2 = b[k];
        }

        // scale coefficients to account for original domain
        let scale = self.normalizer.scale();
        for coeff in &mut b {
            *coeff *= scale;
        }

        let basis = ThirdFormChebyshevBasis::from_normalizer(self.normalizer);
        Ok((basis, b))
    }
}

impl<T: Value> IntegralBasis<T> for SecondFormChebyshevBasis<T> {
    type B2 = ChebyshevBasis<T>;

    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self::B2, Vec<T>)> {
        // Add the constant term and divide all terms by degree
        let mut coefs = Vec::with_capacity(coefficients.len() + 1);
        coefs.push(constant);
        for (i, &c) in coefficients.iter().enumerate() {
            let denom = T::from_positive_int(i + 1);
            coefs.push(c / denom);
        }

        // scale coefficients to account for original domain
        let scale = self.normalizer.scale();
        for coeff in &mut coefs {
            *coeff /= scale;
        }

        let basis = ChebyshevBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}

impl<T: Value> RootFindingBasis<T> for SecondFormChebyshevBasis<T> {
    fn roots(&self, coefs: &[T], x_range: RangeInclusive<T>) -> Result<Vec<Root<T>>> {
        let t = SecondFormChebyshevBasis::first_form_coefficients(coefs);
        ChebyshevBasis::from_normalizer(self.normalizer).roots(&t, x_range)
    }

    fn complex_y(&self, z: Complex<T>, coefs: &[T]) -> Complex<T> {
        let t = SecondFormChebyshevBasis::first_form_coefficients(coefs);
        ChebyshevBasis::from_normalizer(self.normalizer).complex_y(z, &t)
    }
}

#[cfg(test)]
mod test {
    use crate::test::basis_assertions;

    use super::*;

    #[test]
    fn test_chebyshev_second_form() {
        let polyt = ChebyshevBasis::new_polynomial((0.0, 1000.0), &[3.0, 2.0, 1.5, 3.0]).unwrap();
        let c2 = polyt.derivative().unwrap();

        basis_assertions::test_root_finding(&c2, 0.0..=1000.0);
    }
}
