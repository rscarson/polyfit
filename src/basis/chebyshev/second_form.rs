use nalgebra::MatrixViewMut;

use crate::{
    basis::{
        chebyshev::ThirdFormChebyshevBasis, Basis, ChebyshevBasis, DifferentialBasis,
        IntegralBasis, Root, RootFindingBasis,
    },
    display,
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
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

impl<T: Value> RootFindingBasis<T> for SecondFormChebyshevBasis<T> {
    fn roots(&self, coefs: &[T]) -> Result<Vec<Root<T>>> {
        let mut roots = Vec::with_capacity(coefs.len());
        // Xk = cos((2k+1)π/(2n)) for k=0..n-1
        // All roots are real in [-1, 1]
        let n = coefs.len() - 1;
        let nplus1 = T::from_positive_int(n + 1);
        for k in 0..coefs.len() - 1 {
            let k = n - 1 - k; // Reverse order to get ascending roots

            let k = T::from_positive_int(k);

            let x = (T::pi() * k / nplus1).cos();
            let x = self.denormalize_x(x);
            roots.push(Root::Real(x));
        }

        Ok(roots)
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

        let basis = ChebyshevBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}
