use std::fmt::Debug;

use nalgebra::MatrixViewMut;

use crate::{
    basis::Basis,
    display::{self, default_fixed_range, format_coefficient, Sign, Term, DEFAULT_PRECISION},
    statistics::DomainNormalizer,
    value::Value,
};

#[derive(Debug, Clone)]
pub struct FourierBasis<T: Value> {
    normalizer: DomainNormalizer<T>,
}
impl<T: Value> Basis<T> for FourierBasis<T> {
    fn new(data: &[(T, T)]) -> Self {
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
        let fixed_range = default_fixed_range::<T>();
        let (x_min, x_max) = self.normalizer.src_range();
        let min = display::unicode::float(x_min, fixed_range.clone(), DEFAULT_PRECISION);
        let max = display::unicode::float(x_max, fixed_range, DEFAULT_PRECISION);

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = 2(x - a) / (b - a) - 1, a={min}, b={max}"))
    }
}
