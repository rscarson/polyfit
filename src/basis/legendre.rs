use nalgebra::MatrixViewMut;

use crate::{
    basis::{Basis, IntoMonomialBasis},
    display::{self, format_coefficient, PolynomialDisplay, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::{IntClampedCast, Value},
};

/// Normalized Legendre basis for polynomial curves.
///
/// This basis uses the Legendre polynomials, which form an
/// orthogonal family of polynomials on the interval [-1, 1].
/// Orthogonality ensures numerical stability and clean separation
/// of polynomial terms, especially useful for least-squares fitting.
///
/// Inputs are normalized so that the evaluation domain
/// [`x_min`, `x_max`] is mapped onto [-1, 1]. This allows Legendre
/// polynomials to be used naturally with arbitrary input ranges
/// while retaining their orthogonality properties.
///
/// # When to use
/// - Use when fitting polynomials and you want orthogonal basis functions.
/// - Prefer for higher-degree polynomials to reduce numerical error
///   in coefficient estimation.
/// - Useful for physics or engineering applications where
///   Legendre expansions are natural (e.g., potential fields, spherical harmonics).
///
/// # Why Legendre?
/// - Orthogonal over [-1, 1] with uniform weight, simplifying least-squares fitting.
/// - Minimizes coefficient correlation, improving numerical stability.
/// - Provides exact integral properties for weighted projections.
#[derive(Debug, Clone, Copy)]
pub struct LegendreBasis<T: Value = f64> {
    /// Normalizer to map input domain to [-1, 1]
    pub normalizer: DomainNormalizer<T>,
}
impl<T: Value> LegendreBasis<T> {
    /// Creates a new Legendre basis that normalizes inputs from the given range to [-1, 1].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (-T::one(), T::one()));
        Self { normalizer }
    }

    /// Creates a new Legendre polynomial with the given coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Legendre basis is defined.
    /// - `coefficients`: The coefficients for the Legendre basis functions.
    ///
    /// # Returns
    /// A polynomial defined in the Legendre basis.
    ///
    /// # Errors
    /// Returns an error if the polynomial cannot be created with the given basis and coefficients.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::basis::LegendreBasis;
    /// let legendre_poly = LegendreBasis::new_polynomial((-1.0, 1.0), &[1.0, 0.0, -0.5]).unwrap();
    /// ```
    pub fn new_polynomial(
        x_range: (T, T),
        coefficients: &[T],
    ) -> Result<crate::Polynomial<'_, Self, T>> {
        let basis = Self::new(x_range.0, x_range.1);
        crate::Polynomial::<Self, T>::from_basis(basis, coefficients)
    }
}

impl<T: Value> Basis<T> for LegendreBasis<T> {
    fn from_data(data: &[(T, T)]) -> Self {
        let normalizer =
            DomainNormalizer::from_data(data.iter().map(|(x, _)| *x), (-T::one(), T::one()));
        Self { normalizer }
    }

    fn normalize_x(&self, x: T) -> T {
        self.normalizer.normalize(x)
    }

    fn solve_function(&self, j: usize, x: T) -> T {
        match j {
            0 => T::one(),
            1 => x,
            _ => {
                // P_j(x) = SUM_k:0..j/2 [ -1^k * (2j - 2k)! / (2^j * k! * (j - k)! * (j - 2k)! ) * x^(j - 2k) ]
                let mut sum = T::zero();
                for k in 0..=(j / 2) {
                    let sign = if k % 2 == 0 { T::one() } else { -T::one() };

                    let numerator = T::factorial(2 * j - 2 * k);
                    let denom = Value::powi(T::two(), j.clamped_cast())
                        * T::factorial(k)
                        * T::factorial(j - k)
                        * T::factorial(j - 2 * k);
                    let x_factor = Value::powi(x, (j - 2 * k).clamped_cast());

                    sum += sign * numerator / denom * x_factor;
                }

                sum
            }
        }
    }

    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<T, R, C, RS, CS>,
    ) {
        for j in start_index..row.ncols() {
            row[j] = self.solve_function(j, x);
        }
    }
}

impl<T: Value> PolynomialDisplay<T> for LegendreBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        let sign = Sign::from_coef(coef);

        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        let rank = display::unicode::subscript(&degree.to_string());
        let func = if degree > 0 {
            format!("P{rank}({x})")
        } else {
            String::new()
        };
        let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;

        let glue = if coef.is_empty() || func.is_empty() {
            ""
        } else {
            "·"
        };

        let body = format!("{coef}{glue}{func}");
        Some(display::Term::new(sign, body))
    }

    fn format_scaling_formula(&self) -> Option<String> {
        let x = display::unicode::subscript("s");
        let x = format!("x{x}");

        Some(format!("{x} = {}", self.normalizer))
    }
}

impl<T: Value> IntoMonomialBasis<T> for LegendreBasis<T> {
    fn as_monomial(&self, coefficients: &mut [T]) -> Result<()> {
        let n = coefficients.len();
        let mut result = vec![T::zero(); n];

        for j in 0..n {
            let c_j = coefficients[j];
            for k in 0..=(j / 2) {
                let sign = if k % 2 == 0 { T::one() } else { -T::one() };
                let numerator = T::factorial(2 * j - 2 * k);
                let denom = Value::powi(T::two(), j.clamped_cast())
                    * T::factorial(k)
                    * T::factorial(j - k)
                    * T::factorial(j - 2 * k);
                let x_power = j - 2 * k;
                result[x_power] += c_j * sign * numerator / denom;
            }
        }

        //
        // Phase 2 - Un-normalize over x
        result = self.normalizer.denormalize_coefs(&result);

        coefficients.copy_from_slice(&result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        assert_close, assert_fits, score::Aic, statistics::DegreeBound, test_basis_orthogonal,
        LegendreFit, Polynomial,
    };

    use super::*;

    fn get_poly() -> Polynomial<'static, LegendreBasis<f64>> {
        let basis = LegendreBasis::new(0.0, 100.0);
        Polynomial::from_basis(basis, &[1.0, 2.0, -0.5]).unwrap()
    }

    #[test]
    fn test_legendre_solve_function() {
        // Recover the polynomial
        let poly = get_poly();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = LegendreFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit);

        // Orthogonality test points
        let (gauss_xs, gauss_ws) = gauss_legendre_nodes_weights(100, 1e-12);
        test_basis_orthogonal!(
            fit.basis(),
            norm_fn = norm_fn,
            values = gauss_xs,
            weights = gauss_ws,
            n_funcs = 3,
            eps = 1e-12
        );

        // Monomial conversion
        let mono = fit.as_monomial().unwrap();
        assert_fits!(mono, fit);

        // Solve known functions
        let basis = LegendreBasis::new(-1.0, 1.0);
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), 0.5);
        assert_close!(basis.solve_function(2, 0.5), -0.125);
        assert_close!(basis.solve_function(3, 0.5), -0.4375);
    }

    //
    // orthogonal points and weights for Gauss-Legendre quadrature
    //

    fn norm_fn(n: usize) -> f64 {
        2.0 / (2.0 * n as f64 + 1.0)
    }

    fn gauss_legendre_nodes_weights(n: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
        assert!(n > 0);
        let mut xs = Vec::with_capacity(n);
        let mut ws = Vec::with_capacity(n);

        let m = n.div_ceil(2); // roots in [0,1]
        for i in 0..m {
            // Initial guess via cosine
            let theta = std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5);
            let mut x = theta.cos();

            // Newton–Raphson
            loop {
                let (p, dp) = legendre_and_derivative(n, x);
                let dx = -p / dp;
                x += dx;
                if dx.abs() < tol {
                    break;
                }
            }

            let (_, dp) = legendre_and_derivative(n, x);
            let w = 2.0 / ((1.0 - x * x) * dp * dp);

            // Mirror pair
            xs.push(-x);
            ws.push(w);
            if i != m - 1 || n % 2 == 0 {
                xs.push(x);
                ws.push(w);
            }
        }

        let mut combined: Vec<_> = xs.into_iter().zip(ws).collect();
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (xs, ws): (Vec<_>, Vec<_>) = combined.into_iter().unzip();
        (xs, ws)
    }

    // Evaluate P_n(x) and derivative at once
    fn legendre_and_derivative(n: usize, x: f64) -> (f64, f64) {
        let mut p0 = 1.0;
        let mut p1 = x;
        for k in 2..=n {
            let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p2;
        }
        let p = if n == 0 { p0 } else { p1 };
        // Derivative using recurrence
        let dp = (n as f64) * (x * p - p0) / (x * x - 1.0);
        (p, dp)
    }
}
