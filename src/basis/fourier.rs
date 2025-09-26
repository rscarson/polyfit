use nalgebra::MatrixViewMut;

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, MonomialBasis},
    display::{self, format_coefficient, Sign, Term, DEFAULT_PRECISION},
    error::Result,
    statistics::DomainNormalizer,
    value::{IntClampedCast, Value},
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
    polynomial_terms: usize,
}
impl<T: Value> FourierBasis<T> {
    /// Creates a new Fourier basis that normalizes inputs from the given range to [0, 2π].
    pub fn new(x_min: T, x_max: T) -> Self {
        let normalizer = DomainNormalizer::new((x_min, x_max), (T::zero(), T::two_pi()));
        Self {
            normalizer,
            polynomial_terms: 1,
        }
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
        Self {
            normalizer,
            polynomial_terms: 1,
        }
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
        // Because we support calculus, there can be a polynomial series at the start
        // The first [0..polynomial_terms] are monomial terms
        if j < self.polynomial_terms {
            return Value::powi(x, j.clamped_cast());
        }

        let j = j - self.polynomial_terms;
        // Now we have the Fourier terms

        if j % 2 == 0 {
            // Sine terms (odd indices)
            let n = (j + 1).div_ceil(2);

            // Infallible multiplication for the *n term
            let mut angle = x;
            for _ in 1..n {
                angle += x;
            }

            angle.sin()
        } else {
            // Cosine terms (even indices)
            #[allow(
                clippy::manual_div_ceil,
                reason = "Accidentally looking like your function != is your function"
            )]
            let n = (j + 1) / 2;

            // Infallible multiplication for the *n term
            let mut angle = x;
            for _ in 1..n {
                angle += x;
            }

            angle.cos()
        }
    }

    //
    // invariant: polynomial_terms == 1 here
    // We only kerfuffle that in calculus methods
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        mut row: MatrixViewMut<'_, T, R, C, RS, CS>,
    ) {
        let cos_x = x.cos();
        for j in start_index..row.ncols() {
            row[j] = match j {
                0 => T::one(),
                1 => x.sin(), // first sin
                2 => cos_x,   // first cos
                3 => T::two() * cos_x * row[1],
                4 => T::two() * cos_x * row[2] - T::one(),
                _ => T::two() * cos_x * row[j - 2] - row[j - 4],
            }
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

        let (src_min, src_max) = self.normalizer.src_range();
        Some(format!("{x} = T[ {src_min}..{src_max} -> 0..2π ]"))
    }
}

impl<T: Value> IntegralBasis<T> for FourierBasis<T> {
    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self, Vec<T>)> {
        if coefficients.is_empty() {
            return Ok((self.clone(), vec![constant]));
        }

        //
        // We do a monomial integral for the first [0..polynomial_terms]
        let polynomial_terms = &coefficients[..self.polynomial_terms.min(coefficients.len())];
        let (_, mut integral_coeffs) =
            MonomialBasis::default().integral(polynomial_terms, constant)?;

        //
        // The integral is actually not hard here
        // 2- an * sin(nx) -> -an/n * cos(nx)
        // 3- bn * cos(nx) -> bn/n * sin(nx)
        // We can do the last 2 by flipping pairs of coefficients and dividing by n
        let fourier_terms = &coefficients[self.polynomial_terms.min(coefficients.len())..];

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

        let basis = Self {
            normalizer: self.normalizer,
            polynomial_terms: self.polynomial_terms + 1,
        };
        Ok((basis, integral_coeffs))
    }
}

impl<T: Value> DifferentialBasis<T> for FourierBasis<T> {
    fn derivative(&self, coefficients: &[T]) -> Result<(Self, Vec<T>)> {
        if coefficients.len() <= 1 {
            return Ok((self.clone(), vec![T::zero()]));
        }

        //
        // We do a monomial differential for the first [0..polynomial_terms]
        let polynomial_terms = &coefficients[..self.polynomial_terms.min(coefficients.len())];
        let (_, mut derivative_coeffs) = MonomialBasis::default().derivative(polynomial_terms)?;

        let fourier_terms = &coefficients[self.polynomial_terms.min(coefficients.len())..];

        //
        // Similar to integral, we can do this by flipping pairs of coefficients:
        // 2- an * sin(nx) -> n * an * cos(nx)
        // 3- bn * cos(nx) -> -n * bn * sin(nx)

        let mut n = T::one(); // frequency index
        let mut coef_iter = fourier_terms.iter();
        while let Some(a) = coef_iter.next().copied() {
            let b = coef_iter.next().copied().unwrap_or(T::zero());

            // Fourier expects pairs of (sin, cos), so originally we had (a, b)
            // Now under differentiation we need (b, -a) to get the right functions
            derivative_coeffs.push(n * b);
            derivative_coeffs.push(n * -a);

            n += T::one(); // increment frequency index
        }

        let basis = Self {
            normalizer: self.normalizer,
            polynomial_terms: self.polynomial_terms.saturating_sub(1),
        };
        Ok((basis, derivative_coeffs))
    }

    fn critical_points(&self, dx_coefs: &[T]) -> Result<Vec<T>> {
        //
        // We will use tangents to build a monomial form over one period [0, 2π]
        // We can substitute sin(x) = (2t)/(1+t^2) and cos(x) = (1-t^2)/(1+t^2)
        // where t = tan(x/2). This gives us a rational polynomial in t.
        let monomial_coefs = build_tangent_polynomial(dx_coefs);

        // Now we can find the roots of the monomial polynomial
        let mut points = crate::basis::MonomialBasis::default().critical_points(&monomial_coefs)?;

        //
        // Finally, we need to convert back to x by de-normalizing and using x = 2 arctan(t)
        for t in &mut points {
            *t = T::two() * t.atan(); // x in [0, 2π]
            *t = self.normalizer.denormalize(*t); // x in original range
        }

        Ok(points)
    }
}

//
// For critical points, we build a polynomial in t = tan(x/2) using the tangent half-angle formulas
// sin(x) = 2t / (1 + t^2)
fn build_tangent_polynomial<T: Value>(dx_coefs: &[T]) -> Vec<T> {
    let max_n = (dx_coefs.len() - 1) / 2;

    // Initialize base cases: sin(1x) and cos(1x) in terms of t
    let mut sin_poly: Vec<Vec<T>> = vec![vec![T::zero()]; max_n + 1];
    let mut cos_poly: Vec<Vec<T>> = vec![vec![T::zero()]; max_n + 1];

    sin_poly[1] = vec![T::zero(), T::two()]; // sin(x) = 2 t
    cos_poly[1] = vec![T::one(), T::zero(), -T::one()]; // cos(x) = 1 - t^2

    // Build higher harmonics recursively
    for n in 2..=max_n {
        sin_poly[n] = poly_sub(
            &poly_mul(&[T::two(), T::zero(), -T::two()], &sin_poly[n - 1]), // 2 cos(x) * sin((n-1)x)
            &sin_poly[n - 2],
        );
        cos_poly[n] = poly_sub(
            &poly_mul(&[T::two(), T::zero(), -T::two()], &cos_poly[n - 1]), // 2 cos(x) * cos((n-1)x)
            &cos_poly[n - 2],
        );
    }

    // Combine all terms with derivative coefficients
    let mut poly = vec![T::zero()];
    let mut n = 1;
    let mut coef_iter = dx_coefs.iter().skip(1);
    while let Some(a) = coef_iter.next() {
        let b = coef_iter.next().copied().unwrap_or(T::zero());

        poly = poly_add(&poly, &scalar_poly(*a, &sin_poly[n]));
        poly = poly_add(&poly, &scalar_poly(b, &cos_poly[n]));

        n += 1;
    }

    // multiply numerator by (1 + t^2)^max_n to clear denominators
    let denom_poly = vec![T::one(), T::zero(), T::one()]; // 1 + t^2
    let mut full_poly = vec![T::one()]; // start as 1

    for _ in 0..max_n {
        full_poly = poly_mul(&full_poly, &denom_poly);
    }

    poly_mul(&poly, &full_poly)
}

fn scalar_poly<T: Value>(s: T, p: &[T]) -> Vec<T> {
    p.iter().map(|x| s * *x).collect()
}

fn poly_add<T: Value>(p1: &[T], p2: &[T]) -> Vec<T> {
    let n = p1.len().max(p2.len());
    (0..n)
        .map(|i| p1.get(i).copied().unwrap_or(T::zero()) + p2.get(i).copied().unwrap_or(T::zero()))
        .collect()
}

fn poly_sub<T: Value>(p1: &[T], p2: &[T]) -> Vec<T> {
    let n = p1.len().max(p2.len());
    (0..n)
        .map(|i| p1.get(i).copied().unwrap_or(T::zero()) - p2.get(i).copied().unwrap_or(T::zero()))
        .collect()
}

fn poly_mul<T: Value>(p1: &[T], p2: &[T]) -> Vec<T> {
    let mut res = vec![T::zero(); p1.len() + p2.len() - 1];
    for (i, a) in p1.iter().enumerate() {
        for (j, b) in p2.iter().enumerate() {
            res[i + j] += *a * *b;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_close, assert_fits, score::Aic, statistics::DegreeBound, FourierFit, Polynomial,
    };

    fn get_poly() -> Polynomial<'static, FourierBasis<f64>> {
        let basis = FourierBasis::new(0.0, 100.0);
        Polynomial::from_basis(basis, &[1.0, 2.0, -0.5]).unwrap()
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_fourier_basis() {
        // Recover polynomial
        let poly = get_poly();
        let data = poly.solve_range(0.0..=100.0, 1.0);
        let fit = FourierFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit);

        // Solve known values
        let basis = FourierBasis::new(0.0, 2.0 * std::f64::consts::PI);
        assert_close!(basis.solve_function(0, 0.5), 1.0);
        assert_close!(basis.solve_function(1, 0.5), 0.479425538604203);
        assert_close!(basis.solve_function(2, 0.5), 0.8775825618903728);
        assert_close!(basis.solve_function(3, 0.5), 0.8414709848078965);

        // Integrate -> differentiate = Original
        let org_coefs = fit.coefficients();
        let (bi, int_coefs) = fit.basis().integral(org_coefs, 0.0).unwrap();
        let (_, diff_coefs) = bi.derivative(&int_coefs).unwrap();
        assert_close!(org_coefs[0], diff_coefs[0]); // constant term
        for (a, b) in org_coefs.iter().skip(1).zip(diff_coefs.iter().skip(1)) {
            assert_close!(*a, -*b); // Phase-shifted Fourier terms (negated)
        }
    }
}
