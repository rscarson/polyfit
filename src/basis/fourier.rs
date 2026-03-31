use nalgebra::{Complex, ComplexField, DMatrix, MatrixViewMut, Normed};

use crate::{
    basis::{
        trigonometric_polynomial, Basis, DifferentialBasis, IntegralBasis,
        LinearAugmentedFourierBasis, OrthogonalBasis, Root, RootFindingBasis,
    },
    display::{self, Term},
    error::Result,
    statistics::DomainNormalizer,
    value::Value,
    Polynomial,
};

/// Type alias for a Fourier polynomial (`Polynomial<FourierBasis, T>`).
pub type FourierPolynomial<'a, T> = crate::Polynomial<'a, FourierBasis<T>, T>;
impl<T: Value> FourierPolynomial<'_, T> {
    /// Create a new Fourier polynomial with the given constant and Fourier coefficients over the specified x-range.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which the Fourier basis is defined
    /// - `constant`: The constant term of the polynomial
    /// - `terms`: A slice of (`a_n`, `b_n`) pairs representing the sine and cosine coefficients
    ///
    /// # Returns
    /// A polynomial defined in the Fourier basis.
    ///
    /// For example to create a Fourier polynomial:
    /// ```math
    /// f(x) = 3 + 2 sin(2πx) - 0.5 cos(2πx)
    /// ```
    ///
    /// ```rust
    /// use polyfit::FourierPolynomial;
    /// let poly = FourierPolynomial::new((-1.0, 1.0), 3.0, &[(2.0, -0.5)]);
    /// ```
    #[allow(
        clippy::missing_panics_doc,
        reason = "Always has valid coefficients for Fourier basis"
    )]
    pub fn new(x_range: (T, T), constant: T, terms: &[(T, T)]) -> Self {
        let mut coefficients = Vec::with_capacity(1 + terms.len() * 2);
        coefficients.push(constant);
        for (a_n, b_n) in terms {
            coefficients.push(*a_n); // sin term
            coefficients.push(*b_n); // cos term
        }

        let basis = FourierBasis::new(x_range.0, x_range.1);
        Polynomial::from_basis(basis, coefficients).expect("Failed to create Fourier polynomial")
    }

    /// Creates a copy of the fourier series with a specified DC offset (monomial constant term).
    /// This is useful for adjusting the baseline of the function without affecting its oscillatory behavior.
    ///
    /// Useful if you integrate a Fourier series and want the terms to line up with the original function, for example.
    #[must_use]
    pub fn with_dc_offset(&self, offset: T) -> Self {
        let mut coefficients = self.coefficients().to_vec();
        if coefficients.is_empty() {
            coefficients.push(offset);
        } else {
            coefficients[0] = offset;
        }

        unsafe { Polynomial::from_raw(self.basis().clone(), coefficients.into(), self.degree()) }
    }
}

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
impl<T: Value> trigonometric_polynomial::TrigonometricPolynomialBasis<0, T> for FourierBasis<T> {
    fn normalizer(&self) -> &DomainNormalizer<T> {
        &self.normalizer
    }
}
impl<T: Value> FourierBasis<T> {
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

    /// Convert Fourier coefficients to complex monomial coefficients.
    ///
    /// This is useful for certain mathematical operations, such as root finding
    pub fn as_complex_monomial(&self, coefs: &[T]) -> Vec<Complex<T>> {
        let n = (coefs.len() - 1) / 2;

        let mut c = vec![Complex::new(T::zero(), T::zero()); 2 * n + 1];

        // c_0
        c[n] = Complex::from_real(coefs[0]);

        let half = T::one() / T::two();

        for k in 1..=n {
            let b_k = coefs[2 * k - 1]; // sin
            let a_k = coefs[2 * k]; // cos

            // +k
            c[n + k] = Complex::new(a_k * half, -b_k * half);

            // -k
            c[n - k] = Complex::new(a_k * half, b_k * half);
        }

        c
    }
}
impl<T: Value> Basis<T> for FourierBasis<T> {
    fn from_range(x_range: std::ops::RangeInclusive<T>) -> Self {
        let normalizer = trigonometric_polynomial::new_normalizer(*x_range.start(), *x_range.end());
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
    fn k(&self, degree: usize) -> usize {
        trigonometric_polynomial::TrigonometricPolynomialBasis::k(self, degree)
    }

    #[inline(always)]
    fn degree(&self, k: usize) -> Option<usize> {
        trigonometric_polynomial::TrigonometricPolynomialBasis::degree(self, k)
    }

    #[inline(always)]
    fn solve_function(&self, j: usize, x: T) -> T {
        trigonometric_polynomial::TrigonometricPolynomialBasis::solve_function(self, j, x)
    }

    #[inline(always)]
    fn solve(&self, x: T, coefficients: &[T]) -> T {
        trigonometric_polynomial::TrigonometricPolynomialBasis::solve(self, x, coefficients)
    }

    //
    // invariant: polynomial_terms == 1 here
    // We only kerfuffle that in calculus methods
    #[inline(always)]
    fn fill_matrix_row<R: nalgebra::Dim, C: nalgebra::Dim, RS: nalgebra::Dim, CS: nalgebra::Dim>(
        &self,
        start_index: usize,
        x: T,
        row: MatrixViewMut<'_, T, R, C, RS, CS>,
    ) {
        trigonometric_polynomial::TrigonometricPolynomialBasis::fill_matrix_row(
            self,
            start_index,
            x,
            row,
        );
    }
}

impl<T: Value> display::PolynomialDisplay<T> for FourierBasis<T> {
    fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
        trigonometric_polynomial::TrigonometricPolynomialBasis::format_term(self, degree, coef)
    }

    fn format_scaling_formula(&self) -> Option<String> {
        trigonometric_polynomial::TrigonometricPolynomialBasis::format_scaling_formula(self)
    }
}

impl<T: Value> IntegralBasis<T> for FourierBasis<T> {
    type B2 = LinearAugmentedFourierBasis<T>;

    fn integral(&self, coefficients: &[T], constant: T) -> Result<(Self::B2, Vec<T>)> {
        let coefs = trigonometric_polynomial::TrigonometricPolynomialBasis::integral(
            self,
            coefficients,
            constant,
        )?;

        let basis = LinearAugmentedFourierBasis::from_normalizer(self.normalizer);
        Ok((basis, coefs))
    }
}

impl<T: Value> DifferentialBasis<T> for FourierBasis<T> {
    type B2 = Self;

    fn derivative(&self, coefficients: &[T]) -> Result<(Self::B2, Vec<T>)> {
        let coefs =
            trigonometric_polynomial::TrigonometricPolynomialBasis::derivative(self, coefficients)?;
        Ok((self.clone(), coefs))
    }
}

impl<T: Value> OrthogonalBasis<T> for FourierBasis<T> {
    fn gauss_weight(&self, _: T) -> T {
        T::one()
    }

    fn gauss_nodes(&self, n: usize) -> Vec<(T, T)> {
        // Nodes: equispaced in [0, 2π)
        // Weights: uniform (trapezoid rule for periodic functions)
        if n == 0 {
            return vec![(T::zero(), T::one())];
        }

        let two_pi = T::two() * T::pi();
        let w = two_pi / T::from_positive_int(n);
        (0..n)
            .map(|k| {
                let x = T::from_positive_int(k) * w; // x in [0, 2π)
                (x, w)
            })
            .collect()
    }

    fn gauss_normalization(&self, n: usize) -> T {
        if n == 0 {
            // constant term
            T::two() * T::pi()
        } else {
            // all sin/cos terms
            T::pi()
        }
    }
}

impl<T: Value> RootFindingBasis<T> for FourierBasis<T> {
    fn roots(
        &self,
        coefs: &[T],
        x_range: std::ops::RangeInclusive<T>,
    ) -> Result<Vec<super::Root<T>>> {
        let orig_coefs = coefs.to_vec();
        let mut coefs = self.as_complex_monomial(coefs);

        let n = coefs.len() - 1; // degree of polynomial
        if n == 0 {
            return Ok(vec![]);
        }

        let mut companion = DMatrix::zeros(n, n);

        // Find last non-zero (by magnitude)
        let leading_index = coefs.iter().rposition(|c| c.norm() > T::epsilon());

        if let Some(idx) = leading_index {
            let leading = coefs[idx];

            if leading.norm() > T::epsilon() && leading != Complex::from_real(T::one()) {
                for c in &mut coefs {
                    *c /= leading;
                }
            }
        } else {
            // All coefficients are zero
            return Ok(vec![]);
        }

        for i in 1..n {
            companion[(i, i - 1)] = Complex::from_real(T::one());
        }

        let leading = coefs[n];

        for i in 0..n {
            companion[(i, n - 1)] = -coefs[i] / leading;
        }

        let Some(eigs) = companion.eigenvalues() else {
            return Err(crate::error::Error::Algebra(
                "Failed to compute eigenvalues for root finding",
            ));
        };

        let mut eigs = eigs.as_slice().to_vec();
        for z in &mut eigs {
            let norm = z.norm();
            if norm > T::epsilon() {
                *z /= norm;
            }
        }

        // Filter and categorize roots
        let test = eigs
            .iter()
            .map(|e| {
                let arg = e.argument() + T::pi(); // Shift to align with Fourier basis
                self.denormalize_x(arg)
            })
            .collect::<Vec<_>>();
        println!("Eigenvalue arguments: {:?}", test);
        let roots = Root::roots_from_complex(&eigs, |z| self.complex_y(*z, &orig_coefs));

        // Remove roots outside the specified x_range
        let x_min = *x_range.start();
        let x_max = *x_range.end();
        let roots = roots
            .into_iter()
            .filter(|root| match root {
                Root::Real(r) => *r >= x_min && *r <= x_max,
                Root::Complex(_) | Root::ComplexPair(_, _) => true, // Keep complex roots
            })
            .collect();

        Ok(roots)
    }

    fn complex_y(&self, z: Complex<T>, coefs: &[T]) -> Complex<T> {
        let complex_coefs = self.as_complex_monomial(coefs);
        let n = (complex_coefs.len() - 1) / 2;

        let mut result = Complex::new(T::zero(), T::zero());

        for k in 0..=2 * n {
            let exp = k as isize - n as isize;

            let term = if exp >= 0 {
                z.powi(exp as i32)
            } else {
                Complex::from_real(T::one()) / z.powi((-exp) as i32)
            };

            result += complex_coefs[k] * term;
        }
        result
    }
}