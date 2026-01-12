use std::{borrow::Cow, ops::RangeInclusive};

use crate::{
    basis::{
        Basis, CriticalPoint, DifferentialBasis, IntegralBasis, IntoMonomialBasis, OrthogonalBasis,
        Root, RootFindingBasis,
    },
    display::PolynomialDisplay,
    error::{Error, Result},
    statistics,
    value::{CoordExt, FloatClampedCast, SteppedValues, Value},
    MonomialPolynomial,
};

/// Represents a polynomial function in a given basis.
///
/// Unlike [`crate::CurveFit`], this struct is **not tied to any dataset or matrix**, making it a canonical function that
/// can be evaluated for **any x-value** without range restrictions.
///
/// # Type Parameters
/// - `'a`: Lifetime for borrowed basis or coefficients, if used.
/// - `B`: The polynomial basis (e.g., [`crate::basis::MonomialBasis`], [`crate::basis::ChebyshevBasis`]).
/// - `T`: Numeric type for the coefficients, default is `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<'a, B, T: Value = f64>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    degree: usize,
    basis: B,
    coefficients: Cow<'a, [T]>,
}
impl<'a, B, T: Value> Polynomial<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    /// Creates a [`Polynomial`] from a given basis, coefficients, and degree.
    ///
    /// # Safety
    /// This constructor is unsafe because it allows the creation of a polynomial
    /// without enforcing the usual invariants (e.g., degree must match the number
    /// of coefficients expected by the basis).
    ///
    /// The length of coefficients must be equal to `Basis::k(degree)`
    ///
    /// # Parameters
    /// - `basis`: The polynomial basis
    /// - `coefficients`: The coefficients for the polynomial, possibly borrowed or owned
    /// - `degree`: The degree of the polynomial
    ///
    /// # Returns
    /// A new [`Polynomial`] instance with the given basis and coefficients.
    pub const unsafe fn from_raw(basis: B, coefficients: Cow<'a, [T]>, degree: usize) -> Self {
        Self {
            degree,
            basis,
            coefficients,
        }
    }

    /// Creates a new polynomial from a basis and coefficients, inferring the degree.
    ///
    /// # Parameters
    /// - `basis`: The polynomial basis
    /// - `coefficients`: The coefficients for the polynomial, possibly borrowed or owned
    ///
    /// # Returns
    /// A new [`Polynomial`] instance with the given basis and coefficients, or an error if the number of coefficients is invalid for the basis.
    ///
    /// # Errors
    /// Returns an error if the number of coefficients does not correspond to a valid degree for the given basis.
    pub fn from_basis(basis: B, coefficients: impl Into<Cow<'a, [T]>>) -> Result<Self> {
        let coefficients = coefficients.into();
        let degree = basis.degree(coefficients.len()).ok_or(
            crate::error::Error::InvalidNumberOfParameters(coefficients.len()),
        )?;
        Ok(unsafe { Self::from_raw(basis, coefficients, degree) })
    }

    /// Decomposes the polynomial into its basis, coefficients, and degree.
    pub fn into_inner(self) -> (B, Cow<'a, [T]>, usize) {
        (self.basis, self.coefficients, self.degree)
    }

    /// Returns a reference to the polynomial's basis.
    pub(crate) fn basis(&self) -> &B {
        &self.basis
    }

    /// Converts the polynomial into an owned version.
    ///
    /// This consumes the current `Polynomial` and returns a new one with
    /// `'static` lifetime, owning both the basis and the coefficients.
    ///
    /// Useful when you need a fully independent polynomial that does not
    /// borrow from any external data.
    pub fn into_owned(self) -> Polynomial<'static, B, T> {
        Polynomial {
            degree: self.degree,
            basis: self.basis,
            coefficients: Cow::Owned(self.coefficients.into_owned()),
        }
    }

    /// Returns a reference to the polynomial’s coefficients.
    ///
    /// The index of each coefficient the jth basis function.
    ///
    /// For example in a monomial expression `y(x) = 2x^2 - 3x + 1`;
    /// coefficients = [1.0, -3.0, 2.0]
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Formally, for each coefficient *j*, and the jth basis function *`B_j(x)`*, the relationship is:
    /// ```math
    /// y(x) = Σ (c_j * B_j(x))
    /// ```
    /// </div>
    #[must_use]
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }

    /// Returns a mutable reference to the polynomial’s coefficients.
    ///
    /// The index of each coefficient the jth basis function.
    ///
    /// For example in a monomial expression `y(x) = 2x^2 - 3x + 1`;
    /// coefficients = [1.0, -3.0, 2.0]
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Formally, for each coefficient *j*, and the jth basis function *`B_j(x)`*, the relationship is:
    /// ```math
    /// y(x) = Σ (c_j * B_j(x))
    /// ```
    /// </div>
    #[must_use]
    pub fn coefficients_mut(&mut self) -> &mut [T] {
        self.coefficients.to_mut()
    }

    /// Returns the degree of the polynomial.
    ///
    /// The number of actual components, or basis functions, in the expression of a degree is defined by the basis.
    ///
    /// That number is called k. For most basis choices, `k = degree + 1`.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Replaces the coefficients of the polynomial in place with absolute values.
    pub fn abs(&mut self) {
        for c in self.coefficients.to_mut().iter_mut() {
            *c = Value::abs(*c);
        }
    }

    /// Scales all coefficients of the polynomial by a given factor in place.
    pub fn scale(&mut self, factor: T) {
        for c in self.coefficients.to_mut().iter_mut() {
            *c *= factor;
        }
    }

    /// Evaluates the polynomial at a given x-value.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Given [`Basis::k`] coefficients and basis functions, and for each pair of coefficients *`c_j`* and basis function *`B_j(x)`*, this function returns:
    /// ```math
    /// y(x) = Σ (c_j * B_j(x))
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `x`: The point at which to evaluate the polynomial.
    ///
    /// # Returns
    /// The computed y-value using the polynomial basis and coefficients.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{MonomialPolynomial};
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // Represents 1 + 2x + 3x^2
    /// let y = poly.y(2.0); // evaluates 1 + 2*2 + 3*2^2 = 17.0
    /// ```
    pub fn y(&self, x: T) -> T {
        let mut y = T::zero();
        let x = self.basis.normalize_x(x);
        for (i, &coef) in self.coefficients.iter().enumerate() {
            y += coef * self.basis.solve_function(i, x);
        }

        y
    }

    /// Evaluates the polynomial at multiple x-values.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Given [`Basis::k`] coefficients and basis functions, and for each pair of coefficients *`c_j`* and basis function *`B_j(x)`*, this function returns:
    /// ```math
    /// y(x) = Σ (c_j * B_j(x))
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `x`: An iterator of x-values at which to evaluate the polynomial.
    ///
    /// # Returns
    /// A `Vec` of `(x, y)` pairs corresponding to each input value.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// let points = poly.solve(vec![0.0, 1.0, 2.0]);
    /// // points = [(0.0, 1.0), (1.0, 6.0), (2.0, 17.0)]
    /// ```
    pub fn solve(&self, x: impl IntoIterator<Item = T>) -> Vec<(T, T)> {
        x.into_iter().map(|x| (x, self.y(x))).collect()
    }

    /// Evaluates the polynomial over a range of x-values with a fixed step.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Given [`Basis::k`] coefficients and basis functions, and for each pair of coefficients *`c_j`* and basis function *`B_j(x)`*, this function returns:
    /// ```math
    /// y(x) = Σ (c_j * B_j(x))
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `range`: The start and end of the x-values to evaluate.
    /// - `step`: The increment between successive x-values.
    ///
    /// # Returns
    /// A `Vec` of `(x, y)` pairs for each sampled point.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// let points = poly.solve_range(0.0..=2.0, 1.0);
    /// // points = [(0.0, 1.0), (1.0, 6.0), (2.0, 17.0)]
    /// ```
    pub fn solve_range(&self, range: RangeInclusive<T>, step: T) -> Vec<(T, T)> {
        self.solve(SteppedValues::new(range, step))
    }

    /// Calculates the R-squared value for the model compared to provided data.
    ///
    /// R-squared is a statistical measure of how well the polynomial explains
    /// the variance in the data. Values closer to 1 indicate a better fit.
    ///
    /// # Parameters
    /// - `data`: A slice of `(x, y)` pairs to compare against the polynomial fit.
    ///
    /// See [`statistics::r_squared`] for more details.
    ///
    /// # Returns
    /// The R-squared value as type `T`.
    pub fn r_squared(&self, data: &[(T, T)]) -> T {
        let x = data.x_iter();
        let y = data.y_iter();
        let y_fit = self.solve(x).into_iter().map(|(_, y)| y);

        statistics::r_squared(y, y_fit)
    }

    /// Removes leading zero coefficients from the polynomial in place.
    /// For example, a polynomial `y(x) = 0x^2 + x + 3` would become `y(x) = x + 3`
    pub fn remove_leading_zeros(&mut self) {
        while self.degree > 0
            && !self.coefficients.is_empty()
            && Value::abs(self.coefficients[self.degree]) <= T::epsilon()
        {
            self.degree -= 1;
            self.coefficients.to_mut().pop();
        }

        if self.coefficients.is_empty() {
            self.coefficients.to_mut().push(T::zero());
        }
    }

    /// Returns the most-significant (leading) non-zero coefficient of the polynomial.
    ///
    /// If all coefficients are zero, returns zero.
    pub fn leading_coefficient(&self) -> T {
        for c in self.coefficients.iter().rev() {
            if Value::abs(*c) > T::epsilon() {
                return *c;
            }
        }
        T::zero()
    }

    /// Computes the derivative of this polynomial.
    ///
    /// # Type Parameters
    /// - `B2`: The basis type for the derivative (determined by the implementing `DifferentialBasis` trait).
    ///
    /// # Returns
    /// - `Ok(Polynomial<'static, B2, T>)`: The derivative polynomial.
    /// - `Err`: If computing the derivative fails.
    ///
    /// # Requirements
    /// - The polynomial's basis `B` must implement [`DifferentialBasis`].
    ///
    /// # Errors
    /// If the basis cannot compute the derivative coefficients, an error is returned.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::function;
    /// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3);
    /// let deriv = test.derivative().unwrap();
    /// println!("Derivative: {:?}", deriv.coefficients());
    /// ```
    pub fn derivative(&self) -> Result<Polynomial<'static, B::B2, T>>
    where
        B: DifferentialBasis<T>,
    {
        let new_degree = if self.degree == 0 { 0 } else { self.degree - 1 };

        let (db, dc) = self.basis.derivative(&self.coefficients)?;
        let derivative = unsafe { Polynomial::from_raw(db, dc.into(), new_degree) };

        Ok(derivative)
    }

    /// Finds the critical points (where the derivative is zero) of a polynomial in this basis.
    ///
    /// This corresponds to the polynomial's local minima and maxima (The `x` values where curvature changes).
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// The critical points are found by solving the equation `f'(x) = 0`, where `f'(x)` is the derivative of the polynomial.
    ///
    /// This is done with by finding the eigenvalues of the companion matrix of the derivative polynomial.
    /// </div>
    ///
    /// # Returns
    /// A vector of `x` values where the critical points occur.
    ///
    /// # Requirements
    /// - The polynomial's basis `B` must implement [`DifferentialBasis`].
    ///
    /// # Errors
    /// Returns an error if the critical points cannot be found.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// let critical_points = poly.critical_points(0.0..=100.0).unwrap();
    /// ```
    pub fn critical_points(&self, x_range: RangeInclusive<T>) -> Result<Vec<CriticalPoint<T>>>
    where
        B: DifferentialBasis<T>,
        B::B2: DifferentialBasis<T>,
    {
        let dx = self.derivative()?;
        let ddx = dx.derivative()?;
        let roots = dx.real_roots(x_range, None)?;

        let mut points = Vec::with_capacity(roots.len());
        for x in roots {
            let curvature = ddx.y(x);
            let y = self.y(x);

            match curvature {
                c if c > T::zero() => points.push(CriticalPoint::Minima(x, y)),
                c if c < T::zero() => points.push(CriticalPoint::Maxima(x, y)),
                _ => points.push(CriticalPoint::Inflection(x, y)),
            }
        }

        Ok(points)
    }

    /// Finds the roots (zeros) of the polynomial in this basis.
    ///
    /// This corresponds to the `x` values where the polynomial evaluates to zero.
    ///
    /// A root can be either real or complex:
    /// - `Root::Real(x)` indicates a real root at `x`. This is a point where the polynomial crosses or touches the x-axis.
    /// - `Root::ComplexPair(z, z2)` indicates a pair of complex conjugate roots. These do not correspond to x-axis crossings but are important in the polynomial's overall behavior.
    /// - `Root::Complex(z)` indicates a single complex root (not part of a conjugate pair). Should be rare for polynomials with real coefficients.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// The roots are found by solving the equation `f(x) = 0`, where `f(x)` is the polynomial.
    ///
    /// This is done in a basis-specific manner, often involving finding the eigenvalues of the companion matrix of the polynomial.
    ///
    /// </div>
    ///
    /// # Returns
    /// A vector of `Root<T>` representing the roots of the polynomial.
    ///
    /// # Errors
    /// Returns an error if the roots cannot be found.
    pub fn roots(&self) -> Result<Vec<Root<T>>>
    where
        B: RootFindingBasis<T>,
    {
        self.basis.roots(self.coefficients())
    }

    /// Uses a less precise iterative method to find only the real roots of the polynomial.
    ///
    /// This is less precise than [`Self::roots`] and will not find complex roots, but is often faster and more stable for high-degree polynomials
    /// and is available for all bases.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values to search for real roots.
    /// - `max_newton_iterations`: The maximum number of Newton-Raphson iterations to refine each root.
    ///   This helps improve the accuracy of the found roots. If omitted, a sensible value will be calculated
    ///
    /// # Returns
    /// A vector of `T` representing the real roots of the polynomial within the specified range
    ///
    /// # Errors
    /// Returns an error if the derivative cannot be computed.
    pub fn real_roots(
        &self,
        x_range: RangeInclusive<T>,
        max_newton_iterations: Option<usize>,
    ) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        const DEFAULT_SAMPLES: f64 = 5000.0;
        const DEFAULT_ITERATIONS: usize = 20;

        let mut roots = vec![];
        let mut prev_x = *x_range.start();
        let mut prev_y = self.y(prev_x);

        // The aim is ~5000 samples, given a 64bit float precision and a width of domain of 100
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
        let num_samples = DEFAULT_SAMPLES.clamped_cast::<T>() * domain_scalar * precision_scalar;

        // Determine a sensible number of newton iterations if not provided
        // This is based leading coefficient, and type precision
        //
        // 20 iterations is usually sufficient for double precision and well-scaled polynomials
        let max_newton_iterations = if let Some(max) = max_newton_iterations {
            max
        } else {
            let leading_coef = self.leading_coefficient();
            let coef_scalar = Value::min(Value::abs(leading_coef.log10()), T::one());
            let max =
                (DEFAULT_ITERATIONS as f64).clamped_cast::<T>() * coef_scalar * precision_scalar;

            max.as_usize().unwrap_or(DEFAULT_ITERATIONS)
        };

        let dx = self.derivative()?;
        let sqrt_eps = T::epsilon().sqrt();
        for x in SteppedValues::new(x_range, domain_width / num_samples) {
            let y = self.y(x);
            if (prev_y * y).is_sign_negative() || prev_y.is_near_zero() || y.is_near_zero() {
                // Sign change -> root in between
                // Perform bisection to find it more precisely
                let mid_x = (prev_x + x) / T::two();
                let mid_y = (prev_y + y) / T::two();
                let slope = (y - prev_y) / (x - prev_x);
                let neg_recip_slope = if Value::abs(slope) > sqrt_eps {
                    T::one() / slope
                } else {
                    T::zero()
                };

                let x = Value::clamp(mid_x - mid_y * neg_recip_slope, prev_x, x);

                // Newton iterations to refine
                let mut newton_prev_x = x;
                let mut newton_x;
                for _ in 0..max_newton_iterations {
                    let y = self.y(newton_prev_x);
                    let dy = dx.y(newton_prev_x);
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

        Ok(roots)
    }

    /// Computes the indefinite integral of this polynomial.
    ///
    /// # Type Parameters
    /// - `B2`: The basis type for the integral (determined by the implementing `DifferentialBasis` trait).
    ///
    /// # Parameters
    /// - `constant`: Constant of integration (value at x = 0).
    ///
    /// # Requirements
    /// - The polynomial's basis `B` must implement [`IntegralBasis`].
    ///
    /// # Returns
    /// - `Ok(Polynomial<'static, B2, T>)`: The integral polynomial.
    /// - `Err`: If computing the integral fails.
    ///
    /// # Errors
    /// If the basis cannot compute the integral coefficients, an error is returned.
    ///
    /// # Example
    /// ```rust
    /// use polyfit::function;
    /// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3);
    /// let integral = test.integral(Some(1.0)).unwrap();
    /// println!("Integral: {:?}", integral.coefficients());
    /// ```
    pub fn integral(&self, constant: Option<T>) -> Result<Polynomial<'static, B::B2, T>>
    where
        B: IntegralBasis<T>,
    {
        let constant = constant.unwrap_or(T::zero());
        let new_degree = self.degree + 1;

        let (ib, ic) = self.basis.integral(&self.coefficients, constant)?;
        let integral = unsafe { Polynomial::from_raw(ib, ic.into(), new_degree) };

        Ok(integral)
    }

    /// Computes the definite integral (area under the curve) of the fitted polynomial
    /// between `x_min` and `x_max`.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// The area under the curve is computed using the definite integral of the polynomial
    /// between the specified bounds:
    /// ```math
    /// Area = ∫[x_min to x_max] f(x) dx = F(x_max) - F(x_min)
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `x_min`: Lower bound of integration.
    /// - `x_max`: Upper bound of integration.
    /// - `constant`: Constant of integration (value at x = 0) for the indefinite integral.
    ///
    /// # Requirements
    /// - The polynomial's basis `B` must implement [`IntegralBasis`].
    ///
    /// # Returns
    /// - `Ok(T)`: The computed area under the curve between `x_min` and `x_max`.
    /// - `Err`: If computing the integral fails (e.g., basis cannot compute integral coefficients).
    ///
    /// # Errors
    /// If the basis cannot compute the integral coefficients, an error is returned.
    ///
    /// # Example
    /// ```rust
    /// polyfit::function!(poly(x) = 4 x^3 + 2);
    /// let area = poly.area_under_curve(0.0, 3.0, None).unwrap();
    /// println!("Area under curve: {}", area);
    /// ```
    pub fn area_under_curve(&self, x_min: T, x_max: T, constant: Option<T>) -> Result<T>
    where
        B: IntegralBasis<T>,
    {
        let integral = self.integral(constant)?;
        Ok(integral.y(x_max) - integral.y(x_min))
    }

    /// Returns the X-values where the function is not monotone (i.e., where the derivative changes sign).
    ///
    /// # Errors
    /// Returns an error if the derivative cannot be computed.
    ///
    /// # Example
    /// ```rust
    /// polyfit::function!(poly(x) = 4 x^3 + 2);
    /// let area = poly.area_under_curve(0.0, 3.0, None).unwrap();
    /// let violations = poly.monotonicity_violations(0.0..=3.0).unwrap();
    /// ```
    pub fn monotonicity_violations(&self, x_range: RangeInclusive<T>) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        let dx = self.derivative()?;
        let roots = self.real_roots(x_range.clone(), None)?;

        if roots.is_empty() {
            // No critical points -> derivative does not change sign
            return Ok(vec![]);
        }

        let mut violated_at = vec![];

        let mut prev_sign = dx.y(*x_range.start()).f_signum();
        for x in roots {
            let y = dx.y(x);
            if Value::abs(y) > T::epsilon() {
                let sign = y.f_signum();
                if sign != prev_sign {
                    violated_at.push(x);
                }
                prev_sign = sign;
            }
        }

        let sign = dx.y(*x_range.end()).f_signum();
        if sign != prev_sign {
            violated_at.push(*x_range.end());
        }

        Ok(violated_at)
    }

    /// Converts the polynomial into a monomial polynomial.
    ///
    /// This produces a [`MonomialPolynomial`] representation of the curve,
    /// which uses the standard monomial basis `1, x, x^2, …`.
    ///
    /// # Returns
    /// A monomial polynomial with owned coefficients.
    ///
    /// # Errors
    /// Returns an error if the current basis cannot be converted to monomial form.
    /// This requires that the basis implements [`IntoMonomialBasis`].
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit, MonomialPolynomial};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let mono_poly = fit.as_polynomial().as_monomial().unwrap();
    /// let y = mono_poly.y(1.5);
    /// ```
    pub fn as_monomial(&self) -> Result<MonomialPolynomial<'static, T>>
    where
        B: IntoMonomialBasis<T>,
    {
        let mut coefficients = self.coefficients().to_vec();
        self.basis().as_monomial(&mut coefficients)?;
        Ok(MonomialPolynomial::owned(coefficients))
    }

    /// Projects this polynomial onto another basis over a specified x-range.
    ///
    /// This is useful for converting between different polynomial representations.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Gets `15 * k` evenly spaced sample points over the specified range, where `k` is the number of coefficients in the current polynomial.
    /// - 15 observations per degree of freedom - [`crate::statistics::DegreeBound::Conservative`]
    ///
    /// Fits a new polynomial in the target basis to these points using least-squares fitting.
    /// </div>
    ///
    /// # Type Parameters
    /// - `B2`: The target basis type to project onto.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which to perform the projection.
    ///
    /// # Returns
    /// - `Ok(Polynomial<'static, B2, T>)`: The projected polynomial in the new basis.
    ///
    /// # Errors
    /// Returns an error if the projection fails, such as if the fitting process encounters issues.
    pub fn project<B2: Basis<T> + PolynomialDisplay<T>>(
        &self,
        x_range: RangeInclusive<T>,
    ) -> Result<Polynomial<'static, B2, T>> {
        let samples = self.coefficients.len() * 15; // 15 observations per degree of freedom - [`crate::statistics::DegreeBound::Conservative`]
        let step = (*x_range.end() - *x_range.start()) / T::try_cast(samples)?;
        let points = self.solve_range(x_range, step);
        let fit = crate::CurveFit::<B2, T>::new(&points, self.degree())?;
        Ok(fit.into_polynomial())
    }

    /// Projects this polynomial onto an orthogonal basis over a specified x-range.
    ///
    /// This is a -very- stable way to convert between polynomial bases, as it uses Gaussian quadrature
    ///
    /// You can use it to leverage the strengths of different bases, for example:
    /// - If you have very very bad data
    /// - Fit a fourier curve of a moderate degree
    ///   - This smooths out noise, but overexplains outliers and can ring at the edges
    /// - Then project that onto a Chebyshev basis of degree 2(degree of fourier - 1)
    ///   - This makes sure you don't create new information in the transfer
    ///   - It drops 2 parameters to account for noise and the fourier ringing
    /// - The Chebyshev basis is well behaved and numerically stable
    /// - This will far outperform a direct fit to the noisy data
    ///
    /// # Type Parameters
    /// - `B2`: The target orthogonal basis type to project onto.
    /// - `T`: The numeric type for the polynomial coefficients and evaluations.
    ///
    /// # Parameters
    /// - `x_range`: The range of x-values over which to perform the projection.
    /// - `target_degree`: The degree of the target polynomial in the new basis.
    ///
    /// # Returns
    /// - `Ok(Polynomial<'static, B2, T>)`: The projected polynomial in the new orthogonal basis.
    /// - `Err`: If the projection fails, such as if the fitting process encounters issues.
    ///
    /// # Errors
    /// Returns an error if the projection fails, such as if the fitting process encounters issues.
    pub fn project_orthogonal<B2>(
        &self,
        x_range: RangeInclusive<T>,
        target_degree: usize,
    ) -> Result<Polynomial<'static, B2, T>>
    where
        B2: Basis<T> + PolynomialDisplay<T> + OrthogonalBasis<T>,
    {
        let b2 = B2::from_range(x_range.clone());
        let k = b2.k(target_degree);
        let mut coeffs = vec![T::zero(); k];

        let nodes = b2.gauss_nodes(k);

        for i in 0..k {
            let mut numerator = T::zero();
            let mut denominator = T::zero();

            for j in 0..k {
                // f(x) is defined in the x_range domain
                // and the basis functions in b2 are defined in the b2 domain
                // so we need to convert between them
                // First set the target node in real space
                let x2 = b2.denormalize_x(nodes[j].0);

                let w_j = nodes[j].1;

                let y_i = self.y(x2); // Solve f(x) in `x_range` domain
                let phi_i = b2.solve_function(i, nodes[j].0); // Solve `B_i(x)` in b2 domain

                numerator += w_j * y_i * phi_i;
                denominator += w_j * phi_i * phi_i;
            }

            coeffs[i] = numerator / denominator;
        }

        let poly = unsafe { Polynomial::from_raw(b2, coeffs.into(), target_degree) };
        Ok(poly)
    }

    /// Checks if the polynomial's basis is orthogonal.
    ///
    /// Can be used to determine if methods that require orthogonality can be applied.
    /// Returns true if the basis is orthogonal, false otherwise, like in the case of integrated Fourier series.
    pub fn is_orthogonal(&self) -> bool
    where
        B: OrthogonalBasis<T>,
    {
        self.basis.is_orthogonal()
    }

    /// Computes the energy contribution of each coefficient in an orthogonal basis.
    ///
    /// This is a measure of how much each basis function contributes to the resulting polynomial.
    ///
    /// It can be useful for understanding the significance of each term
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// For an orthogonal basis, the energy contribution of each coefficient is calculated as:
    /// ```math
    /// E_j = c_j^2 * N_j
    /// ```
    /// where:
    /// - `E_j` is the energy contribution of the jth coefficient.
    /// - `c_j` is the jth coefficient.
    /// - `N_j` is the normalization factor for the jth basis function, provided by the basis.
    ///
    /// </div>
    ///
    /// # Returns
    /// A vector of energy contributions for each coefficient.
    ///
    /// # Errors
    /// Returns an error if the basis is not orthogonal. This can be checked with [`Polynomial::is_orthogonal`].
    /// Can happen for integrated Fourier series
    pub fn coefficient_energies(&self) -> Result<Vec<T>>
    where
        B: OrthogonalBasis<T>,
    {
        if !self.basis.is_orthogonal() {
            return Err(Error::NotOrthogonal);
        }

        Ok(self
            .coefficients()
            .iter()
            .enumerate()
            .map(|(degree, &c)| {
                let norm = self.basis.gauss_normalization(degree);
                c * c * norm
            })
            .collect())
    }

    /// Computes a smoothness metric for the polynomial.
    ///
    /// This metric quantifies how "smooth" the polynomial is, with lower values indicating smoother curves.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// The smoothness is calculated as a weighted average of the coefficient energies, where higher-degree coefficients are penalized more heavily.
    /// The formula used is:
    /// ```math
    /// Smoothness = (Σ (k^2 * E_k)) / (Σ E_k)
    /// ```
    /// where:
    /// - `k` is the degree of the basis function.
    /// - `E_k` is the energy contribution of the k-th coefficient.
    /// </div>
    ///
    /// # Returns
    /// A smoothness value, where lower values indicate a smoother polynomial.
    ///
    /// # Errors
    /// Returns an error if the basis is not orthogonal. This can be checked with [`Polynomial::is_orthogonal`].
    pub fn smoothness(&self) -> Result<T>
    where
        B: OrthogonalBasis<T>,
    {
        let energies = self.coefficient_energies()?;

        let mut smoothness = T::zero();
        let mut total_energy = T::zero();
        for (degree, &energy) in energies.iter().enumerate() {
            let k = T::from_positive_int(degree);
            smoothness += k * k * energy;
            total_energy += energy;
        }

        if total_energy.is_near_zero() {
            return Ok(T::zero());
        }

        Ok(smoothness / total_energy)
    }

    /// Applies a spectral energy filter to the polynomial.
    ///
    /// This uses the properties of a orthogonal basis to de-noise the polynomial by removing higher-degree terms that contribute little to the overall energy.
    /// Terms are split into "signal" and "noise" based on their energy contributions, and the polynomial is truncated to only include the signal components.
    ///
    /// Remaining terms are smoothly attenuated to prevent ringing artifacts from a hard cutoff.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// The energy of each coefficient is calculated using the formula:
    ///
    /// ```math
    /// E_j = c_j^2 * N_j
    /// ```
    /// where:
    /// - `E_j` is the energy contribution of the jth coefficient.
    /// - `c_j` is the jth coefficient.
    /// - `N_j` is the normalization factor for the jth basis function, provided by the basis.
    ///
    /// Generalized Cross-Validation (GCV) is used to determine the optimal cutoff degree `K` that minimizes the prediction error using:
    /// `GCV(K) = (suffix[0] - suffix[K]) / K^2`, where `suffix` is the suffix sum of energies.
    ///
    /// A Lanczos Sigma filter with p=1 is applied to smoothly attenuate coefficients up to the cutoff degree, reducing Gibbs ringing artifacts.
    /// </div>
    ///
    /// # Notes
    /// - This method modifies the polynomial in place.
    ///
    /// # Errors
    /// Returns an error if the basis is not orthogonal. This can be checked with [`Polynomial::is_orthogonal`].
    pub fn spectral_energy_filter(&mut self) -> Result<()>
    where
        B: OrthogonalBasis<T>,
    {
        let n = self.coefficients().len();
        let energies = self.coefficient_energies()?;
        let mut total_energy = T::zero();
        for &e in &energies {
            total_energy += e;
        }

        if total_energy.is_near_zero() || n <= 1 {
            return Ok(()); // Nothing to filter
        }

        // compute suffix-sums for R(K)
        let mut suffix = vec![T::zero(); n + 1];
        for i in (0..n).rev() {
            suffix[i] = suffix[i + 1] + energies[i];
        }

        // GCV(K) = (suffix[0] - suffix[K]) / K^2
        // Chooses the K that minimizes this
        let mut best_score = T::infinity();
        let mut best_k = None;
        for k in 1..n {
            let total = suffix[0] - suffix[k];
            let kt = T::from_positive_int(k);
            let score = total / (kt * kt);
            if score < best_score {
                best_k = Some(k);
                best_score = score;
            }
        }

        // Use Lanczos Sigma to smooth in the cutoff
        let Some(k_keep) = best_k else {
            return Ok(());
        };
        let m = T::from_positive_int(k_keep + 1);
        for k in 1..=k_keep.saturating_sub(1) {
            let x = T::from_positive_int(k) / m;
            let sinc = if x.is_near_zero() {
                T::one()
            } else {
                (T::pi() * x).sin() / (T::pi() * x)
            };

            self.coefficients_mut()[k] = self.coefficients()[k] * sinc;
        }

        for i in (k_keep + 1)..n {
            self.coefficients_mut()[i] = T::zero();
        }

        Ok(())
    }

    /// Returns a human-readable string of the polynomial equation.
    ///
    /// The output shows the polynomial in standard mathematical notation, for example:
    /// ```text
    /// y = 1.0x^3 + 2.0x^2 + 3.0x + 4.0
    /// ```
    ///
    /// # Notes
    /// - Requires the basis to implement [`PolynomialDisplay`] for formatting.
    /// - This operation is infallible and guaranteed to succeed, hence no error return.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// println!("{}", poly.equation());
    /// ```
    #[expect(clippy::missing_panics_doc, reason = "Infallible operation")]
    #[must_use]
    pub fn equation(&self) -> String {
        let mut output = String::new();
        self.basis
            .format_polynomial(&mut output, self.coefficients())
            .expect("String should be infallible");
        output
    }
}

impl<'a, B, T: Value> AsRef<Polynomial<'a, B, T>> for Polynomial<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn as_ref(&self) -> &Polynomial<'a, B, T> {
        self
    }
}

impl<B, T: Value> std::fmt::Display for Polynomial<'_, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.equation())
    }
}

impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> std::ops::Mul<T> for Polynomial<'_, B, T> {
    type Output = Polynomial<'static, B, T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut result = self.clone().into_owned();
        result.scale(rhs);
        result
    }
}
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> std::ops::MulAssign<T> for Polynomial<'_, B, T> {
    fn mul_assign(&mut self, rhs: T) {
        self.scale(rhs);
    }
}

impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> std::ops::Div<T> for Polynomial<'_, B, T> {
    type Output = Polynomial<'static, B, T>;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self.clone().into_owned();
        result.scale(T::one() / rhs);
        result
    }
}
impl<B: Basis<T> + PolynomialDisplay<T>, T: Value> std::ops::DivAssign<T> for Polynomial<'_, B, T> {
    fn div_assign(&mut self, rhs: T) {
        self.scale(T::one() / rhs);
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_all_close, assert_close, assert_y};

    use super::*;

    #[test]
    fn test_y() {
        function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2);
        assert_y!(&test, 0.0, 8.0);
        assert_y!(&test, 1.0, 21.0);
        assert_y!(&test, 2.0, 46.0);
    }

    #[test]
    fn test_solve() {
        function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2);
        let points: Vec<_> = test.solve(vec![0.0, 1.0, 2.0]).y();
        assert_all_close!(points, &[8.0, 21.0, 46.0]);
    }

    #[test]
    fn test_solve_range() {
        function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2);
        let points = test.solve_range(0.0..=3.0, 1.0).y();
        assert_all_close!(points, &[8.0, 21.0, 46.0, 83.0]);
    }

    #[test]
    fn test_area_under_curve() {
        function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2);
        let area = test
            .area_under_curve(0.0, 3.0, None)
            .expect("Failed to compute area");
        assert_close!(area, 109.5);
    }
}
