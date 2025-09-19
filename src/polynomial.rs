use std::{
    borrow::Cow,
    ops::{Range, RangeInclusive},
};

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, IntoMonomialBasis, MonomialBasis},
    display::PolynomialDisplay,
    error::Result,
    statistics,
    value::{CoordExt, SteppedValues, Value},
};

/// A monomial polynomial of the form `y = a_n * x^n + ... + a_1 * x + a_0`.
///
/// This is the what most people imagine when they hear "polynomial".
///
/// # Type Parameters
/// - `'a`: Lifetime of borrowed coefficients (if used).
/// - `T`: Numeric type (default `f64`).
pub type MonomialPolynomial<'a, T = f64> = Polynomial<'a, MonomialBasis<T>, T>;

impl<'a, T: Value> MonomialPolynomial<'a, T> {
    /// Creates a new borrowed monomial polynomial from a slice of coefficients.
    ///
    /// # Parameters
    /// - `coefficients`: Slice of coefficients, starting from the constant term.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::borrowed(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// ```
    pub const fn borrowed(coefficients: &'a [T]) -> Self {
        let degree = coefficients.len() - 1;
        unsafe {
            Self::from_raw(
                MonomialBasis::default(),
                Cow::Borrowed(coefficients),
                degree,
            )
        } // Safety: Monomials expect k+1 coefficients
    }

    /// Creates a new owned monomial polynomial from a vector of coefficients.
    ///
    /// # Parameters
    /// - `coefficients`: Vec of coefficients, starting from the constant term.
    ///
    /// # Example
    /// ```
    /// # use polyfit::MonomialPolynomial;
    /// let poly = MonomialPolynomial::owned(vec![1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
    /// ```
    #[must_use]
    pub const fn owned(coefficients: Vec<T>) -> Self {
        let degree = coefficients.len() - 1;
        unsafe { Self::from_raw(MonomialBasis::default(), Cow::Owned(coefficients), degree) }
        // Safety: Monomials expect k+1 coefficients
    }
}

/// Represents a polynomial function in a given basis.
///
/// Unlike [`crate::CurveFit`], this struct is **not tied to any dataset or matrix**, making it a canonical function that
/// can be evaluated for **any x-value** without range restrictions.
///
/// # Type Parameters
/// - `'a`: Lifetime for borrowed basis or coefficients, if used.
/// - `B`: The polynomial basis (e.g., [`MonomialBasis`], [`crate::basis::ChebyshevBasis`]).
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

    /// Returns a reference to the polynomial's basis.
    pub(crate) fn basis(&self) -> &B {
        &self.basis
    }

    /// Converts the polynomial into its basis and coefficients.
    pub(crate) fn into_inner(self) -> (B, Cow<'a, [T]>) {
        (self.basis, self.coefficients)
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
    /// let points = poly.solve_range(0.0..2.0, 1.0);
    /// // points = [(0.0, 1.0), (1.0, 6.0), (2.0, 17.0)]
    /// ```
    pub fn solve_range(&self, range: Range<T>, step: T) -> Vec<(T, T)> {
        self.solve(SteppedValues::new(range.start..=range.end, step))
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
    pub fn derivative(&self) -> Result<Polynomial<'static, B, T>>
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
    /// let critical_points = poly.critical_points().unwrap();
    /// ```
    pub fn critical_points(&self) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        let dx = self.derivative()?;
        self.basis.critical_points(dx.coefficients())
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
    pub fn integral(&self, constant: Option<T>) -> Result<Polynomial<'static, B, T>>
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
        let critical_points = self.basis.critical_points(dx.coefficients())?;

        if critical_points.is_empty() {
            // No critical points -> derivative does not change sign
            return Ok(vec![]);
        }

        let mut violated_at = vec![];

        let mut prev_sign = dx.y(*x_range.start()).f_signum();
        for &x in &critical_points {
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

#[cfg(test)]
mod tests {
    use crate::{assert_all_close, assert_close, assert_y, function};

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
        let points = test.solve_range(0.0..3.0, 1.0).y();
        assert_all_close!(points, &[8.0, 21.0, 46.0]);
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
