use std::ops::{Range, RangeInclusive};

use nalgebra::{DMatrix, DVector, SVD};

use crate::{
    basis::{Basis, DifferentialBasis, IntoMonomialBasis},
    display::PolynomialDisplay,
    error::{Error, Result},
    statistics::{self, Confidence, ConfidenceBand, DegreeBound, ScoringMethod},
    value::{CoordExt, Value, ValueRange},
    MonomialPolynomial, Polynomial,
};

/// Fourier series curve
///
/// Uses a Fourier series basis, which is particularly well-suited for modeling periodic functions.
/// The basis functions include sine and cosine terms, allowing for effective representation of oscillatory behavior.
pub type FourierFit<T = f64> = CurveFit<crate::basis::FourierBasis<T>, T>;

/// Normalized Chebyshev polynomial curve
///
/// Uses the Chebyshev polynomials, which are orthogonal polynomials defined on the interval \[-1, 1\].
/// These polynomials are particularly useful for minimizing Runge's phenomenon in polynomial interpolation.
pub type ChebyshevFit<T = f64> = CurveFit<crate::basis::ChebyshevBasis<T>, T>;

/// Non-normalized monomial polynomial curve
///
/// Uses the standard monomial functions: 1, x, x^2, ..., x^n
///
/// It is the most basic form of polynomial basis and is not normalized.
/// It can lead to numerical instability for high-degree polynomials.
pub type MonomialFit<T = f64> = CurveFit<crate::basis::MonomialBasis<T>, T>;

/// Represents the covariance matrix and derived statistics for a curve fit.
///
/// Provides tools to evaluate the uncertainty of coefficients and predictions
/// of a fitted polynomial or other basis function model.
///
/// # Type Parameters
/// - `'a`: Lifetime of the reference to the original curve fit.
/// - `B`: Basis type used by the curve fit (implements `Basis<T>`).
/// - `T`: Numeric type (defaults to `f64`) implementing `Value`.
pub struct CurveFitCovariance<'a, B, T: Value = f64>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fit: &'a CurveFit<B, T>,
    covariance: DMatrix<T>,
}
impl<'a, B, T: Value> CurveFitCovariance<'a, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    /// Creates a new covariance matrix for a curve fit.
    ///
    /// See [`CurveFit::covariance`]
    ///
    /// # Errors
    /// Returns an error if the covariance matrix cannot be computed.
    pub fn new(fit: &'a CurveFit<B, T>) -> Result<Self> {
        let covariance = Self::fill_matrix(fit)?;
        Ok(Self { fit, covariance })
    }

    /// Computes the covariance matrix for the curve fit.
    fn fill_matrix(fit: &CurveFit<B, T>) -> Result<DMatrix<T>> {
        let n = fit.data.len();
        let k = fit.coefficients().len();

        let mut x_matrix = DMatrix::zeros(n, k);
        for (i, (x, _)) in fit.data.iter().enumerate() {
            let x = fit.basis().normalize_x(*x);
            for j in 0..k {
                x_matrix[(i, j)] = fit.basis().solve_function(j, x);
            }
        }

        // Compute (X^T X)^-1
        let xtx = &x_matrix.transpose() * &x_matrix;
        let xtx_reg = &xtx + DMatrix::<T>::identity(k, k) * T::epsilon();
        let svd = xtx_reg.svd(true, true);
        let xtx_inv = svd.pseudo_inverse(T::epsilon()).map_err(Error::Algebra)?;

        let res_var = fit.residual_variance();
        Ok(xtx_inv * res_var)
    }

    /// Computes the standard error of the coefficient at j.
    ///
    /// Returns None if the coefficient is not valid.
    ///
    /// This is the estimated standard deviation of the coefficient, providing
    /// a measure of its uncertainty.
    #[must_use]
    pub fn coefficient_standard_error(&self, j: usize) -> Option<T> {
        let cell = self.covariance.get((j, j))?;
        Some(cell.sqrt())
    }

    /// Computes the standard error of the coefficients.
    ///
    /// This is the estimated standard deviation of the coefficients, providing
    /// a measure of their uncertainty.
    #[must_use]
    pub fn coefficient_standard_errors(&self) -> Vec<T> {
        let cov = &self.covariance;
        (0..cov.ncols())
            .filter_map(|i| self.coefficient_standard_error(i))
            .collect()
    }

    /// Computes the variance of the predicted y value at `x`.
    ///
    /// This quantifies the uncertainty in the prediction at a specific point.
    ///
    /// The square root of the variance gives the standard deviation.
    pub fn prediction_variance(&self, x: T) -> T {
        let k = self.fit.basis().k(self.fit.degree());
        let x = self.fit.basis().normalize_x(x);
        let phi_x =
            DVector::from_iterator(k, (0..k).map(|j| self.fit.basis().solve_function(j, x)));
        (phi_x.transpose() * &self.covariance * phi_x)[(0, 0)]
    }

    /// Computes the confidence band for an x value.
    ///
    /// Returns (lower-bound, upper-bound)
    ///
    /// This estimates the uncertainty in the predicted y value at a specific x
    /// location, providing a range within which the true value is likely to fall.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn confidence_band(&self, x: T, confidence_level: Confidence) -> Result<ConfidenceBand<T>> {
        let y_var = self.prediction_variance(x);
        let y_se = y_var.sqrt();
        let value = self.fit.y(x)?;

        let z = confidence_level.try_cast::<T>()?;
        let lower = value - z * y_se;
        let upper = value + z * y_se;
        Ok(ConfidenceBand {
            value,
            lower,
            upper,
            level: confidence_level,
        })
    }

    /// Computes the confidence intervals for all data points in the original dataset.
    ///
    /// This evaluates the fitted model at each `x` from the original data and returns
    /// a `ConfidenceBand` for each point, quantifying the uncertainty of predictions.
    ///
    /// # Parameters
    /// - `confidence_level`: Desired confidence level (e.g., P95).
    ///
    /// # Returns
    /// - `Ok(Vec<ConfidenceBand<T>>)` containing one confidence band per data point.
    /// - `Err` if any prediction or type conversion fails.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn solution_confidence(
        &self,
        confidence_level: Confidence,
    ) -> Result<Vec<ConfidenceBand<T>>> {
        let x = self.fit.data().iter().map(|(x, _)| *x);
        x.map(|x| self.confidence_band(x, confidence_level))
            .collect()
    }
}

/// Represents a polynomial curve fit for a set of data points.
///
/// `CurveFit` computes a polynomial that best fits a given dataset using a
/// specified polynomial basis (e.g., monomial, Chebyshev). It stores both the
/// original data and the resulting coefficients.
///
/// # For beginners
/// Most users do **not** need to construct this directly. Use one of the
/// specialized type aliases for common bases:
/// - [`crate::MonomialFit`] — uses the standard monomial basis (`1, x, x², …`).
/// - [`crate::ChebyshevFit`] — uses Chebyshev polynomials to reduce oscillation.
///
/// # How it works
/// - Builds a **basis matrix** with shape `[rows, k]` where `rows`
///   is the number of data points and `k` is the number of basis functions.
/// - Forms a **column vector** `b` from the `y` values of the dataset.
/// - Solves the linear system `A * x = b` using the **SVD** of the basis matrix.
///   The solution `x` is the vector of polynomial coefficients.
///
/// # Type parameters
/// - `B`: The basis type, implementing [`Basis<T>`].
/// - `T`: Numeric type (default `f64`) implementing [`Value`].
///
/// # Example
/// ```
/// # use polyfit::MonomialFit;
/// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
/// let fit = MonomialFit::new(data, 2).unwrap();
/// println!("Coefficients: {:?}", fit.coefficients());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CurveFit<B, T: Value = f64>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    data: Vec<(T, T)>,
    x_range: RangeInclusive<T>,
    function: Polynomial<'static, B, T>,

    matrix: DMatrix<T>,
    k: T,
}
impl<T: Value, B> CurveFit<B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    /// Creates a new polynomial curve fit for the given data and degree.
    ///
    /// You can also use [`CurveFit::new_auto`] to automatically select the best degree.
    ///
    /// This constructor fits a polynomial to the provided `(x, y)` points using
    /// the chosen basis type `B`. For most users, `B` will be a Chebyshev or
    /// Monomial basis.
    ///
    /// # Parameters
    /// - `data`: Slice of `(x, y)` points to fit.
    /// - `degree`: Desired polynomial degree.
    ///
    /// # Returns
    /// Returns `Ok(Self)` if the fit succeeds.
    ///
    /// # Errors
    /// Returns an [`Error`] in the following cases:
    /// - `Error::NoData`: `data` is empty.
    /// - `Error::DegreeTooHigh`: `degree >= data.len()`.
    /// - `Error::Algebra`: the linear system could not be solved.
    /// - `Error::CastFailed`: a numeric value could not be cast to the target type.
    ///
    /// # Behavior
    /// - Builds the basis matrix internally and fills each row using
    ///   [`Basis::fill_matrix_row`].
    /// - Computes `x_range` as the inclusive range of `x` values in the data.
    /// - Solves the linear system `A * x = b` to determine polynomial coefficients.
    ///
    /// # Example
    /// ```
    /// # use polyfit::ChebyshevFit;
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// println!("Coefficients: {:?}", fit.coefficients());
    /// ```
    pub fn new(data: &[(T, T)], degree: usize) -> Result<Self> {
        // Cannot fit a polynomial of degree 0 or if there is no data.
        if data.is_empty() {
            return Err(Error::NoData);
        } else if degree >= data.len() {
            return Err(Error::DegreeTooHigh(degree));
        }

        let basis = B::new(data);
        let data = data.to_vec();
        let k = basis.k(degree);

        let min_x = data
            .iter()
            .map(|&(x, _)| x)
            .fold(T::infinity(), <T as nalgebra::RealField>::min);
        let max_x = data
            .iter()
            .map(|&(x, _)| x)
            .fold(T::neg_infinity(), <T as nalgebra::RealField>::max);
        let x_range = min_x..=max_x;

        let mut matrix = DMatrix::zeros(data.len(), k);
        let coefs = Self::fill_matrix(&basis, &mut matrix, &data, 0)?;
        let function = unsafe { Polynomial::from_raw(basis, coefs.into(), degree) }; // Safety: The coefs were generated by the basis

        Ok(Self {
            data,
            x_range,
            function,

            matrix,
            k: T::try_cast(k)?,
        })
    }

    /// Automatically selects the best polynomial degree and creates a curve fit.
    ///
    /// This function fits polynomials of increasing degree to the provided dataset
    /// and selects the “best” degree according to the specified scoring method.
    ///
    /// # Parameters
    /// - `data`: Slice of `(x, y)` points to fit.
    /// - `method`: [`ScoringMethod`] to evaluate model quality.  
    ///   - `AIC`: Akaike Information Criterion (uses `AICc` if `n/k < 4`)  
    ///   - `BIC`: Bayesian Information Criterion
    ///
    /// # Choosing a scoring method
    /// - Consider the size of your dataset: If you have a small dataset, prefer `AIC` as it penalizes complexity more gently.
    /// - If your dataset is large, `BIC` may be more appropriate as it imposes a harsher penalty on complexity.
    ///
    /// # Returns
    /// Returns `Ok(Self)` with the fit at the optimal degree.
    ///
    /// # Errors
    /// Returns [`Error`] if:
    /// - `data` is empty (`Error::NoData`)  
    /// - A numeric value cannot be represented in the target type (`Error::CastFailed`)
    ///
    /// # Behavior
    /// - Starts with degree 0 and iteratively fits higher degrees up to `data.len() - 1`.
    /// - Evaluates each fit using `model_score(method)`.
    /// - Stops when the score no longer improves.
    /// - Returns the model with the best score.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, statistics::{DegreeBound, ScoringMethod}};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
    /// println!("Selected degree: {}", fit.degree());
    /// ```
    pub fn new_auto(
        data: &[(T, T)],
        max_degree: impl Into<DegreeBound>,
        method: ScoringMethod,
    ) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::NoData);
        }

        let max_degree = max_degree.into().max_degree(data.len());
        let mut min_score = T::infinity();
        let mut models = Vec::with_capacity(max_degree + 1);

        // Pass 1 - generate models, get score_min
        for degree in 0..=max_degree {
            let model = Self::new(data, degree)?;
            let score = model.model_score(method);

            models.push((model, score));
            if score < min_score {
                min_score = score;
            }
        }

        // Pass 2 - get delta_score
        // Re: Burnham and Anderson, use the first delta <=2 (P = 0.37)
        // Statistically indistinguishable from the top model
        for (model, score) in models {
            let delta = score - min_score;
            if delta <= T::two() {
                return Ok(model);
            }
        }

        Err(Error::NoModel)
    }

    /// Prunes coefficients that are statistically insignificant based on a t-test.
    ///
    /// # Parameters
    /// - `confidence`: Confidence level for determining significance (e.g., P95, P99)
    ///
    /// # Errors
    /// Returns an error if the covariance matrix cannot be computed.
    ///
    /// # Returns
    /// A vector of `(index, coefficient)` for all pruned coefficients.
    ///
    /// # Notes
    /// - Modifies `self` in-place, zeroing out insignificant coefficients.
    /// - Uses the standard errors derived from the covariance matrix.
    /// - Ignores coefficients whose absolute value is smaller than `T::epsilon()`.
    pub fn prune_insignificant(&mut self, confidence: Confidence) -> Result<Vec<(usize, T)>> {
        let covariance = self.covariance()?;
        let se = covariance.coefficient_standard_errors();
        let coeffs = self.coefficients();

        let df = self.data().len().saturating_sub(coeffs.len());
        let t_crit = confidence.t_score(df);
        let t_crit = T::try_cast(t_crit)?;

        let mut insignificant = Vec::new();
        for (i, (&c, s)) in coeffs.iter().zip(se).enumerate() {
            let t = Value::abs(c) / s;
            if t < t_crit && c > T::epsilon() {
                insignificant.push((i, c));
            }
        }

        let coefs_mut = self.function.coefficients_mut();
        for (i, _) in &insignificant {
            coefs_mut[*i] = T::zero();
        }

        Ok(insignificant)
    }

    /// Fills the given Vandermonde matrix for the specified data points.
    ///
    /// Returns the coefficients for the polynomial fit.
    fn fill_matrix(
        basis: &B,
        matrix: &mut DMatrix<T>,
        data: &[(T, T)],
        from_degree: usize,
    ) -> Result<Vec<T>> {
        for (i, (x, _)) in data.iter().enumerate() {
            let row = matrix.row_mut(i);
            let x = basis.normalize_x(*x);
            basis.fill_matrix_row(from_degree, x, row);
        }

        Self::compute_coefficients(matrix, data)
    }

    /// Recomputes the polynomial coefficients.
    fn compute_coefficients(matrix: &DMatrix<T>, data: &[(T, T)]) -> Result<Vec<T>> {
        let size = matrix.shape();

        // Create a column vector of the y values
        let y_values = data.iter().map(|(_, y)| *y);
        let b = DVector::from_iterator(size.0, y_values);

        // Calculate the singular value decomposition of the matrix
        let decomp = SVD::new(matrix.clone(), true, true);

        // Calculate epsilon value
        // ~= machine_epsilon * max(size) * max_singular
        let machine_epsilon = T::epsilon();
        let max_size = size.0.max(size.1);
        let sigma_max = decomp.singular_values.max();
        let epsilon = machine_epsilon * T::try_cast(max_size)? * sigma_max;

        // Solve for X in `SVD * X = b`
        let big_x = decomp.solve(&b, epsilon).map_err(Error::Algebra)?;
        let coefficients: Vec<_> = big_x.data.into();

        // Make sure the coefficients are valid
        if coefficients.iter().any(|c| c.is_nan()) {
            return Err(Error::Algebra("NaN in coefficients"));
        }

        Ok(coefficients)
    }

    /// Returns a reference to the basis function.
    pub(crate) fn basis(&self) -> &B {
        self.function.basis()
    }

    /// Changes the polynomial degree of an existing curve fit.
    ///
    /// This allows you to **increase or decrease the degree** of a polynomial fit
    /// after it has been created. The function tries to reuse the existing basis
    /// matrix where possible to improve performance.
    ///
    /// # Behavior
    /// - **Upsizing (`new_degree` > `old_degree`):** recomputes the matrix rows for the
    ///   additional columns and solves the new linear system from scratch.
    /// - **Downsizing (`new_degree` < `old_degree`):** reuses the existing matrix and
    ///   discards higher-degree columns, then recomputes the coefficients.
    /// - If `new_degree == old_degree`, returns the original instance unchanged.
    ///
    /// # Errors
    /// Returns [`Error`] if:
    /// - A numeric value cannot be represented in the target type (`CastFailed`).
    /// - The final linear solve fails (`Algebra`).
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let lower_fit = fit.resize(1).unwrap(); // decrease degree to 1
    /// ```
    pub fn resize(mut self, new_degree: usize) -> Result<Self> {
        let old_degree = self.degree();
        if new_degree == old_degree {
            return Ok(self);
        }

        let k = self.basis().k(new_degree);
        self.k = T::try_cast(k)?;

        self.matrix = self.matrix.resize_horizontally(k, T::zero());
        let (basis, _) = self.function.into_inner();
        let coefs = Self::fill_matrix(&basis, &mut self.matrix, &self.data, new_degree)?;
        let function = unsafe { Polynomial::from_raw(basis, coefs.into(), new_degree) }; // Safety: The coefs were generated by the basis

        self.function = function;
        Ok(self)
    }

    /// Computes the covariance matrix and related statistics for this curve fit.
    ///
    /// The returned [`CurveFitCovariance`] provides:
    /// - Covariance matrix of the fitted coefficients.
    /// - Standard errors of the coefficients.
    /// - Prediction variance at a specific `x`.
    /// - Confidence intervals for predictions.
    ///
    /// # Returns
    /// - `Ok(CurveFitCovariance<'_, B, T>)` on success.
    ///
    /// # Errors
    /// Returns `Err(Error::Algebra)` if `(Xᵀ X)` is singular or nearly singular,
    /// i.e., the pseudo-inverse cannot be computed. Causes include too few data points
    /// relative to parameters or collinear/linearly dependent basis functions.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::statistics::Confidence;
    /// # use polyfit::MonomialFit;
    /// # let model = MonomialFit::new(&[(0.0, 0.0), (1.0, 1.0)], 1).unwrap();
    /// let cov = model.covariance().unwrap();
    /// let se = cov.coefficient_standard_errors();
    /// let band = cov.confidence_band(1.0, Confidence::P95).unwrap();
    /// println!("Predicted CI at x=1: {} - {}", band.lower, band.upper);
    /// ```
    pub fn covariance(&self) -> Result<CurveFitCovariance<'_, B, T>> {
        CurveFitCovariance::new(self)
    }

    /// Computes the critical points of the fitted polynomial.
    ///
    /// This returns the x-values where the polynomial has local minima or maxima.
    ///
    /// # Errors
    /// Returns an error if the critical points cannot be computed.
    pub fn critical_points(&self) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        self.function.critical_points()
    }

    /// Returns the X-values where the function is not monotone (i.e., where the derivative changes sign).
    ///
    /// # Errors
    /// Returns an error if the derivative cannot be computed.
    ///
    /// # Example
    /// ```ignore
    /// let fit = Fit::new(...);
    /// let violations = fit.monotonicity_violations().unwrap();
    /// ```
    pub fn monotonicity_violations(&self) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        let dx = self.function.derivative()?;
        let critical_points = dx.basis().critical_points(dx.coefficients())?;

        if critical_points.is_empty() {
            // No critical points -> derivative does not change sign
            return Ok(vec![]);
        }

        let mut violated_at = vec![];

        let x_range = self.x_range();
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

    /// Computes the quality score of the polynomial fit using the specified method.
    ///
    /// This evaluates how well the fitted polynomial represents the data, taking
    /// into account both the fit error and model complexity.
    ///
    /// # Parameters
    /// - `method`: [`ScoringMethod`] to use for scoring.  
    ///   - `AIC`: Akaike Information Criterion (uses `AICc` if `n/k < 4`)  
    ///   - `BIC`: Bayesian Information Criterion
    ///
    /// # Returns
    /// The score as a numeric value (`T`). Lower scores indicate better models.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, statistics::ScoringMethod};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let score = fit.model_score(ScoringMethod::AIC);
    /// println!("Model score: {}", score);
    /// ```
    pub fn model_score(&self, method: ScoringMethod) -> T {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        method.calculate(y, y_fit, self.k)
    }

    /// Computes the residuals of the fit.
    pub fn residuals(&self) -> Vec<T> {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        y.zip(y_fit).map(|(y, y_fit)| y - y_fit).collect()
    }

    /// Computes the residual variance of the model's predictions.
    ///
    /// Residual variance is the unbiased estimate of the variance of the
    /// errors (σ²) after fitting a model. It's used for confidence intervals
    /// and covariance estimates of the fitted parameters.
    pub fn residual_variance(&self) -> T {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        statistics::residual_variance(y, y_fit, self.k)
    }

    /// Computes the mean squared error (MSE) of this fit against its source data.
    ///
    /// See [`statistics::mean_squared_error`].
    ///
    /// MSE measures the average squared difference between the observed and predicted values.
    /// Lower values indicate a better fit.
    pub fn mean_squared_error(&self) -> T {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        statistics::mean_squared_error(y, y_fit)
    }

    /// Computes the root mean squared error (RMSE) of this fit against its source data.
    ///
    /// See [`statistics::root_mean_squared_error`].
    ///
    /// RMSE is the square root of the MSE, giving error in the same units as the original data.
    /// Lower values indicate a closer fit.
    pub fn root_mean_squared_error(&self) -> T {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        statistics::root_mean_squared_error(y, y_fit)
    }

    /// Computes the mean absolute error (MAE) of this fit against its source data.
    ///
    /// See [`statistics::mean_absolute_error`].
    ///
    /// MAE measures the average absolute difference between observed and predicted values.
    /// Lower values indicate a better fit.
    pub fn mean_absolute_error(&self) -> T {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        statistics::mean_absolute_error(y, y_fit)
    }

    /// Computes the coefficient of determination (R²) for the polynomial fit.
    ///
    /// The R² value measures how well the model explains the variance in the data.
    /// - `R² = 1.0` indicates a perfect fit.
    /// - `R² = 0.0` indicates that the model does no better than the mean of the data.
    ///
    /// # Parameters
    /// - `data`: Slice of `(x, y)` points to compare against the model.
    ///
    /// # Returns
    /// R² as a numeric value of type `T`.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let r2 = fit.r_squared(data);
    /// println!("R² = {}", r2);
    /// ```
    pub fn r_squared(&self, data: &[(T, T)]) -> T {
        let y = data.iter().map(|&(_, y)| y);
        let y_fit = self.solution().into_iter().map(|(_, y)| y);

        statistics::r_squared(y, y_fit)
    }

    /// Computes how well this curve fit matches a target polynomial.
    ///
    /// This calculates R² by evaluating the target polynomial at the same `x`
    /// values as the model’s data and comparing the predicted `y` values.
    ///
    /// # Parameters
    /// - `function`: The target [`Polynomial`] to compare against.
    ///
    /// # Returns
    /// R² as a numeric value of type `T`.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, MonomialPolynomial};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let target = MonomialPolynomial::borrowed(&[1.0, 2.0, 1.0]);
    /// let r2 = fit.r_squared_against(&target);
    /// println!("R² vs target polynomial = {}", r2);
    /// ```
    pub fn r_squared_against<C>(&self, function: &Polynomial<C, T>) -> T
    where
        C: Basis<T>,
        C: PolynomialDisplay<T>,
    {
        let data: Vec<_> = self
            .data()
            .iter()
            .map(|&(x, _)| (x, function.y(x)))
            .collect();
        self.r_squared(&data)
    }

    /// Returns the degree of the polynomial
    pub fn degree(&self) -> usize {
        self.function.degree()
    }

    /// Returns a reference to the coefficients of the polynomial.
    ///
    /// The coefficient at index `i` corresponds to `x^i` in the polynomial.
    pub fn coefficients(&self) -> &[T] {
        self.function.coefficients()
    }

    /// Returns a reference to the data points used for fitting.
    ///
    /// Each element is a `(x, y)` tuple representing a measured data point.
    pub fn data(&self) -> &[(T, T)] {
        &self.data
    }

    /// Returns the inclusive range of x-values in the dataset.
    pub fn x_range(&self) -> RangeInclusive<T> {
        self.x_range.clone()
    }

    /// Returns the inclusive range of y-values in the dataset.
    ///
    /// This is computed dynamically from the stored data points.
    pub fn y_range(&self) -> RangeInclusive<T> {
        let min_y = self
            .data
            .iter()
            .map(|&(_, y)| y)
            .fold(T::infinity(), <T as nalgebra::RealField>::min);
        let max_y = self
            .data
            .iter()
            .map(|&(_, y)| y)
            .fold(T::neg_infinity(), <T as nalgebra::RealField>::max);
        min_y..=max_y
    }

    /// Evaluates the polynomial curve at a given x-value.
    ///
    /// # Parameters
    /// - `x`: The x-coordinate to evaluate.
    ///
    /// # Returns
    /// The corresponding y-value as `T` if `x` is within the valid range.
    ///
    /// # Errors
    /// Returns [`Error::DataRange`] if `x` is outside the original data bounds.
    ///
    /// # Notes
    /// - Polynomial fits are generally only stable within the x-range used for fitting.
    /// - To evaluate outside the original bounds, use [`CurveFit::as_polynomial`] to get
    ///   a pure polynomial function that ignores the original x-range.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let y = fit.y(1.0).unwrap();
    /// println!("y(1.0) = {}", y);
    /// ```
    pub fn y(&self, x: T) -> Result<T> {
        if !self.x_range.contains(&x) {
            return Err(Error::DataRange(
                format!("{}", self.x_range.start()),
                format!("{}", self.x_range.end()),
            ));
        }

        Ok(self.function.y(x))
    }

    /// Returns the fitted y-values corresponding to the original x-values.
    ///
    /// This produces a vector of `(x, y)` pairs for the same x-values used in
    /// the source data. It is guaranteed to succeed because all x-values are
    /// within the curve's valid range.
    ///
    /// # Notes
    /// - Useful for quickly plotting or analyzing the fitted curve against the
    ///   original data points.
    /// - The method internally calls [`CurveFit::y`] but is infallible because
    ///   it only evaluates x-values within the valid range.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let points = fit.solution();
    /// for (x, y) in points {
    ///     println!("x = {}, y = {}", x, y);
    /// }
    /// ```
    #[expect(clippy::missing_panics_doc, reason = "Infallible operation")]
    pub fn solution(&self) -> Vec<(T, T)> {
        self.data()
            .iter()
            .map(|&(x, _)| (x, self.y(x).expect("data range check")))
            .collect()
    }

    /// Evaluates the curve at multiple x-values.
    ///
    /// # Parameters
    /// - `x`: An iterator of x-values to evaluate.
    ///
    /// # Returns
    /// A vector of `(x, y)` pairs corresponding to each input x-value.
    ///
    /// # Errors
    /// Returns [`Error::DataRange`] if any x-value is outside the original data range.
    ///
    /// # Notes
    /// - Curve fits are generally only stable within the x-range used for fitting.
    /// - To evaluate outside the original bounds, use [`CurveFit::as_polynomial`]
    ///   to get a pure polynomial function that ignores the original x-range.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let points = fit.solve([0.0, 0.5, 1.0]).unwrap();
    /// for (x, y) in points {
    ///     println!("x = {}, y = {}", x, y);
    /// }
    /// ```
    pub fn solve(&self, x: impl IntoIterator<Item = T>) -> Result<Vec<(T, T)>> {
        x.into_iter().map(|x| Ok((x, self.y(x)?))).collect()
    }

    /// Evaluates the curve at evenly spaced points over a range.
    ///
    /// # Parameters
    /// - `range`: The start and end x-values to evaluate.
    /// - `step`: The increment between points.
    ///
    /// # Returns
    /// A vector of `(x, y)` pairs corresponding to each x-value in the range.
    ///
    /// # Errors
    /// Returns [`Error::DataRange`] if any x-value is outside the original data range.
    ///
    /// # Notes
    /// - Curve fits are only stable within the x-range used for fitting.
    /// - To evaluate outside the original bounds, use [`CurveFit::as_polynomial`]
    ///   to get a pure polynomial function.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let points = fit.solve_range(0.0..2.0, 0.5).unwrap();
    /// for (x, y) in points {
    ///     println!("x = {}, y = {}", x, y);
    /// }
    /// ```
    pub fn solve_range(&self, range: Range<T>, step: T) -> Result<Vec<(T, T)>> {
        self.solve(ValueRange::new(range.start, range.end, step))
    }

    /// Returns a pure polynomial representation of the curve fit.
    ///
    /// This allows evaluation of the polynomial at **any x-value**, without
    /// restriction to the original data range. Unlike [`CurveFit::y`] or
    /// [`CurveFit::solve`], this does not perform range checks, so use with
    /// caution outside the fit’s stable region.
    ///
    /// The [`Polynomial`] form is considered a canonical function, not a fit estimate.
    ///
    /// # Returns
    /// A [`Polynomial`] that borrows the basis and coefficients from this fit.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let poly = fit.as_polynomial();
    /// let y = poly.y(10.0); // can evaluate outside original x-range
    /// ```
    pub fn as_polynomial(&self) -> &Polynomial<'_, B, T> {
        &self.function
    }

    /// Returns a pure polynomial representation of the curve fit.
    ///
    /// This is primarily a memory optimization in practice; You become responsible for
    /// maintaining the stability around the x-bounds, but the copy of the original data,
    /// and the Vandermonde matrix are dropped.
    ///
    /// This allows evaluation of the polynomial at **any x-value**, without
    /// restriction to the original data range. Unlike [`CurveFit::y`] or
    /// [`CurveFit::solve`], this does not perform range checks, so use with
    /// caution outside the fit’s stable region.
    ///
    /// The [`Polynomial`] form is considered a canonical function, not a fit estimate.
    ///
    /// # Returns
    /// A [`Polynomial`] that borrows the basis and coefficients from this fit.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let poly = fit.as_polynomial();
    /// let y = poly.y(10.0); // can evaluate outside original x-range
    /// ```
    pub fn into_polynomial(self) -> Polynomial<'static, B, T> {
        self.function
    }

    /// Converts the curve fit into a monomial polynomial.
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
    /// let mono_poly = fit.as_monomial().unwrap();
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
    /// # use polyfit::{ChebyshevFit, CurveFit};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// println!("{}", fit.equation());
    /// ```
    #[expect(clippy::missing_panics_doc, reason = "Infallible operation")]
    pub fn equation(&self) -> String {
        let mut output = String::new();
        self.basis()
            .format_polynomial(&mut output, self.coefficients())
            .expect("String should be infallible");
        output
    }

    /// Returns the properties of the curve fit.
    ///
    /// This is a comprehensive summary of the fit's characteristics.
    pub fn properties(&self) -> FitProperties<T> {
        FitProperties {
            degree: self.degree(),
            data_points: self.data().len(),
            coefficients: self.coefficients().to_vec(),
            coefficient_errors: self
                .covariance()
                .map(|cov| cov.coefficient_standard_errors())
                .ok(),
            mse: self.mean_squared_error(),
            r_squared: self.r_squared(self.data()),
        }
    }
}

impl<B, T: Value> AsRef<Polynomial<'_, B, T>> for CurveFit<B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn as_ref(&self) -> &Polynomial<'static, B, T> {
        &self.function
    }
}

impl<T: Value, B> std::fmt::Display for CurveFit<B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.equation())
    }
}

/// A set of diagnostic properties for a curve fit.
///
/// Can be serialize to JSON or other formats.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FitProperties<T: Value> {
    /// The degree of the fitted polynomial.
    pub degree: usize,

    /// The number of data points used in the fit.
    pub data_points: usize,

    /// The coefficients of the fitted polynomial.
    pub coefficients: Vec<T>,

    /// The standard errors of the fitted coefficients.
    pub coefficient_errors: Option<Vec<T>>,

    /// The mean squared error of the fit.
    pub mse: T,

    /// The R² value of the fit, if available.
    pub r_squared: T,
}

#[cfg(test)]
mod tests {
    use crate::{assert_close, function, transforms::ApplyNoise, MonomialFit};

    use super::*;

    #[test]
    fn test_curvefit_new_and_coefficients() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let coefs = fit.coefficients();
        assert_eq!(coefs.len(), 3);
    }

    #[test]
    fn test_curvefit_y_in_range() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let y: f64 = fit.y(1.0).unwrap();
        assert!(y.is_finite());
    }

    #[test]
    fn test_curvefit_y_out_of_range() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let y = fit.y(-1.0);
        assert!(y.is_err());
    }

    #[test]
    fn test_curvefit_solution_matches_data_len() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let solution = fit.solution();
        assert_eq!(solution.len(), data.len());
    }

    #[test]
    fn test_curvefit_resize_increase_and_decrease() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0), (3.0, 13.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let fit_higher = fit.clone().resize(3).unwrap();
        assert_eq!(fit_higher.degree(), 3);
        let fit_lower = fit_higher.resize(1).unwrap();
        assert_eq!(fit_lower.degree(), 1);
    }

    #[test]
    fn test_curvefit_covariance_and_standard_errors() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let cov = fit.covariance().unwrap();
        let errors = cov.coefficient_standard_errors();
        assert_eq!(errors.len(), 3);
        for err in errors {
            assert!(err >= 0.0);
        }
    }

    #[test]
    fn test_curvefit_confidence_band() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let cov = fit.covariance().unwrap();
        let band = cov.confidence_band(1.0, Confidence::P95).unwrap();
        assert!(band.lower <= band.upper);
        assert!(band.value >= band.lower && band.value <= band.upper);
    }

    #[test]
    fn test_curvefit_model_score_and_r_squared() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let score: f64 = fit.model_score(ScoringMethod::AIC);
        let r2 = fit.r_squared(data);
        assert!(score.is_finite());
        assert_close!(r2, 1.0);

        function!(mono(x) = 1.0 + 2.0 x^1); // strictly increasing
        let data = mono
            .solve_range(0.0..1000.0, 1.0)
            .apply_normal_noise(0.3, None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert!(fit.r_squared(&data) < 1.0);
        assert!(fit.model_score(ScoringMethod::AIC).is_finite());
    }

    #[test]
    fn test_curvefit_as_polynomial_and_into_polynomial() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let poly_ref = fit.as_polynomial();
        let poly_owned = fit.clone().into_polynomial();
        assert_eq!(poly_ref.coefficients(), poly_owned.coefficients());
    }

    #[test]
    fn test_curvefit_properties() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let props = fit.properties();
        assert_eq!(props.degree, 2);
        assert_eq!(props.data_points, 3);
        assert_eq!(props.coefficients.len(), 3);
        assert!(props.mse >= 0.0);
        assert!(props.r_squared <= 1.0 && props.r_squared >= 0.0);
    }

    #[test]
    fn test_curvefit_new_auto_selects_best_degree() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0), (3.0, 13.0)];
        let fit = MonomialFit::new_auto(data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert!(fit.degree() < data.len());
    }

    #[test]
    fn test_curvefit_solve_and_solve_range() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let xs = vec![0.0, 1.0, 2.0];
        let points = fit.solve(xs.clone()).unwrap();
        assert_eq!(points.len(), xs.len());
        let range_points = fit.solve_range(0.0..2.0, 1.0).unwrap();
        assert_eq!(range_points.len(), 2);
    }

    #[test]
    fn test_fourier() {
        let basis = crate::basis::FourierBasis::new(&[(0.0, 1.0), (100.0, 1.0)]);
        let poly = unsafe {
            crate::Polynomial::from_raw(
                basis,
                std::borrow::Cow::Borrowed(&[10.0, 5.0, 3.0, 6.0, 4.0]),
                2,
            )
        };
        let data = poly.solve_range(0.0..101.0, 1.0);
        let fit =
            crate::FourierFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();

        println!("{fit}");
        println!("{poly}");

        crate::plot!(&fit, functions = [&poly]);
    }
}
