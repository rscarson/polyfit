use std::{borrow::Cow, ops::RangeInclusive};

use nalgebra::{DMatrix, DVector, SVD};

use crate::{
    basis::{Basis, DifferentialBasis, IntegralBasis, IntoMonomialBasis},
    display::PolynomialDisplay,
    error::{Error, Result},
    score::ModelScoreProvider,
    statistics::{self, Confidence, ConfidenceBand, DegreeBound, Tolerance},
    value::{CoordExt, SteppedValues, Value},
    MonomialPolynomial, Polynomial,
};

/// Logarithmic series curve
///
/// Uses logarithmic basis functions, which are particularly useful for modeling data that exhibits logarithmic growth or decay.
/// The basis functions include terms like 1, ln(x), (ln(x))^2, ..., (ln(x))^n.
pub type LogarithmicFit<'data, T = f64> = CurveFit<'data, crate::basis::LogarithmicBasis<T>, T>;

/// Laguerre series curve
///
/// Uses Laguerre polynomials, which are orthogonal polynomials defined on the interval \[0, ∞\].
/// These polynomials are particularly useful in quantum mechanics and numerical analysis.
pub type LaguerreFit<'data, T = f64> = CurveFit<'data, crate::basis::LaguerreBasis<T>, T>;

/// Physicists' Hermite series curve
///
/// Uses Physicists' Hermite polynomials, which are orthogonal polynomials defined on the interval \[-∞, ∞\].
/// These polynomials are particularly useful in probability, combinatorics, and physics, especially in quantum mechanics.
pub type PhysicistsHermiteFit<'data, T = f64> =
    CurveFit<'data, crate::basis::PhysicistsHermiteBasis<T>, T>;

/// Probabilists' Hermite series curve
///
/// Uses Probabilists' Hermite polynomials, which are orthogonal polynomials defined on the interval \[-∞, ∞\].
/// These polynomials are particularly useful in probability theory and statistics, especially in the context of Gaussian distributions.
pub type ProbabilistsHermiteFit<'data, T = f64> =
    CurveFit<'data, crate::basis::ProbabilistsHermiteBasis<T>, T>;

/// Legendre series curve
///
/// Uses Legendre polynomials, which are orthogonal polynomials defined on the interval \[-1, 1\].
/// These polynomials are particularly useful for minimizing oscillation in polynomial interpolation.
pub type LegendreFit<'data, T = f64> = CurveFit<'data, crate::basis::LegendreBasis<T>, T>;

/// Fourier series curve
///
/// Uses a Fourier series basis, which is particularly well-suited for modeling periodic functions.
/// The basis functions include sine and cosine terms, allowing for effective representation of oscillatory behavior.
pub type FourierFit<'data, T = f64> = CurveFit<'data, crate::basis::FourierBasis<T>, T>;

/// Normalized Chebyshev polynomial curve
///
/// Uses the Chebyshev polynomials, which are orthogonal polynomials defined on the interval \[-1, 1\].
/// These polynomials are particularly useful for minimizing Runge's phenomenon in polynomial interpolation.
pub type ChebyshevFit<'data, T = f64> = CurveFit<'data, crate::basis::ChebyshevBasis<T>, T>;

/// Non-normalized monomial polynomial curve
///
/// Uses the standard monomial functions: 1, x, x^2, ..., x^n
///
/// It is the most basic form of polynomial basis and is not normalized.
/// It can lead to numerical instability for high-degree polynomials.
pub type MonomialFit<'data, T = f64> = CurveFit<'data, crate::basis::MonomialBasis<T>, T>;

/// Represents the covariance matrix and derived statistics for a curve fit.
///
/// Provides tools to evaluate the uncertainty of coefficients and predictions
/// of a fitted polynomial or other basis function model.
///
/// # Type Parameters
/// - `'a`: Lifetime of the reference to the original curve fit.
/// - `B`: Basis type used by the curve fit (implements `Basis<T>`).
/// - `T`: Numeric type (defaults to `f64`) implementing `Value`.
pub struct CurveFitCovariance<'a, 'data, B, T: Value = f64>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fit: &'a CurveFit<'data, B, T>,
    covariance: DMatrix<T>,
}
impl<'a, 'data, B, T: Value> CurveFitCovariance<'a, 'data, B, T>
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
    pub fn new(fit: &'a CurveFit<'data, B, T>) -> Result<Self> {
        let n = fit.data.len();
        let k = fit.basis().k(fit.degree());

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
        let covariance = xtx_inv * res_var;
        Ok(Self { fit, covariance })
    }

    /// Computes the standard error of the coefficient at j.
    ///
    /// Returns None if the coefficient does not exist.
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
    /// Returns a confidence band representing the uncertainty in the predicted y value
    ///
    /// # Parameters
    /// - `x`: The x value to compute the confidence band for.
    /// - `confidence_level`: Desired confidence level (e.g., P95).
    /// - `noise_tolerance`: Optional additional variance to add to the prediction variance,
    ///   (e.g., to account for measurement noise).
    ///
    /// This estimates the uncertainty in the predicted y value at a specific x
    /// location, providing a range within which the true value is likely to fall.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn confidence_band(
        &self,
        x: T,
        confidence_level: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> Result<ConfidenceBand<T>> {
        let mut y_var = self.prediction_variance(x);
        let value = self.fit.y(x)?;

        match noise_tolerance {
            Some(Tolerance::Absolute(tol)) => {
                y_var += tol;
            }
            Some(Tolerance::Relative(rel)) => {
                let (data_sdev, _) = statistics::stddev_and_mean(self.fit.data().y_iter());
                let noise_tolerance = data_sdev * rel;
                y_var += noise_tolerance * noise_tolerance;
            }
            None => {}
        }

        let y_se = y_var.sqrt();

        let z = confidence_level.try_cast::<T>()?;
        let lower = value - z * y_se;
        let upper = value + z * y_se;
        Ok(ConfidenceBand {
            value,
            lower,
            upper,
            level: confidence_level,
            tolerance: noise_tolerance,
        })
    }

    /// Computes the confidence intervals for all data points in the original dataset.
    ///
    /// This evaluates the fitted model at each `x` from the original data and returns
    /// a `ConfidenceBand` for each point, quantifying the uncertainty of predictions.
    ///
    /// # Parameters
    /// - `confidence_level`: Desired confidence level (e.g., P95).
    /// - `noise_tolerance`: Optional additional variance to add to the prediction variance,
    ///
    /// # Returns
    /// - `Ok(Vec<(T, ConfidenceBand<T>)>)` containing one confidence band per data point.
    /// - `Err` if any prediction or type conversion fails.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn solution_confidence(
        &self,
        confidence_level: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> Result<Vec<(T, ConfidenceBand<T>)>> {
        let x = self.fit.data().iter().map(|(x, _)| *x);
        x.map(|x| {
            Ok((
                x,
                self.confidence_band(x, confidence_level, noise_tolerance)?,
            ))
        })
        .collect()
    }

    /// Identifies outliers in the original dataset based on the confidence intervals.
    ///
    /// An outlier is defined as a data point where the actual `y` value falls outside
    /// the computed confidence band for its corresponding `x`.
    ///
    /// The confidence level determines the width of the confidence band. Higher confidence levels have wider bands,
    /// making it less likely for points to be classified as outliers.
    ///
    /// # Parameters
    /// - `confidence_level`: Confidence level used to determine the bounds (e.g., P95).
    ///
    /// # Returns
    /// - `Ok(Vec<((T, T, ConfidenceBand<T>))>)` containing the index and `(x, y, confidence_band)` of each outlier.
    /// - `Err` if confidence intervals cannot be computed.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn outliers(
        &self,
        confidence_level: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> Result<Vec<(T, T, ConfidenceBand<T>)>> {
        let bands = self.solution_confidence(confidence_level, noise_tolerance)?;
        let mut outliers = Vec::new();

        for ((x, y), (_, band)) in self.fit.data().iter().zip(bands) {
            if *y < band.lower || *y > band.upper {
                outliers.push((*x, *y, band));
            }
        }

        Ok(outliers)
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
pub struct CurveFit<'data, B, T: Value = f64>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    data: Cow<'data, [(T, T)]>,
    x_range: RangeInclusive<T>,
    function: Polynomial<'static, B, T>,
    k: T,
}
impl<'data, T: Value, B> CurveFit<'data, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    /// Turns a dataset portion into a basis matrix and y-values vector.
    fn create_matrix(data: &[(T, T)], basis: &B, k: usize) -> (DMatrix<T>, DVector<T>) {
        let mut bigx = DMatrix::zeros(data.len(), k);
        let b = DVector::from_iterator(data.len(), data.iter().map(|&(_, y)| y));

        for (row, (x, _)) in bigx.row_iter_mut().zip(data.iter()) {
            let x = basis.normalize_x(*x);
            basis.fill_matrix_row(0, x, row);
        }

        (bigx, b)
    }

    /// If appropriate, creates the basis matrix in parallel, using the normal equation in chunks
    /// Otherwise, falls back to the normal `create_matrix`.
    ///
    /// The bool indicates if parallel processing was used - needed for `new_auto` to know to slice the matrix properly
    fn create_parallel_matrix(
        data: &[(T, T)],
        basis: &B,
        k: usize,
    ) -> (DMatrix<T>, DVector<T>, bool) {
        #[cfg(not(feature = "parallel"))]
        {
            let (m, b) = Self::create_matrix(data, basis, k);
            return (m, b, false);
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            const MIN_ROWS_TO_PARALLEL: usize = 500_000;

            if data.len() < MIN_ROWS_TO_PARALLEL {
                let (m, b) = Self::create_matrix(data, basis, k);
                return (m, b, false);
            }

            // Each thread builds the (xtx, xtb) pair for its chunk, reducing an NxK to KxK problem
            let threads = rayon::current_num_threads();
            let chunk_size = (data.len() / threads).max(1);
            let thread_data: Vec<&[(T, T)]> = data.chunks(chunk_size).collect();
            let mut partial_results: Vec<(DMatrix<T>, DVector<T>)> = thread_data
                .into_par_iter()
                .map(|chunk| {
                    let (m, b) = Self::create_matrix(chunk, basis, k);
                    Self::invert_matrix(&m, &b)
                })
                .collect();

            // Now accumulate the partial results to get the full (xtx, xtb)
            let (mut xtx, mut xtb) = partial_results.pop().unwrap_or_else(|| {
                (
                    DMatrix::<T>::zeros(k, k), // No data, zero matrix
                    DVector::<T>::zeros(k),    // No data, zero vector
                )
            });

            // We use kahan summation here to reduce numerical error
            let mut xtx_c = DMatrix::<T>::zeros(k, k);
            let mut xtb_c = DVector::<T>::zeros(k);
            for (part_xtx, part_xtb) in partial_results {
                for i in 0..k {
                    let y = part_xtb[i] - xtb_c[i];
                    let t = xtb[i] + y;
                    xtb_c[i] = (t - xtb[i]) - y;
                    xtb[i] = t;

                    for j in 0..k {
                        let y = part_xtx[(i, j)] - xtx_c[(i, j)];
                        let t = xtx[(i, j)] + y;
                        xtx_c[(i, j)] = (t - xtx[(i, j)]) - y;
                        xtx[(i, j)] = t;
                    }
                }
            }

            (xtx, xtb, true)
        }
    }

    /// Reduce the n by k / 1 by n into a k by k and k by 1 system.
    fn invert_matrix(matrix: &DMatrix<T>, b: &DVector<T>) -> (DMatrix<T>, DVector<T>) {
        let xtx = matrix.transpose() * matrix;
        let xtb = matrix.transpose() * b;
        (xtx, xtb)
    }

    /// Solves the linear system using SVD.
    fn solve_matrix(xtx: DMatrix<T>, xtb: &DVector<T>) -> Result<Vec<T>> {
        let size = xtx.shape();

        // Calculate the singular value decomposition of the matrix
        let decomp = SVD::new_unordered(xtx, true, true);

        // Calculate epsilon value
        // ~= machine_epsilon * max(size) * max_singular
        let machine_epsilon = T::epsilon();
        let max_size = size.0.max(size.1);
        let sigma_max = decomp.singular_values.max();
        let epsilon = machine_epsilon * T::try_cast(max_size)? * sigma_max;

        // Solve for X in `SVD * X = b`
        let big_x = decomp.solve(xtb, epsilon).map_err(Error::Algebra)?;
        let coefficients: Vec<_> = big_x.data.into();

        // Make sure the coefficients are valid
        if coefficients.iter().any(|c| c.is_nan()) {
            return Err(Error::Algebra("NaN in coefficients"));
        }

        Ok(coefficients)
    }

    /// Creates a new polynomial curve fit from raw components.
    fn from_raw(
        data: Cow<'data, [(T, T)]>,
        x_range: RangeInclusive<T>,
        basis: B,
        coefs: Vec<T>,
        degree: usize,
    ) -> Result<Self> {
        let k = T::try_cast(coefs.len())?;
        let function = unsafe { Polynomial::from_raw(basis, coefs.into(), degree) }; // Safety: The coefs were generated by the basis

        Ok(Self {
            data,
            x_range,
            function,
            k,
        })
    }

    /// Returns an owned version of this curve fit, with a full copy of the data.
    #[must_use]
    pub fn to_owned(&self) -> CurveFit<'static, B, T> {
        CurveFit {
            data: Cow::Owned(self.data.to_vec()),
            x_range: self.x_range.clone(),
            function: self.function.clone(),
            k: self.k,
        }
    }

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
    /// # Warning
    /// If the `parallel` feature is enabled, and the dataset is > 500,000 rows,
    /// the basis matrix will be constructed in parallel, and the normal equation
    /// will be used to reduce the size of the system.
    ///
    /// This can reduce numerical accuracy for very high-degree polynomials, and
    /// should be used with caution.
    ///
    /// # Example
    /// ```
    /// # use polyfit::ChebyshevFit;
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// println!("Coefficients: {:?}", fit.coefficients());
    /// ```
    pub fn new(data: impl Into<Cow<'data, [(T, T)]>>, degree: usize) -> Result<Self> {
        let data: Cow<_> = data.into();

        // Cannot fit a polynomial of degree 0 or if there is no data.
        if data.is_empty() {
            return Err(Error::NoData);
        } else if degree >= data.len() {
            return Err(Error::DegreeTooHigh(degree));
        }

        let x_range = data.x_range().ok_or(Error::NoData)?;
        let basis = B::from_range(x_range.clone());
        let k = basis.k(degree);

        let (m, b, _) = Self::create_parallel_matrix(&data, &basis, k);
        let coefs = Self::solve_matrix(m, &b)?;
        Self::from_raw(data, x_range, basis, coefs, degree)
    }

    /// Automatically selects the best polynomial degree and creates a curve fit.
    ///
    /// This function fits polynomials of increasing degree to the provided dataset
    /// and selects the “best” degree according to the specified scoring method.
    ///
    /// # Parameters
    /// - `data`: Slice of `(x, y)` points to fit.
    /// - `method`: [`crate::score`] to evaluate model quality.  
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
    /// # Warning
    /// If the `parallel` feature is enabled, and the dataset is > 500,000 rows,
    /// the basis matrix will be constructed in parallel, and the normal equation
    /// will be used to reduce the size of the system.
    ///
    /// This can reduce numerical accuracy for very high-degree polynomials, and
    /// should be used with caution.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, statistics::DegreeBound, score::Aic};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, &Aic).unwrap();
    /// println!("Selected degree: {}", fit.degree());
    /// ```
    pub fn new_auto(
        data: impl Into<Cow<'data, [(T, T)]>>,
        max_degree: impl Into<DegreeBound>,
        method: &impl ModelScoreProvider,
    ) -> Result<Self> {
        let data: Cow<_> = data.into();
        let max_degree = max_degree.into().max_degree(data.len());
        if data.is_empty() {
            return Err(Error::NoData);
        }

        // Step 1 - Create the basis and matrix once
        let x_range = data.x_range().ok_or(Error::NoData)?;
        let basis = B::from_range(x_range.clone());
        let max_k = basis.k(max_degree);
        let (xtx, xtb, normal_eq) = Self::create_parallel_matrix(&data, &basis, max_k);

        #[cfg(not(feature = "parallel"))]
        let (min_score, model_scores) = {
            // Step 2 - Build models using increasingly narrow slices of the matrix
            let mut min_score = T::infinity();
            let mut model_scores: Vec<(Self, T)> = Vec::with_capacity(max_degree + 1);
            for degree in 0..=max_degree {
                let k = basis.k(degree);

                let height = if normal_eq { k } else { xtx.nrows() };
                let m = xtx.view((0, 0), (height, k)).into_owned();
                let Ok(coefs) = Self::solve_matrix(m, &xtb) else {
                    continue;
                };

                let Ok(model) =
                    Self::from_raw(data.clone(), x_range.clone(), basis.clone(), coefs, degree)
                else {
                    continue;
                };

                let score = model.model_score(method);
                model_scores.push((model, score));
                if score < min_score {
                    min_score = score;
                }
            }

            (min_score, model_scores)
        };

        #[cfg(feature = "parallel")]
        let (min_score, model_scores) = {
            use rayon::prelude::*;

            let mut model_scores: Vec<(Self, T)> = (0..=max_degree)
                .into_par_iter()
                .filter_map(|degree| {
                    let k = basis.k(degree);

                    let height = if normal_eq { k } else { xtx.nrows() };
                    let m = xtx.view((0, 0), (height, k)).into_owned();
                    let coefs = Self::solve_matrix(m, &xtb).ok()?;

                    let model =
                        Self::from_raw(data.clone(), x_range.clone(), basis.clone(), coefs, degree)
                            .ok()?;

                    let score = model.model_score(method);
                    Some((model, score))
                })
                .collect();

            // Sort by degree ascending
            model_scores.sort_by_key(|(m, _)| m.degree());

            let min_score = model_scores
                .iter()
                .map(|(_, score)| *score)
                .fold(T::infinity(), nalgebra::RealField::min);

            (min_score, model_scores)
        };

        // Step 3 - get delta_score
        // Re: Burnham and Anderson, use the first delta <=2 (P = 0.37)
        // Statistically indistinguishable from the top model
        for (model, score) in model_scores {
            let delta = score - min_score;
            if delta <= T::two() {
                return Ok(model);
            }
        }

        Err(Error::NoModel)
    }

    /// Creates a new polynomial curve fit using K-fold cross-validation to select the best degree.
    ///
    /// This function splits the dataset into `folds` subsets, using each subset as a validation set while training on the remaining data.
    /// It evaluates polynomial fits of increasing degree and selects the best degree based on the specified scoring method.
    ///
    /// This method helps prevent overfitting by ensuring that the selected model generalizes well to unseen data, and is particularly useful for small datasets
    /// or those with outliers.
    ///
    /// # Parameters
    /// - `data`: Slice of `(x, y)` points to fit.
    /// - `folds`: Number of folds for cross-validation (must be at least 2).
    /// - `max_degree`: Maximum polynomial degree to consider.
    /// - `method`: [`ModelScoreProvider`] to evaluate model quality.
    ///
    /// # Returns
    /// Returns `Ok(Self)` with the fit at the optimal degree.
    ///
    /// # Errors
    /// Returns [`Error`] if:
    /// - `data` is empty or `folds < 2` (`Error::NoData`)  
    /// - A numeric value cannot be represented in the target type (`Error::CastFailed`)
    /// - No valid model could be fitted (`Error::NoModel`)
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, statistics::DegreeBound, score::Aic};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new_kfold_cross_validated(data, 2, DegreeBound::Relaxed, &Aic).unwrap();
    /// println!("Selected degree: {}", fit.degree());
    /// ```
    #[expect(
        clippy::many_single_char_names,
        reason = "It's math what do you want from me"
    )]
    pub fn new_kfold_cross_validated(
        data: impl Into<Cow<'data, [(T, T)]>>,
        folds: usize,
        max_degree: impl Into<DegreeBound>,
        method: &impl ModelScoreProvider,
    ) -> Result<Self> {
        let data: Cow<_> = data.into();
        let fold_size = data.len() / folds;
        let max_degree = max_degree.into().max_degree(data.len());
        if data.is_empty() || folds < 2 {
            return Err(Error::NoData);
        }

        // Step 1 - Create the basis and matrix once
        let x_range = data.x_range().ok_or(Error::NoData)?;
        let basis = B::from_range(x_range.clone());
        let k = basis.k(max_degree);
        let (m, b) = Self::create_matrix(data.as_ref(), &basis, k);

        // Step 2 - Precalculate fold boundaries
        let mut fold_ranges = Vec::with_capacity(folds);
        for i in 0..folds {
            let start = i * fold_size;
            let end = if i == folds - 1 {
                data.len()
            } else {
                (i + 1) * fold_size
            };
            fold_ranges.push(start..end);
        }

        // Step 3 - Use `folds` views into m and b for training and test sets
        // We do this for each degree candidate, and each fold
        let mut min_score = T::infinity();
        let mut candidates = Vec::with_capacity(max_degree + 1);
        for degree in 0..=max_degree {
            let k = basis.k(degree);
            let m = m.view((0, 0), (m.nrows(), k)).into_owned();

            // Evaluate this degree with K-fold cross-validation
            let mut mean_score = T::zero();
            let mut n = T::zero();
            for i in 0..folds {
                let test_range = &fold_ranges[i];
                let test_data = &data[test_range.clone()];

                let mut fold_m = DMatrix::zeros(data.len() - test_range.len(), k);
                let fold_b = DVector::from_iterator(
                    data.len() - test_data.len(),
                    b.iter().enumerate().filter_map(|(j, &y)| {
                        if test_range.contains(&j) {
                            None
                        } else {
                            Some(y)
                        }
                    }),
                );

                // Copy relevent rows from m into fold_m
                for (i, src) in m.row_iter().enumerate() {
                    if !test_range.contains(&i) {
                        let dst_index = if i < test_range.start {
                            i
                        } else {
                            i - test_range.len()
                        };
                        let mut dst = fold_m.row_mut(dst_index);
                        dst.copy_from(&src);
                    }
                }

                let Ok(coefs) = Self::solve_matrix(fold_m, &fold_b) else {
                    continue;
                };

                let Ok(model) =
                    Self::from_raw(data.clone(), x_range.clone(), basis.clone(), coefs, degree)
                else {
                    continue;
                };

                let y = test_data.y_iter();
                let predicted = model.as_polynomial().solve(test_data.x_iter());
                let y_fit = predicted.y_iter();
                mean_score += method.score(y, y_fit, model.k);
                n += T::one();
            }

            mean_score /= n;
            candidates.push((degree, mean_score));
            if mean_score < min_score {
                min_score = mean_score;
            }
        }

        // Step 4 - Select the best model within 2 AIC units of the minimum (Burnham and Anderson 2002)
        for (degree, score) in candidates {
            let delta = score - min_score;
            if delta <= T::two() {
                return Self::new(data, degree);
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

    /// Returns a reference to the basis function.
    pub(crate) fn basis(&self) -> &B {
        self.function.basis()
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
    /// let band = cov.confidence_band(1.0, Confidence::P95, None).unwrap();
    /// println!("Predicted CI at x=1: {} - {}", band.min(), band.max());
    /// ```
    pub fn covariance(&self) -> Result<CurveFitCovariance<'_, '_, B, T>> {
        CurveFitCovariance::new(self)
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
    /// # use polyfit::statistics::Confidence;
    /// # use polyfit::MonomialFit;
    /// # let model = MonomialFit::new(&[(0.0, 0.0), (1.0, 1.0)], 1).unwrap();
    /// let critical_points = model.critical_points().unwrap();
    /// ```
    pub fn critical_points(&self) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        self.function.critical_points()
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
    /// # use polyfit::statistics::Confidence;
    /// # use polyfit::MonomialFit;
    /// # let model = MonomialFit::new(&[(0.0, 0.0), (1.0, 1.0)], 1).unwrap();
    /// let area = model.area_under_curve(0.0, 3.0, None).unwrap();
    /// println!("Area under curve: {}", area);
    /// ```
    pub fn area_under_curve(&self, x_min: T, x_max: T, constant: Option<T>) -> Result<T>
    where
        B: IntegralBasis<T>,
    {
        self.function.area_under_curve(x_min, x_max, constant)
    }

    /// Returns the X-values where the function is not monotone (i.e., where the derivative changes sign).
    ///
    /// # Errors
    /// Returns an error if the derivative cannot be computed.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::MonomialFit;
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = MonomialFit::new(data, 2).unwrap();
    /// let violations = fit.monotonicity_violations().unwrap();
    /// ```
    pub fn monotonicity_violations(&self) -> Result<Vec<T>>
    where
        B: DifferentialBasis<T>,
    {
        self.function.monotonicity_violations(self.x_range.clone())
    }

    /// Computes the quality score of the polynomial fit using the specified method.
    ///
    /// This evaluates how well the fitted polynomial represents the data, taking
    /// into account both the fit error and model complexity.
    ///
    /// # Parameters
    /// - `method`: [`ModelScoreProvider`] to use for scoring.  
    ///   - `AIC`: Akaike Information Criterion (uses `AICc` if `n/k < 4`)  
    ///   - `BIC`: Bayesian Information Criterion
    ///
    /// # Returns
    /// The score as a numeric value (`T`). Lower scores indicate better models.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{ChebyshevFit, score::Aic};
    /// let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    /// let fit = ChebyshevFit::new(data, 2).unwrap();
    /// let score = fit.model_score(&Aic);
    /// println!("Model score: {}", score);
    /// ```
    pub fn model_score(&self, method: &impl ModelScoreProvider) -> T {
        let y = self.data.y_iter();
        let y_fit = self.solution().into_iter().map(|(_, y)| y);
        method.score(y, y_fit, self.k)
    }

    /// Computes the residuals of the fit.
    ///
    /// Residuals are the differences between the observed `y` values and the predicted `y` values from the fitted polynomial.
    /// They provide insight into the fit quality and can be used for diagnostic purposes.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// residual_i = y_i - f(x_i)
    /// where
    ///   y_i = observed value, f(x_i) = predicted value from the polynomial at x_i
    /// ```
    /// </div>
    ///
    /// # Returns
    /// A vector of residuals, where each element corresponds to a data point.
    ///
    pub fn residuals(&self) -> Vec<(T, T)> {
        let y = self.data.y_iter();
        y.zip(self.solution())
            .map(|(y, (x, y_fit))| (x, y - y_fit))
            .collect()
    }

    /// Computes the residuals of the fit, filtering out small residuals likely due to floating point noise.
    ///
    /// This form can help minimize the impact of floating point precision and rounding.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// max_val = max(|y_i|, |f(x_i)|, 1)
    /// epsilon = machine_epsilon * sqrt(n) * max_val
    /// residual_i = y_i - f(x_i) if |y_i - f(x_i)| >= epsilon else 0
    /// where
    ///   y_i = observed value, f(x_i) = predicted value from the polynomial at x_i, n = number of data points
    /// ```
    /// </div>
    ///
    /// # Returns
    /// A vector of scaled residuals, where each element corresponds to a data point.
    pub fn filtered_residuals(&self) -> Vec<(T, T)> {
        // Get max(|y|, |y_fit|, 1)
        let max_val = self
            .data
            .iter()
            .chain(self.solution().iter())
            .map(|(_, y)| y.abs())
            .fold(T::zero(), nalgebra::RealField::max);
        let max_val = nalgebra::RealField::max(max_val, T::one());

        // Residual epsilon
        let root_n = T::from_positive_int(self.data.len()).sqrt();
        let epsilon = T::epsilon() * root_n * max_val;

        let y = self.data.y_iter();
        y.zip(self.solution())
            .filter_map(|(y, (x, y_fit))| {
                let r = y - y_fit;
                let r = if nalgebra::ComplexField::abs(r) < epsilon {
                    None?
                } else {
                    r
                };
                Some((x, r))
            })
            .collect()
    }

    /// Computes the residual variance of the model's predictions.
    ///
    /// See [`statistics::residual_variance`].
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

    /// Calculates the R-squared value for the model compared to provided function.
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

    /// Returns the degree of the polynomial.
    ///
    /// The number of actual components, or basis functions, in the expression of a degree is defined by the basis.
    ///
    /// That number is called k. For most basis choices, `k = degree + 1`.
    pub fn degree(&self) -> usize {
        self.function.degree()
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
    pub fn coefficients(&self) -> &[T] {
        self.function.coefficients()
    }

    /// Returns a reference to the data points used for fitting.
    ///
    /// Each element is a `(x, y)` tuple representing a data point.
    pub fn data(&self) -> &[(T, T)] {
        &self.data
    }

    /// Returns the inclusive range of x-values in the dataset.
    pub fn x_range(&self) -> RangeInclusive<T> {
        self.x_range.clone()
    }

    /// Returns the inclusive range of y-values in the dataset.
    ///
    /// This is computed dynamically from the stored data points. Use sparingly
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
    /// let points = fit.solve_range(0.0..=2.0, 0.5).unwrap();
    /// for (x, y) in points {
    ///     println!("x = {}, y = {}", x, y);
    /// }
    /// ```
    pub fn solve_range(&self, range: RangeInclusive<T>, step: T) -> Result<Vec<(T, T)>> {
        self.solve(SteppedValues::new(range, step))
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
    /// A reference to the [`Polynomial`] that this fit uses internally.
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
    /// The [`Polynomial`] that this fit uses internally
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

impl<B, T: Value> AsRef<Polynomial<'_, B, T>> for CurveFit<'_, B, T>
where
    B: Basis<T>,
    B: PolynomialDisplay<T>,
{
    fn as_ref(&self) -> &Polynomial<'static, B, T> {
        &self.function
    }
}

impl<T: Value, B> std::fmt::Display for CurveFit<'_, B, T>
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
    use crate::{
        assert_close, assert_fits, function, score::Aic, transforms::ApplyNoise, MonomialFit,
    };

    use super::*;

    #[test]
    fn test_curvefit_new_and_coefficients() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let coefs = fit.coefficients();
        assert_eq!(coefs.len(), 3);
    }

    #[test]
    fn test_big() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2 + 4.0 x^3 + 5.0 x^4 + 6.0 x^5);
        let data = poly.solve_range(0.0..=10_000_000.0, 1.0);
        let fit = ChebyshevFit::new(&data, 5).unwrap();
        assert_fits!(poly, fit, 0.999);
        crate::plot!(fit, prefix = "big");
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
        let band = cov.confidence_band(1.0, Confidence::P95, None).unwrap();
        assert!(band.lower <= band.upper);
        assert!(band.value >= band.lower && band.value <= band.upper);
    }

    #[test]
    fn test_curvefit_model_score_and_r_squared() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let score: f64 = fit.model_score(&Aic);
        let r2 = fit.r_squared(data);
        assert!(score.is_finite());
        assert_close!(r2, 1.0);

        function!(mono(x) = 1.0 + 2.0 x^1); // strictly increasing
        let data = mono
            .solve_range(0.0..=1000.0, 1.0)
            .apply_normal_noise(Tolerance::Relative(0.3), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert!(fit.r_squared(&data) < 1.0);
        assert!(fit.model_score(&Aic).is_finite());
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
        let fit = MonomialFit::new_auto(data, DegreeBound::Relaxed, &Aic).unwrap();
        assert!(fit.degree() < data.len());
    }

    #[test]
    fn test_curvefit_solve_and_solve_range() {
        let data = &[(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
        let fit = MonomialFit::new(data, 2).unwrap();
        let xs = vec![0.0, 1.0, 2.0];
        let points = fit.solve(xs.clone()).unwrap();
        assert_eq!(points.len(), xs.len());
        let range_points = fit.solve_range(0.0..=2.0, 1.0).unwrap();
        assert_eq!(range_points.len(), 3);
    }

    #[test]
    fn test_kfold() {
        function!(mono(x) = 5 x^5 - 3 x^3 + 2 x^2 + 1.0);
        let data = mono
            .solve_range(0.0..=1000.0, 1.0)
            .apply_salt_pepper_noise(0.01, -10000.0, 10000.0, None)
            .apply_poisson_noise(10.0, None);
        let fit =
            MonomialFit::new_kfold_cross_validated(&data, 5, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(mono, fit);
    }
}
