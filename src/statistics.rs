//! Functions and tools for evaluating polynomial fits and scoring models
//!
//! This module provides functions and types to evaluate how well a polynomial model
//! fits a dataset, and to score models when automatically selecting polynomial degrees.
//!
//! # Model Fit / Regression Diagnostics
//! - [`r_squared`]: Proportion of variance explained by the model. Higher is better (0 to 1).
//! - [`adjusted_r_squared`]: R² adjusted for number of predictors. Use to compare models of different degrees.
//! - [`residual_variance`]: Unbiased estimate of variance of errors after fitting. Used for confidence intervals.
//! - [`residual_normality`]: Likelihood that the residuals are normally distributed. Results near 0 or 1 indicate non-normality, higher results do not guarantee normality.
//!
//! # Confidence Intervals
//! - [`ConfidenceBand`]: Represents a confidence interval with lower and upper bounds, determined by a given probability.
//! - [`Confidence`]: Enum for common confidence levels (68%, 95%, 99%).
//!
//! # Model Selection
//! - [`ScoringMethod`]: Enum for model selection criteria (AIC, BIC) with methods to calculate scores.
//! - [`DegreeBound`]: Enum to specify constraints on polynomial degree when automatically selecting it.
//!
//! # Error Metrics
//! - [`mean_absolute_error`]: Average absolute difference between observed and predicted values. Lower is better.
//! - [`mean_squared_error`]: Average squared difference between observed and predicted values. Lower is better.
//! - [`root_mean_squared_error`]: Square root of MSE, giving error in same units as observed values. Lower is better.
//! - [`huber_log_likelihood`]: Robust error metric less sensitive to outliers. Higher is better.
//!
//! # Descriptive Statistics
//! - [`mean`]: Arithmetic mean of a dataset.
//! - [`stddev_and_mean`]: Standard deviation and mean of a dataset.
//! - [`median_absolute_deviation`]: Average absolute deviation from the mean.
//! - [`spread`]: Difference between maximum and minimum values in a dataset.
//! - [`skewness_and_kurtosis`]: Measures of asymmetry and "tailedness" of the distribution.
//!
//! # Model Fit vs Model Selection
//!
//! - **Model Fit**: How well does the model explain the data? Use [`r_squared`] or [`residual_variance`].
//!   - Returns a value between 0 and 1.
//!   - 0 = model explains none of the variance.
//!   - 1 = model perfectly fits the data.
//!
//! - **Model selection**: Choosing the best polynomial degree to avoid overfitting.
//!   - Use [`ScoringMethod`] with [`ScoringMethod::calculate`].
//!   - Options:
//!     - `AIC`: Akaike Information Criterion, more lenient penalty for complexity.
//!     - `BIC`: Bayesian Information Criterion, stricter penalty for complexity.
//!   - Lower scores are better; but not a measure of goodness-of-fit outside the context of model selection.
//!
//! # Examples
//!
//! ```rust
//! use polyfit::statistics::{r_squared, ScoringMethod};
//!
//! let y = vec![1.0, 2.0, 3.0];
//! let y_fit = vec![1.1, 1.9, 3.05];
//!
//! // Goodness-of-fit
//! let r2 = r_squared(y.iter().copied(), y_fit.iter().copied());
//! println!("R² = {r2}");
//!
//! // Model scoring
//! let score = ScoringMethod::AIC.calculate(y.into_iter(), y_fit.into_iter(), 3.0);
//! println!("AIC score = {score}");
//! ```
use crate::value::{IntClampedCast, Value};

/// Computes the residual variance of a model's predictions.
///
/// Residual variance is the unbiased estimate of the variance of the
/// errors (σ²) after fitting a model. It's used for confidence intervals
/// and covariance estimates of the fitted parameters.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// σ² = Σ (y_i - y_fit_i)² / (n - k)
/// where
///   y_i = observed values, y_fit_i = predicted values,
///   n = number of observations, k = number of model parameters
/// ```
/// </div>
///
/// # Parameters
/// - `y`: Iterator over the observed (actual) values.
/// - `y_fit`: Iterator over the predicted values from the model.
/// - `k`: Number of model parameters (degrees of freedom used by the fit).
///
/// # Returns
/// The residual variance as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::residual_variance;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![0.9, 2.1, 2.95];
/// let k = 2.0; // e.g., linear fit
/// let variance = residual_variance(y.into_iter(), y_fit.into_iter(), k);
/// ```
pub fn residual_variance<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
    k: T,
) -> T {
    let mut ss_total = T::zero();
    let mut n = T::zero();
    for (y, y_fit) in y.zip(y_fit) {
        ss_total += Value::powi(y - y_fit, 2);
        n += T::one();
    }

    if n == k {
        return T::zero();
    }
    ss_total / (n - k)
}

/// Calculate the R-squared value for a set of data.
///
/// R-squared is a number between 0 and 1 that tells you how well the model explains the data:
/// - `0` means the model explains none of the variation.
/// - `1` means the model explains all the variation.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// R-squared is calculated as:
/// ```math
/// R² = 1 - (SS_res / SS_tot)
/// where
///   SS_res = Σ (y_i - y_fit_i)²
///   SS_tot = Σ (y_i - y_mean)²
/// ```
/// </div>
///
/// # Parameters
/// - `y`: The actual (observed) values.
/// - `y_fit`: The predicted values from the model.
///
/// # Returns
/// The proportion of variance explained by the model.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::r_squared;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let r2 = r_squared(y.into_iter(), y_fit.into_iter());
/// ```
pub fn r_squared<T: Value>(y: impl Iterator<Item = T>, y_fit: impl Iterator<Item = T>) -> T {
    let (r2, _) = r_squared_with_n(y, y_fit);
    r2
}

/// Computes the arithmetic mean of a sequence of values.
///
/// This is the average value, calculated as the sum of all values divided by the count.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// Mean = (Σ x_i) / N
/// where
///   x_i = each value in the dataset, N = total number of values
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `data`: An iterator over values of type `T`.
///
/// # Returns
/// The arithmetic mean of all elements in `data`.
/// - Returns zero if the iterator yields no elements.
///
/// # Examples
/// ```rust
/// let values = vec![1.0, 2.0, 3.0];
/// let m = polyfit::statistics::mean(values.into_iter());
/// assert_eq!(m, 2.0);
/// ```
pub fn mean<T: Value>(data: impl Iterator<Item = T>) -> T {
    let mut sum = T::zero();
    let mut count = T::zero();
    for value in data {
        sum += value;
        count += T::one();
    }
    sum / count
}

/// Computes the standard deviation of a sequence of values.
/// - Uses the population formula (divides by `N`) rather than `N-1`.
///
/// The standard deviation measures the amount of variation or dispersion in a dataset. This function also returns the mean, for performance reasons.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// σ = sqrt( (Σ (x_i - Mean)²) / N )
/// where
///   x_i = each value in the dataset, Mean = arithmetic mean
///   N = total number of values
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `data`: An iterator over values of type `T`.
///
/// # Returns
/// The standard deviation of the elements in `data`.
///
/// Returns zero if the iterator yields no elements.
///
/// # Examples
/// ```rust
/// let values = vec![1.0, 2.0, 3.0];
/// let (s, _) = polyfit::statistics::stddev_and_mean(values.into_iter());
/// assert_eq!(s, 0.816496580927726); // sqrt(2/3)
/// ```
pub fn stddev_and_mean<T: Value>(data: impl Iterator<Item = T>) -> (T, T) {
    let data: Vec<_> = data.collect();
    let mean = mean(data.iter().copied());
    let mut sum_sq_diff = T::zero();
    let mut count = T::zero();
    for value in data {
        sum_sq_diff += Value::powi(value - mean, 2);
        count += T::one();
    }
    let dev = (sum_sq_diff / count).sqrt();

    (dev, mean)
}

/// Computes the skewness and excess kurtosis of a dataset.
///
/// Skewness measures the asymmetry of the distribution:
/// - Positive skew → tail to the right
/// - Negative skew → tail to the left
///
/// Kurtosis measures the "tailedness" of the distribution:
/// - Excess kurtosis = kurtosis - 3 (so that a normal distribution has excess kurtosis of 0)
/// - Positive excess kurtosis → heavier tails than normal
/// - Negative excess kurtosis → lighter tails than normal
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// Skewness = (Σ ((x_i - Mean) / StdDev)³) / N
/// Kurtosis = (Σ ((x_i - Mean) / StdDev)⁴) / N - 3
/// where
///   x_i = each value in the dataset, Mean = arithmetic mean
///   StdDev = standard deviation, N = total number of values
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `residuals`: A slice of values representing the dataset (e.g., residuals from a fit).
///
/// # Returns
/// A tuple `(skewness, kurtosis)`:
/// - `skewness`: Measure of asymmetry of the distribution.
///     - Positive → tail to the right, Negative → tail to the left.
/// - `kurtosis`: Excess kurtosis (kurtosis minus 3, so 0 for a normal distribution).
///
/// # Behavior
/// - If all values are identical (zero standard deviation), both skewness and kurtosis return `0`.
/// - Uses population formulas (divide by `N` rather than `N-1`).
///
/// # Examples
/// ```rust
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let (skew, kurt) = polyfit::statistics::skewness_and_kurtosis(&data);
/// println!("Skewness = {}, Kurtosis = {}", skew, kurt);
/// ```
pub fn skewness_and_kurtosis<T: Value>(residuals: &[T]) -> (T, T) {
    let (stddev, mean) = stddev_and_mean(residuals.iter().copied());

    // Trivial case
    if stddev == T::zero() {
        return (T::zero(), T::zero());
    }

    let mut skewness = T::zero();
    let mut kurtosis = T::zero();
    let mut n = T::zero();
    for &r in residuals {
        let diff = (r - mean) / stddev;
        skewness += Value::powi(diff, 3);
        kurtosis += Value::powi(diff, 4);
        n += T::one();
    }
    skewness /= n;
    kurtosis = kurtosis / n - (T::two() + T::one());

    (skewness, kurtosis)
}

/// Returns a score measuring if the residuals can be normally distributed.
/// - Normality refers to how closely the residuals follow a normal (Gaussian) distribution, and can check how well a model fits the data.
/// - Values near 0 are said to 'reject' normality, while values near 1 'do not reject' normality.
///   - In practice, values below 0.05 are often considered to reject normality.
///   - This means any value below 0.05 indicates the residuals are likely not normally distributed.
///   - Values above 0.05 do not guarantee normality, just that we do not have strong evidence to reject it.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// k² = (skewness / σ_skew)² + (kurtosis / σ_kurt)²
/// score = exp(-k² / 2)
/// where
///   σ_skew = √(6 / N), σ_kurt = √(24 / N)
///   N = number of residuals
/// ```
/// skewness and kurtosis as defined in [`skewness_and_kurtosis`]
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `residuals`: A slice of values representing the residuals from a model fit.
///
/// # Returns
/// A likelihood score between 0 and 1 indicating how likely the residuals are normally distributed.
///
/// # Panics
/// If T cannot hold 24, which would be silly
///
/// # Examples
/// ```rust
/// let residuals = vec![0.5, -0.2, 0.1, -0.4, 0.3];
/// let normality_score = polyfit::statistics::residual_normality(&residuals);
/// println!("Normality Score = {}", normality_score);
/// ```
pub fn residual_normality<T: Value>(residuals: &[T]) -> T {
    let six = T::try_cast(6.0).unwrap();
    let twentyfour = T::try_cast(24.0).unwrap();

    let mut n = T::zero();
    for _ in residuals {
        n += T::one();
    }

    if n == T::zero() {
        return T::one();
    }

    let (skew, kurt) = skewness_and_kurtosis(residuals);
    let se_skew = (six / n).sqrt();
    let se_kurt = (twentyfour / n).sqrt();

    let z_skew = skew / se_skew;
    let z_kurt = kurt / se_kurt;
    let k_squared = z_skew * z_skew + z_kurt * z_kurt;

    (-k_squared / T::two()).exp()
}

/// Computes the range (spread) of a dataset
/// - The spread is the difference between the maximum and minimum values.
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `data`: A slice of values.
///
/// # Returns
/// The difference `max - min` of all elements in `data`.
///
/// # Notes
/// - Returns `T::infinity() - T::neg_infinity()` if `data` is empty  
///
/// # Examples
/// ```rust
/// let values = vec![2.0, 5.0, 1.0, 9.0];
/// let r = polyfit::statistics::spread(values.iter().copied());
/// assert_eq!(r, 8.0); // 9 - 1
/// ```
pub fn spread<T: Value>(data: impl Iterator<Item = T>) -> T {
    let mut min = T::infinity();
    let mut max = T::neg_infinity();
    for value in data {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
    }
    max - min
}

/// Computes the adjusted R-squared value.
///
/// Adjusted R² accounts for the number of predictors in a model, penalizing
/// overly complex models. Use it to compare models of different degrees.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// R²_adj = R² - (1 - R²) * (k / (n - k))
/// where
///   n = number of observations, k = number of model parameters
/// ```
/// [`r_squared`] is used to compute R²
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over observed values.
/// - `y_fit`: Iterator over predicted values.
/// - `k`: Number of model parameters (usually `degree + 1`).
///
/// # Returns
/// The adjusted R² as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::adjusted_r_squared;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let k = 3.0;
/// let r2_adj = adjusted_r_squared(y.into_iter(), y_fit.into_iter(), k);
/// ```
pub fn adjusted_r_squared<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
    k: T,
) -> T {
    let (r2, n) = r_squared_with_n(y, y_fit);
    r2 - (T::one() - r2) * k / (n - k)
}

/// Internal performance implementation - returns (R², n)
fn r_squared_with_n<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> (T, T) {
    let y: Vec<T> = y.collect();
    let y_fit: Vec<T> = y_fit.collect();

    //
    // Mean of the canonical values
    let mut y_mean = T::zero();
    let mut y_n = T::zero();
    for y in &y {
        y_mean += *y;
        y_n += T::one();
    }
    y_mean /= y_n;

    //
    // Sum of (y_canonical - y_fit)^2
    // Sum of (y_canonical - y_mean)^2
    let mut ss_total = T::zero();
    let mut ss_residual = T::zero();
    for (y, y_fit) in y.into_iter().zip(y_fit) {
        ss_total += Value::powi(y - y_mean, 2);
        ss_residual += Value::powi(y - y_fit, 2);
    }

    (T::one() - (ss_residual / ss_total), y_n)
}

/// Computes the mean absolute error (MAE) between two sets of values.
///
/// MAE measures the average absolute difference between observed (`y`)
/// and predicted (`y_fit`) values. Lower values indicate a closer fit.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// MAE = (Σ |y_i - y_fit_i|) / N
/// where
///   y_i = observed values, y_fit_i = predicted values,
///   N = number of observations
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over observed values.
/// - `y_fit`: Iterator over predicted values.
///
/// # Returns
/// The mean absolute error as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::mean_absolute_error;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let mae = mean_absolute_error(y.into_iter(), y_fit.into_iter());
/// ```
pub fn mean_absolute_error<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> T {
    let mut total = T::zero();
    let mut n = T::zero();
    for (y, y_fit) in y.zip(y_fit) {
        total += Value::abs(y - y_fit);
        n += T::one();
    }
    total / n
}

/// Computes the root mean squared error (RMSE) between two sets of values.
///
/// RMSE is the square root of the mean squared error, giving the error
/// in the same units as the observed values. Lower values indicate a better fit.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// RMSE = √( (Σ (y_i - y_fit_i)²) / N )
/// where
///   y_i = observed values, y_fit_i = predicted values,
///   N = number of observations
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over observed values.
/// - `y_fit`: Iterator over predicted values.
///
/// # Returns
/// The root mean squared error as a `T`.
///
/// # Example
/// ```
/// # use polyfit::statistics::root_mean_squared_error;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let rmse = root_mean_squared_error(y.into_iter(), y_fit.into_iter());
/// ```
pub fn root_mean_squared_error<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> T {
    let (mse, _) = mse_with_n(y, y_fit);
    mse.sqrt()
}

/// Computes the mean squared error (MSE) between two sets of values.
///
/// For comparing 2 fits, prefer:
/// - [`ScoringMethod::BIC`] to prevent overfitting (fitting the noise instead of the real trend) or
/// - [`r_squared`] to evaluate goodness-of-fit.
///
/// MSE is a measure of the average squared difference between the
/// observed (`y`) and predicted (`y_fit`) values. Lower values indicate
/// a better fit.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// MSE = (Σ (y_i - y_fit_i)²) / N
/// where
///   y_i = observed values, y_fit_i = predicted values,
///   N = number of observations
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over the observed (actual) values.
/// - `y_fit`: Iterator over the predicted values from a model.
///
/// # Returns
/// The mean squared error as a `T`.
///
/// # Example
/// ```
/// # use polyfit::statistics::mean_squared_error;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let mse = mean_squared_error(y.into_iter(), y_fit.into_iter());
/// ```
pub fn mean_squared_error<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> T {
    let (mse, _) = mse_with_n(y, y_fit);
    mse
}

/// Computes the log-likelihood of the Huber loss for a set of data points.
/// - Huber loss is a robust error metric that is less sensitive to outliers than MSE.
/// - Higher values indicate a better fit.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// - Uses `1.345` as the Huber constant, which is standard for 95% efficiency with normally distributed errors.
/// - This is the value recommended by Peter J. Huber in his original paper "Robust Estimation of a Location Parameter" (1964).
/// ```math
/// delta = 1.345 * MAD
/// LogLikelihood = - (Σ huber_loss(y_i - y_fit_i, delta)) / N
/// huber_loss(r, delta) = { 0.5 * r²                    if |r| ≤ delta
///                        { delta * (|r| - 0.5 * delta) if |r| > delta
/// where
///   y_i = observed values, y_fit_i = predicted values,
///   N = number of observations
/// ```
/// [`median_absolute_deviation`] is used to compute MAD
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over observed values.
/// - `y_fit`: Iterator over predicted values.
///
/// # Returns
/// The Huber log-likelihood as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::huber_log_likelihood;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let ll = huber_log_likelihood(y.into_iter(), y_fit.into_iter());
/// ```
pub fn huber_log_likelihood<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> T {
    let (log_likelihood, _) = robust_mse_with_n(y, y_fit);
    log_likelihood
}

/// Internal impl for performance
///
/// Returns (mean squared error, count)
fn mse_with_n<T: Value>(y: impl Iterator<Item = T>, y_fit: impl Iterator<Item = T>) -> (T, T) {
    let mut total = T::zero();
    let mut n = T::zero();
    for (y, y_fit) in y.zip(y_fit) {
        total += Value::powi(y - y_fit, 2);
        n += T::one();
    }
    (total / n, n)
}

/// Internal impl of MSE-like log-likelihood using Huber loss
fn robust_mse_with_n<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> (T, T) {
    // This should only fail for very low precision types
    let huber_const = T::try_cast(1.345).unwrap_or(T::one());

    let y: Vec<_> = y.collect();
    let y_fit: Vec<_> = y_fit.collect();

    let mad = median_absolute_deviation(y.iter().copied(), y_fit.iter().copied());
    let delta = huber_const * mad;

    let mut total_loss = T::zero();
    let mut n = T::zero();
    for (yi, fi) in y.into_iter().zip(y_fit) {
        total_loss += huber_loss(yi - fi, delta);
        n += T::one();
    }

    (total_loss / n, n)
}

fn huber_loss<T: Value>(r: T, delta: T) -> T {
    let r = Value::abs(r);
    let half = T::one() / T::two();
    if r <= delta {
        half * r * r
    } else {
        delta * (r - half * delta)
    }
}

/// Computes the median absolute deviation (MAD) between two sets of values.
/// - MAD is a measure of variability that is robust to outliers.
/// - Lower values indicate a closer fit.
/// - Uses the median of the absolute deviations from the median.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// MAD = median( |y_i - y_fit_i| )
/// where
///   y_i = observed values, y_fit_i = predicted values
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over the observed (actual) values.
/// - `y_fit`: Iterator over the predicted values from a model.
///
/// # Returns
/// The median absolute deviation as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::median_absolute_deviation;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let mad = median_absolute_deviation(y.into_iter(), y_fit.into_iter());
/// ```
pub fn median_absolute_deviation<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> T {
    let mut residuals: Vec<T> = y.zip(y_fit).map(|(yi, fi)| Value::abs(yi - fi)).collect();
    let midpoint = residuals.len() / 2;
    if residuals.is_empty() {
        return T::zero();
    }

    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = residuals[midpoint];

    for r in &mut residuals {
        *r = Value::abs(*r - median);
    }
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    residuals[midpoint]
}

/// Methods for scoring a polynomial model to evaluate its fit.
///
/// These are used when automatically selecting the "best" polynomial degree.
///
/// # Selecting a method
/// - `AIC`: Picks a slightly more complex model if it fits better.
/// - `BIC`: Prefers simpler models, even if the fit is slightly worse.
///
/// - AIC is a good default for general use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringMethod {
    /// Akaike Information Criterion. Uses a more lenient penalty for model complexity
    /// - Picks a slightly more complex model if it fits better.
    ///
    /// This is the default choice for most applications.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// AIC is calculated as:
    /// ```math
    /// AIC = { n * ln(L) + 2k
    ///       { n * ln(L) + 2k + (2k(k+1)) / (n - k - 1)  if (n / k < 2 + 2) AND (n > k + 1)
    /// where
    ///   L = likelihood of the model (using Huber loss)
    ///   n = number of observations, k = number of model parameters
    /// ```
    /// </div>
    AIC,

    /// Bayesian Information Criterion. Uses a stricter penalty for model complexity.
    /// - Prefers simpler models, even if the fit is slightly worse.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// BIC is calculated as:
    /// ```math
    /// BIC = n * ln(L) + k * ln(n)
    /// where
    ///   L = likelihood of the model (using Huber loss)
    ///   n = number of observations, k = number of model parameters
    /// ```
    /// </div>
    BIC,
}
impl ScoringMethod {
    /// Calculate the model's score using this scoring method.
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// In place of MSE, which is susceptible to outliers, we use a robust loss function based on the
    /// Huber loss. This combines the ideas of MSE and MAE, providing a balance between the two.
    ///
    /// For more details, see the documentation for [`huber_log_likelihood`], [`ScoringMethod::AIC`], and [`ScoringMethod::BIC`].
    /// </div>
    ///
    /// # Notes
    /// - Lower scores indicate a "better" choice for automatically selecting the polynomial degree.
    /// - This is **not** a measure of how well the model fits your data. For that, use `r_squared`.
    ///
    /// # Type Parameters
    /// - `T`: A numeric type implementing the `Value` trait.
    ///
    /// # Parameters
    /// - `y`: Iterator over the observed (actual) values.
    /// - `y_fit`: Iterator over the predicted values from the model.
    /// - `k`: Number of model parameters (degrees of freedom used by the fit).
    ///
    /// # Returns
    /// The computed score as a `T`.
    ///
    /// # Errors
    /// Returns an error if the huber loss constant cannot be cast to the required type.
    ///
    /// # Example
    /// ```
    /// # use polyfit::{statistics::ScoringMethod, value::Value};
    /// # let y = vec![1.0, 2.0, 3.0];
    /// # let y_fit = vec![1.1, 1.9, 3.05];
    /// let score = ScoringMethod::AIC.calculate(y.into_iter(), y_fit.into_iter(), 3.0);
    /// ```
    pub fn calculate<T: Value>(
        self,
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
        k: T,
    ) -> T {
        let (log_likelihood, n) = robust_mse_with_n(y, y_fit);
        let log_likelihood = nalgebra::RealField::max(log_likelihood, T::epsilon());
        if n == T::zero() {
            return T::nan();
        }

        match self {
            ScoringMethod::AIC => {
                let mut aic = n * log_likelihood.ln() + T::two() * k;
                if n / k < T::two() + T::two() && n > k + T::one() {
                    // Apply AICc correction
                    aic += T::two() * k * (k + T::one()) / (n - k - T::one());
                }

                aic
            }
            ScoringMethod::BIC => n * log_likelihood.ln() + k * n.ln(),
        }
    }
}

/// In order to find the best fitting polynomial degree, we need to limit the maximum degree considered.
/// The choice of degree bound can significantly impact the model's performance and its ability to generalize.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// The maximum degree is chosen as the minimum of four constraints:
///
/// 1. Theoretical maximum for non-interpolating fits: `n - 1`, where `n` is the number of observations.
///
/// 2. A hard cap to prevent excessively high degrees:
/// - Conservative: 8
/// - Relaxed: 15
///
/// 3. Smoothness (`s`):
/// ```math
/// lim_smooth = n ^ (1 / (2s + 1))
/// where
///   s = assumed smoothness of the underlying function
///   n = number of observations
/// ```
///
/// 4. Observations per parameter:
/// ```math
/// lim_obs = (n / n_k_ratio_limit) - 1
/// where
///   n_k_ratio_limit = minimum required number of observations per coefficient
///   n = number of observations
/// ```
/// </div>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreeBound {
    /// Limits model complexity more aggressively, recommended for small datasets or when overfitting is a major concern.
    ///   - Assumes the data is smoother (s=2)
    ///   - Requires more observations per parameter (15)
    ///   - Lower hard cap (8)
    ///   - Hard cap reached when n ~= 32,000
    Conservative,

    /// Allows for higher complexity, useful when the underlying function may be more complex and the dataset is moderate in size.
    /// - Assumes the data is less 'smooth' (s=1)
    /// - Allows for fewer observations per parameter (8)
    /// - Higher hard cap (15)
    /// - Hard cap reached when n ~= 3,375
    Relaxed,

    /// User-specified maximum degree. Use only if you understand the implications for overfitting and numerical stability.
    Custom(usize),
}
impl From<usize> for DegreeBound {
    fn from(value: usize) -> Self {
        DegreeBound::Custom(value)
    }
}
impl DegreeBound {
    /// Computes the maximum polynomial degree to use for fitting based on the selected [`DegreeBound`]
    /// and the number of observations `n`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn max_degree(self, n: usize) -> usize {
        let theoretical_max = n.saturating_sub(1);
        match self {
            DegreeBound::Custom(d) => d.min(theoretical_max),
            DegreeBound::Conservative | DegreeBound::Relaxed => {
                let (hard_cap, max_n_per_k, est_smoothness) = match self {
                    DegreeBound::Conservative => (8, 15, 2),
                    DegreeBound::Relaxed => (15, 8, 1),
                    DegreeBound::Custom(_) => unreachable!(),
                };

                let smooth_lim = (n as f64)
                    .powf(1.0 / (2.0 * f64::from(est_smoothness) + 1.0))
                    .floor();

                let smooth_lim = smooth_lim as usize;

                let obs_lim = (n / max_n_per_k).saturating_sub(1);

                smooth_lim.min(obs_lim).min(hard_cap).min(theoretical_max)
            }
        }
    }
}

/// Standard Z-score confidence levels for fitted models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Confidence {
    /// 90% confidence level
    P90,

    /// 95% confidence level
    P95,

    /// 98% confidence level
    P98,

    /// 99% confidence level
    P99,

    /// Custom confidence level
    Custom(f64),
}

impl Confidence {
    #[allow(clippy::approx_constant)]
    #[rustfmt::skip]
    const T_TABLE: &[&[f64]] = &[
        /*  0.9     0.95    0.98,      0.99 */
        &[6.314,    12.71,  31.82,    63.66],
        &[2.92,     4.303,  6.965,    9.925],
        &[2.353,    3.182,  4.541,    5.841],
        &[2.132,    2.776,  3.747,    4.604],
        &[2.015,    2.571,  3.365,    4.032],
        &[1.943,    2.447,  3.143,    3.707],
        &[1.895,    2.365,  2.998,    3.499],
        &[1.86,     2.306,  2.896,    3.355],
        &[1.833,    2.262,  2.821,    3.25],
        &[1.812,    2.228,  2.764,    3.169],
        &[1.796,    2.201,  2.718,    3.106],
        &[1.782,    2.179,  2.681,    3.055],
        &[1.771,    2.16,   2.65,     3.012],
        &[1.761,    2.145,  2.624,    2.977],
        &[1.753,    2.131,  2.602,    2.947],
        &[1.746,    2.12,   2.583,    2.921],
        &[1.74,     2.11,   2.567,    2.898],
        &[1.734,    2.101,  2.552,    2.878],
        &[1.729,    2.093,  2.539,    2.861],
        &[1.725,    2.086,  2.528,    2.845],
        &[1.721,    2.08,   2.518,    2.831],
        &[1.717,    2.074,  2.508,    2.819],
        &[1.714,    2.069,  2.5,      2.807],
        &[1.711,    2.064,  2.492,    2.797],
        &[1.708,    2.06,   2.485,    2.787],
        &[1.706,    2.056,  2.479,    2.779],
        &[1.703,    2.052,  2.473,    2.771],
        &[1.701,    2.048,  2.467,    2.763],
        &[1.699,    2.045,  2.462,    2.756],
        &[1.697,    2.042,  2.457,    2.75],
    ];

    /// Returns the confidence level as a percentage.
    #[must_use]
    pub fn percentage(&self) -> f64 {
        match self {
            Confidence::P90 => 0.9,
            Confidence::P95 => 0.95,
            Confidence::P98 => 0.98,
            Confidence::P99 => 0.99,
            Confidence::Custom(z) => *z,
        }
    }

    /// Returns the alpha level (1 - confidence level).
    #[must_use]
    pub fn alpha(&self) -> f64 {
        1.0 - self.percentage()
    }

    /// Returns the T-score associated with the confidence level and degrees of freedom.
    ///
    /// For very large degrees of freedom, the T-score approaches the Z-score.
    /// Also approximates using T-score for custom confidence levels.
    #[must_use]
    pub fn t_score(self, df: usize) -> f64 {
        match df {
            0 => f64::INFINITY,
            i if i >= Self::T_TABLE.len() => self.z_score(),
            i => {
                let col = match self {
                    Confidence::P90 => 0,
                    Confidence::P95 => 1,
                    Confidence::P98 => 2,
                    Confidence::P99 => 3,
                    Confidence::Custom(_) => return self.z_score(),
                };

                Self::T_TABLE[i - 1][col]
            }
        }
    }

    /// Returns the Z-score associated with the confidence level.
    #[must_use]
    pub fn z_score(self) -> f64 {
        match self {
            Confidence::P90 => 1.645,
            Confidence::P95 => 1.960,
            Confidence::P98 => 2.326,
            Confidence::P99 => 2.576,
            Confidence::Custom(z) => z,
        }
    }

    /// Converts the Z-score to the numeric type `T`.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn try_cast<T: Value>(self) -> crate::error::Result<T> {
        T::try_cast(self.z_score())
    }
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Confidence::P90 => write!(f, "90%"),
            Confidence::P95 => write!(f, "95%"),
            Confidence::P98 => write!(f, "98%"),
            Confidence::P99 => write!(f, "99%"),
            Confidence::Custom(z) => write!(f, "{z}σ"),
        }
    }
}

/// Specifies a tolerance level for numerical comparisons.
///
/// Can be either an absolute unit value, or a percentage of the variance or standard deviation (depending on context).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance<T: Value> {
    /// An absolute tolerance value.
    Absolute(T),

    /// A relative tolerance value (percentage).
    Relative(T),
}

// A confidence band for a fitted model.
///
/// Represents a predicted range for model outputs at a given confidence level.
/// The band contains the central estimate (`value`) and the upper and lower bounds.
///
/// Created by a [`crate::CurveFit`]
///
/// # Type Parameters
/// - `T`: Numeric type that implements `Value` (e.g., `f64`, `f32`).
///
/// # Fields
/// - `level`: Confidence level (e.g., 95%) as a [`Confidence`] enum.
/// - `value`: Central predicted value of the model.
/// - `lower`: Lower bound of the confidence band.
/// - `upper`: Upper bound of the confidence band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceBand<T: Value> {
    pub(crate) level: Confidence,
    pub(crate) tolerance: Option<Tolerance<T>>,
    pub(crate) value: T,
    pub(crate) lower: T,
    pub(crate) upper: T,
}

impl<T: Value> ConfidenceBand<T> {
    /// Returns the tolerance used to compute the confidence band, if any.
    pub fn tolerance(&self) -> Option<Tolerance<T>> {
        self.tolerance
    }

    /// Returns the confidence level of the band.
    pub fn confidence(&self) -> Confidence {
        self.level
    }

    /// Returns the central predicted value of the model.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the lower bound of the confidence band.
    pub fn min(&self) -> T {
        self.lower
    }

    /// Returns the upper bound of the confidence band.
    pub fn max(&self) -> T {
        self.upper
    }

    /// Returns the midpoint of the confidence band.
    pub fn center(&self) -> T {
        (self.lower + self.upper) / T::two()
    }

    /// Returns the width of the confidence band (upper - lower).
    pub fn width(&self) -> T {
        self.upper - self.lower
    }
}

impl<T: Value> std::fmt::Display for ConfidenceBand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (min, y, max) = (self.min(), self.center(), self.max());
        let confidence = self.level;
        write!(f, "{y} ({min}, {max}) [confidence = {confidence}]")
    }
}

/// Normalizes values from one range to another.
///
/// Destination range use infinity to indicate unbounded ranges.
/// - `(f64::NEG_INFINITY, f64::INFINITY)` means no normalization.
#[derive(Debug, Clone, Copy)]
pub struct DomainNormalizer<T: Value> {
    src_range: (T, T),
    dst_range: (T, T),
}
impl<T: Value> DomainNormalizer<T> {
    /// Creates a new `DomainNormalizer` for the given source and destination ranges.
    pub fn new(src_range: (T, T), dst_range: (T, T)) -> Self {
        Self {
            src_range,
            dst_range,
        }
    }

    /// Creates a new `DomainNormalizer` from the range of the given data and a destination range.
    pub fn from_data(data: impl Iterator<Item = T>, dst_range: (T, T)) -> Self {
        let (min, max) = data.fold((T::infinity(), T::neg_infinity()), |(lo, hi), v| {
            (
                nalgebra::RealField::min(lo, v),
                nalgebra::RealField::max(hi, v),
            )
        });

        Self {
            src_range: (min, max),
            dst_range,
        }
    }

    /// Returns the source range of the normalizer.
    pub fn src_range(&self) -> (T, T) {
        self.src_range
    }

    /// Returns the destination range of the normalizer.
    pub fn dst_range(&self) -> (T, T) {
        self.dst_range
    }

    /// Normalizes a value from the source range to the destination range.
    pub fn normalize(&self, x: T) -> T {
        let (src_min, src_max) = self.src_range;
        let (dst_min, dst_max) = self.dst_range;

        if dst_min == T::neg_infinity() && dst_max == T::infinity() {
            return x;
        }

        if dst_min == T::neg_infinity() {
            // We have a maximum only
            // Adjust x by - src_max, then add dst_max
            return x - src_max + dst_max;
        }

        if dst_max == T::infinity() {
            // We have a minimum only
            // Adjust x by - src_min, then add dst_min
            return x - src_min + dst_min;
        }

        let value = dst_min + (x - src_min) * (dst_max - dst_min) / (src_max - src_min);
        nalgebra::RealField::clamp(value, dst_min, dst_max)
    }

    /// Denormalizes a value from the destination range back to the source range.
    pub fn denormalize(&self, x: T) -> T {
        let (src_min, src_max) = self.src_range;
        let (dst_min, dst_max) = self.dst_range;

        if dst_min == T::neg_infinity() && dst_max == T::infinity() {
            return x;
        }

        if dst_min == T::neg_infinity() {
            // We have a maximum only
            // Adjust x by - dst_max, then add src_max
            return x - dst_max + src_max;
        }

        if dst_max == T::infinity() {
            // We have a minimum only
            // Adjust x by - dst_min, then add src_min
            return x - dst_min + src_min;
        }

        let value = src_min + (x - dst_min) * (src_max - src_min) / (dst_max - dst_min);
        nalgebra::RealField::clamp(value, src_min, src_max)
    }

    /// Denormalizes a slice of polynomial coefficients from the destination range back to the source range.
    ///
    /// The coefficients are assumed to be in ascending order (constant term first).
    #[must_use]
    pub fn denormalize_coefs(&self, coefs: &[T]) -> Vec<T> {
        let (x_min, x_max) = self.src_range();
        let (d_min, d_max) = self.dst_range();
        let (alpha, beta) = if d_min == T::neg_infinity() && d_max == T::infinity() {
            (T::one(), T::zero()) // no change
        } else if d_min == T::neg_infinity() {
            // We have a maximum only
            // Adjust x by - src_max, then add dst_max
            let beta = d_max - x_max;
            (T::one(), beta)
        } else if d_max == T::infinity() {
            // We have a minimum only... shift only
            let beta = d_min - x_min;
            (T::one(), beta)
        } else {
            let alpha = (d_max - d_min) / (x_max - x_min);
            let beta = d_min - alpha * x_min;
            (alpha, beta)
        };

        let mut unnorm = vec![T::zero(); coefs.len()];
        for (i, &c) in coefs.iter().enumerate() {
            for j in 0..=i {
                let binom = T::factorial(i) / (T::factorial(j) * T::factorial(i - j));
                unnorm[j] += c
                    * binom
                    * Value::powi(alpha, j.clamped_cast())
                    * Value::powi(beta, (i - j).clamped_cast());
            }
        }
        unnorm
    }
}
impl<T: Value> std::fmt::Display for DomainNormalizer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (src_min, src_max) = self.src_range;
        let (dst_min, dst_max) = self.dst_range;

        let dst_min = if dst_min == T::neg_infinity() {
            "-∞".to_string()
        } else {
            dst_min.to_string()
        };

        let dst_max = if dst_max == T::infinity() {
            "∞".to_string()
        } else {
            dst_max.to_string()
        };

        write!(f, "T[ {src_min}..{src_max} -> {dst_min}..{dst_max} ]")
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn residual_variance_zero_error() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 1.0);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn residual_variance_simple_case() {
        // y = [1, 2], y_fit = [0, 0], k = 1
        // errors: [1, 2], squared = [1, 4], sum = 5
        // n=2, n-k=1, variance = 5
        let y = vec![1.0, 2.0];
        let y_fit = vec![0.0, 0.0];
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 1.0);
        assert_eq!(var, 5.0);
    }

    #[test]
    fn residual_variance_mse_equivalence() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![0.0, 0.0, 0.0];
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 0.0);

        // manual MSE: (1² + 2² + 3²) / 3 = 14/3
        assert!((var - 14.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn residual_variance_invalid_degrees_of_freedom() {
        let y = vec![1.0, 2.0];
        let y_fit = vec![1.0, 2.0];
        // n = 2, k = 2 → division by zero
        let var = residual_variance::<f64>(y.into_iter(), y_fit.into_iter(), 2.0);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn r_squared_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(r2, 1.0);
    }

    #[test]
    fn r_squared_bad_fit() {
        // y = [1, 2, 3], y_fit = [2, 2, 2]
        // mean(y) = 2
        // SST = (1-2)² + (2-2)² + (3-2)² = 2
        // SSE = same, 2
        // R² = 1 - SSE/SST = 0
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn r_squared_partial_fit() {
        // y = [1, 2, 3], y_fit = [0, 2, 4]
        // mean(y) = 2
        // SST = (1-2)² + (2-2)² + (3-2)² = 2
        // SSE = (1-0)² + (2-2)² + (3-4)² = 2
        // R² = 0
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![0.0, 2.0, 4.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn r_squared_negative_case() {
        // y = [1, 2, 3], y_fit = [10, 10, 10]
        // mean(y) = 2, SST = 2
        // SSE = (1-10)² + (2-10)² + (3-10)² = 9² + 8² + 7² = 194
        // R² = 1 - 194/2 = -96
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![10.0, 10.0, 10.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(r2, -96.0);
    }

    #[test]
    fn r_squared_constant_y() {
        let y = vec![2.0, 2.0, 2.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let r2 = r_squared::<f64>(y.into_iter(), y_fit.into_iter());
        assert!(r2.is_nan()); // or adjust depending on intended behavior
    }

    #[test]
    fn adjusted_r_squared_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = y.clone(); // perfect fit
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 2.0);
        assert_eq!(r2_adj, 1.0);
    }

    #[test]
    fn adjusted_r_squared_equals_r2_when_k1() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y_fit = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let r2 = r_squared(y.clone().into_iter(), y_fit.clone().into_iter());
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 1.0);
        assert!((r2 - r2_adj).abs() < 1e-12);
    }

    #[test]
    fn adjusted_r_squared_penalizes_complexity() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = vec![1.1, 1.9, 3.2, 3.8]; // decent fit
        let r2 = r_squared(y.clone().into_iter(), y_fit.clone().into_iter());
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 3.0);
        assert!(r2_adj < r2);
    }

    #[test]
    fn adjusted_r_squared_negative_case() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = vec![10.0, 10.0, 10.0, 10.0]; // terrible fit
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 2.0);
        assert!(r2_adj < 0.0);
    }

    #[test]
    fn adjusted_r_squared_invalid_degrees_of_freedom() {
        let y = vec![1.0, 2.0];
        let y_fit = vec![1.0, 2.0];
        // n = 2, k = 2 → division by zero
        let r2_adj = adjusted_r_squared::<f64>(y.into_iter(), y_fit.into_iter(), 2.0);
        assert!(r2_adj.is_nan());
    }

    #[test]
    fn mae_zero_error() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(mae, 0.0);
    }

    #[test]
    fn mae_simple_case() {
        // y = [1, 2, 3], y_fit = [2, 2, 2]
        // abs diffs = [1, 0, 1], sum = 2, mean = 2/3
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert!((mae - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mae_symmetric() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mae1 = mean_absolute_error::<f64>(y.clone().into_iter(), y_fit.clone().into_iter());
        let mae2 = mean_absolute_error::<f64>(y_fit.into_iter(), y.into_iter());
        assert_eq!(mae1, mae2);
    }

    #[test]
    fn mae_with_negatives() {
        // y = [-1, -2], y_fit = [1, 2]
        // diffs = [2, 4], mean = 3
        let y = vec![-1.0, -2.0];
        let y_fit = vec![1.0, 2.0];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(mae, 3.0);
    }

    #[test]
    fn mae_empty_input() {
        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let mae = mean_absolute_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert!(mae.is_nan());
    }

    #[test]
    fn mse_zero_error() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.0, 2.0, 3.0];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(mse, 0.0);
    }

    #[test]
    fn mse_simple_case() {
        // y = [1, 2, 3], y_fit = [2, 2, 2]
        // squared diffs = [1, 0, 1], sum = 2, mean = 2/3
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert!((mse - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn mse_with_negatives() {
        // y = [-1, -2], y_fit = [1, 2]
        // diffs = [-2, -4], squared = [4, 16], mean = 10
        let y = vec![-1.0, -2.0];
        let y_fit = vec![1.0, 2.0];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert_eq!(mse, 10.0);
    }

    #[test]
    fn mse_symmetric() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![2.0, 2.0, 2.0];
        let mse1 = mean_squared_error::<f64>(y.clone().into_iter(), y_fit.clone().into_iter());
        let mse2 = mean_squared_error::<f64>(y_fit.into_iter(), y.into_iter());
        assert_eq!(mse1, mse2);
    }

    #[test]
    fn mse_empty_input_returns_nan() {
        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let mse = mean_squared_error::<f64>(y.into_iter(), y_fit.into_iter());
        assert!(mse.is_nan());
    }

    #[test]
    fn scoring_perfect_fit() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let y_fit = y.clone();
        let k = 2.0;
        let aic = ScoringMethod::AIC.calculate::<f64>(
            y.clone().into_iter(),
            y_fit.clone().into_iter(),
            k,
        );
        let bic = ScoringMethod::BIC.calculate::<f64>(y.into_iter(), y_fit.into_iter(), k);
        assert!(aic.is_finite());
        assert!(bic.is_finite());
    }

    #[test]
    fn scoring_aicc_correction() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.8, 2.7, 3.6];
        let k = 2.0; // n/k = 1.5 < 4 → triggers correction
        let score = ScoringMethod::AIC.calculate::<f64>(y.into_iter(), y_fit.into_iter(), k);
        assert!(score.is_finite());
    }

    #[test]
    fn scoring_higher_error_higher_score() {
        let y = vec![1.0, 2.0, 3.0];
        let y_fit_good = vec![1.0, 2.0, 3.0];
        let y_fit_bad = vec![0.0, 0.0, 0.0];
        let k = 2.0;
        let score_good =
            ScoringMethod::BIC.calculate::<f64>(y.clone().into_iter(), y_fit_good.into_iter(), k);
        let score_bad =
            ScoringMethod::BIC.calculate::<f64>(y.into_iter(), y_fit_bad.into_iter(), k);
        assert!(score_bad > score_good);
    }
    #[test]
    fn scoring_empty_input_returns_nan() {
        let y: Vec<f64> = vec![];
        let y_fit: Vec<f64> = vec![];
        let k = 2.0;
        let aic = ScoringMethod::AIC.calculate::<f64>(
            y.clone().into_iter(),
            y_fit.clone().into_iter(),
            k,
        );
        let bic = ScoringMethod::BIC.calculate::<f64>(y.into_iter(), y_fit.into_iter(), k);
        assert!(aic.is_nan());
        assert!(bic.is_nan());
    }

    #[test]
    fn scoring_bic_stricter_than_aic() {
        let y = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
        ];
        let y_fit = vec![
            1.1, 2.1, 2.9, 4.0, 5.05, 1.1, 2.1, 2.9, 4.0, 5.05, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0,
            3.0, 4.0, 5.0,
        ];
        let k = 3.0;
        let aic = ScoringMethod::AIC.calculate::<f64>(
            y.clone().into_iter(),
            y_fit.clone().into_iter(),
            k,
        );
        let bic = ScoringMethod::BIC.calculate::<f64>(y.into_iter(), y_fit.into_iter(), k);

        assert!(bic >= aic);
    }
}
