use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    error::Error,
    score::{Bic, ModelScoreProvider},
    statistics::{
        accumulator::{
            HuberLogLikelihoodAccumulator, MeanAbsoluteErrorAccumulator, MeanAccumulator,
            MeanSquaredErrorAccumulator, NormalityAccumulator, RSquaredAccumulator,
        },
        UncertainValue,
    },
    value::{CoordExt, Value},
};

//
// Basic metrics
//

/// Computes the standard deviation and mean of a sequence of values.
///
/// Returns both values since the mean is not typically useful without the standard deviation:
/// - The mean gives a central value, but the standard deviation indicates how much the values vary around that mean.
/// - A low standard deviation means the values are close to the mean, while a high standard deviation indicates more spread.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// Uses Welford's online algorithm for numerical stability, which allows computing the mean and standard deviation in a single pass
///
/// ```math
/// mean = sum / n
/// var = sum_sq / n - mean²
/// stddev = sqrt(var)
///
/// where
///     sum = Σ x_i
///     sum_sq = Σ x_i²
///     n = number of elements
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
/// Returns `None` if the iterator yields no elements.
///
/// # Examples
/// ```rust
/// let values = vec![1.0, 2.0, 3.0];
/// let (s, _) = polyfit::statistics::stddev_and_mean(values.into_iter());
/// assert_eq!(s, 0.816496580927726); // sqrt(2/3)
/// ```
pub fn mean<T: Value>(data: impl Iterator<Item = T>) -> Option<UncertainValue<T>> {
    let acc: MeanAccumulator<T> = data.collect();
    acc.mean()
}

/// Computes the median of a sequence of values.
///
/// The median is the middle value when the data is sorted.
///
/// Note that this will reorder the input slice, but it does not require a full sort, and is O(n) on average.
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `data`: A slice of values of type `T`.
///
/// # Returns
/// The median value of the elements in `data`.
#[must_use]
pub fn median<T: Value>(data: &mut [T]) -> Option<T> {
    if data.is_empty() {
        return None;
    }

    let n = data.len();
    let i = n / 2;
    let (l, mid, _) = data.select_nth_unstable_by(i, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });

    if n % 2 == 0 {
        // Even number of elements, so n/2 is actually the upper middle
        let lower_mid = l
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        Some((*lower_mid + *mid) / T::two())
    } else {
        Some(*mid)
    }
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
/// - Returns 0 if `data` is empty
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
    let mut n = 0;
    for value in data {
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
        n += 1;
    }

    if n == 0 {
        return T::zero();
    }

    max - min
}

//
// Deviation metrics
//

/// Computes the median absolute deviation (MAD) between two sets of values.
/// - MAD is a measure of variability that is robust to outliers.
/// - Lower values indicate a closer fit.
/// - Uses the median of the absolute deviations from the median.
///
/// Note that this function will allocate in order to compute the median
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// median_residual = median( |y_i - y_fit_i| )
/// MAD = median( |y_i - median_residual| )
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
/// The median absolute deviation as a `T`, or `None` if there are no values to compare.
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
) -> Option<T> {
    let mut residuals: Vec<T> = y.zip(y_fit).map(|(yi, fi)| Value::abs(yi - fi)).collect();
    let median_r = median(&mut residuals)?;

    for r in &mut residuals {
        *r = Value::abs(*r - median_r);
    }

    median(&mut residuals)
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
/// The mean absolute error as a `T`, or `None` if there are no values to compare.
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
) -> Option<T> {
    let acc: MeanAbsoluteErrorAccumulator<T> = y.zip(y_fit).collect();
    acc.mean_absolute_error()
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
/// The root mean squared error as a `T`, or `None` if there are no values to compare.
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
) -> Option<T> {
    let acc: MeanSquaredErrorAccumulator<T> = y.zip(y_fit).collect();
    acc.mean_squared_error().map(T::sqrt)
}

/// Computes the mean squared error (MSE) between two sets of values.
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
/// The mean squared error as a `T`, or `None` if there are no values to compare.
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
) -> Option<T> {
    let acc: MeanSquaredErrorAccumulator<T> = y.zip(y_fit).collect();
    acc.mean_squared_error()
}

//
// R squared variants
//

/// Computes the adjusted R-squared value.
///
/// Adjusted R² accounts for the number of predictors in a model, penalizing
/// overly complex models. Use it to compare models of different degrees.
///
/// Returns `None` if there are not enough data points to compute the adjusted R² (i.e., when n <= k).
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
) -> Option<T> {
    let acc: RSquaredAccumulator<T> = y.zip(y_fit).collect();
    let r2 = acc.r_squared()?;
    let n = acc.count();
    if n <= k {
        return None; // Not enough data points to compute adjusted R²
    }

    Some(r2 - (T::one() - r2) * k / (n - k))
}

/// Calculate the R-squared value for a set of data.
///
/// R-squared is a number between 0 and 1 that tells you how well the model explains the data:
/// - `0` means the model explains none of the variation.
/// - `1` means the model explains all the variation.
///
/// Returns `None` if there are no data points (i.e., when the count is zero).
///
/// If there is no variance in the observed values (i.e., all `y_i` are the same), then we return:
/// - 1 if the predicted values are exactly the same as the observed values (perfect fit)
/// - -inf if the predicted values are not the same as the observed values (since the model does not explain any of the variance)
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
pub fn r_squared<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> Option<T> {
    let acc: RSquaredAccumulator<T> = y.zip(y_fit).collect();
    acc.r_squared()
}

/// Uses huber loss to compute a robust R-squared value.
/// - More robust to outliers than traditional R².
/// - Values closer to 1 indicate a better fit.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// R²_robust = 1 - (Σ huber_loss(y_i - y_fit_i, delta)) / (Σ (y_i - y_fit_i)²)
/// where
///   huber_loss(r, delta) = { 0.5 * r²                    if |r| ≤ delta
///                          { delta * (|r| - 0.5 * delta) if |r| > delta
///  delta = 1.345 * MAD
///  MAD = median( |y_i - y_fit_i| )
///  where
///    y_i = observed values, y_fit_i = predicted values
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
/// The robust R² as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::robust_r_squared;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let r2_robust = robust_r_squared(y.into_iter(), y_fit.into_iter());
/// ```
pub fn robust_r_squared<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> Option<T> {
    let y: Vec<_> = y.collect();
    let y_fit: Vec<_> = y_fit.collect();

    let mad = median_absolute_deviation(y.iter().copied(), y_fit.iter().copied())?;
    let mut acc = HuberLogLikelihoodAccumulator::new(mad, None);

    //
    // Get:
    //    TSE - Σ huber_loss(y_i - y_fit_i, delta))
    //    TSS - Σ (y_i - y_fit_i)²
    let mut tss = T::zero();
    for (&y, &y_fit) in y.iter().zip(y_fit.iter()) {
        acc.add(y, y_fit);
        tss += Value::powi(y - y_fit, 2);
    }

    let tse = acc.log_likelihood_sum()?;
    Some(T::one() - (tse / tss))
}

//
// Normality metrics
//

pub use crate::statistics::accumulator::Normality;

/// Returns a set of metrics indicating if the set of values can be normally distributed.
/// - Normality refers to how closely the set of values follow a normal (Gaussian) distribution, and can check how well a model fits the data.
/// - Likelihoods near 0 are said to 'reject' normality, while likelihoods near 1 'do not reject' normality.
///   - In practice, values below 0.05 are often considered to reject normality.
///   - This means any value below 0.05 indicates the set of values are likely not normally distributed.
///   - Values above 0.05 do not guarantee normality, just that we do not have strong evidence to reject it.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// Uses an extension of the Knuth and Welford online algorithm to compute skewness and kurtosis in a single pass
///
/// A normality score is then computed based on the skewness and kurtosis, using an estimate of the D'Agostino's K² test using a normal approximation
///
/// See <https://www.johndcook.com/blog/skewness_kurtosis/#:~:text=Computing%20skewness%20and%20kurtosis%20in,Kurtosis> for the algorithm used for skewness and kurtosis
///
/// The likelihood is computed as:
/// ```math
/// se_skew = sqrt(6 / n)
/// se_kurt = sqrt(24 / n)
/// z_skew = skewness / se_skew
/// z_kurt = kurtosis / se_kurt
/// K² = z_skew² + z_kurt²
/// p_value = e^(-K² / 2)
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
/// A `Normality` struct containing:
/// - `likelihood`: A score between 0 and 1 indicating how likely the set of values are normally distributed.
/// - `skewness`: Measure of asymmetry of the distribution.
/// - `kurtosis`: Excess kurtosis (kurtosis minus 3, so 0 for a normal distribution).
/// - `mean`: The mean of the set of values.
/// - `variance`: The variance of the set of values.
/// - `standard_deviation`: The standard deviation of the set of values.
/// - `count`: The number of values.
///
/// Returns `None` if there are no values (i.e., when the count is zero).
///
/// # Examples
/// ```rust
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let normality_score = polyfit::statistics::residual_normality(y.into_iter(), y_fit.into_iter());
/// println!("Normality Score = {}", normality_score.likelihood);
/// ```
pub fn normality<T: Value>(values: impl Iterator<Item = T>) -> Option<Normality<T>> {
    let acc: NormalityAccumulator<T> = values.collect();
    acc.normality()
}

/// Returns a set of metrics indicating if the residuals can be normally distributed.
/// - Normality refers to how closely the residuals follow a normal (Gaussian) distribution, and can check how well a model fits the data.
/// - Likelihoods near 0 are said to 'reject' normality, while likelihoods near 1 'do not reject' normality.
///   - In practice, values below 0.05 are often considered to reject normality.
///   - This means any value below 0.05 indicates the residuals are likely not normally distributed.
///   - Values above 0.05 do not guarantee normality, just that we do not have strong evidence to reject it.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// Uses an extension of the Knuth and Welford online algorithm to compute skewness and kurtosis in a single pass
///
/// A normality score is then computed based on the skewness and kurtosis, using an estimate of the D'Agostino's K² test using a normal approximation
///
/// See <https://www.johndcook.com/blog/skewness_kurtosis/#:~:text=Computing%20skewness%20and%20kurtosis%20in,Kurtosis> for the algorithm used for skewness and kurtosis
///
/// The likelihood is computed as:
/// ```math
/// se_skew = sqrt(6 / n)
/// se_kurt = sqrt(24 / n)
/// z_skew = skewness / se_skew
/// z_kurt = kurtosis / se_kurt
/// K² = z_skew² + z_kurt²
/// p_value = e^(-K² / 2)
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
/// A `Normality` struct containing:
/// - `likelihood`: A score between 0 and 1 indicating how likely the residuals are normally distributed.
/// - `skewness`: Measure of asymmetry of the distribution.
/// - `kurtosis`: Excess kurtosis (kurtosis minus 3, so 0 for a normal distribution).
/// - `mean`: The mean of the residuals.
/// - `variance`: The variance of the residuals.
/// - `standard_deviation`: The standard deviation of the residuals.
/// - `count`: The number of residuals.
///
/// Returns `None` if there are no residuals (i.e., when the count is zero).
///
/// # Examples
/// ```rust
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let normality_score = polyfit::statistics::residual_normality(y.into_iter(), y_fit.into_iter());
/// println!("Normality Score = {}", normality_score.likelihood);
/// ```
pub fn residual_normality<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> Option<Normality<T>> {
    let acc: NormalityAccumulator<T> = y.zip(y_fit).map(|(y, y_fit)| y - y_fit).collect();
    acc.normality()
}

/// Computes the residual variance of a model's predictions.
///
/// Residual variance is the unbiased estimate of the variance of the
/// errors (σ²) after fitting a model. It's used for confidence intervals
/// and covariance estimates of the fitted parameters.
///
/// This means (eli5) that it gives us an estimate of how much the predicted values deviate from the actual values,
/// adjusted for the number of parameters in the model. A lower residual variance indicates a better fit,
/// while a higher residual variance suggests that the model's predictions are more spread out from the actual values.
///
/// if n <= k, we return `None` because we cannot compute the residual variance (division by zero or negative degrees of freedom).
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
) -> Option<T> {
    let acc: MeanSquaredErrorAccumulator<T> = y.zip(y_fit).collect();
    if acc.count() <= k {
        return None; // Not enough data points to compute residual variance
    }

    let ss_total = acc.mean_squared_error_sum();
    Some(ss_total / (acc.count() - k))
}

//
// Huber loss core
//

/// Computes the log-likelihood of the Huber loss for a set of data points.
/// - Huber loss is a robust error metric that is less sensitive to outliers than MSE.
/// - Higher values indicate a better fit.
///
/// Note that this function will allocate in order to compute the median absolute deviation (MAD) used for the Huber loss calculation.
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
) -> Option<T> {
    let y: Vec<_> = y.collect();
    let y_fit: Vec<_> = y_fit.collect();

    let mad = median_absolute_deviation(y.iter().copied(), y_fit.iter().copied())?;
    let mut acc = HuberLogLikelihoodAccumulator::new(mad, None);
    acc.add_iter(y.into_iter(), y_fit.into_iter());

    acc.log_likelihood()
}

//
// Model aware metrics
//

/// Computes the Bayes factor between two models.
///
/// The result indicates how much more likely the data is under one model compared to the other:
/// - < 1.0 → Model 2 is favored
/// - 1.0 → Both models are equally likely
/// - 1.0 to 3.0 → Weak evidence for Model 1
/// - 3.0 to 10.0 → Moderate evidence for Model 1
/// - > 10.0 → Strong evidence for Model 1
///
/// Essentially, the Bayes factor quantifies how well each model explains the data, while penalizing for model complexity (number of parameters).
///
/// # Errors
/// Returns `Err` if either model fails to solve for the given data, indicating the supplied data is out of the model's domain
pub fn bayes_factor<T: Value, B1, B2>(
    m1: &crate::CurveFit<B1, T>,
    m2: &crate::CurveFit<B2, T>,
    data: &[(T, T)],
) -> Result<T, Error>
where
    B1: Basis<T> + PolynomialDisplay<T>,
    B2: Basis<T> + PolynomialDisplay<T>,
{
    let y1 = m1.solve(data.x_iter())?.y();
    let y2 = m2.solve(data.x_iter())?.y();

    let k1 = T::from_positive_int(m1.coefficients().len());
    let bic1 = Bic
        .score(m1, y1.into_iter(), data.y_iter(), k1)
        .ok_or(Error::NoData)?;

    let k2 = T::from_positive_int(m2.coefficients().len());
    let bic2 = Bic
        .score(m2, y2.into_iter(), data.y_iter(), k2)
        .ok_or(Error::NoData)?;

    Ok(((bic2 - bic1) / T::two()).exp())
}
