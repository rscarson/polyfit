use crate::{
    statistics::{
        accumulator::{Accumulator, DefaultAccumulatorFactory, ExecutionRouter, FittedValue},
        MeanAccumulator, MeanSquaredErrorAccumulator,
    },
    value::Value,
};

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
    match ExecutionRouter::route(values) {
        ExecutionRouter::Eager(data) => NormalityAccumulator::eager_total(data),
        ExecutionRouter::Streaming(data) => NormalityAccumulator::new_from_iter(data).total(),
    }
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
    let residuals = y.zip(y_fit).map(|(y, y_fit)| y - y_fit);
    NormalityAccumulator::new_from_iter(residuals).total()
}

/// Computes the residual variance of a model's predictions.
///
/// Residual variance is the unbiased estimate of the variance of the
/// errors (σ²) after fitting a model. It's used for confidence intervals
/// and covariance estimates of the fitted parameters.
///
/// This means that it gives us an estimate of how much the predicted values deviate from the actual values,
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
    let values = FittedValue::from_iters(y, y_fit);
    match ExecutionRouter::route(values) {
        ExecutionRouter::Eager(data) => {
            let mut count = T::zero();
            let mut sum_sq_error = T::zero();

            for value in data {
                count += T::one();
                let res = value.residual();
                sum_sq_error += res * res; // Squaring removes the sign and fits FMA registers
            }

            if count <= k {
                return None; // Not enough degrees of freedom
            }

            Some(sum_sq_error / (count - k))
        }

        ExecutionRouter::Streaming(data) => {
            // For large datasets, we can use a more efficient algorithm that uses less memory
            let acc = MeanSquaredErrorAccumulator::new_from_iter(data);
            if acc.count <= k {
                return None; // Not enough data points to compute residual variance
            }
            let ss_total = acc.sum_sq_error.total();
            Some(ss_total / (acc.count - k))
        }
    }
}

/// Struct representing the normality of a set of residuals. This is a measure of how closely the residuals follow a normal distribution.
/// In practice, this can detect if a given fit is missing structured elements, or if the errors made by the fit are random gaussian noise.
///
/// The most important metric is the `likelihood`, which gives a measure of how likely it is that the residuals are normally distributed.
///
/// High likelihood values indicate that the residuals are likely to be normally distributed, but in practice even a score of 0.05 (5% likelihood)
/// typically indicates decent evidence of normality.
///
/// It is important that this is not the probability of the residuals being normally distributed,
/// but rather the likelihood that you can reject the null hypothesis of normality (i.e. that the residuals are not normally distributed).
///
/// So a likelihood of 0.3 means that there is 30% chance that can reject the idea that the residuals are not normally distributed
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normality<T: Value> {
    /// The likelihood that the residuals are normally distributed, based on D'Agostino's K-squared test.
    ///
    /// This is a value between 0 and 1, where higher values indicate a higher likelihood of normality.
    ///
    /// In practice, a likelihood of 0.05 (5%) or higher is often considered evidence of normality, but this can depend on the context and specific requirements
    pub likelihood: T,

    /// The skewness of the residuals, which measures the asymmetry of the distribution.
    /// A skewness of 0 indicates a perfectly symmetric distribution, while positive or negative values indicate skewness to the right or left, respectively.
    ///
    /// This means that if you plotted the residuals, a positive skewness would be visibly skewed to the right, while a negative skewness would be visibly skewed to the left.
    pub skewness: T,

    /// The excess kurtosis of the residuals, which measures the "tailedness" of the distribution.
    /// A kurtosis of 0 indicates a normal distribution, while positive values indicate heavier tails.
    ///
    /// In this context a 'tail' refers to the presence of outliers in the distribution. A high kurtosis indicates that there are more extreme values (outliers) than
    /// would be expected in a normal distribution, while a low kurtosis indicates that there are fewer extreme values than expected.
    pub kurtosis: T,

    /// The mean of the residuals, which should ideally be close to zero for a good fit.
    pub mean: T,

    /// The standard deviation of the residuals, which is the square root of the variance and gives a measure of the typical size of the residuals.
    pub standard_deviation: T,

    /// The number of residuals that were used to calculate the normality metrics.
    ///
    /// Important for understanding the reliability of the normality metrics, as a small number of residuals can lead to unreliable estimates of normality.
    pub count: T,
}

/// Accumulator for calculating normality metrics (skewness, kurtosis, mean, variance) in a single pass through the data
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NormalityAccumulator<T: Value> {
    pub count: T,
    pub mean: T,
    pub m2: T,
    pub m3: T,
    pub m4: T,
}
impl<T: Value> Default for NormalityAccumulator<T> {
    fn default() -> Self {
        NormalityAccumulator {
            count: T::zero(),
            mean: T::zero(),
            m2: T::zero(),
            m3: T::zero(),
            m4: T::zero(),
        }
    }
}
impl<T: Value> NormalityAccumulator<T> {
    /// Calculates the likelihood of normality using D'Agostino's K-squared test, given the skewness, kurtosis, and count of the residuals.
    pub fn k_squared_likelihood(skewness: T, kurtosis: T, count: T) -> T {
        // The standard error of skewness and kurtosis can be calculated using the formulas:
        // SE_skewness = sqrt(6/n), SE_kurtosis = sqrt(24/n), where n is the sample size (count).
        let se_skew = (T::from_positive_int(6) / count).sqrt();
        let se_kurt = (T::from_positive_int(24) / count).sqrt();

        // Now we can calculate the z-scores for skewness and kurtosis:
        // z_skew = skewness / SE_skewness, z_kurt = kurtosis / SE_kurtosis
        let z_skew = skewness / se_skew;
        let z_kurt = kurtosis / se_kurt;

        // Finally, we can calculate the combined K-squared statistic:
        // K^2 = z_skew^2 + z_kurt^2
        // p = 1 - CDF(K^2, 2), which has the close form p = exp(-K^2/2) for 2 degrees of freedom
        let k_squared = z_skew * z_skew + z_kurt * z_kurt;
        (-k_squared / T::two()).exp()
    }
}
impl<T: Value> DefaultAccumulatorFactory for NormalityAccumulator<T> {}
impl<T: Value> Accumulator<T, T, Normality<T>> for NormalityAccumulator<T> {
    fn add(&mut self, x: T) {
        let three = T::from_positive_int(3);
        let four = T::from_positive_int(4);
        let six = T::from_positive_int(6);

        let n1 = self.count + T::one();
        let delta = x - self.mean;
        let delta_n = delta / n1; // Streaming hardware division penalty
        let delta_n2 = delta_n * delta_n;
        let term = delta * delta_n * self.count;

        self.mean += delta_n;

        let m3_old = self.m3;
        let m2_old = self.m2;

        self.m4 += term * delta_n2 * (n1 * n1 - three * n1 + three) + six * delta_n2 * m2_old
            - four * delta_n * m3_old;

        self.m3 += term * delta_n * (n1 - T::two()) - three * delta_n * m2_old;

        self.m2 += term;
        self.count = n1;
    }

    fn merge(&mut self, other: Self) {
        let three = T::from_positive_int(3);
        let four = T::from_positive_int(4);
        let six = T::from_positive_int(6);

        let n1 = self.count;
        let n2 = other.count;
        let n = n1 + n2;

        if n == T::zero() {
            return;
        }

        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta3 * delta;

        // mean
        self.mean += delta * (n2 / n);

        // M2
        let m2 = self.m2 + other.m2 + delta2 * (n1 * n2 / n);

        // M3
        let m3 = self.m3
            + other.m3
            + delta3 * (n1 * n2 * (n1 - n2) / (n * n))
            + three * delta * (n1 * other.m2 - n2 * self.m2) / n;

        // M4
        let m4 = self.m4
            + other.m4
            + delta4 * (n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2) / (n * n * n))
            + six * delta2 * (n1 * n1 * other.m2 + n2 * n2 * self.m2) / (n * n)
            + four * delta * (n1 * other.m3 - n2 * self.m3) / n;

        self.m2 = m2;
        self.m3 = m3;
        self.m4 = m4;
        self.count = n;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn total(&self) -> Option<Normality<T>> {
        if self.count < T::from_positive_int(4) {
            return None;
        }

        //
        // Fill out the stuff we accumulated
        let mean = self.mean;
        let variance = self.m2 / (self.count - T::one());
        let std_dev = variance.sqrt();

        // Sanity check
        if variance.is_zero() {
            return Some(Normality {
                likelihood: T::one(),
                skewness: T::zero(),
                kurtosis: T::zero(),
                mean,
                standard_deviation: T::zero(),
                count: self.count,
            });
        }

        let one_p_five = T::from_positive_int(3) / T::two();
        let skewness = (self.count.sqrt() * self.m3) / self.m2.powf(one_p_five);
        let kurtosis = self.count * self.m4 / (self.m2 * self.m2) - T::from_positive_int(3);

        //
        // Get the likelihood of normality using D'Agostino's K-squared
        let likelihood = Self::k_squared_likelihood(skewness, kurtosis, self.count);

        Some(Normality {
            likelihood,
            skewness,
            kurtosis,
            mean,
            standard_deviation: std_dev,
            count: self.count,
        })
    }

    fn eager_total_with_config(
        values: impl IntoIterator<Item = T>,
        _config: Self::Config,
    ) -> Option<Normality<T>> {
        let values: Vec<_> = values.into_iter().collect();
        let n = values.len();
        if n < 4 {
            return None;
        }

        let mean = MeanAccumulator::eager_total(values.iter().copied())?;
        let n = T::from_positive_int(n);
        if mean.std_dev == T::zero() {
            return Some(Normality {
                likelihood: T::one(),
                skewness: T::zero(),
                kurtosis: T::zero(),
                mean: mean.mean,
                standard_deviation: mean.std_dev,
                count: n,
            });
        }

        let mut skewness = T::zero();
        let mut kurtosis = T::zero();
        for r in values {
            let diff = (r - mean.mean) / mean.std_dev;
            skewness += diff * diff * diff;
            kurtosis += diff * diff * diff * diff;
        }

        skewness /= n;
        kurtosis = kurtosis / n - (T::two() + T::one());

        // Get the likelihood of normality using D'Agostino's K-squared
        let likelihood = NormalityAccumulator::k_squared_likelihood(skewness, kurtosis, n);

        Some(Normality {
            likelihood,
            skewness,
            kurtosis,
            mean: mean.mean,
            standard_deviation: mean.std_dev,
            count: n,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_close, statistics::mean, value::CoordExt};

    use super::*;

    function!(const SRC(x) = 2.0 x^2 + 1.0 x + 5.0);

    #[test]
    fn test_normality() {
        fn legacy_skewness_and_kurtosis<T: Value>(residuals: &[T]) -> (T, T) {
            let mean = mean(residuals.iter().copied()).unwrap();
            let (mean, stddev) = (mean.mean, mean.std_dev);

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

        let small_dataset = SRC.solve_range(0.0..=1000.0, 1.0).y();
        let large_dataset = SRC.solve_range(0.0..=1000.0, 0.01).y();

        let legacy_small = legacy_skewness_and_kurtosis(&small_dataset);
        let legacy_large = legacy_skewness_and_kurtosis(&large_dataset);

        let new_small = normality(small_dataset.into_iter()).unwrap();
        let new_large = normality(large_dataset.into_iter()).unwrap();

        assert_close!(
            legacy_small.0,
            new_small.skewness,
            epsilon = 1e-12,
            "Small dataset skewness mismatch compared to legacy implementation"
        );
        assert_close!(
            legacy_small.1,
            new_small.kurtosis,
            epsilon = 1e-12,
            "Small dataset kurtosis mismatch compared to legacy implementation"
        );

        assert_close!(
            legacy_large.0,
            new_large.skewness,
            epsilon = 1e-12,
            "Large dataset skewness mismatch compared to legacy implementation"
        );
        assert_close!(
            legacy_large.1,
            new_large.kurtosis,
            epsilon = 1e-12,
            "Large dataset kurtosis mismatch compared to legacy implementation"
        );

        // -----------------------------------------------------------------
        // Exact Ground-Truth Validation (Uniform Distribution)
        // -----------------------------------------------------------------
        // A perfectly symmetric uniform range has an exact mathematical skewness of 0.0
        let uniform_data: Vec<f64> = (1..=101).map(f64::from).collect();
        let norm_uniform = normality(uniform_data.into_iter()).unwrap();

        assert_close!(
            norm_uniform.skewness,
            0.0,
            epsilon = 1e-12,
            "Uniform distribution skewness must be exactly 0.0"
        );
        // Excess kurtosis of a discrete uniform distribution approaches -1.2 as N grows
        assert!(
            norm_uniform.kurtosis < 0.0,
            "Uniform distributions must have negative excess kurtosis (light tails)"
        );
        assert!(
            norm_uniform.likelihood < 0.05,
            "A uniform distribution should reject normality layout"
        );

        // -----------------------------------------------------------------
        // Mathematical Edge Cases & Safety Limits
        // -----------------------------------------------------------------
        // Edge Case A: Perfect flat signal (std_dev = 0). Must not divide-by-zero or panic.
        let flat_data = vec![5.5, 5.5, 5.5, 5.5, 5.5];
        let norm_flat = normality(flat_data.into_iter()).unwrap();
        assert_close!(norm_flat.skewness, 0.0, epsilon = 1e-12);
        assert_close!(norm_flat.kurtosis, 0.0, epsilon = 1e-12);
        assert_close!(
            norm_flat.likelihood,
            1.0,
            epsilon = 1e-12,
            "Zero variance implies perfect normal certainty mapping"
        );

        // Edge Case B: Insufficient data bounds (N < 4). Must safely return None.
        let tiny_data = vec![1.0, 2.0, 3.0];
        assert!(
            normality(tiny_data.into_iter()).is_none(),
            "Normality metrics are mathematically undefined for fewer than 4 elements"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "This test is long because it covers multiple edge cases."
    )]
    fn test_residual_variance() {
        fn legacy_residual_variance<T: Value>(
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

        let small_dataset = SRC.solve_range(0.0..=1000.0, 1.0).y();
        let large_dataset = SRC.solve_range(0.0..=1000.0, 0.01).y();

        let k = 3.0; // Quadratic model has 3 parameters

        let small_fit = SRC.solve_range(500.0..=1500.0, 1.0).y();
        let large_fit = SRC.solve_range(500.0..=1500.0, 0.01).y();

        let legacy_small =
            legacy_residual_variance(small_dataset.iter().copied(), small_fit.iter().copied(), k);
        let legacy_large =
            legacy_residual_variance(large_dataset.iter().copied(), large_fit.iter().copied(), k);

        let new_small =
            residual_variance(small_dataset.iter().copied(), small_fit.iter().copied(), k).unwrap();
        let new_large =
            residual_variance(large_dataset.iter().copied(), large_fit.iter().copied(), k).unwrap();

        let small_eps = 1e-12 * legacy_small.abs().max(1.0);
        let large_eps = 1e-12 * legacy_large.abs().max(1.0);

        assert_close!(
            legacy_small,
            new_small,
            epsilon = small_eps,
            "Small dataset residual variance mismatch compared to legacy implementation"
        );
        assert_close!(
            legacy_large,
            new_large,
            epsilon = large_eps,
            "Large dataset residual variance mismatch compared to legacy implementation"
        );

        // 1. Small exact target verification (Linear model: k = 2.0)
        // Observed: [1.0, 2.0, 3.0]
        // Predicted: [1.1, 1.9, 3.2]
        // Residuals: [-0.1, 0.1, -0.2] -> Squared: [0.01, 0.01, 0.04]
        // sum_sq_error = 0.06. Divisor = N - k = 3 - 2 = 1
        // Expected residual variance = 0.06 / 1 = 0.06
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.1, 1.9, 3.2];
        let k_linear = 2.0;
        let var_small = residual_variance(y.into_iter(), y_fit.into_iter(), k_linear).unwrap();
        assert_close!(
            var_small,
            0.06,
            epsilon = 1e-12,
            "Small exact residual variance failed"
        );

        // 2. Threshold scale checking using your uniform layout
        let uniform_data: Vec<f64> = (0..=1000).map(f64::from).collect(); // 1001 items
        let uniform_fit: Vec<f64> = uniform_data.iter().map(|v| v + 3.0).collect(); // Strict +3.0 error
        let k_mock = 5.0; // Mock 5 parameters

        // Total squared error = 1001 * (3^2) = 9009
        // Expected = 9009 / (1001 - 5) = 9009 / 996 = 9.045180722891566
        let var_large =
            residual_variance(uniform_data.into_iter(), uniform_fit.into_iter(), k_mock).unwrap();
        assert_close!(
            var_large,
            9009.0 / 996.0,
            epsilon = 1e-12,
            "Large scale residual variance mismatch"
        );

        // 3. Negative degrees of freedom boundary safety check (N <= k must return None)
        let tiny_y = vec![1.0, 2.0];
        let tiny_fit = vec![1.1, 1.9];
        let k_too_high = 3.0; // 3 parameters > 2 elements
        assert!(
            residual_variance(tiny_y.into_iter(), tiny_fit.into_iter(), k_too_high).is_none(),
            "Function must safely return None when elements are insufficient for parameters"
        );

        // 4. Zero degrees of freedom boundary safety check (N == k must return None)
        let exact_y = vec![1.0, 2.0, 3.0];
        let exact_fit = vec![1.0, 2.0, 3.0];
        let k_exact = 3.0; // 3 parameters == 3 elements
        assert!(
            residual_variance(exact_y.into_iter(), exact_fit.into_iter(), k_exact).is_none(),
            "Function must safely return None when elements equal parameters"
        );

        // 5. Zero variance case (all residuals are zero, N > k)
        let zero_y = vec![1.0, 2.0, 3.0, 4.0];
        let zero_fit = vec![1.0, 2.0, 3.0, 4.0]; // Perfect fit
        let k_valid = 2.0; // 2 parameters < 4 elements
        let var_zero =
            residual_variance(zero_y.into_iter(), zero_fit.into_iter(), k_valid).unwrap();
        assert_close!(
            var_zero,
            0.0,
            epsilon = 1e-12,
            "A perfect fit with valid degrees of freedom should return Some(0.0)"
        );

        // 6. Perfect fit with more elements than parameters (variance should be zero)
        let perfect_y = vec![1.0, 2.0, 3.0, 4.0];
        let perfect_fit = vec![1.0, 2.0, 3.0, 4.0]; // Perfect fit
        let k_less = 3.0; // 3 parameters < 4 elements
        let var_perfect =
            residual_variance(perfect_y.into_iter(), perfect_fit.into_iter(), k_less).unwrap();
        assert_close!(
            var_perfect,
            0.0,
            epsilon = 1e-12,
            "Perfect fit with more elements than parameters should yield zero variance"
        );

        // 7. Empty input case (should return None)
        let empty_y: Vec<f64> = vec![];
        let empty_fit: Vec<f64> = vec![];
        let k_empty = 1.0; // Any k
        assert!(
            residual_variance(empty_y.into_iter(), empty_fit.into_iter(), k_empty).is_none(),
            "Function must safely return None for empty input"
        );
    }
}
