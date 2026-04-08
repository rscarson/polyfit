use crate::{statistics::UncertainValue, value::Value};

/// Kahan summation algorithm for improved numerical stability when summing floating-point numbers
///
/// Reduces the numerical error that can occur when adding a sequence of finite precision floating-point numbers
pub struct KahanAccumulator<T: Value> {
    sum: T,
    kahan: T,
}
impl<T: Value> Default for KahanAccumulator<T> {
    fn default() -> Self {
        KahanAccumulator {
            sum: T::zero(),
            kahan: T::zero(),
        }
    }
}
impl<T: Value> KahanAccumulator<T> {
    /// Adds a value to the accumulator using the Kahan summation algorithm
    pub fn add(&mut self, value: T) {
        let y = value - self.kahan;
        let t = self.sum + y;
        self.kahan = (t - self.sum) - y;
        self.sum = t;
    }

    /// Returns the current sum of the accumulated values
    pub fn sum(&self) -> T {
        self.sum
    }
}

/// Accumulator for calculating R-squared values in a single pass through the data
///
/// Uses Kahan summation to improve numerical stability when summing large numbers of values, which can be important for accurate R-squared calculations.
pub struct RSquaredAccumulator<T: Value> {
    count: T,
    sum: KahanAccumulator<T>,
    sum_sq: KahanAccumulator<T>,
    sum_res_sq: KahanAccumulator<T>,
}
impl<T: Value> Default for RSquaredAccumulator<T> {
    fn default() -> Self {
        RSquaredAccumulator {
            count: T::zero(),
            sum: KahanAccumulator::default(),
            sum_sq: KahanAccumulator::default(),
            sum_res_sq: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> RSquaredAccumulator<T> {
    /// Adds a pair of observed and predicted values to the accumulator
    pub fn add(&mut self, y: T, y_fit: T) {
        self.count += T::one();
        self.sum.add(y);
        self.sum_sq.add(y * y);
        self.sum_res_sq.add(Value::powi(y - y_fit, 2));
    }

    /// Calculates and returns the R-squared value based on the accumulated data
    ///
    /// Returns `None` if no data has been added (i.e., count is zero)
    ///
    /// If the total sum of squares (TSS) is zero (which happens when all observed values are the same), then traditional R-squared is undefined
    ///
    /// However, we can still provide a meaningful R-squared value in this case:
    /// - If the residual sum of squares (RSS) is also zero, then we have a perfect fit and R-squared should be 1.
    /// - If the RSS is not zero, then the model has error but there is no variance in the observed data, so R-squared can be considered -infinity
    pub fn r_squared(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }

        let mean = self.sum.sum() / self.count;
        let tss = self.sum_sq.sum() - self.count * mean * mean;
        if tss.is_zero() {
            let residual_sum = self.sum_res_sq.sum();
            if residual_sum.is_zero() {
                // Perfect fit, R² = 1
                return Some(T::one());
            }

            // No variance in observed data but model has error, R² is undefined (could be considered 0 or negative infinity)
            return Some(T::neg_infinity());
        }

        let r2 = T::one() - (self.sum_res_sq.sum() / tss);
        Some(r2)
    }

    /// Returns the number of data points that have been added to the accumulator
    pub fn count(&self) -> T {
        self.count
    }
}
impl<T: Value> std::iter::FromIterator<(T, T)> for RSquaredAccumulator<T> {
    fn from_iter<I: IntoIterator<Item = (T, T)>>(iter: I) -> Self {
        let mut acc = RSquaredAccumulator::default();
        for (y, y_fit) in iter {
            acc.add(y, y_fit);
        }
        acc
    }
}

/// Accumulator for calculating mean and standard deviation in a single pass through the data
///
/// Uses Welford's algorithm for numerically stable calculation of mean and standard deviation
pub struct MeanAccumulator<T: Value> {
    count: T,
    sum: T,
    sum_sq: T,
}
impl<T: Value> Default for MeanAccumulator<T> {
    fn default() -> Self {
        MeanAccumulator {
            count: T::zero(),
            sum: T::zero(),
            sum_sq: T::zero(),
        }
    }
}
impl<T: Value> MeanAccumulator<T> {
    pub fn add(&mut self, value: T) {
        self.count += T::one();
        self.sum += value;
        self.sum_sq += value * value;
    }

    pub fn mean(&self) -> Option<UncertainValue<T>> {
        if self.count == T::zero() {
            return None;
        }

        let mean = self.sum / self.count;
        let variance = self.sum_sq / self.count - mean * mean;
        let std_dev = variance.sqrt();

        Some(UncertainValue::new(mean, std_dev))
    }
}
impl<T: Value> std::iter::FromIterator<T> for MeanAccumulator<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut acc = MeanAccumulator::default();
        for value in iter {
            acc.add(value);
        }
        acc
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

    /// The variance of the residuals, which gives a measure of how much the residuals vary around the mean.
    pub variance: T,

    /// The standard deviation of the residuals, which is the square root of the variance and gives a measure of the typical size of the residuals.
    pub standard_deviation: T,

    /// The number of residuals that were used to calculate the normality metrics.
    ///
    /// Important for understanding the reliability of the normality metrics, as a small number of residuals can lead to unreliable estimates of normality.
    pub count: T,
}

/// Accumulator for calculating normality metrics (skewness, kurtosis, mean, variance) in a single pass through the data
pub struct NormalityAccumulator<T: Value> {
    count: T,
    m1: KahanAccumulator<T>,
    m2: KahanAccumulator<T>,
    m3: KahanAccumulator<T>,
    m4: KahanAccumulator<T>,
}
impl<T: Value> Default for NormalityAccumulator<T> {
    fn default() -> Self {
        NormalityAccumulator {
            count: T::zero(),
            m1: KahanAccumulator::default(),
            m2: KahanAccumulator::default(),
            m3: KahanAccumulator::default(),
            m4: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> NormalityAccumulator<T> {
    /// Adds a pair of observed and predicted values to the accumulator, updating the necessary moments for normality calculations
    pub fn add(&mut self, value: T) {
        let three = T::from_positive_int(3);
        let four = T::from_positive_int(4);
        let six = T::from_positive_int(6);

        self.count += T::one();

        let delta = value - self.m1.sum();
        let delta_n = delta / self.count;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * self.count;

        self.m1.add(delta_n);
        self.m4.add(
            term1 * delta_n2 * (self.count * self.count - three * self.count + three)
                + six * delta_n2 * self.m2.sum()
                - four * delta_n * self.m3.sum(),
        );
        self.m3
            .add(term1 * delta_n * (self.count - T::two()) - three * delta_n * self.m2.sum());
        self.m2.add(term1);
    }

    /// Calculates and returns the normality metrics based on the accumulated data
    pub fn normality(&self) -> Option<Normality<T>> {
        if self.count < T::from_positive_int(2) {
            return None;
        }

        //
        // Fill out the stuff we accumulated
        let mean = self.m1.sum();
        let variance = self.m2.sum() / self.count;
        let std_dev = variance.sqrt();

        // Sanity check
        if variance.is_zero() {
            return Some(Normality {
                likelihood: T::one(),
                skewness: T::zero(),
                kurtosis: T::zero(),
                mean,
                variance: T::zero(),
                standard_deviation: T::zero(),
                count: self.count,
            });
        }

        let skewness = self.count * self.m3.sum() / (self.m2.sum().powf(T::from_f64(1.5)?));
        let kurtosis =
            self.count * self.m4.sum() / (self.m2.sum() * self.m2.sum()) - T::from_positive_int(3);

        //
        // Get the likelihood of normality using D'Agostino's K-squared

        let six = T::from_positive_int(6);
        let twentyfour = T::from_positive_int(24);

        let se_skew = (six / self.count).sqrt();
        let se_kurt = (twentyfour / self.count).sqrt();

        let z_skew = skewness / se_skew;
        let z_kurt = kurtosis / se_kurt;
        let k_squared = z_skew * z_skew + z_kurt * z_kurt;

        let likelihood = (-k_squared / T::two()).exp();

        Some(Normality {
            likelihood,
            skewness,
            kurtosis,
            mean,
            variance,
            standard_deviation: std_dev,
            count: self.count,
        })
    }
}
impl<T: Value> std::iter::FromIterator<T> for NormalityAccumulator<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut acc = NormalityAccumulator::default();
        for value in iter {
            acc.add(value);
        }
        acc
    }
}

/// Accumulator for calculating mean squared error in a single pass through the data
pub struct MeanSquaredErrorAccumulator<T: Value> {
    count: T,
    sum_sq_error: KahanAccumulator<T>,
}
impl<T: Value> Default for MeanSquaredErrorAccumulator<T> {
    fn default() -> Self {
        MeanSquaredErrorAccumulator {
            count: T::zero(),
            sum_sq_error: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> MeanSquaredErrorAccumulator<T> {
    /// Adds a pair of observed and predicted values to the accumulator, updating the sum of squared errors
    pub fn add(&mut self, y: T, y_fit: T) {
        self.count += T::one();
        let error = y - y_fit;
        self.sum_sq_error.add(error * error);
    }

    /// Returns the number of data points that have been added to the accumulator
    pub fn count(&self) -> T {
        self.count
    }

    /// Calculates and returns the mean squared error based on the accumulated data
    ///
    /// Returns `None` if no data has been added (i.e., count is zero).
    pub fn mean_squared_error(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }
        Some(self.sum_sq_error.sum() / self.count)
    }

    /// Returns the sum of squared errors accumulated so far, without dividing by the count
    pub fn mean_squared_error_sum(&self) -> T {
        self.sum_sq_error.sum()
    }
}
impl<T: Value> std::iter::FromIterator<(T, T)> for MeanSquaredErrorAccumulator<T> {
    fn from_iter<I: IntoIterator<Item = (T, T)>>(iter: I) -> Self {
        let mut acc = MeanSquaredErrorAccumulator::default();
        for (y, y_fit) in iter {
            acc.add(y, y_fit);
        }
        acc
    }
}

/// Accumulator for calculating mean absolute error in a single pass through the data
pub struct MeanAbsoluteErrorAccumulator<T: Value> {
    count: T,
    sum_abs_error: KahanAccumulator<T>,
}
impl<T: Value> Default for MeanAbsoluteErrorAccumulator<T> {
    fn default() -> Self {
        MeanAbsoluteErrorAccumulator {
            count: T::zero(),
            sum_abs_error: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> MeanAbsoluteErrorAccumulator<T> {
    /// Adds a pair of observed and predicted values to the accumulator, updating the sum of absolute errors
    pub fn add(&mut self, y: T, y_fit: T) {
        self.count += T::one();
        let error = Value::abs(y - y_fit);
        self.sum_abs_error.add(error);
    }

    /// Calculates and returns the mean absolute error based on the accumulated data
    ///
    /// Returns `None` if no data has been added (i.e., count is zero).
    pub fn mean_absolute_error(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }
        Some(self.sum_abs_error.sum() / self.count)
    }
}
impl<T: Value> std::iter::FromIterator<(T, T)> for MeanAbsoluteErrorAccumulator<T> {
    fn from_iter<I: IntoIterator<Item = (T, T)>>(iter: I) -> Self {
        let mut acc = MeanAbsoluteErrorAccumulator::default();
        for (y, y_fit) in iter {
            acc.add(y, y_fit);
        }
        acc
    }
}

/// Accumulator for calculating the Huber loss in a single pass through the data
pub struct HuberLogLikelihoodAccumulator<T: Value> {
    count: T,
    sum_log_likelihood: KahanAccumulator<T>,
    delta: T,
}
impl<T: Value> HuberLogLikelihoodAccumulator<T> {
    /// Creates a new `HuberLogLikelihoodAccumulator`
    ///
    /// # Parameters
    /// - `median_absolute_deviation`: The median absolute deviation of the residuals
    /// - `huber_const`: An optional constant to use in the Huber loss function. If not provided, 1.345 or the nearest precision supported value will be used.
    pub fn new(median_absolute_deviation: T, huber_const: Option<T>) -> Self {
        let huber_const = huber_const.unwrap_or_else(|| T::from_f64(1.345).unwrap_or(T::one()));
        let delta = huber_const * median_absolute_deviation;

        HuberLogLikelihoodAccumulator {
            count: T::zero(),
            sum_log_likelihood: KahanAccumulator::default(),
            delta,
        }
    }

    pub fn add_iter(&mut self, y: impl Iterator<Item = T>, y_fit: impl Iterator<Item = T>) {
        for (y, y_fit) in y.zip(y_fit) {
            self.add(y, y_fit);
        }
    }

    /// Adds a pair of observed and predicted values to the accumulator, updating the sum of log-likelihoods based on the Huber loss function
    pub fn add(&mut self, y: T, y_fit: T) {
        let value = Value::abs(y - y_fit);

        let half = T::one() / T::two();
        let value = if value <= self.delta {
            half * value * value
        } else {
            self.delta * (value - half * self.delta)
        };

        self.count += T::one();
        self.sum_log_likelihood.add(value);
    }

    /// Calculates and returns the average log-likelihood based on the accumulated data
    pub fn log_likelihood_sum(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }
        Some(self.sum_log_likelihood.sum())
    }

    /// Calculates and returns the average log-likelihood based on the accumulated data
    pub fn log_likelihood(&self) -> Option<T> {
        self.log_likelihood_sum().map(|sum| sum / self.count)
    }

    /// Returns the number of data points that have been added to the accumulator
    pub fn count(&self) -> T {
        self.count
    }
}
