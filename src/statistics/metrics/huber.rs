use crate::{
    statistics::{
        accumulator::{Accumulator, AccumulatorFactory, KahanAccumulator},
        median_absolute_deviation,
    },
    value::Value,
};

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
pub fn huber_log_likelihood<T: Value>(y: &[T], y_fit: &[T]) -> Option<T> {
    let residuals: Vec<_> = y
        .iter()
        .zip(y_fit.iter())
        .map(|(&y, &y_fit)| y - y_fit)
        .collect();
    let mad = median_absolute_deviation(residuals.clone())?;

    HuberLogLikelihoodAccumulator::new_from_iter_with_config(
        residuals,
        HuberLogLikelihoodConfig {
            median_absolute_deviation: mad,
            huber_const: None,
        },
    )
    .total()
}

/// Accumulator for calculating the Huber loss in a single pass through the data
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HuberLogLikelihoodConfig<T: Value> {
    pub median_absolute_deviation: T,
    pub huber_const: Option<T>,
}

/// Accumulator for calculating the Huber loss in a single pass through the data
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HuberLogLikelihoodAccumulator<T: Value> {
    pub count: T,
    pub sum_log_likelihood: KahanAccumulator<T>,
    pub delta: T,
}
impl<T: Value> AccumulatorFactory for HuberLogLikelihoodAccumulator<T> {
    type Config = HuberLogLikelihoodConfig<T>;

    /// Creates a new `HuberLogLikelihoodAccumulator2`
    ///
    /// # Parameters
    /// - `median_absolute_deviation`: The median absolute deviation of the residuals
    /// - `huber_const`: An optional constant to use in the Huber loss function. If not provided, 1.345 or the nearest precision supported value will be used.
    fn create(config: Self::Config) -> Self {
        let huber_const = config
            .huber_const
            .unwrap_or_else(|| T::from_f64(1.345).unwrap_or(T::one()));
        let delta = huber_const * config.median_absolute_deviation;

        Self {
            count: T::zero(),
            sum_log_likelihood: KahanAccumulator::default(),
            delta,
        }
    }
}
impl<T: Value> Accumulator<T> for HuberLogLikelihoodAccumulator<T> {
    fn add(&mut self, residual: T) {
        let residual = Value::abs(residual);

        let half = T::one() / T::two();
        let log_likelihood = if residual <= self.delta {
            half * residual * residual
        } else {
            self.delta * (residual - half * self.delta)
        };

        self.count += T::one();
        self.sum_log_likelihood.add(log_likelihood);
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum_log_likelihood.merge(other.sum_log_likelihood);
    }

    fn reset(&mut self) {
        self.count = T::zero();
        self.sum_log_likelihood = KahanAccumulator::default();
    }

    fn total(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }
        Some(self.sum_log_likelihood.total() / self.count)
    }
}
