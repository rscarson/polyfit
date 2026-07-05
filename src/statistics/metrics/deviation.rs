use crate::{
    statistics::{
        accumulator::{
            Accumulator, DefaultAccumulatorFactory, ExecutionRouter, FittedValue, KahanAccumulator,
        },
        median,
    },
    value::Value,
};

/// Computes the median absolute deviation (MAD) of a set of values.
/// - MAD is a measure of variability that is robust to outliers.
/// - Lower values indicate a closer fit.
/// - Uses the median of the absolute deviations from the median.
///
/// **This function consumes the input `Vec`, completely avoiding any new heap allocations.**
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// median = median( y )
/// MAD = median( |y_i - median| )
/// where
///   y = observed values
/// ```
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `data`: A vector of values of type `T`.
///
/// # Returns
/// The median absolute deviation as a `T`, or `None` if there are no values to compare.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::median_absolute_deviation;
/// let data = vec![1.0, 2.0, 3.0];
/// let mad = median_absolute_deviation(data);
/// ```
#[must_use]
pub fn median_absolute_deviation<T: Value>(mut data: Vec<T>) -> Option<T> {
    let median_r = median(&mut data)?;

    // Avoid trait calls for optimization
    for r in &mut data {
        let diff = *r - median_r;
        *r = if diff < T::zero() {
            T::zero() - diff
        } else {
            diff
        };
    }

    median(&mut data)
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
    let values = FittedValue::from_iters(y, y_fit);
    match ExecutionRouter::route(values) {
        ExecutionRouter::Eager(data) => MeanAbsoluteErrorAccumulator::eager_total(data),
        ExecutionRouter::Streaming(data) => {
            MeanAbsoluteErrorAccumulator::new_from_iter(data).total()
        }
    }
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
    let values = FittedValue::from_iters(y, y_fit);
    match ExecutionRouter::route(values) {
        ExecutionRouter::Eager(data) => MeanSquaredErrorAccumulator::eager_total(data),
        ExecutionRouter::Streaming(data) => {
            MeanSquaredErrorAccumulator::new_from_iter(data).total()
        }
    }
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
    mean_squared_error(y, y_fit).map(T::sqrt)
}

/// Accumulator for calculating mean absolute error in a single pass through the data
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MeanAbsoluteErrorAccumulator<T: Value> {
    pub count: T,
    pub sum_abs_error: KahanAccumulator<T>,
}
impl<T: Value> Default for MeanAbsoluteErrorAccumulator<T> {
    fn default() -> Self {
        MeanAbsoluteErrorAccumulator {
            count: T::zero(),
            sum_abs_error: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> DefaultAccumulatorFactory for MeanAbsoluteErrorAccumulator<T> {}
impl<T: Value> Accumulator<T, FittedValue<T>> for MeanAbsoluteErrorAccumulator<T> {
    fn add(&mut self, value: FittedValue<T>) {
        self.count += T::one();
        let error = Value::abs(value.residual());
        self.sum_abs_error.add(error);
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum_abs_error.merge(other.sum_abs_error);
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn total(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }
        Some(self.sum_abs_error.total() / self.count)
    }

    fn eager_total_with_config(
        values: impl IntoIterator<Item = FittedValue<T>>,
        _config: Self::Config,
    ) -> Option<T> {
        let mut count = T::zero();
        let mut sum_abs_error = T::zero();
        // the hope is branchless absolute value check via direct subtraction ordering
        // to allow LLVM to auto-vectorize this loop using SIMD registers
        for value in values {
            count += T::one();
            let res = value.residual();
            sum_abs_error += if res < T::zero() {
                T::zero() - res
            } else {
                res
            };
        }

        if count == T::zero() {
            return None;
        }

        Some(sum_abs_error / count)
    }
}

/// Accumulator for calculating mean squared error in a single pass through the data
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MeanSquaredErrorAccumulator<T: Value> {
    pub count: T,
    pub sum_sq_error: KahanAccumulator<T>,
}
impl<T: Value> Default for MeanSquaredErrorAccumulator<T> {
    fn default() -> Self {
        MeanSquaredErrorAccumulator {
            count: T::zero(),
            sum_sq_error: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> DefaultAccumulatorFactory for MeanSquaredErrorAccumulator<T> {}
impl<T: Value> Accumulator<T, FittedValue<T>> for MeanSquaredErrorAccumulator<T> {
    fn add(&mut self, value: FittedValue<T>) {
        self.count += T::one();
        let error = value.residual();
        self.sum_sq_error.add(error * error);
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum_sq_error.merge(other.sum_sq_error);
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn total(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }
        Some(self.sum_sq_error.total() / self.count)
    }

    fn eager_total_with_config(
        values: impl IntoIterator<Item = FittedValue<T>>,
        _config: Self::Config,
    ) -> Option<T> {
        let mut count = T::zero();
        let mut sum_squared_error = T::zero();
        for value in values {
            count += T::one();
            let res = value.residual();
            sum_squared_error += res * res;
        }

        if count == T::zero() {
            return None;
        }

        Some(sum_squared_error / count)
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_close, value::CoordExt};

    use super::*;

    function!(const SRC(x) = 2.0 x^2 + 1.0 x + 5.0);

    #[test]
    fn test_mad() {
        //
        // We do not test against a legacy implementation here, because of a known issue with the legacy implementation of MAD
        // Previously we were taking the absolute residuals in step 1, which is incorrect.
        //
        // There also isn't top level branching for small/large datasets

        // 1. Mixed-sign odd-length verification (The core correctness proof)
        // Data: [-10.0, -2.0, 3.0, 4.0, 15.0]
        // Raw Median = 3.0
        // Absolute Deviations from 3.0: [|-10 - 3|, |-2 - 3|, |3 - 3|, |4 - 3|, |15 - 3|]
        // Unsorted Deviations: [13.0, 5.0, 0.0, 1.0, 12.0]
        // Sorted Deviations:   [0.0, 1.0, 5.0, 12.0, 13.0] -> Expected MAD = 5.0
        let mixed_odd = vec![-10.0, -2.0, 3.0, 4.0, 15.0];
        let res_odd = median_absolute_deviation(mixed_odd).unwrap();
        assert_close!(
            res_odd,
            5.0,
            epsilon = 1e-12,
            "Mixed-sign odd MAD calculation failed"
        );

        // 2. Mixed-sign even-length verification (Testing even averages)
        // Data: [-5.0, 2.0, 8.0, 14.0] -> Raw Median = (2.0 + 8.0) / 2.0 = 5.0
        // Absolute Deviations from 5.0: [|-5 - 5|, |2 - 5|, |8 - 5|, |14 - 5|]
        // Unsorted Deviations: [10.0, 3.0, 3.0, 9.0]
        // Sorted Deviations:   [3.0, 3.0, 9.0, 10.0] -> Expected MAD = (3.0 + 9.0) / 2.0 = 6.0
        let mixed_even = vec![-5.0, 2.0, 8.0, 14.0];
        let res_even = median_absolute_deviation(mixed_even).unwrap();
        assert_close!(
            res_even,
            6.0,
            epsilon = 1e-12,
            "Mixed-sign even MAD calculation failed"
        );

        // 3. Single-value verification
        let single_value = vec![42.0];
        let res_single = median_absolute_deviation(single_value).unwrap();
        assert_close!(
            res_single,
            0.0,
            epsilon = 1e-12,
            "Single-value MAD calculation failed"
        );

        // 4. Empty dataset verification
        let empty_dataset: Vec<f64> = vec![];
        let res_empty = median_absolute_deviation(empty_dataset);
        assert!(
            res_empty.is_none(),
            "Empty dataset should return None for MAD"
        );
    }

    #[test]
    fn test_mean_absolute_error() {
        fn legacy_mean_absolute_error<T: Value>(
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

        let small_dataset = SRC.solve_range(0.0..=1000.0, 1.0);
        let large_dataset = SRC.solve_range(0.0..=1000.0, 0.01);

        let small_fit = SRC.solve_range(500.0..=1500.0, 1.0);
        let large_fit = SRC.solve_range(500.0..=1500.0, 0.01);

        let legacy_small = legacy_mean_absolute_error(small_dataset.y_iter(), small_fit.y_iter());
        let legacy_large = legacy_mean_absolute_error(large_dataset.y_iter(), large_fit.y_iter());

        let new_small = mean_absolute_error(small_dataset.y_iter(), small_fit.y_iter()).unwrap();
        let new_large = mean_absolute_error(large_dataset.y_iter(), large_fit.y_iter()).unwrap();

        assert_close!(
            legacy_small,
            new_small,
            epsilon = 1e-12,
            "Small dataset MAE mismatch"
        );
        assert_close!(
            legacy_large,
            new_large,
            epsilon = 1e-12,
            "Large dataset MAE mismatch"
        );

        // 1. Small exact target test
        // Observed: [1.0, 2.0, 3.0], Predicted: [1.5, 1.5, 3.2] -> Errors: [0.5, 0.5, 0.2]
        // Total = 1.2 / 3 = 0.4 expected MAE
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.5, 1.5, 3.2];
        let res_small = mean_absolute_error(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_close!(
            res_small,
            0.4,
            epsilon = 1e-12,
            "Small exact MAE calculation failed"
        );

        // 2. Large scale-up simulation matching your cubic benchmark layouts
        function!(const F(x) = 5.3 x^3 - 2.1 x^2 + 0.5 x - 1.0);
        let dataset = F.solve_range(0.0..=10.0, 0.1); // 101 values

        // Add a strict, clean +3.5 offset error across the board
        let y_fit_large: Vec<f64> = dataset.y_iter().map(|v| v + 3.5).collect();
        let res_large = mean_absolute_error(dataset.y_iter(), y_fit_large.into_iter()).unwrap();
        assert_close!(
            res_large,
            3.5,
            epsilon = 1e-12,
            "Large scale threshold MAE verification failed"
        );

        // 3. Empty data safety guard
        let empty_y: Vec<f64> = vec![];
        let empty_fit: Vec<f64> = vec![];
        assert!(
            mean_absolute_error(empty_y.into_iter(), empty_fit.into_iter()).is_none(),
            "Empty dataset MAE must return None"
        );
    }

    #[test]
    fn test_mean_squared_error() {
        fn legacy_mean_squared_error<T: Value>(
            y: impl Iterator<Item = T>,
            y_fit: impl Iterator<Item = T>,
        ) -> T {
            let mut total = T::zero();
            let mut n = T::zero();
            for (y, y_fit) in y.zip(y_fit) {
                total += Value::powi(y - y_fit, 2);
                n += T::one();
            }
            total / n
        }

        let small_dataset = SRC.solve_range(0.0..=1000.0, 1.0);
        let large_dataset = SRC.solve_range(0.0..=1000.0, 0.01);

        let small_fit = SRC.solve_range(500.0..=1500.0, 1.0);
        let large_fit = SRC.solve_range(500.0..=1500.0, 0.01);

        let legacy_small = legacy_mean_squared_error(small_dataset.y_iter(), small_fit.y_iter());
        let legacy_large = legacy_mean_squared_error(large_dataset.y_iter(), large_fit.y_iter());

        let new_small = mean_squared_error(small_dataset.y_iter(), small_fit.y_iter()).unwrap();
        let new_large = mean_squared_error(large_dataset.y_iter(), large_fit.y_iter()).unwrap();

        let small_eps = 1e-12 * legacy_small.abs().max(1.0);
        let large_eps = 1e-12 * legacy_large.abs().max(1.0);

        assert_close!(
            legacy_small,
            new_small,
            epsilon = small_eps,
            "Small dataset MSE mismatch compared to legacy implementation"
        );
        assert_close!(
            legacy_large,
            new_large,
            epsilon = large_eps,
            "Large dataset MSE mismatch compared to legacy implementation"
        );

        // 1. Basic small exact mathematical validation
        // Observed: [1.0, 2.0, 3.0]
        // Predicted: [1.5, 1.1, 3.2]
        // Residuals: [-0.5, 0.9, -0.2] -> Squared Errors: [0.25, 0.81, 0.04]
        // Total Error = 1.10 / 3 elements = 0.36666666666666664 expected MSE
        let y = vec![1.0, 2.0, 3.0];
        let y_fit = vec![1.5, 1.1, 3.2];
        let mse_small = mean_squared_error(y.into_iter(), y_fit.into_iter()).unwrap();
        assert_close!(
            mse_small,
            1.10 / 3.0,
            epsilon = 1e-12,
            "Small exact MSE calculation failed"
        );

        // 2. Large dataset validation crossing cutover thresholds
        // Using your cubic curve framework to verify structural accuracy
        function!(const F(x) = 5.3 x^3 - 2.1 x^2 + 0.5 x - 1.0);
        let dataset = F.solve_range(0.0..=10.0, 0.1); // 101 values

        // Add a strict, flat +2.5 scalar error offset to the predictions
        let y_fit_large: Vec<f64> = dataset.y_iter().map(|v| v + 2.5).collect();

        // If every single point is exactly 2.5 units off, the MSE must be 2.5^2 = 6.25
        let mse_large = mean_squared_error(dataset.y_iter(), y_fit_large.into_iter()).unwrap();
        assert_close!(
            mse_large,
            6.25,
            epsilon = 1e-12,
            "Large scale threshold MSE verification failed"
        );

        // 3. Empty iterator boundary handling
        let empty_y: Vec<f64> = vec![];
        let empty_fit: Vec<f64> = vec![];
        assert!(
            mean_squared_error(empty_y.into_iter(), empty_fit.into_iter()).is_none(),
            "Empty dataset MSE must return None"
        );
    }
}
