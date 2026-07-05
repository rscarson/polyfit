use crate::{
    statistics::{
        accumulator::{Accumulator, DefaultAccumulatorFactory, ExecutionRouter, BIG_DATA_SIZE},
        UncertainValue,
    },
    value::Value,
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
    match ExecutionRouter::route(data) {
        ExecutionRouter::Eager(data) => MeanAccumulator::eager_total(data),
        ExecutionRouter::Streaming(data) => MeanAccumulator::new_from_iter(data).total(),
    }
}

/// Computes the median of a sequence of values.
///
/// The median is the middle value when the data is sorted.
///
/// Note that this will reorder the input slice, but it does not always require a full sort, and is O(n) on average.
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
    #[inline(always)]
    fn cmp<T: Value>(a: &T, b: &T) -> std::cmp::Ordering {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    }

    if data.is_empty() {
        return None;
    }

    //
    // for small sets, we can just sort and pick the middle value
    if data.len() <= BIG_DATA_SIZE {
        data.sort_by(cmp);
        let n = data.len();
        if n % 2 == 0 {
            return Some((data[n / 2 - 1] + data[n / 2]) / T::two());
        }

        return Some(data[n / 2]);
    }

    let n = data.len();
    let i = n / 2;
    let (l, mid, _) = data.select_nth_unstable_by(i, cmp);
    let mid = *mid;

    if n % 2 == 0 {
        // Even number of elements, so n/2 is actually the upper middle
        // select i - 1 to get the lower middle
        let lower_mid = unsafe { l.iter().copied().max_by(cmp).unwrap_unchecked() }; // Safe because l has length i (where i >= 1 when n > BIG_DATA_SIZE)
        Some((lower_mid + mid) / T::two())
    } else {
        Some(mid)
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
/// - Returns None if `data` is empty
///
/// # Examples
/// ```rust
/// let values = vec![2.0, 5.0, 1.0, 9.0];
/// let r = polyfit::statistics::spread(values.iter().copied()).unwrap();
/// assert_eq!(r, 8.0); // 9 - 1
/// ```
pub fn spread<T: Value>(data: impl Iterator<Item = T>) -> Option<T> {
    match ExecutionRouter::route(data) {
        ExecutionRouter::Eager(data) => {
            //
            // Faster path for small datasets, could maybe leverage SIMD?
            let values: Vec<T> = data.collect();
            if values.is_empty() {
                return None;
            }
            let first = values[0];
            let (min, max) = values
                .iter()
                .copied()
                .fold((first, first), |(min, max), x| {
                    // I'm hoping LLVM can emit branchless conditional move
                    let new_min = if x < min { x } else { min };
                    let new_max = if x > max { x } else { max };
                    (new_min, new_max)
                });

            Some(max - min)
        }

        ExecutionRouter::Streaming(data) => {
            //
            // For large datasets, we can just iterate through the values and keep track of min/max
            let mut min = T::infinity();
            let mut max = T::neg_infinity();
            let mut has_elements = false;

            for value in data {
                has_elements = true;
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
            }

            if !has_elements {
                return None;
            }

            Some(max - min)
        }
    }
}

/// Accumulator for calculating mean and standard deviation in a single pass through the data
///
/// Uses Welford's algorithm for numerically stable calculation of mean and standard deviation
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MeanAccumulator<T: Value> {
    pub count: T,
    pub m: T,
    pub s: T,
}
impl<T: Value> Default for MeanAccumulator<T> {
    fn default() -> Self {
        MeanAccumulator {
            count: T::zero(),
            m: T::zero(),
            s: T::zero(),
        }
    }
}
impl<T: Value> DefaultAccumulatorFactory for MeanAccumulator<T> {}
impl<T: Value> Accumulator<T, T, UncertainValue<T>> for MeanAccumulator<T> {
    fn add(&mut self, value: T) {
        self.count += T::one();
        let old_m = self.m;
        self.m += (value - self.m) / self.count;
        self.s += (value - self.m) * (value - old_m);
    }

    fn merge(&mut self, other: Self) {
        if self.count == T::zero() {
            *self = other;
            return;
        }
        if other.count == T::zero() {
            return;
        }

        let n1 = self.count;
        let n2 = other.count;
        let new_n = n1 + n2;

        let delta = other.m - self.m;
        self.m += delta * (n2 / new_n);
        self.s += other.s + delta * delta * (n1 * n2 / new_n);
        self.count = new_n;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn total(&self) -> Option<UncertainValue<T>> {
        if self.count == T::zero() {
            return None;
        }

        let mean = self.m;
        let variance = self.s / self.count;
        let std_dev = variance.sqrt();

        Some(UncertainValue::new(mean, std_dev))
    }

    fn eager_total_with_config(
        values: impl IntoIterator<Item = T>,
        _config: Self::Config,
    ) -> Option<UncertainValue<T>> {
        let mut values = values.into_iter();

        let anchor = values.next()?;
        let mut sum_shift = T::zero();
        let mut sum_sq_shift = T::zero();

        let mut count = T::one();
        for value in values {
            count += T::one();
            let shifted = value - anchor;
            sum_shift += shifted;
            sum_sq_shift += shifted * shifted;
        }

        // Correct the mean and variance back to the original scale
        let mean = anchor + sum_shift / count;

        // Adjusted variance calculation using the shifted values
        let variance = (sum_sq_shift - (sum_shift * sum_shift) / count) / count;

        Some(UncertainValue::new(mean, variance.sqrt()))
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_close, value::CoordExt};

    use super::*;

    function!(const SRC(x) = 2.0 x^2 + 1.0 x + 5.0);

    #[test]
    fn test_mean() {
        const LEGACY_EPSILON: f64 = 1e-12; // Legacy implementation used a different algorithm for computing mean and stddev,
                                           // so we need to use a larger epsilon for the tests to pass
                                           // If anything the new method is more accurate

        fn legacy_stddev_and_mean<T: Value>(data: impl Iterator<Item = T>) -> (T, T) {
            fn mean<T: Value>(data: impl Iterator<Item = T>) -> T {
                let mut sum = T::zero();
                let mut count = T::zero();
                for value in data {
                    sum += value;
                    count += T::one();
                }
                sum / count
            }

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

        let small_dataset = SRC.solve_range(0.0..=1_000.0, 1.0);
        let large_dataset = SRC.solve_range(0.0..=1_000.0, 0.01);

        let (ds, m) = legacy_stddev_and_mean(small_dataset.y_iter());
        let legacy_small = UncertainValue::new(m, ds);

        let (dl, m) = legacy_stddev_and_mean(large_dataset.y_iter());
        let legacy_large = UncertainValue::new(m, dl);

        let new_small = mean(small_dataset.y_iter()).unwrap();
        let new_large = mean(large_dataset.y_iter()).unwrap();

        let small_epsilon = LEGACY_EPSILON * ds.abs();
        let large_epsilon = LEGACY_EPSILON * dl.abs();

        assert_close!(
            legacy_small.mean,
            new_small.mean,
            epsilon = small_epsilon,
            "Small dataset mean do not match legacy implementation"
        );
        assert_close!(
            legacy_small.std_dev,
            new_small.std_dev,
            epsilon = small_epsilon,
            "Small dataset stddev do not match legacy implementation"
        );

        assert_close!(
            legacy_large.mean,
            new_large.mean,
            epsilon = large_epsilon,
            "Large dataset mean do not match legacy implementation"
        );
        assert_close!(
            legacy_large.std_dev,
            new_large.std_dev,
            epsilon = large_epsilon,
            "Large dataset stddev do not match legacy implementation"
        );

        //
        // Now we test known means and stddevs for a uniform distribution
        let uniform_data: Vec<f64> = (0..=100).map(f64::from).collect();
        let uniform = mean(uniform_data.iter().copied()).unwrap();
        let known_stddev = ((50.0f64 * 51.0) / 3.0).sqrt(); // stdev = sqrt((half_n * (half_n + 1)) / 3)
        assert_close!(
            uniform.mean,
            50.0,
            epsilon = 1e-12,
            "Uniform distribution mean does not match expected value"
        );
        assert_close!(
            uniform.std_dev,
            known_stddev,
            epsilon = 1e-12,
            "Uniform distribution stddev does not match expected value"
        );

        //
        // And a large uniform distribution
        let large_uniform_data: Vec<f64> = (0..=100_000).map(f64::from).collect();
        let large_uniform = mean(large_uniform_data.iter().copied()).unwrap();
        let known_large_stddev = ((50_000.0f64 * 50_001.0) / 3.0).sqrt(); // stdev = sqrt((half_n * (half_n + 1)) / 3)
        assert_close!(
            large_uniform.mean,
            50_000.0,
            epsilon = 1e-12,
            "Large uniform distribution mean does not match expected value"
        );
        assert_close!(
            large_uniform.std_dev,
            known_large_stddev,
            epsilon = 1e-12,
            "Large uniform distribution stddev does not match expected value"
        );

        //
        // Now empty datasets
        let empty_data: Vec<f64> = vec![];
        let empty = mean(empty_data.iter().copied());
        assert!(
            empty.is_none(),
            "Empty dataset should return None for mean and stddev"
        );
    }

    #[test]
    fn test_median() {
        fn legacy_median<T: Value>(data: &[T]) -> T {
            let mut data = data.to_vec();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = data.len();
            if n == 0 {
                return T::zero();
            }
            if n % 2 == 1 {
                data[n / 2]
            } else {
                (data[n / 2 - 1] + data[n / 2]) / T::two()
            }
        }

        let small_dataset = SRC.solve_range(0.0..=1_000.0, 1.0);
        let large_dataset = SRC.solve_range(0.0..=1_000.0, 0.01);

        let legacy_small = legacy_median(small_dataset.y_iter().collect::<Vec<_>>().as_slice());
        let legacy_large = legacy_median(large_dataset.y_iter().collect::<Vec<_>>().as_slice());

        let new_small = median(&mut small_dataset.y()).unwrap();
        let new_large = median(&mut large_dataset.y()).unwrap();

        let small_epsilon = 1e-12 * legacy_small.abs();
        let large_epsilon = 1e-12 * legacy_large.abs();

        assert_close!(
            legacy_small,
            new_small,
            epsilon = small_epsilon,
            "Small dataset median do not match legacy implementation"
        );
        assert_close!(
            legacy_large,
            new_large,
            epsilon = large_epsilon,
            "Large dataset median do not match legacy implementation"
        );

        //
        // Now we test known medians for a uniform distribution
        let mut uniform_data: Vec<f64> = (0..=100).map(f64::from).collect();
        let uniform_median = median(&mut uniform_data).unwrap();
        assert_close!(
            uniform_median,
            50.0,
            epsilon = 1e-12,
            "Uniform distribution median does not match expected value"
        );

        //
        // And a large uniform distribution
        let mut large_uniform_data: Vec<f64> = (0..=100_000).map(f64::from).collect();
        let large_uniform_median = median(&mut large_uniform_data).unwrap();
        assert_close!(
            large_uniform_median,
            50_000.0,
            epsilon = 1e-12,
            "Large uniform distribution median does not match expected value"
        );

        //
        // Now we repeat all of these with an even number of elements to test the averaging behavior
        //

        let small_dataset = SRC.solve_range(1.0..=1_000.0, 1.0);
        let large_dataset = SRC.solve_range(1.0..=1_000.0, 0.01);

        let legacy_small = legacy_median(small_dataset.y_iter().collect::<Vec<_>>().as_slice());
        let legacy_large = legacy_median(large_dataset.y_iter().collect::<Vec<_>>().as_slice());

        let new_small = median(&mut small_dataset.y()).unwrap();
        let new_large = median(&mut large_dataset.y()).unwrap();

        let small_epsilon = 1e-12 * legacy_small.abs();
        let large_epsilon = 1e-12 * legacy_large.abs();

        assert_close!(
            legacy_small,
            new_small,
            epsilon = small_epsilon,
            "Small dataset median do not match legacy implementation"
        );
        assert_close!(
            legacy_large,
            new_large,
            epsilon = large_epsilon,
            "Large dataset median do not match legacy implementation"
        );

        //
        // Now we test known medians for a uniform distribution
        let mut uniform_data: Vec<f64> = (1..=100).map(f64::from).collect();
        let uniform_median = median(&mut uniform_data).unwrap();
        assert_close!(
            uniform_median,
            50.5,
            epsilon = 1e-12,
            "Uniform distribution median does not match expected value"
        );

        //
        // And a large uniform distribution
        let mut large_uniform_data: Vec<f64> = (1..=100_000).map(f64::from).collect();
        let large_uniform_median = median(&mut large_uniform_data).unwrap();
        assert_close!(
            large_uniform_median,
            50_000.5,
            epsilon = 1e-12,
            "Large uniform distribution median does not match expected value"
        );

        //
        // Empty datasets should return None
        let mut empty_data: Vec<f64> = vec![];
        let empty = median(&mut empty_data);
        assert!(
            empty.is_none(),
            "Empty dataset should return None for median"
        );
    }

    #[test]
    fn test_spread() {
        let small_dataset = SRC.solve_range(0.0..=1_000.0, 1.0);
        let large_dataset = SRC.solve_range(0.0..=1_000.0, 0.01);

        let small = spread(small_dataset.y_iter()).unwrap();
        let large = spread(large_dataset.y_iter()).unwrap();

        assert_close!(
            small,
            large,
            epsilon = 1e-12,
            "Spread values do not match for small and large datasets"
        );
    }
}
