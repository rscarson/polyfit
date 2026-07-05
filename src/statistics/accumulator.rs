#![allow(
    dead_code,
    reason = "Parallel mostly, but we might make this public later"
)]
use crate::value::Value;

/// The maximum size of a dataset that will be processed in-memory before switching to an online algorithm.
pub const BIG_DATA_SIZE: usize = 2500;

/// The minimum size of a dataset that will be processed in parallel before switching to an online algorithm.
pub const PAR_DATA_SIZE: usize = 1_000_000;

/// Enum representing the execution strategy for processing data, based on the size of the dataset and available features
///
/// The default, safe behavior is to use the `Streaming` variant for large datasets, which processes data in a streaming fashion
/// with stable methods or Kahan summation
///
/// If the iterator can prove a maximum size below a certain threshold, the `Eager` variant will be used, which may allocate and processes
/// faster for small sets, but may be less stable for large datasets
///
/// If the "parallel" feature is enabled, and the iterator can prove a minimum size above a certain threshold, the `Parallel` variant will be used,
/// which will use Rayon to process the data in parallel, which is faster for large datasets.
///
/// ## Justifications
/// This is safe even for badly behaved iterators, so there's no assumption of truth in the `size_hint`.
///
/// - If the iterator can prove a maximum size below threshold, eager is always safe and faster
/// - If the iterator cannot prove a max, and `size_hint` is worstcase O(n), then for large sets the check is worth it for the memory and stability benefits of streaming
/// - If the iterator can prove a minimum size above threshold, parallel is always faster and more numerically stable
/// - If no upper bound exists, streaming is always safer
/// - If the iterator is badly behaved and returns a `size_hint` of (0, None), then streaming is always safer
/// - If the iterator is badly behaved and returns an underestimate of the size, the result will be equivalent but less precise, and still correct
/// - If the iterator is badly behaved and returns an overestimate of the size, the result will be equivalent and precise, but may be less efficient
pub enum ExecutionRouter<I: Iterator> {
    /// The faster path for small datasets, which may allocate and process faster, but may be less stable for large datasets
    ///
    /// This will only be selected if the iterator can prove a maximum size below a certain threshold
    Eager(I),

    /// The default, safe path for large datasets, which processes data in a streaming fashion with stable methods or Kahan summation
    Streaming(I),
    //    #[cfg(feature = "parallel")]
    //    Parallel(I),
}
impl<I: Iterator> ExecutionRouter<I> {
    /// Routes the given iterator to the appropriate execution strategy based on its size hint and available features
    pub fn route(data: I) -> Self {
        match data.size_hint() {
            (_, Some(n)) if n <= BIG_DATA_SIZE => Self::Eager(data),

            //    #[cfg(feature = "parallel")]
            //    (n, _) if n >= PAR_DATA_SIZE => Self::Parallel(data),
            _ => Self::Streaming(data),
        }
    }
}

/// Kahan-Neumaier summation algorithm for improved numerical stability when summing floating-point numbers
///
/// Reduces the numerical error that can occur when adding a sequence of finite precision floating-point numbers
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KahanAccumulator<T: Value> {
    sum: T,
    c: T,
}
impl<T: Value> Default for KahanAccumulator<T> {
    fn default() -> Self {
        KahanAccumulator {
            sum: T::zero(),
            c: T::zero(),
        }
    }
}
impl<T: Value> KahanAccumulator<T> {
    pub fn add(&mut self, value: T) {
        let t = self.sum + value;
        if Value::abs(self.sum) >= Value::abs(value) {
            self.c += (self.sum - t) + value; // If sum is bigger, low-order digits of input[i] are lost.
        } else {
            self.c += (value - t) + self.sum; // Else low-order digits of sum are lost.
        }
        self.sum = t;
    }

    pub fn merge(&mut self, other: Self) {
        let t = self.sum + other.sum;
        if Value::abs(self.sum) >= Value::abs(other.sum) {
            self.c += (self.sum - t) + other.sum;
        } else {
            self.c += (other.sum - t) + self.sum;
        }
        self.sum = t;
        self.c += other.c; // Linearly combine error terms directly
    }

    pub fn total(&self) -> T {
        self.sum + self.c
    }
}

pub trait AccumulatorFactory
where
    Self: Sized + Send + Sync,
{
    type Config: Clone + Send + Sync;

    fn create(config: Self::Config) -> Self;
}

pub trait DefaultAccumulatorFactory: Default + Send + Sync {
    fn new_from_par_iter<I: IntoIterator<Item = V>, T: Value, V, O>(iter: I) -> Self
    where
        Self: Accumulator<T, V, O> + AccumulatorFactory,
        <Self as AccumulatorFactory>::Config: Default,
        I::IntoIter: Send,
        V: Send + Copy,
    {
        Self::new_from_par_iter_with_config(iter, <Self as AccumulatorFactory>::Config::default())
    }

    fn eager_total<I, T: Value, V: Copy, O>(values: I) -> Option<O>
    where
        Self: Accumulator<T, V, O> + AccumulatorFactory,
        <Self as AccumulatorFactory>::Config: Default,
        I: IntoIterator<Item = V>,
    {
        Self::eager_total_with_config(values, <Self as AccumulatorFactory>::Config::default())
    }

    fn new_from_iter<I, T: Value, V: Copy, O>(iter: I) -> Self
    where
        Self: Accumulator<T, V, O> + AccumulatorFactory,
        <Self as AccumulatorFactory>::Config: Default,
        I: IntoIterator<Item = V>,
    {
        Self::new_from_iter_with_config(iter, <Self as AccumulatorFactory>::Config::default())
    }
}
impl<F> AccumulatorFactory for F
where
    F: DefaultAccumulatorFactory + Default + Send + Sync,
{
    type Config = ();

    fn create(_config: Self::Config) -> Self {
        Self::default()
    }
}

/// Trait for accumulators that can accumulate values and compute a total
pub trait Accumulator<T: Value, V: Copy = T, O = T>: AccumulatorFactory + Copy {
    /// Adds a value to the accumulator
    fn add(&mut self, value: V);

    /// Merges another accumulator into this one
    fn merge(&mut self, other: Self);

    /// Resolve a small dataset quickly, without the overhead of the full accumulator
    ///
    /// By default just calls `add_iter` on the values, but can be overridden for more efficient implementations
    ///
    /// This will often allocate the iterator, but does not always have to.
    fn eager_total_with_config(
        values: impl IntoIterator<Item = V>,
        config: Self::Config,
    ) -> Option<O>
    where
        Self: AccumulatorFactory,
    {
        Self::new_from_iter_with_config(values, config).total()
    }

    /// Gets the total value from the accumulator, if available
    fn total(&self) -> Option<O>;

    /// Resets the accumulator to its initial state
    fn reset(&mut self);

    /// Creates a new accumulator from an iterator of values
    fn new_from_iter_with_config<I: IntoIterator<Item = V>>(iter: I, config: Self::Config) -> Self
    where
        Self: AccumulatorFactory,
    {
        let mut acc = Self::create(config);
        acc.add_iter(iter);
        acc
    }

    /// Adds an iterator of values to the accumulator
    fn add_iter<I: IntoIterator<Item = V>>(&mut self, iter: I) {
        for value in iter {
            self.add(value);
        }
    }

    /// Adds a parallel iterator of values to the accumulator, if the "parallel" feature is enabled
    #[cfg(feature = "parallel")]
    fn new_from_par_iter_with_config<I: IntoIterator<Item = V>>(
        iter: I,
        config: Self::Config,
    ) -> Self
    where
        Self: AccumulatorFactory,
        I::IntoIter: Send,
        V: Send,
    {
        use rayon::iter::ParallelBridge;
        use rayon::iter::ParallelIterator;

        let iter = iter.into_iter().par_bridge();
        let acc: Self = iter
            .fold(
                || Self::create(config.clone()),
                |mut acc, val| {
                    acc.add(val);
                    acc
                },
            )
            .reduce(
                || Self::create(config.clone()),
                |mut acc1, acc2| {
                    acc1.merge(acc2);
                    acc1
                },
            );

        acc
    }
}

/// Struct representing a fitted value, which includes both the observed value and the fitted (predicted) value from a model
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FittedValue<T: Value> {
    /// The observed value from the dataset
    pub y: T,
    /// The fitted (predicted) value from the model
    pub y_fit: T,
}
impl<T: Value> FittedValue<T> {
    /// Creates a new `FittedValue` with the given observed and fitted values
    pub fn new(y: T, y_fit: T) -> Self {
        FittedValue { y, y_fit }
    }

    /// Creates an iterator of `FittedValue` from two iterators: one for observed values and one for fitted values
    pub fn from_iters(
        y: impl Iterator<Item = T>,
        y_fit: impl Iterator<Item = T>,
    ) -> impl Iterator<Item = Self> {
        y.zip(y_fit).map(|(y, y_fit)| FittedValue { y, y_fit })
    }

    /// Returns the residual (difference between observed and fitted value)
    pub fn residual(&self) -> T {
        self.y - self.y_fit
    }
}
