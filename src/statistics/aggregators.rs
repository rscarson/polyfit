use crate::value::Value;

/// Represents a type which can calculate some statistic over input values.
///
/// For example, calling [`Aggregator::inspect`] repeatedly on a [`Variance`]
/// will calculate the running mean & variance of those inputs.
pub trait Aggregator<T> {
    /// Processes all the elements in the given slice.
    fn inspect_slice(&mut self, elements: &[T]);

    /// Processes a single element.
    fn inspect(&mut self, element: T) {
        self.inspect_slice(std::slice::from_ref(&element));
    }

    /// Returns an aggregator that has processed zero items.
    ///
    /// This is also the identity element for [`Self::merge`].
    fn empty() -> Self
    where
        Self: Sized;

    /// Merges two instances to produce a combined result.
    ///
    /// This is most useful for parallel scenarios, where rather than feeding all
    /// results to one aggregator, you can split up the input into different chunks,
    /// run the aggregator on each chunk separately, then merge the results.
    fn merge(a: Self, b: Self) -> Self
    where
        Self: Sized;

    /// Produces this aggregator's result for the elements in the given slice.
    fn from_slice(elements: &[T]) -> Self
    where
        Self: Sized,
    {
        let mut agg = Self::empty();
        agg.inspect_slice(elements);
        agg
    }

    /// Produces this aggregator's -- usually trivial -- result for a single item.
    fn from_scalar(element: T) -> Self
    where
        Self: Sized,
    {
        let mut agg = Self::empty();
        agg.inspect(element);
        agg
    }
}

fn _assert_aggregator_is_dyn_compatible(agg: &mut dyn Aggregator<i32>) {
    agg.inspect(1);
}

macro_rules! impl_aggregator_for_tuple {
    ($($i:ident)* / $($n:tt)*) => {
        impl<T: Value $(, $i: Aggregator<T>)* > Aggregator<T> for ( $( $i,)* )
            where
        {
            fn inspect_slice(&mut self, elements: &[T]) {
                $(
                    self.$n.inspect_slice(elements);
                )*
            }
            fn empty() -> Self {
                (
                    $(
                        <$i>::empty(),
                    )*
                )
            }
            fn merge(a: Self, b: Self) -> Self {
                (
                    $(
                        <$i>::merge(a.$n, b.$n),
                    )*
                )
            }
        }
    };
}
impl_aggregator_for_tuple!(A / 0);
impl_aggregator_for_tuple!(A B / 0 1);
impl_aggregator_for_tuple!(A B D / 0 1 2);
impl_aggregator_for_tuple!(A B C D / 0 1 2 3);
impl_aggregator_for_tuple!(A B C D E / 0 1 2 3 4);
impl_aggregator_for_tuple!(A B C D E F / 0 1 2 3 4 5);
impl_aggregator_for_tuple!(A B C D E F G / 0 1 2 3 4 5 6);
impl_aggregator_for_tuple!(A B C D E F G H / 0 1 2 3 4 5 6 7);

/// Tracks the range of values seen.
#[derive(Debug, Copy, Clone)]
pub struct MinMax<T> {
    /// The least value seen thus far.
    pub min: T,
    /// The greatest value seen thus far.
    pub max: T,
}
impl<T: Value> Aggregator<T> for MinMax<T> {
    fn inspect_slice(&mut self, elements: &[T]) {
        for element in elements {
            self.min = <T as Value>::min(self.min, *element);
            self.max = <T as Value>::max(self.max, *element);
        }
    }
    fn empty() -> Self {
        Self {
            min: T::infinity(),
            max: T::neg_infinity(),
        }
    }
    fn merge(a: Self, b: Self) -> Self {
        Self {
            min: <T as Value>::min(a.min, b.min),
            max: <T as Value>::max(a.max, b.max),
        }
    }
}

/// Tracks the mean and variance of the values seen.
///
/// Uses a parallel online version of Welford's algorithm for numerical stability.
#[derive(Debug, Copy, Clone)]
pub struct Variance<T> {
    w: T,
    m: T,
    m2: T,
}
impl<T: Value> Variance<T> {
    /// The total weight of values seen so far.
    ///
    /// If one isn't using customized weights, this is the count of values.
    pub fn weight(&self) -> T {
        self.w
    }
    /// The mean (μ) of values seen so far.
    pub fn mean(&self) -> T {
        self.m
    }
    /// The variance (σ²) of values seen so far.
    ///
    /// This is the *population* variance (not the sample-corrected variance).
    pub fn variance(&self) -> T {
        self.m2 / self.w
    }
    /// The standard deviation (σ) of values seen so far.
    ///
    /// This is the *population* standard deviation (not the sample-corrected version).
    pub fn stdev(&self) -> T {
        self.variance().sqrt()
    }

    /// Process an element `x` with a custom weight `w`.
    pub fn inspect_weighted(&mut self, x: T, w: T) {
        *self = Self::merge(*self, Self::from_scalar_and_weight(x, w));
    }

    /// Represents a single value with a custom weight.
    pub fn from_scalar_and_weight(x: T, w: T) -> Self {
        Self {
            w,
            m: x,
            m2: T::zero(),
        }
    }

    fn from_short_slice(elements: &[T]) -> Self {
        let weight = T::from_positive_int(elements.len());

        let mut sum = T::zero();
        for element in elements {
            sum += *element;
        }
        let mean = sum / weight;

        let mut m2 = T::zero();
        for element in elements {
            m2 += Value::powi(*element - mean, 2);
        }

        Self {
            w: weight,
            m: mean,
            m2,
        }
    }
}
impl<T: Value> Aggregator<T> for Variance<T> {
    fn empty() -> Self
    where
        Self: Sized,
    {
        Self {
            w: T::zero(),
            m: T::zero(),
            m2: T::zero(),
        }
    }
    fn merge(a: Self, b: Self) -> Self {
        let w = a.w + b.w;
        let m = (a.w * a.m + b.w * b.m) / w;
        let delta = b.m - a.m;
        let m2 = (a.m2 + b.m2) + (delta * delta) * (a.w * b.w) / w;
        Self { w, m, m2 }
    }
    fn inspect_slice(&mut self, elements: &[T]) {
        *self = Self::merge(*self, Self::from_slice(elements));
    }

    fn from_scalar(x: T) -> Self {
        Self::from_scalar_and_weight(x, T::one())
    }
    fn from_slice(elements: &[T]) -> Self {
        // Chunk things up to avoid using the `merge` approach all the time.
        // TODO: parallelize the chunks if big enough to be worth it.
        const CHUNK_SIZE: usize = 1 << 10;
        let (prefix, chunks) = elements.as_rchunks::<CHUNK_SIZE>();
        chunks
            .iter()
            .map(|chunk| Self::from_short_slice(chunk))
            .fold(Self::from_short_slice(prefix), Self::merge)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_variance() {
        let agg = Variance::from_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!(agg.weight(), 8.0);
        assert_eq!(agg.mean(), 5.0);
        assert_eq!(agg.variance(), 4.0);
        assert_eq!(agg.stdev(), 2.0);
    }

    #[test]
    fn test_weighted_variance() {
        let mut agg = Variance::empty();

        agg.inspect_weighted(3.0, 2.0);
        assert_eq!(agg.weight(), 2.0);
        assert_eq!(agg.mean(), 3.0);
        assert_eq!(agg.variance(), 0.0);

        agg.inspect_weighted(7.5, 4.0);
        assert_eq!(agg.weight(), 6.0);
        assert_eq!(agg.mean(), 6.0);
        assert_eq!(agg.variance(), 4.5);
    }

    #[test]
    fn test_tupled_aggregators() {
        let (mm, var) = <(MinMax<_>, Variance<_>)>::from_slice(&[1.0, 2.0, 4.0, 5.0]);
        assert_eq!(mm.min, 1.0);
        assert_eq!(mm.max, 5.0);
        assert_eq!(var.weight(), 4.0);
        assert_eq!(var.mean(), 3.0);
        assert_eq!(var.variance(), 2.5);
    }
}
