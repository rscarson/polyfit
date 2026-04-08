use crate::{
    statistics::{mean, Confidence, ConfidenceBand},
    value::Value,
};
/// A value with an associated amount of uncertainty, represented by a mean
/// and a standard deviation.
///
/// This is useful when a quantity is computed multiple times and the results
/// vary, such as error estimates from cross-validation.
///
/// The `mean` represents the typical value, and `std_dev` indicates how much
/// the value tends to vary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UncertainValue<T: Value> {
    /// The mean (average) value. This is a point estimate of the quantity.
    pub mean: T,

    /// The standard deviation, representing the uncertainty or variability
    /// around the mean.
    pub std_dev: T,
}
impl<T: Value> UncertainValue<T> {
    /// Creates a new `UncertainValue` with the given mean and standard deviation.
    pub fn new(mean: T, std_dev: T) -> Self {
        UncertainValue { mean, std_dev }
    }

    /// Creates a new `UncertainValue` from an iterator of values.
    ///
    /// Computes the mean and standard deviation of the provided values.
    pub fn new_from_values(values: impl Iterator<Item = T>) -> Option<Self> {
        mean(values)
    }

    /// Returns a likely range of values at the given confidence level.
    ///
    /// This computes a symmetric range around the mean based on the standard
    /// deviation. Larger confidence levels produce wider ranges.
    ///
    /// # Parameters
    /// - `confidence`: The confidence level for the range calculation.
    ///
    /// # Returns
    /// A tuple containing the lower and upper bounds of the range.
    pub fn range(&self, confidence: Confidence) -> (T, T) {
        let z = T::from_f64(confidence.z_score()).unwrap_or(T::one());
        let margin = z * self.std_dev;
        (self.mean - margin, self.mean + margin)
    }

    /// Returns a confidence band representing the likely range of values
    /// at the given confidence level.
    ///
    /// # Parameters
    /// - `confidence`: The confidence level for the band calculation.
    ///
    /// # Returns
    /// A `ConfidenceBand` containing the mean value and the computed range.
    pub fn confidence_band(&self, confidence: Confidence) -> ConfidenceBand<T> {
        let (min, max) = self.range(confidence);
        ConfidenceBand {
            level: confidence,
            tolerance: None,
            value: self.mean,
            lower: min,
            upper: max,
        }
    }
}
