use crate::{
    statistics::{self, DomainNormalizer},
    transforms::Transform,
    value::Value,
};

/// Transformations around normalization or otherwise controlling range
pub enum NormalizationTransform<T: Value> {
    /// Normalizes the dataset to a specified range.
    ///
    /// Each element is linearly scaled to fit within `[min, max]`. Useful for
    /// mapping values to a standard range before further processing or ML workflows.
    ///
    /// ![Domain example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/domain_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = (x - x_min) / (x_max - x_min) * (max - min) + min
    /// where
    ///   x_min and x_max are the minimum and maximum of the original dataset.
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `min`: Minimum value of the target range.
    /// - `max`: Maximum value of the target range.
    Domain {
        /// Minimum value of the target range
        min: T,

        /// Maximum value of the target range
        max: T,
    },

    /// Restricts all values in the dataset to a specified range.
    ///
    /// Any element smaller than `min` is set to `min`, and any element larger than
    /// `max` is set to `max`. Useful for bounding outliers or enforcing hard limits.
    ///
    /// ![Clip example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/clip_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Element-wise operation:
    ///
    /// ```math
    /// xₙ = min(max(x, min), max)
    /// where
    ///   min and max are the specified bounds.
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `min`: Lower bound of the allowed range.
    /// - `max`: Upper bound of the allowed range.
    Clip {
        /// Lower bound of the allowed range
        min: T,

        /// Upper bound of the allowed range
        max: T,
    },

    /// Centers the dataset by subtracting its mean from every element.
    ///
    /// After this transformation, the dataset has a mean of zero, but its variance
    /// and overall shape remain unchanged. Useful as a preprocessing step in
    /// statistics and machine learning.
    ///
    /// ![Mean subtraction example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/mean_subtraction_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x - mean(x)
    /// ```
    /// </div>
    MeanSubtraction,

    /// Normalizes the dataset to zero mean and unit variance.
    ///
    /// Each element is centered by subtracting the dataset mean, then scaled by
    /// dividing with the standard deviation. This is a common preprocessing step
    /// in statistics and machine learning to make features comparable.
    ///
    /// ![Z-Score example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/zscore_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Element-wise operation:
    ///
    /// ```math
    /// xₙ = (x - mean(x)) / std(x)
    /// where
    ///   mean(x) is the average of the dataset
    ///   std(x) is the standard deviation of the dataset
    /// ```
    /// </div>
    ZScore,
}

impl<T: Value> Transform<T> for NormalizationTransform<T> {
    fn apply<'a>(&self, data: impl Iterator<Item = &'a mut T>) {
        let data: Vec<_> = data.collect();
        match self {
            Self::Domain { min, max } => {
                let normalizer =
                    DomainNormalizer::from_data(data.iter().map(|d| **d), (*min, *max));
                for value in data {
                    *value = normalizer.normalize(*value);
                }
            }

            Self::Clip { min, max } => {
                for value in data {
                    *value = nalgebra::RealField::clamp(*value, *min, *max);
                }
            }

            Self::MeanSubtraction => {
                let mean = statistics::mean(data.iter().map(|d| **d));
                for value in data {
                    *value -= mean;
                }
            }

            Self::ZScore => {
                let (s, m) = statistics::stddev_and_mean(data.iter().map(|d| **d));
                for value in data {
                    *value = (*value - m) / s;
                }
            }
        }
    }
}

/// Trait for applying normalization techniques to a dataset.
pub trait ApplyNormalization<T: Value> {
    /// Normalizes the dataset to a specified range.
    ///
    /// Each element is linearly scaled to fit within `[min, max]`. Useful for
    /// mapping values to a standard range before further processing or ML workflows.
    ///
    /// ![Domain example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/domain_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = (x - x_min) / (x_max - x_min) * (max - min) + min
    /// where
    ///   x_min and x_max are the minimum and maximum of the original dataset.
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `min`: Minimum value of the target range.
    /// - `max`: Maximum value of the target range.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyNormalization;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_domain_normalization(0.0, 2.5);
    /// ```
    #[must_use]
    fn apply_domain_normalization(self, min: T, max: T) -> Self;

    /// Restricts all values in the dataset to a specified range.
    ///
    /// Any element smaller than `min` is set to `min`, and any element larger than
    /// `max` is set to `max`. Useful for bounding outliers or enforcing hard limits.
    ///
    /// ![Clip example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/clip_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Element-wise operation:
    ///
    /// ```math
    /// xₙ = min(max(x, min), max)
    /// where
    ///   min and max are the specified bounds.
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `min`: Lower bound of the allowed range.
    /// - `max`: Upper bound of the allowed range.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyNormalization;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_clipping(0.0, 2.5);
    /// ```
    #[must_use]
    fn apply_clipping(self, min: T, max: T) -> Self;

    /// Centers the dataset by subtracting its mean from every element.
    ///
    /// After this transformation, the dataset has a mean of zero, but its variance
    /// and overall shape remain unchanged. Useful as a preprocessing step in
    /// statistics and machine learning.
    ///
    /// ![Mean subtraction example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/mean_subtraction_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x - mean(x)
    /// ```
    /// </div>
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyNormalization;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_mean_subtraction();
    /// ```
    #[must_use]
    fn apply_mean_subtraction(self) -> Self;

    /// Normalizes the dataset to zero mean and unit variance.
    ///
    /// Each element is centered by subtracting the dataset mean, then scaled by
    /// dividing with the standard deviation. This is a common preprocessing step
    /// in statistics and machine learning to make features comparable.
    ///
    /// ![Z-Score example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/zscore_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Element-wise operation:
    ///
    /// ```math
    /// xₙ = (x - mean(x)) / std(x)
    /// where
    ///   mean(x) is the average of the dataset
    ///   std(x) is the standard deviation of the dataset
    /// ```
    /// </div>
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyNormalization;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_z_score_normalization();
    /// ```
    #[must_use]
    fn apply_z_score_normalization(self) -> Self;
}
impl<T: Value> ApplyNormalization<T> for Vec<(T, T)> {
    fn apply_domain_normalization(mut self, min: T, max: T) -> Self {
        NormalizationTransform::Domain { min, max }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_clipping(mut self, min: T, max: T) -> Self {
        NormalizationTransform::Clip { min, max }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_mean_subtraction(mut self) -> Self {
        NormalizationTransform::MeanSubtraction.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_z_score_normalization(mut self) -> Self {
        NormalizationTransform::ZScore.apply(self.iter_mut().map(|(_, y)| y));
        self
    }
}

/// Smoothing transformations for time series data.
pub enum SmoothingTransform<T: Value> {
    /// Applies moving average smoothing to the dataset.
    ///
    /// Each element is replaced with the average of values within a fixed-size
    /// sliding window centered on that element.
    ///
    /// ![Moving average example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/moving_average_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = (1/k) Σⱼ xⱼ
    /// where
    ///   k = window_size
    /// ```
    /// - Near the boundaries, the window is truncated.
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `window_size`: Number of neighboring points (including the current point)
    ///   used to compute the average. Larger values increase smoothing but blur
    ///   sharp features.
    MovingAverage {
        /// Number of neighboring points (including the current point) used to compute the average.
        /// Larger values increase smoothing but blur sharp features.
        window_size: usize,
    },

    /// Applies Gaussian smoothing to the dataset.
    ///
    /// Each element is replaced with a weighted average of its neighbors,
    /// where weights follow a Gaussian (normal) distribution centered on
    /// the element. This smooths noise while preserving overall shape.
    ///
    /// Normalization ensures the weights sum to 1.
    ///
    /// ![Gaussian smoothing example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/gaussian_smoothing_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// wⱼ = exp( - ( (xᵢ - xⱼ)² ) / (2 σ²) )
    /// xₙ = ( Σⱼ wⱼ · xⱼ ) / ( Σⱼ wⱼ )
    /// where
    ///   xⱼ are points within ±3σ of xᵢ
    ///   wⱼ are Gaussian weights based on distance from xᵢ
    ///   σ is the sigma parameter
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `sigma`: Standard deviation of the Gaussian kernel.
    ///   Larger values apply stronger smoothing over a wider neighborhood.
    Gaussian {
        /// Standard deviation of the Gaussian kernel.
        /// Larger values apply stronger smoothing over a wider neighborhood.
        sigma: T,
    },
}
impl<T: Value> Transform<T> for SmoothingTransform<T> {
    fn apply<'a>(&self, data: impl Iterator<Item = &'a mut T>) {
        let data: Vec<_> = data.collect();
        match self {
            Self::MovingAverage { window_size } => {
                let n = data.len();
                if *window_size == 0 || n == 0 {
                    return;
                }
                let mut result = vec![T::zero(); n];
                let half = *window_size / 2;

                for i in 0..n {
                    let start = i.saturating_sub(half);
                    let end = usize::min(i + half + 1, n);
                    let count = T::from(end - start).unwrap();

                    let mut sum = T::zero();
                    for j in start..end {
                        sum += *data[j];
                    }
                    result[i] = sum / count;
                }

                for (v, r) in data.into_iter().zip(result) {
                    *v = r;
                }
            }

            Self::Gaussian { sigma } => {
                let n = data.len();
                if n == 0 {
                    return;
                }

                let sigma_f = *sigma;
                if sigma_f <= T::zero() {
                    return;
                }

                // ±3σ kernel size
                let radius = nalgebra::ComplexField::ceil((T::two() + T::one()) * sigma_f)
                    .as_usize()
                    .unwrap_or_default();
                let kernel_size = 2 * radius + 1;

                // Precompute Gaussian weights
                let mut kernel = Vec::with_capacity(kernel_size);
                let mut sum = T::zero();
                for i in 0..kernel_size {
                    let x = T::from(i as f64).unwrap() - T::from(radius as f64).unwrap();
                    let w = (-(x * x) / (T::from(2.0).unwrap() * sigma_f * sigma_f)).exp();
                    kernel.push(w);
                    sum += w;
                }

                // Normalize
                for w in &mut kernel {
                    *w /= sum;
                }

                // Apply convolution
                let mut result = vec![T::zero(); n];
                for i in 0..n {
                    let mut acc = T::zero();
                    for j in 0..kernel_size {
                        let idx = (i + j).saturating_sub(radius);
                        if idx < n {
                            acc += *data[idx] * kernel[j];
                        }
                    }
                    result[i] = acc;
                }

                // Copy back
                for (v, r) in data.into_iter().zip(result) {
                    *v = r;
                }
            }
        }
    }
}

/// Smoothing transformations for time series data.
pub trait ApplySmoothing<T: Value> {
    /// Applies moving average smoothing to the dataset.
    ///
    /// Each element is replaced with the average of values within a fixed-size
    /// sliding window centered on that element.
    ///
    /// ![Moving average example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/moving_average_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = (1/k) Σⱼ xⱼ
    /// where
    ///   k = window_size
    /// ```
    /// - Near the boundaries, the window is truncated.
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `window_size`: Number of neighboring points (including the current point)
    ///   used to compute the average. Larger values increase smoothing but blur
    ///   sharp features.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplySmoothing;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)].apply_moving_average_smoothing(2);
    /// ```
    #[must_use]
    fn apply_moving_average_smoothing(self, window_size: usize) -> Self;

    /// Applies Gaussian smoothing to the dataset.
    ///
    /// Each element is replaced with a weighted average of its neighbors,
    /// where weights follow a Gaussian (normal) distribution centered on
    /// the element. This smooths noise while preserving overall shape.
    ///
    /// ![Gaussian smoothing example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/gaussian_smoothing_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// wⱼ = exp( - ( (xᵢ - xⱼ)² ) / (2 σ²) )
    /// xₙ = ( Σⱼ wⱼ · xⱼ ) / ( Σⱼ wⱼ )
    /// where
    ///   xⱼ are points within ±3σ of xᵢ
    ///   wⱼ are Gaussian weights based on distance from xᵢ
    ///   σ is the sigma parameter
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `sigma`: Standard deviation of the Gaussian kernel.
    ///   Larger values apply stronger smoothing over a wider neighborhood.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplySmoothing;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_gaussian_smoothing(1.0);
    /// ```
    #[must_use]
    fn apply_gaussian_smoothing(self, sigma: T) -> Self;
}
impl<T: Value> ApplySmoothing<T> for Vec<(T, T)> {
    fn apply_moving_average_smoothing(mut self, window_size: usize) -> Self {
        SmoothingTransform::MovingAverage { window_size }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_gaussian_smoothing(mut self, sigma: T) -> Self {
        SmoothingTransform::Gaussian { sigma }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use crate::transforms::{ApplyNormalization, ApplySmoothing};

    #[test]
    fn test_domain_normalization() {
        let data = vec![(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)];
        let normalized = data.clone().apply_domain_normalization(0.0, 1.0);
        let expected = vec![(1.0, 0.0), (2.0, 0.5), (3.0, 1.0)];
        assert_eq!(normalized, expected);
    }

    #[test]
    fn test_clipping() {
        let data = vec![(1.0, -1.0), (2.0, 2.5), (3.0, 5.0)];
        let clipped = data.clone().apply_clipping(0.0, 3.0);
        let expected = vec![(1.0, 0.0), (2.0, 2.5), (3.0, 3.0)];
        assert_eq!(clipped, expected);
    }

    #[test]
    fn test_mean_subtraction() {
        let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
        let mean_subtracted = data.clone().apply_mean_subtraction();
        let expected = vec![(1.0, -2.0), (2.0, 0.0), (3.0, 2.0)];
        assert_eq!(mean_subtracted, expected);
    }

    #[test]
    fn test_z_score_normalization() {
        let data = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
        let z_scored = data.clone().apply_z_score_normalization();
        let expected = [
            (1.0f64, -1.224744871391589f64),
            (2.0f64, 0.0f64),
            (3.0f64, 1.224744871391589f64),
        ];
        for (a, b) in z_scored.iter().zip(expected.iter()) {
            assert!((a.1 - b.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_moving_average_smoothing() {
        let data = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)];
        let smoothed = data.clone().apply_moving_average_smoothing(3);
        let expected: Vec<(f64, f64)> =
            vec![(1.0, 1.5), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 4.5)];
        for (a, b) in smoothed.iter().zip(expected.iter()) {
            assert!((a.1 - b.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gaussian_smoothing() {
        let data = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)];
        let smoothed = data.clone().apply_gaussian_smoothing(1.0);
        let expected: Vec<(f64, f64)> = vec![
            (1.0, 1.3633465391466744),
            (2.0, 2.062871678972902),
            (3.0, 2.977834759123781),
            (4.0, 3.644935167038806),
            (5.0, 3.1342791599844624),
        ];
        for (a, b) in smoothed.iter().zip(expected.iter()) {
            assert!((a.1 - b.1).abs() < 0.01);
        }
    }
}
