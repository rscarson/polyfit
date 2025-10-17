//! Utilities for adding transformations to data
//!
//! Data can be transformed by anything implementing the [`Transform`] trait, which applies the transformation over a set of values.
//!
//! The [`Transformable`] trait is a convenient wrapper that allows you to apply transformations to your data more easily.
//!
//! Transformations can include operations like scaling, normalization, and noise addition.
//!
//! Predefined transformations are available for common use cases:
//!
//! # Noise: [`NoiseTransform`] / [`ApplyNoise`]
//! - Gaussian noise: [`NoiseTransform::CorrelatedGaussian`]
//!   - Applies correlated Gaussian noise to the data.
//!   - [`ApplyNoise::apply_correlated_noise`] allows you to apply it to the Y channel of an (X, Y) dataset
//!   - [`ApplyNoise::apply_normal_noise`] is similar, but applies uncorrelated Gaussian noise.
//! - Uniform noise: [`NoiseTransform::Uniform`]
//!   - Applies uniform noise to the data.
//!   - [`ApplyNoise::apply_uniform_noise`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Poisson noise: [`NoiseTransform::Poisson`]
//!   - Applies Poisson noise to the data.
//!   - [`ApplyNoise::apply_poisson_noise`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Impulse noise: [`NoiseTransform::Impulse`]
//!   - Applies impulse noise to the data.
//!   - [`ApplyNoise::apply_impulse_noise`] allows you to apply it to the Y channel of an (X, Y) dataset
//!   - [`ApplyNoise::apply_salt_pepper_noise`] is similar, but pins the noise to the edges of the range.
//!
//! # Scaling: [`ScaleTransform`] / [`ApplyScale`]
//! - Shift scaling: [`ScaleTransform::Shift`]
//!   - Applies shift scaling to the data.
//!   - [`ApplyScale::apply_shift_scale`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Linear scaling: [`ScaleTransform::Linear`]
//!   - Applies linear scaling to the data.
//!   - [`ApplyScale::apply_linear_scale`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Quadratic scaling: [`ScaleTransform::Quadratic`]
//!   - Applies quadratic scaling to the data.
//!   - [`ApplyScale::apply_quadratic_scale`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Cubic Scaling: [`ScaleTransform::Cubic`]
//!   - Applies cubic scaling to the data.
//!   - [`ApplyScale::apply_cubic_scale`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Polynomial scaling: [`ApplyScale::apply_polynomial_scale`]
//!   - Generalized case applying a polynomial function in any basis to each value.
//!
//! # Smoothing: [`SmoothingTransform`] / [`ApplySmoothing`]
//! - Moving-Average Smoothing: [`SmoothingTransform::MovingAverage`]
//!   - Applies moving-average smoothing to the data.
//!   - [`ApplySmoothing::apply_moving_average_smoothing`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Gaussian Smoothing: [`SmoothingTransform::Gaussian`]
//!   - Applies Gaussian smoothing to the data.
//!   - [`ApplySmoothing::apply_gaussian_smoothing`] allows you to apply it to the Y channel of an (X, Y) dataset
//!
//! # Normalization: [`NormalizationTransform`] / [`ApplyNormalization`]
//! - Domain Normalization: [`NormalizationTransform::Domain`]
//!   - Normalizes the dataset to a specified range.
//!   - [`ApplyNormalization::apply_domain_normalization`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Clipping: [`NormalizationTransform::Clip`]
//!   - Restricts all values in the dataset to a specified range.
//!   - [`ApplyNormalization::apply_clipping`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Mean Subtraction: [`NormalizationTransform::MeanSubtraction`]
//!   - Centers the dataset by subtracting its mean from every element.
//!   - [`ApplyNormalization::apply_mean_subtraction`] allows you to apply it to the Y channel of an (X, Y) dataset
//! - Z-Score Normalization: [`NormalizationTransform::ZScore`]
//!   - Normalizes the dataset to zero mean and unit variance.
//!   - [`ApplyNormalization::apply_z_score_normalization`] allows you to apply it to the Y channel of an (X, Y) dataset
use crate::value::Value;

mod noise;
pub use noise::{ApplyNoise, NoiseTransform, Strength};

mod scale;
pub use scale::{ApplyScale, ScaleTransform};

mod normalization;
pub use normalization::{ApplyNormalization, NormalizationTransform};
pub use normalization::{ApplySmoothing, SmoothingTransform};

pub use rand;
pub use rand_distr;

/// Trait for applying transformations to data.
pub trait Transform<T: Value> {
    /// Applies the transformation to the given data.
    fn apply<'a>(&self, data: impl Iterator<Item = &'a mut T>);
}

/// Trait for transforming data.
pub trait Transformable<T: Value> {
    /// Transforms the data in place.
    fn transform<R: Transform<T>>(&mut self, transform: &R);
}
impl<T: Value> Transformable<T> for Vec<(T, T)> {
    fn transform<R: Transform<T>>(&mut self, transform: &R) {
        transform.apply(self.iter_mut().map(|(_, y)| y));
    }
}
