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

    /// Applies the transformation to a single data point.
    fn apply_to(&self, point: &mut T) {
        self.apply(std::iter::once(point));
    }
}

/// Trait for transforming data.
pub trait Transformable<T: Value> {
    /// Transforms the data in place.
    fn transform<R: Transform<T>>(&mut self, transform: &R);

    /// Returns a transformed copy of the data.
    #[must_use]
    fn transformed<R: Transform<T>>(&self, transform: &R) -> Self
    where
        Self: Sized + Clone,
    {
        let mut new_data = self.clone();
        new_data.transform(transform);
        new_data
    }
}
impl<T: Value> Transformable<T> for Vec<(T, T)> {
    fn transform<R: Transform<T>>(&mut self, transform: &R) {
        transform.apply(self.iter_mut().map(|(_, y)| y));
    }
}

thread_local! {
    // Mutexed Vec of seeds
    static SEED_VAULT: std::sync::Mutex<Vec<u64>> = const { std::sync::Mutex::new(Vec::new()) };
}

/// Source of random seeds for debugging purposes
///
/// Any seeds generated will be stored in a thread-local vault and can be retrieved with [`SeedSource::all_seeds`] later
///
/// Every `assert_` macro in the test suite will print seeds on failure if they were generated during the test
///
/// You can use a custom [`SeedSource`] to replay those seeds for debugging purposes, by using [`SeedSource::from_seeds`]
#[derive(Debug)]
pub struct SeedSource {
    replay: Vec<u64>,

    rng: rand::rngs::ThreadRng,
}
impl Default for SeedSource {
    fn default() -> Self {
        Self::new()
    }
}
impl SeedSource {
    /// Create a new [`SeedSource`]
    #[must_use]
    pub fn new() -> Self {
        Self {
            replay: Vec::new(),
            rng: rand::rng(),
        }
    }

    /// Create a new [`SeedSource`] that will replay the given seeds
    ///
    /// Debug/testing use only. Not intended for production use.
    ///
    /// This function will override any random seed generation and return the given seeds in order
    /// until they are exhausted, at which point it will revert to random generation.
    #[must_use]
    pub fn from_seeds(seeds: impl Into<Vec<u64>>) -> Self {
        Self {
            replay: seeds.into(),
            rng: rand::rng(),
        }
    }

    /// Reset the seed vault, clearing all stored seeds for this thread
    ///
    /// # Panics
    /// Panics if the thread-local vault cannot be locked
    pub fn reset() {
        SEED_VAULT.with(|vault| {
            let mut vault = vault.lock().expect("Failed to lock SEED_VAULT");
            vault.clear();
        });
    }

    /// Get all seeds generated so far in this thread by any [`SeedSource`]
    ///
    /// # Panics
    /// Panics if the thread-local vault cannot be locked
    #[must_use]
    pub fn all_seeds() -> Vec<u64> {
        SEED_VAULT.with(|vault| {
            let vault = vault.lock().expect("Failed to lock SEED_VAULT");
            vault.clone()
        })
    }

    /// Get a new random seed. This is meant for debugging purposes, to make random tests reproducible.
    ///
    /// Any seeds generated will be stored in a thread-local vault and can be retrieved with [`SeedSource::all_seeds`]
    ///
    /// # Panics
    /// Panics if the thread-local vault cannot be locked
    pub fn seed(&mut self) -> u64 {
        let seed: u64 = if self.replay.is_empty() {
            rand::Rng::random(&mut self.rng)
        } else {
            self.replay.remove(0)
        };

        SEED_VAULT.with(|vault| {
            let mut vault = vault.lock().expect("Failed to lock SEED_VAULT");
            vault.push(seed);
        });
        seed
    }

    /// Print the seeds used in this test thread so far, formatted for easy replaying
    ///
    /// This is meant for debugging purposes, to make random tests reproducible.
    /// Any seeds generated will be stored in a thread-local vault and can be retrieved with [`SeedSource::all_seeds`]
    #[must_use]
    #[rustfmt::skip]
    pub fn print_seeds() -> Option<String> {
        use std::fmt::Write;

        let mut out = String::new();
        let seeds = Self::all_seeds();
        if seeds.is_empty() {
            return None;
        }

        let seeds = seeds
            .iter()
            .map(|s| format!("0x{s:x}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(out, "Seeds used in this test thread: [{seeds}]").ok()?;

        writeln!(out, "You can replay this test with:").ok()?;
        writeln!(out, "    let mut src = polyfit::transforms::SeedSource::from_seeds([{seeds}]);").ok()?;
        writeln!(out, "    data.apply_poisson_noise(Strength::Absolute(0.1), Some(src.seed()); // Poisson used for example").ok()?;

        Some(out)
    }
}
