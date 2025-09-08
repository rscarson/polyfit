//! Utilities for adding transformations to data
use crate::value::Value;

mod noise;
pub use noise::{ApplyNoise, NoiseTransform};

mod scale;
pub use scale::{ApplyScale, ScaleTransform};

mod normalization;
pub use normalization::{ApplyNormalization, NormalizationTransform};
pub use normalization::{ApplySmoothing, SmoothingTransform};

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
