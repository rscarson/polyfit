use crate::{
    statistics::{self, DomainNormalizer},
    transforms::Transform,
    value::Value,
};

pub enum NormalizationTransform<T: Value> {
    Domain { min: T, max: T },

    Clip { min: T, max: T },

    MeanSubtraction,
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
                    *value = value.clamp(*min, *max);
                }
            }

            Self::MeanSubtraction => {
                let mean = statistics::mean(data.iter().map(|d| **d));
                for value in data {
                    *value = *value - mean;
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

pub trait ApplyNormalization<T: Value> {
    fn apply_domain_normalization(&mut self, min: T, max: T);
    fn apply_clipping(&mut self, min: T, max: T);
    fn apply_mean_subtraction(&mut self);
    fn apply_z_score_normalization(&mut self);
}
impl<T: Value> ApplyNormalization<T> for Vec<(T, T)> {
    fn apply_domain_normalization(&mut self, min: T, max: T) {
        NormalizationTransform::Domain { min, max }.apply(self.iter_mut().map(|(_, y)| y));
    }

    fn apply_clipping(&mut self, min: T, max: T) {
        NormalizationTransform::Clip { min, max }.apply(self.iter_mut().map(|(_, y)| y));
    }

    fn apply_mean_subtraction(&mut self) {
        NormalizationTransform::MeanSubtraction.apply(self.iter_mut().map(|(_, y)| y));
    }

    fn apply_z_score_normalization(&mut self) {
        NormalizationTransform::ZScore.apply(self.iter_mut().map(|(_, y)| y));
    }
}

pub enum SmoothingTransform<T: Value> {
    MovingAverageSmoothing { window_size: usize },
    GaussianSmoothing { sigma: T },
}
impl<T: Value> Transform<T> for SmoothingTransform<T> {
    fn apply<'a>(&self, data: impl Iterator<Item = &'a mut T>) {
        let data: Vec<_> = data.collect();
        match self {
            Self::MovingAverageSmoothing { window_size } => {
                todo!()
            }

            Self::GaussianSmoothing { sigma } => {
                todo!()
            }
        }
    }
}
pub trait ApplySmoothing<T: Value> {
    fn apply_moving_average_smoothing(&mut self, window_size: usize);
    fn apply_gaussian_smoothing(&mut self, sigma: T);
}
impl<T: Value> ApplySmoothing<T> for Vec<(T, T)> {
    fn apply_moving_average_smoothing(&mut self, window_size: usize) {
        SmoothingTransform::MovingAverageSmoothing { window_size }
            .apply(self.iter_mut().map(|(_, y)| y));
    }

    fn apply_gaussian_smoothing(&mut self, sigma: T) {
        SmoothingTransform::GaussianSmoothing { sigma }.apply(self.iter_mut().map(|(_, y)| y));
    }
}
