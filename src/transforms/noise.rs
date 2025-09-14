use rand::SeedableRng;
use rand_distr::{Bernoulli, Beta, Distribution, Normal, Poisson, Uniform};

use crate::{statistics::DomainNormalizer, transforms::Transform, value::Value};

/// Types of noise based transforms for data
pub enum NoiseTransform<T: Value> {
    /// Adds correlated Gaussian noise to a signal or dataset.
    ///
    /// Gaussian noise is the familiar "bell curve" distribution.
    ///
    /// This variant introduces *correlation* between neighboring values, so the
    /// noise isn’t purely independent at each point. Instead, it varies smoothly,
    /// making it useful for simulating natural processes or measurement systems
    /// where consecutive samples are not independent.
    ///
    /// ![Correlated Gaussian example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/gaussian_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// - Each value is drawn from a normal distribution `N(0, strength²)`.  
    /// - Correlation is introduced by mixing the new sample with the previous one:  
    ///
    /// ```math
    /// xₙ = ρ * xₙ₋₁ + √(1 − ρ²) * εₙ
    /// where
    ///   εₙ ~ N(0, strength²), ρ = correlation factor
    ///   xₙ₋₁ = previous noisy value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `rho`: Correlation factor between consecutive samples.  
    ///   - Values near `0` → mostly independent noise.  
    ///   - Values near `1` → highly correlated, slow-changing noise.  
    ///
    /// - `strength`: Multiplier for the standard deviation (spread) of the Gaussian distribution.  
    ///
    /// - `seed` *(optional)*: Fixes the RNG seed for reproducibility.
    ///   If not provided, a system RNG will be used each run.
    CorrelatedGaussian {
        /// Correlation factor between consecutive samples.  
        /// - Values near `0` → mostly independent noise.  
        /// - Values near `1` → highly correlated, slow-changing noise.  
        rho: T,

        /// Standard deviation (spread) of the Gaussian distribution.
        strength: T,

        /// `seed` *(optional)*: Fixes the RNG seed for reproducibility.
        /// If not provided, a system RNG will be used each run.
        seed: Option<u64>,
    },

    /// Adds impulse noise (also called *salt-and-pepper noise*) to a signal or dataset.
    ///
    /// Impulse noise randomly replaces some values with sharp outliers, either at the
    /// extremes of a range or drawn from a secondary distribution. This is useful for
    /// testing robustness against corrupted measurements, bit errors, or sudden spikes.
    ///
    /// ![Impulse example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/impulse_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Impulse noise can be modeled as a mixture distribution:  
    ///
    /// ```math
    /// X = { original_value with probability (1 − p)
    ///     { impulse_distribution with probability p
    /// where
    ///   p = probability of an impulse
    ///   impulse_distribution = values drawn according to (alpha, beta) within [min, max]
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `probability`: The chance that any given value is replaced with an impulse.  
    ///   - `0.0` → no impulses, original signal unchanged.  
    ///   - `1.0` → every sample is replaced.  
    ///
    /// - `alpha`: Shape parameter for the impulse distribution.  
    /// - `beta`: Secondary shape/scale parameter.
    ///   Together `alpha` and `beta` control how impulse magnitudes are drawn.
    ///   (For example, they may define a Beta distribution or similar skewed law,
    ///   depending on implementation.)
    ///
    /// - `min`: Lower bound for impulses.  
    /// - `max`: Upper bound for impulses.  
    ///
    /// - `seed` *(optional)*: Fixes the RNG seed for reproducibility.
    ///   If not provided, a system RNG will be used each run.
    ///
    /// Typical cases:  
    /// - With `alpha = beta = 1`, impulses are uniformly distributed in `[min, max]`.  
    /// - With other values, impulses can be skewed toward one side or clustered around the center.  
    ///
    /// This makes impulse noise more flexible than simple salt-and-pepper noise, which
    /// only flips values to hard extremes.
    Impulse {
        /// The chance that any given value is replaced with an impulse.  
        /// - `0.0` → no impulses, original signal unchanged.  
        /// - `1.0` → every sample is replaced.  
        probability: f64,

        /// Shape parameter for the impulse distribution.
        /// Together `alpha` and `beta` control how impulse magnitudes are drawn.
        alpha: T,

        /// Secondary shape/scale parameter.
        /// Together `alpha` and `beta` control how impulse magnitudes are drawn.
        beta: T,

        /// Lower bound for impulses.
        min: T,

        /// Upper bound for impulses.
        max: T,

        /// Fixes the RNG seed for reproducibility.
        /// If not provided, a system RNG will be used each run.
        seed: Option<u64>,
    },

    /// Adds uniform noise to a signal or dataset.
    ///
    /// Uniform noise is random variation drawn from a flat distribution where
    /// every value in the range has the same probability. This makes it useful
    /// for simulating simple "background fuzz" or nondirectional noise.
    ///
    /// ![Uniform example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/uniform_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + εₙ
    /// where
    ///   εₙ ~ U(−strength, +strength), x = uncorrupted value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `strength`: Controls the maximum deviation from the original value.
    ///   Noise is sampled from the interval `[-strength, +strength]`.  
    ///
    /// - `seed` *(optional)*: Allows you to fix the random number generator seed
    ///   for reproducibility. If not provided, a system RNG will be used each run.
    Uniform {
        /// Controls the maximum deviation from the original value.
        /// Noise is sampled from the interval `[-strength, +strength]`
        strength: T,

        /// Allows you to fix the random number generator seed for reproducibility.
        /// If not provided, a system RNG will be used each run.
        seed: Option<u64>,
    },

    /// Adds Poisson noise to a signal or dataset.
    ///
    /// The Poisson distribution describes the probability of observing `k` events
    /// in a fixed interval given a rate `λ`:
    ///
    /// Poisson noise is a type of random variation commonly used to simulate
    /// real-world counting processes, like photon arrivals in sensors or packet
    /// arrivals in a network.
    ///
    /// ![Poisson example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/poisson_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + εₙ
    /// where
    ///   εₙ ~ Poisson(λ), x = uncorrupted value
    /// ```
    ///
    /// # Parameters
    ///
    /// - `lambda`: Controls the intensity of the noise.  
    ///   - Small `lambda` → sparse, spiky noise with frequent zeros.  
    ///   - Large `lambda` → smoother noise that starts to resemble Gaussian.  
    ///
    /// - `seed` *(optional)*: Allows you to fix the random number generator seed
    ///   for reproducibility. If not provided, a system RNG will be used each run.
    /// </div>
    Poisson {
        /// Controls the intensity of the noise.
        /// - Small `lambda` → sparse, spiky noise with frequent zeros.
        /// - Large `lambda` → smoother noise that starts to resemble Gaussian.
        lambda: f64,

        /// Allows you to fix the random number generator seed for reproducibility.
        /// If not provided, a system RNG will be used each run.
        seed: Option<u64>,
    },
}
impl<T: Value> NoiseTransform<T> {
    fn seed(&self) -> Option<u64> {
        match self {
            NoiseTransform::CorrelatedGaussian { seed, .. }
            | NoiseTransform::Impulse { seed, .. }
            | NoiseTransform::Uniform { seed, .. }
            | NoiseTransform::Poisson { seed, .. } => *seed,
        }
    }

    fn rng(seed: Option<u64>) -> rand::rngs::SmallRng {
        match seed {
            Some(s) => rand::rngs::SmallRng::seed_from_u64(s),
            None => rand::rngs::SmallRng::from_rng(&mut rand::rng()),
        }
    }
}
impl<T: Value> Transform<T> for NoiseTransform<T>
where
    T: num_traits::Float + num_traits::FloatConst + rand_distr::uniform::SampleUniform,
    rand_distr::StandardUniform: rand_distr::Distribution<T>,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
    rand_distr::Exp1: rand_distr::Distribution<T>,
    rand_distr::Open01: rand_distr::Distribution<T>,
{
    fn apply<'a>(&self, data: impl Iterator<Item = &'a mut T>) {
        let mut rng = Self::rng(self.seed());
        match self {
            NoiseTransform::CorrelatedGaussian { rho, strength, .. } => {
                let data = data.collect::<Vec<_>>();

                let mut mean = T::zero();
                let mut n = T::zero();
                for v in &data {
                    mean += **v;
                    n += T::one();
                }
                mean /= n;

                let mut std_dev = T::zero();
                for v in &data {
                    std_dev += Value::powi(**v - mean, 2);
                }
                std_dev = num_traits::Float::sqrt(std_dev / n);

                let mut noise_std = std_dev * *strength;

                if noise_std == T::zero() {
                    noise_std = *strength;
                }

                if !num_traits::Float::is_finite(noise_std) {
                    noise_std = T::zero();
                }

                let gaussian = Normal::new(T::zero(), noise_std)
                    .map_err(|e| e.to_string())
                    .expect("std must be finite!");

                let mut state = gaussian.sample(&mut rng); // start from a plain Gaussian
                for v in data {
                    let drive =
                        gaussian.sample(&mut rng) * num_traits::Float::sqrt(T::one() - *rho * *rho);
                    state = *rho * state + drive;
                    *v += state;
                }
            }

            NoiseTransform::Impulse {
                probability,
                alpha,
                beta,
                min,
                max,
                ..
            } => {
                let probability = probability.clamp(0.0, 1.0);
                let alpha = *alpha;
                let beta = *beta;

                let flip = Bernoulli::new(probability).expect("p not in 0..1");
                if alpha <= T::zero() && beta <= T::zero() {
                    let dist = Bernoulli::new(0.5).unwrap();
                    for v in data {
                        if flip.sample(&mut rng) {
                            *v = if dist.sample(&mut rng) { *max } else { *min };
                        }
                    }
                } else {
                    let alpha = num_traits::Float::max(alpha, <T as num_traits::Float>::epsilon());
                    let beta = num_traits::Float::max(beta, <T as num_traits::Float>::epsilon());

                    let dist = Beta::new(alpha, beta).expect("alpha or beta <= 0");
                    let normalizer = DomainNormalizer::new((T::zero(), T::one()), (*min, *max));
                    for v in data {
                        if flip.sample(&mut rng) {
                            let x = dist.sample(&mut rng); // [0,1]
                            *v = normalizer.normalize(x);
                        }
                    }
                }
            }

            NoiseTransform::Uniform { strength, .. } => {
                let strength = num_traits::Float::abs(*strength);
                let strength =
                    num_traits::Float::max(strength, <T as num_traits::Float>::epsilon());

                let uniform = Uniform::new(-strength, strength)
                    .map_err(|e| e.to_string())
                    .expect("Invalid uniform distribution");
                data.for_each(|v| {
                    let noise = uniform.sample(&mut rng);
                    *v += noise;
                });
            }

            NoiseTransform::Poisson { lambda, .. } => {
                let lambda = lambda.clamp(f64::EPSILON, Poisson::MAX_LAMBDA);
                let lambda = T::try_cast(lambda).unwrap_or(<T as num_traits::Float>::epsilon());

                let poisson = Poisson::new(lambda).expect("Invalid Poisson distribution");
                data.for_each(|v| {
                    let noise = poisson.sample(&mut rng);
                    *v += noise;
                });
            }
        }
    }
}

/// Trait for applying noise to data.
pub trait ApplyNoise<T: Value>
where
    Self: Sized,
{
    /// Adds Gaussian noise to a signal or dataset.
    ///
    /// Gaussian noise is the familiar "bell curve" distribution.
    ///
    /// This version corresponds to `rho=0` in the sample:
    ///
    /// ![Correlated Gaussian example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/gaussian_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// - Each value is drawn from a normal distribution `N(0, strength²)`.  
    ///
    /// ```math
    /// xₙ = x + εₙ
    /// where
    ///   εₙ ~ N(0, strength²), x = uncorrupted value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `strength`: Multiplier for the standard deviation (spread) of the Gaussian distribution.  
    ///
    /// - `seed` *(optional)*: Fixes the RNG seed for reproducibility.
    ///   If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_normal_noise(0.1, None);
    /// ```
    #[must_use]
    fn apply_normal_noise(self, strength: T, seed: Option<u64>) -> Self;

    /// Adds Gaussian noise to a signal or dataset.
    ///
    /// Gaussian noise is the familiar "bell curve" distribution.
    ///
    /// This variant introduces *correlation* between neighboring values, so the
    /// noise isn’t purely independent at each point. Instead, it varies smoothly,
    /// making it useful for simulating natural processes or measurement systems
    /// where consecutive samples are not independent.
    ///
    /// ![Correlated Gaussian example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/gaussian_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// - Each value is drawn from a normal distribution `N(0, strength²)`.  
    /// - Correlation is introduced by mixing the new sample with the previous one:  
    ///
    /// ```math
    /// xₙ = ρ * xₙ₋₁ + √(1 − ρ²) * εₙ
    /// where
    ///   εₙ ~ N(0, strength²), ρ = correlation factor
    ///   xₙ₋₁ = previous noisy value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `rho`: Correlation factor between consecutive samples.  
    ///   - Values near `0` → mostly independent noise.  
    ///   - Values near `1` → highly correlated, slow-changing noise.  
    ///
    /// - `strength`: Multiplier for the standard deviation (spread) of the Gaussian distribution.  
    ///
    /// - `seed` *(optional)*: Fixes the RNG seed for reproducibility.
    ///   If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_correlated_noise(0.1, 0.5, None);
    /// ```
    #[must_use]
    fn apply_correlated_noise(self, strength: T, rho: T, seed: Option<u64>) -> Self;

    /// Adds uniform noise to a signal or dataset.
    ///
    /// Uniform noise is random variation drawn from a flat distribution where
    /// every value in the range has the same probability. This makes it useful
    /// for simulating simple "background fuzz" or nondirectional noise.
    ///
    /// ![Uniform example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/uniform_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + εₙ
    /// where
    ///   εₙ ~ U(−strength, +strength), x = uncorrupted value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `strength`: Controls the maximum deviation from the original value.
    ///   Noise is sampled from the interval `[-strength, +strength]`.  
    ///
    /// - `seed` *(optional)*: Allows you to fix the random number generator seed
    ///   for reproducibility. If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_uniform_noise(0.05, None);
    /// ```
    #[must_use]
    fn apply_uniform_noise(self, strength: T, seed: Option<u64>) -> Self;

    /// Adds Poisson noise to a signal or dataset.
    ///
    /// Poisson noise is a type of random variation commonly used to simulate
    /// real-world counting processes, like photon arrivals in sensors or packet
    /// arrivals in a network.
    ///
    /// ![Poisson example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/poisson_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + εₙ
    /// where
    ///   εₙ ~ Poisson(λ), x = uncorrupted value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `lambda`: Controls the intensity of the noise.  
    ///   - Small `lambda` → sparse, spiky noise with frequent zeros.  
    ///   - Large `lambda` → smoother noise that starts to resemble Gaussian.  
    ///
    /// - `seed` *(optional)*: Allows you to fix the random number generator seed
    ///   for reproducibility. If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_poisson_noise(2.0, None);
    /// ```
    #[must_use]
    fn apply_poisson_noise(self, lambda: f64, seed: Option<u64>) -> Self;

    /// Adds salt-and-pepper noise to a signal or dataset.
    ///
    /// This is a special case of impulse noise where the impulses are limited to
    /// two values: the minimum and maximum bounds.
    ///
    /// ![Impulse example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/impulse_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Impulse noise can be modeled as a mixture distribution:  
    ///
    /// ```math
    /// X = { original_value with probability (1 − p)
    ///     { impulse_distribution with probability p
    /// where
    ///   p = probability of an impulse
    ///   impulse_distribution = values drawn according to (alpha ~=0 , beta ~= 0) within [min, max]
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `probability`: The chance that any given value is replaced with an impulse.  
    ///   - `0.0` → no impulses, original signal unchanged.  
    ///   - `1.0` → every sample is replaced.  
    ///
    /// - `min`: Lower bound for impulses.  
    /// - `max`: Upper bound for impulses.  
    ///
    /// - `seed` *(optional)*: Fixes the RNG seed for reproducibility.
    ///   If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_salt_pepper_noise(0.1, 0.0, 5.0, None);
    /// ```
    #[must_use]
    fn apply_salt_pepper_noise(self, amount: f64, min: T, max: T, seed: Option<u64>) -> Self;

    /// Adds impulse noise (also called *salt-and-pepper noise*) to a signal or dataset.
    ///
    /// Impulse noise randomly replaces some values with sharp outliers, either at the
    /// extremes of a range or drawn from a secondary distribution. This is useful for
    /// testing robustness against corrupted measurements, bit errors, or sudden spikes.
    ///
    /// ![Impulse example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/impulse_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// Impulse noise can be modeled as a mixture distribution:  
    ///
    /// ```math
    /// X = { original_value with probability (1 − p)
    ///     { impulse_distribution with probability p
    /// where
    ///   p = probability of an impulse
    ///   impulse_distribution = values drawn according to (alpha, beta) within [min, max]
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `probability`: The chance that any given value is replaced with an impulse.  
    ///   - `0.0` → no impulses, original signal unchanged.  
    ///   - `1.0` → every sample is replaced.  
    ///
    /// - `alpha`: Shape parameter for the impulse distribution.  
    /// - `beta`: Secondary shape/scale parameter.
    ///   Together `alpha` and `beta` control how impulse magnitudes are drawn.
    ///   (For example, they may define a Beta distribution or similar skewed law,
    ///   depending on implementation.)
    ///
    /// - `min`: Lower bound for impulses.  
    /// - `max`: Upper bound for impulses.  
    ///
    /// - `seed` *(optional)*: Fixes the RNG seed for reproducibility.
    ///   If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_impulse_noise(0.1, 0.0, 5.0, 1.0, 1.0, None);
    /// ```
    #[must_use]
    fn apply_impulse_noise(
        self,
        amount: f64,
        min: T,
        max: T,
        alpha: T,
        beta: T,
        seed: Option<u64>,
    ) -> Self;
}
impl<T: Value> ApplyNoise<T> for Vec<(T, T)>
where
    T: num_traits::Float + num_traits::FloatConst + rand_distr::uniform::SampleUniform,
    rand_distr::StandardUniform: rand_distr::Distribution<T>,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
    rand_distr::Exp1: rand_distr::Distribution<T>,
    rand_distr::Open01: rand_distr::Distribution<T>,
{
    fn apply_normal_noise(mut self, strength: T, seed: Option<u64>) -> Self {
        let strength = num_traits::Float::max(strength, <T as num_traits::Float>::epsilon());

        NoiseTransform::CorrelatedGaussian {
            rho: T::zero(),
            strength,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_correlated_noise(mut self, strength: T, rho: T, seed: Option<u64>) -> Self {
        let strength = num_traits::Float::max(strength, <T as num_traits::Float>::epsilon());
        let rho = num_traits::Float::clamp(rho, -T::one(), T::one());

        NoiseTransform::CorrelatedGaussian {
            rho,
            strength,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_uniform_noise(mut self, strength: T, seed: Option<u64>) -> Self {
        NoiseTransform::Uniform { strength, seed }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_poisson_noise(mut self, lambda: f64, seed: Option<u64>) -> Self {
        NoiseTransform::Poisson { lambda, seed }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_salt_pepper_noise(mut self, amount: f64, min: T, max: T, seed: Option<u64>) -> Self {
        NoiseTransform::Impulse {
            probability: amount,
            alpha: T::zero(),
            beta: T::zero(),
            min,
            max,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_impulse_noise(
        mut self,
        amount: f64,
        min: T,
        max: T,
        alpha: T,
        beta: T,
        seed: Option<u64>,
    ) -> Self {
        NoiseTransform::Impulse {
            probability: amount,
            alpha,
            beta,
            min,
            max,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //
    // These should all test the mean and stddev of the noise added
    // to ensure it matches the expected distribution.

    #[test]
    fn test_correlated_gaussian() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data.clone().apply_correlated_noise(0.1, 0.9, Some(42));

        let mut diffs = Vec::new();
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            diffs.push(y2 - y1);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_dev: f64 =
            (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        assert!(mean.abs() < 0.1);
        assert!((std_dev - 0.1).abs() < 0.05);
    }

    #[test]
    fn test_uniform() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data.clone().apply_uniform_noise(0.1, Some(42));

        let mut diffs = Vec::new();
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            diffs.push(y2 - y1);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_dev: f64 =
            (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        assert!(mean.abs() < 0.1);
        assert!((std_dev - (0.1 / (3.0f64).sqrt())).abs() < 0.05);
    }

    #[test]
    fn test_poisson() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data.clone().apply_poisson_noise(2.0, Some(42));

        let mut diffs = Vec::new();
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            diffs.push(y2 - y1);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_dev: f64 =
            (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        assert!((mean - 2.0).abs() < 0.5);
        assert!((std_dev - (2.0f64).sqrt()).abs() < 0.5);
    }

    #[test]
    fn test_salt_pepper() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data
            .clone()
            .apply_salt_pepper_noise(0.1, 0.0, 5.0, Some(42));

        let mut impulse_count = 0;
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            if (y2 - y1).abs() > f64::EPSILON {
                impulse_count += 1;
                assert!(*y2 == 0.0 || (*y2 - 5.0).abs() < f64::EPSILON);
            }
        }

        let impulse_ratio = f64::from(impulse_count) / data.len() as f64;
        assert!((impulse_ratio - 0.1).abs() < 0.05);
    }

    #[test]
    fn test_impulse() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data
            .clone()
            .apply_impulse_noise(0.1, 0.0, 5.0, 2.0, 5.0, Some(42));

        let mut impulse_count = 0;
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            if (y2 - y1).abs() > f64::EPSILON {
                impulse_count += 1;
                assert!(*y2 >= 0.0 && *y2 <= 5.0);
            }
        }

        let impulse_ratio = f64::from(impulse_count) / data.len() as f64;
        assert!((impulse_ratio - 0.1).abs() < 0.05);
    }
}
