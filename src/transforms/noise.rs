use rand::SeedableRng;
use rand_distr::{Bernoulli, Beta, Distribution, Normal, Poisson, Uniform};

use crate::{
    statistics::DomainNormalizer,
    transforms::{SeedSource, Transform},
    value::{FloatClampedCast, Value},
};

/// Strength of noise to add
#[derive(Copy, Clone, Debug)]
pub enum Strength<T> {
    /// Absolute strength - This is the std-dev of the gaussian distribution
    Absolute(T),

    /// Relative strength - multiplied by the std-dev of the data to get the std-dev of the gaussian distribution
    Relative(T),
}
impl<T: Value> Strength<T> {
    /// Get the inner value
    pub fn inner(&self) -> T {
        match self {
            Strength::Absolute(tol) => *tol,
            Strength::Relative(rel) => *rel,
        }
    }

    /// Get a strength-ajusted std-dev for some data
    ///
    /// Does not mutate data - but this is the form the transforms get
    pub(crate) fn into_stddev(self, data: &[&mut T]) -> T {
        let mut std_dev = match self {
            Strength::Absolute(tol) => tol,
            Strength::Relative(rel) => {
                let mut mean = T::zero();
                let mut n = T::zero();
                for v in data {
                    mean += **v;
                    n += T::one();
                }
                mean /= n;

                let mut std_dev = T::zero();
                for v in data {
                    std_dev += Value::powi(**v - mean, 2);
                }
                std_dev = T::sqrt(std_dev / n);

                std_dev * rel
            }
        };

        if !Value::is_finite(std_dev) {
            std_dev = T::zero();
        }

        std_dev
    }

    /// Get a strength-adjusted scaling factor for a given sample
    pub(crate) fn into_point(self, point: T) -> T {
        match self {
            Strength::Absolute(_) => T::one(),
            Strength::Relative(rel) => rel * Value::abs(point),
        }
    }
}

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
    /// Basically as rho gets closer to 1, instead of bouncing around randomly, the noise will tend to meander in one direction for a
    /// while before switching direction.
    ///
    /// ![Correlated Gaussian example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/correlated_gaussian_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// - Each value is drawn from a normal distribution `N(0, strength²)`.  
    /// - Correlation is introduced by mixing the new sample with the previous one:  
    /// - The initial state is drawn from the same stationary distribution
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
        strength: Strength<T>,

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
    /// - `min`: Lower bound for impulses. Relative strengths are scaled by the data's standard deviation.
    /// - `max`: Upper bound for impulses. Relative strengths are scaled by the data's standard deviation.
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
        min: Strength<T>,

        /// Upper bound for impulses.
        max: Strength<T>,

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
    /// Upper and lower bounds can be set independently to create asymmetric noise.
    ///
    /// Absolute strengths are used directly as the bounds for the uniform distribution,
    /// while relative strengths are interpreted as fractions of the data's standard deviation.
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
    ///   εₙ ~ U(−lower, +upper), x = uncorrupted value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `upper`: Controls the maximum deviation from the original value in the upper direction.
    /// - `lower`: Controls the maximum deviation from the original value in the lower direction.
    /// - `seed` *(optional)*: Allows you to fix the random number generator seed
    ///   for reproducibility. If not provided, a system RNG will be used each run.
    Uniform {
        /// Controls the maximum deviation from the original value in the upper direction.
        upper: Strength<T>,

        /// Controls the maximum deviation from the original value in the lower direction.
        lower: Strength<T>,

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
    /// Lambda controls the average rate of events and variance, and thus the intensity.
    /// - Small `lambda` → sparse, spiky noise with frequent zeros.
    /// - Large `lambda` → smoother noise that starts to resemble Gaussian.
    ///
    /// The noise can be absolute (standard centered Poisson) or relative (scaled by the data point).
    ///
    /// ![Poisson example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/poisson_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + (εₙ - λ) [ absolute λ ]
    /// xₙ = x + (λ * |x|) * εₙ [ relative λ ]
    /// where
    ///   εₙ ~ Poisson(λ), x = uncorrupted value
    ///
    /// ```
    ///
    /// # Parameters
    ///
    /// - `lambda`: Controls the intensity and rate of noise.  
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
        ///
        /// Absolute values are used directly, while relative values produce noise scaled by each data point.
        lambda: Strength<T>,

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
        let seed = seed.unwrap_or_else(|| SeedSource::new().seed());

        rand::rngs::SmallRng::seed_from_u64(seed)
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

                let rho = num_traits::Float::clamp(*rho, -T::one(), T::one());
                let mut std_dev = strength.into_stddev(data.as_slice());
                std_dev = Value::max(std_dev, num_traits::Float::epsilon());
                let gaussian = Normal::new(T::zero(), std_dev)
                    .map_err(|e| e.to_string())
                    .expect("std must be finite!");

                let mut state = gaussian.sample(&mut rng); // start from a plain Gaussian
                for v in data {
                    let drive =
                        gaussian.sample(&mut rng) * num_traits::Float::sqrt(T::one() - rho * rho);
                    state = rho * state + drive;
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

                //
                // Get the min and max values
                // Relative strengths calculate bounds based on data std-dev
                let data = data.collect::<Vec<_>>();
                let min = min.into_stddev(data.as_slice());
                let max = max.into_stddev(data.as_slice());

                // The activation possibility
                let flip = Bernoulli::new(probability).expect("p not in 0..1");

                // Special case: salt and pepper noise
                if alpha <= T::zero() && beta <= T::zero() {
                    let dist = Bernoulli::new(0.5).unwrap();
                    for v in data {
                        if flip.sample(&mut rng) {
                            *v = if dist.sample(&mut rng) { max } else { min };
                        }
                    }
                    return;
                }

                // Clamped to avoid invalid distributions
                let alpha = num_traits::Float::max(alpha, <T as num_traits::Float>::epsilon());
                let beta = num_traits::Float::max(beta, <T as num_traits::Float>::epsilon());

                let dist = Beta::new(alpha, beta).expect("alpha or beta <= 0");
                let normalizer = DomainNormalizer::new((T::zero(), T::one()), (min, max));
                for v in data {
                    if flip.sample(&mut rng) {
                        let x = dist.sample(&mut rng); // [0,1]
                        *v = normalizer.normalize(x); // [min, max]
                    }
                }
            }

            NoiseTransform::Uniform { upper, lower, .. } => {
                let data = data.collect::<Vec<_>>();
                let upper = upper.into_stddev(data.as_slice());
                let lower = lower.into_stddev(data.as_slice());

                let uniform = Uniform::new(-lower, upper)
                    .map_err(|e| e.to_string())
                    .expect("Invalid uniform distribution");
                for v in data {
                    let noise = uniform.sample(&mut rng);
                    *v += noise;
                }
            }

            NoiseTransform::Poisson { lambda, .. } => {
                let l_max = Poisson::MAX_LAMBDA.clamped_cast::<T>();
                let lambda_abs = Value::clamp(lambda.inner(), num_traits::Float::epsilon(), l_max);

                let poisson = Poisson::new(lambda_abs).expect("Invalid Poisson distribution");
                data.for_each(|v| {
                    let noise = poisson.sample(&mut rng) - lambda_abs; // center around 0
                    *v += noise * lambda.into_point(*v);
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
    /// ![Correlated Gaussian example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/normal_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// For relative strength, the standard deviation of the series is computed as a fraction of the standard deviation of the original data:
    /// ```math
    /// std_dev_data = sqrt( (1/N) * Σ (xᵢ - mean)² )
    /// strength = rel * std_dev_data
    /// ```
    ///
    /// - For absolute strength, the standard deviation is simply the provided value.
    ///
    /// - Each value is drawn from a normal distribution `N(0, strength)`.  
    ///
    /// ```math
    /// xₙ = x + εₙ
    /// where
    ///   εₙ ~ N(0, strength), x = uncorrupted value
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
    /// # use polyfit::transforms::Strength;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_normal_noise(Strength::Relative(0.1), None);
    /// ```
    #[must_use]
    fn apply_normal_noise(self, strength: Strength<T>, seed: Option<u64>) -> Self;

    /// Adds Gaussian noise to a signal or dataset.
    ///
    /// Gaussian noise is the familiar "bell curve" distribution.
    ///
    /// This variant introduces *correlation* between neighboring values, so the
    /// noise isn’t purely independent at each point. Instead, it varies smoothly,
    /// making it useful for simulating natural processes or measurement systems
    /// where consecutive samples are not independent.
    ///
    /// Basically as rho gets closer to 1, instead of bouncing around randomly, the noise will tend to meander in one direction for a
    /// while before switching direction.
    ///
    /// ![Correlated Gaussian example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/correlated_gaussian_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// For relative strength, the standard deviation of the series is computed as a fraction of the standard deviation of the original data:
    /// ```math
    /// std_dev_data = sqrt( (1/N) * Σ (xᵢ - mean)² )
    /// strength = rel * std_dev_data
    /// ```
    ///
    /// - For absolute strength, the standard deviation is simply the provided value.
    ///
    /// - Each value is drawn from a normal distribution `N(0, strength)`.  
    /// - Correlation is introduced by mixing the new sample with the previous one:  
    /// - The initial state is drawn from the same stationary distribution
    ///
    /// ```math
    /// xₙ = ρ * xₙ₋₁ + √(1 − ρ²) * εₙ
    /// where
    ///   εₙ ~ N(0, strength), ρ = correlation factor
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
    /// # use polyfit::transforms::Strength;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_correlated_noise(Strength::Relative(0.1), 0.5, None);
    /// ```
    #[must_use]
    fn apply_correlated_noise(self, strength: Strength<T>, rho: T, seed: Option<u64>) -> Self;

    /// Adds uniform noise to a signal or dataset.
    ///
    /// Uniform noise is random variation drawn from a flat distribution where
    /// every value in the range has the same probability. This makes it useful
    /// for simulating simple "background fuzz" or nondirectional noise.
    ///
    /// Strengths can be absolute or relative:
    /// - Absolute strengths are used directly as the bounds for the uniform distribution,
    /// - Relative strengths are interpreted as fractions of the data's standard deviation.
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
    ///   εₙ ~ U(−lower, +upper), x = uncorrupted value
    /// ```
    /// </div>
    ///
    /// # Parameters
    ///
    /// - `lower`: Controls the maximum deviation from the original value in the lower direction.
    ///   Noise is sampled from the interval `[-lower, +upper]`.  
    ///
    /// - `upper`: Controls the maximum deviation from the original value in the upper direction.
    ///   Noise is sampled from the interval `[-lower, +upper]`.  
    ///
    /// - `seed` *(optional)*: Allows you to fix the random number generator seed
    ///   for reproducibility. If not provided, a system RNG will be used each run.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::{Strength, ApplyNoise};
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_uniform_noise(Strength::Relative(0.05), Strength::Relative(0.05), None);
    /// ```
    #[must_use]
    fn apply_uniform_noise(self, lower: Strength<T>, upper: Strength<T>, seed: Option<u64>)
        -> Self;

    /// Adds Poisson noise to a signal or dataset.
    ///
    /// Poisson noise is a type of random variation commonly used to simulate
    /// real-world counting processes, like photon arrivals in sensors or packet
    /// arrivals in a network.
    ///
    /// Lambda controls the average rate of events and variance, and thus the intensity.
    /// - Small `lambda` → sparse, spiky noise with frequent zeros.
    /// - Large `lambda` → smoother noise that starts to resemble Gaussian.
    ///
    /// The noise can be absolute (standard centered Poisson) or relative (scaled by the data point).
    ///
    /// ![Poisson example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/poisson_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + (εₙ - λ) [ absolute λ ]
    /// xₙ = x + (λ * |x|) * εₙ [ relative λ ]
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
    /// # use polyfit::transforms::{Strength, ApplyNoise};
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_poisson_noise(Strength::Relative(2.0), true, None);
    /// ```
    #[must_use]
    fn apply_poisson_noise(self, lambda: Strength<T>, seed: Option<u64>) -> Self;

    /// Adds salt-and-pepper noise to a signal or dataset.
    ///
    /// This is a special case of impulse noise where the impulses are limited to
    /// two values: the minimum and maximum bounds.
    ///
    /// ![Impulse example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/salt_and_pepper_example.png)
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
    /// - `min`: Lower bound for impulses. Relative strengths are scaled by the data's standard deviation.
    /// - `max`: Upper bound for impulses. Relative strengths are scaled by the data's standard deviation.
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
    fn apply_salt_pepper_noise(
        self,
        amount: f64,
        min: Strength<T>,
        max: Strength<T>,
        seed: Option<u64>,
    ) -> Self;

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
    /// - `min`: Lower bound for impulses. Relative strengths are scaled by the data's standard deviation.
    /// - `max`: Upper bound for impulses. Relative strengths are scaled by the data's standard deviation.
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
        min: Strength<T>,
        max: Strength<T>,
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
    fn apply_normal_noise(mut self, strength: Strength<T>, seed: Option<u64>) -> Self {
        NoiseTransform::CorrelatedGaussian {
            rho: T::zero(),
            strength,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_correlated_noise(mut self, strength: Strength<T>, rho: T, seed: Option<u64>) -> Self {
        NoiseTransform::CorrelatedGaussian {
            rho,
            strength,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_uniform_noise(
        mut self,
        lower: Strength<T>,
        upper: Strength<T>,
        seed: Option<u64>,
    ) -> Self {
        NoiseTransform::Uniform { lower, upper, seed }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_poisson_noise(mut self, lambda: Strength<T>, seed: Option<u64>) -> Self {
        NoiseTransform::Poisson { lambda, seed }.apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_salt_pepper_noise(
        mut self,
        amount: f64,
        min: Strength<T>,
        max: Strength<T>,
        seed: Option<u64>,
    ) -> Self {
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
        min: Strength<T>,
        max: Strength<T>,
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
        let noisy_data =
            data.clone()
                .apply_correlated_noise(Strength::Absolute(0.1), 0.9, Some(42));

        let mut diffs = Vec::new();
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            diffs.push(y2 - y1);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_dev: f64 =
            (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        // Mean should be near 0, stddev near 0.1
        assert!(mean.abs() < 0.1);
        assert!((std_dev - 0.1).abs() < 0.05);
    }

    #[test]
    fn test_uniform() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data.clone().apply_uniform_noise(
            Strength::Absolute(0.1),
            Strength::Absolute(0.1),
            Some(42),
        );

        let mut diffs = Vec::new();
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            diffs.push(y2 - y1);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_dev: f64 =
            (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        // Mean should be near 0, stddev near 0.1/sqrt(3)
        assert!(mean.abs() < 0.1);
        assert!((std_dev - (0.1 / (3.0f64).sqrt())).abs() < 0.05);
    }

    #[test]
    fn test_poisson() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data
            .clone()
            .apply_poisson_noise(Strength::Relative(2.0), Some(42));

        let mut diffs = Vec::new();
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            diffs.push(y2 - y1);
        }

        let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_dev: f64 =
            (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        // Mean should be near 0, stddev near sqrt(2)
        assert!(mean.abs() < 0.5);
        assert!((std_dev - (2.0f64).sqrt()).abs() < 0.5);
    }

    #[test]
    fn test_salt_pepper() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data.clone().apply_salt_pepper_noise(
            0.1,
            Strength::Absolute(0.0),
            Strength::Absolute(5.0),
            Some(42),
        );

        let mut impulse_count = 0;
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            if (y2 - y1).abs() > f64::EPSILON {
                impulse_count += 1;
                assert!(*y2 == 0.0 || (*y2 - 5.0).abs() < f64::EPSILON);
            }
        }

        // Approximately 10% of values should be impulses
        let impulse_ratio = f64::from(impulse_count) / data.len() as f64;
        assert!((impulse_ratio - 0.1).abs() < 0.05);
    }

    #[test]
    fn test_impulse() {
        let data = vec![(1.0, 2.0); 1000];
        let noisy_data = data.clone().apply_impulse_noise(
            0.1,
            Strength::Absolute(0.0),
            Strength::Absolute(5.0),
            2.0,
            5.0,
            Some(42),
        );

        let mut impulse_count = 0;
        for ((_, y1), (_, y2)) in data.iter().zip(noisy_data.iter()) {
            if (y2 - y1).abs() > f64::EPSILON {
                impulse_count += 1;
                assert!(*y2 >= 0.0 && *y2 <= 5.0);
            }
        }

        // Approximately 10% of values should be impulses
        let impulse_ratio = f64::from(impulse_count) / data.len() as f64;
        assert!((impulse_ratio - 0.1).abs() < 0.05);
    }
}
