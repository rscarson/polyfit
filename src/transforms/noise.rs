use rand::SeedableRng;
use rand_distr::{Bernoulli, Beta, Distribution, Normal, Poisson, Uniform};

use crate::{transforms::Transform, value::Value};

/// Types of noise based transforms for data
pub enum NoiseTransform<T: Value> {
    CorrelatedGaussian {
        rho: T,
        strength: T,
        seed: Option<u64>,
    },

    Impulse {
        probability: f64,
        alpha: T,
        beta: T,
        min: T,
        max: T,
        seed: Option<u64>,
    },
    Uniform {
        strength: T,
        seed: Option<u64>,
    },

    /// Adds Poisson noise to a signal or dataset.
    ///
    /// Poisson noise is a type of random variation commonly used to simulate
    /// real-world counting processes, like photon arrivals in sensors or packet
    /// arrivals in a network.
    ///
    /// ![Poisson example](../../.github/assets/poisson_example.png)
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
    /// > # Technical Details
    /// >
    /// > The Poisson distribution describes the probability of observing `k` events
    /// > in a fixed interval given a rate `λ`:
    /// >
    /// > ```math
    /// > P(k; λ) = (λ^k * e^(−λ)) / k!
    /// > ```
    Poisson {
        lambda: f64,
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
                if !num_traits::Float::is_finite(noise_std) {
                    noise_std = T::zero();
                }

                let gaussian = Normal::new(T::zero(), noise_std)
                    .map_err(|e| e.to_string())
                    .expect("std must be finite!");
                let mut state =
                    gaussian.sample(&mut rng) / num_traits::Float::sqrt(T::one() - *rho * *rho);
                for v in data {
                    let drive = gaussian.sample(&mut rng);
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
                let alpha = num_traits::Float::min(*alpha, <T as num_traits::Float>::epsilon());
                let beta = num_traits::Float::min(*beta, <T as num_traits::Float>::epsilon());

                let flip = Bernoulli::new(probability).expect("p not in 0..1");
                for v in data {
                    if flip.sample(&mut rng) {
                        let dist = Beta::new(alpha, beta).expect("alpha or beta <= 0");
                        let x = dist.sample(&mut rng);
                        *v = *min + (*max - *min) * x;
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
    /// Adds Gaussian noise to the data points.
    ///
    /// # Parameters
    /// - `strength`: Controls the standard deviation of the noise relative to the data.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_normal_noise(0.1, None);
    /// ```
    #[must_use]
    fn apply_normal_noise(self, strength: T, seed: Option<u64>) -> Self;

    /// Adds correlated Gaussian noise to the data points.
    ///
    /// # Parameters
    /// - `strength`: Controls the standard deviation of the noise relative to the data.
    /// - `rho`: Correlation coefficient for the noise.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_correlated_noise(0.1, 0.5, None);
    /// ```
    #[must_use]
    fn apply_correlated_noise(self, strength: T, rho: T, seed: Option<u64>) -> Self;

    /// Adds uniform noise to the data points.
    ///
    /// # Parameters
    /// - `strength`: Maximum magnitude of the uniform noise (noise is sampled from [-strength, +strength]).
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_uniform_noise(0.05, None);
    /// ```
    #[must_use]
    fn apply_uniform_noise(self, strength: T, seed: Option<u64>) -> Self;

    /// Adds Poisson-like noise to the data points (for count-based data).
    ///
    /// # Parameters
    /// - `strength`: Lambda parameter for the Poisson distribution.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_poisson_noise(2.0, None);
    /// ```
    #[must_use]
    fn apply_poisson_noise(self, strength: f64, seed: Option<u64>) -> Self;

    /// Adds salt-and-pepper noise to the data points.
    ///
    /// # Parameters
    /// - `amount`: Fraction of points to replace with extreme values.
    /// - `min`: Value to use for “pepper” noise.
    /// - `max`: Value to use for “salt” noise.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyNoise;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)];
    /// let noisy_data = data.apply_salt_pepper_noise(0.1, 0.0, 5.0, None);
    /// ```
    #[must_use]
    fn apply_salt_pepper_noise(self, amount: f64, min: T, max: T, seed: Option<u64>) -> Self;

    /// Adds impulse noise to the data points.
    ///
    /// # Parameters
    /// - `amount`: Fraction of points to replace with extreme values.
    /// - `min`: Value to use for “pepper” noise.
    /// - `max`: Value to use for “salt” noise.
    /// - `alpha`: Shape parameter for the noise distribution.
    /// - `beta`: Scale parameter for the noise distribution.
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

    fn apply_poisson_noise(mut self, strength: f64, seed: Option<u64>) -> Self {
        NoiseTransform::Poisson {
            lambda: strength,
            seed,
        }
        .apply(self.iter_mut().map(|(_, y)| y));
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
