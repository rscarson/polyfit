/// In order to find the best fitting polynomial degree, we need to limit the maximum degree considered.
/// The choice of degree bound can significantly impact the model's performance and its ability to generalize.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// The maximum degree is chosen as the minimum of three constraints:
///
/// 1. A hard cap to prevent excessively high degrees:
/// - Conservative: 8
/// - Relaxed: 15
///
/// 2. Smoothness (`s`):
/// ```math
/// lim_smooth = n ^ (1 / (2s + 1))
/// where
///   s = assumed smoothness of the underlying function
///   n = number of observations
/// ```
///
/// 3. Observations per parameter:
/// ```math
/// lim_obs = (n / n_k_ratio_limit) - 1
/// where
///   n_k_ratio_limit = minimum required number of observations per coefficient
///   n = number of observations
/// ```
///
/// One argument against this technique is that it may be more conservative for some bases than others (Fourier for example will allow higher degrees than monomial for the same n).
/// However, while statistically different in that case, the orthogonal nature of the Fourier basis means that it is much less likely to encounter numerical instability
/// or overfitting for reasonable degrees, and this matches user intuition about degree more closely than the exact case by degree bounds for each basis would.
/// </div>
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegreeBound {
    /// Limits model complexity more aggressively, recommended for small datasets or when overfitting is a major concern.
    ///   - Assumes the data is smoother (s=2)
    ///   - Requires more observations per parameter (15)
    ///   - Lower hard cap (8)
    ///   - Hard cap reached when n ~= 32,000
    Conservative,

    /// Allows for higher complexity, useful when the underlying function may be more complex and the dataset is moderate in size.
    /// - Assumes the data is less 'smooth' (s=1)
    /// - Allows for fewer observations per parameter (8)
    /// - Higher hard cap (15)
    /// - Hard cap reached when n ~= 3,375
    Relaxed,

    /// Similar to Relaxed but with no smoothness assumption, or hard cap. Use this if you are trying to approximate a dataset exactly,
    ///  or if you have a very large dataset and want to explore higher degrees.
    ///
    /// <div class="warning">
    ///     Use with caution, as it can lead to overfitting and numerical instability, especially with small datasets.
    /// </div>
    ///
    /// In nearly every case, you should use [`DegreeBound::Relaxed`] instead of this option,
    /// unless you understand the implications and have a specific reason for allowing such high degrees.
    ///
    /// - Assumes the data is not smooth (s=0)
    /// - No hard cap, but the theoretical maximum of n-1 still applies
    /// - Same observation per parameter limit as Relaxed (8)
    Aggressive,

    /// User-specified maximum degree. Use only if you understand the implications for overfitting and numerical stability.
    Custom(usize),
}
impl From<usize> for DegreeBound {
    fn from(value: usize) -> Self {
        DegreeBound::Custom(value)
    }
}
impl DegreeBound {
    /// Computes the maximum polynomial degree to use for fitting based on the selected [`DegreeBound`]
    /// and the number of observations `n`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn max_degree(self, n: usize) -> usize {
        let theoretical_max = n.saturating_sub(1);
        match self {
            DegreeBound::Custom(d) => d.min(theoretical_max),
            DegreeBound::Conservative | DegreeBound::Relaxed | DegreeBound::Aggressive => {
                let (hard_cap, max_n_per_k, est_smoothness) = match self {
                    DegreeBound::Conservative => (8, 15, 2),
                    DegreeBound::Relaxed => (15, 8, 1),
                    DegreeBound::Aggressive => (theoretical_max, 8, 0),
                    DegreeBound::Custom(_) => unreachable!(),
                };

                let smooth_lim = (n as f64)
                    .powf(1.0 / (2.0 * f64::from(est_smoothness) + 1.0))
                    .floor();

                let smooth_lim = smooth_lim as usize;

                let obs_lim = (n / max_n_per_k).saturating_sub(1);

                smooth_lim.min(obs_lim).min(hard_cap).min(theoretical_max)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theoretical_bounds_fourier() {
        //
        // So we will go from n=1 to n=100k
        // For each, we will find the max degree for each bound
        // and verify it is <= (n-1)/2 -> theoretical max for Fourier basis
        for n in 1..=100_000 {
            let max = DegreeBound::Relaxed.max_degree(n);
            let theoretical_max = (n - 1) / 2;
            assert!(
                max <= theoretical_max,
                "Degree bound exceeded theoretical max for n={n}: {max} > {theoretical_max}"
            );

            let hard_max = n - 1;
            assert!(
                max == 0 || max < hard_max,
                "Degree bound hit hard max for n={n}: {max} > {hard_max}"
            );

            // Now the first test again for the aggressive bound
            let max = DegreeBound::Aggressive.max_degree(n);
            let theoretical_max = n - 1;
            assert!(
                max <= theoretical_max,
                "Degree bound exceeded theoretical max for n={n}: {max} > {theoretical_max}"
            );
        }
    }
}
