use crate::{
    statistics::{
        accumulator::{
            Accumulator, AccumulatorFactory, DefaultAccumulatorFactory, FittedValue,
            KahanAccumulator,
        },
        median_absolute_deviation, HuberLogLikelihoodAccumulator, HuberLogLikelihoodConfig,
    },
    value::Value,
};

/// Computes the adjusted R-squared value.
///
/// Adjusted R² accounts for the number of predictors in a model, penalizing
/// overly complex models. Use it to compare models of different degrees.
///
/// Returns `None` if there are not enough data points to compute the adjusted R² (i.e., when n <= k).
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// R²_adj = R² - (1 - R²) * (k / (n - k))
/// where
///   n = number of observations, k = number of model parameters
/// ```
/// [`r_squared`] is used to compute R²
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: Iterator over observed values.
/// - `y_fit`: Iterator over predicted values.
/// - `k`: Number of model parameters (usually `degree + 1`).
///
/// # Returns
/// The adjusted R² as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::adjusted_r_squared;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let k = 3.0;
/// let r2_adj = adjusted_r_squared(y.into_iter(), y_fit.into_iter(), k);
/// ```
pub fn adjusted_r_squared<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
    k: T,
) -> Option<T> {
    let values = FittedValue::from_iters(y, y_fit);
    let acc = RSquaredAccumulator::new_from_iter(values);
    let r2 = acc.total()?;
    let n = acc.count;
    if n <= k {
        return None; // Not enough data points to compute adjusted R²
    }

    Some(r2 - (T::one() - r2) * k / (n - k))
}

/// Calculate the R-squared value for a set of data.
///
/// R-squared is a number between 0 and 1 that tells you how well the model explains the data:
/// - `0` means the model explains none of the variation.
/// - `1` means the model explains all the variation.
///
/// Returns `None` if there are no data points (i.e., when the count is zero).
///
/// If there is no variance in the observed values (i.e., all `y_i` are the same), then we return:
/// - 1 if the predicted values are exactly the same as the observed values (perfect fit)
/// - -inf if the predicted values are not the same as the observed values (since the model does not explain any of the variance)
///
/// <div class="warning">
///
/// **Technical Details**
///
/// R-squared is calculated as:
/// ```math
/// SS_res = Σ (y_i - y_fit_i)²
/// SS_tot = Σ (y_i - y_mean)²
///
/// R² = { 1 - (SS_res / SS_tot)    if SS_tot > 0
///      { 1                        if SS_tot = 0 and SS_res = 0
///      { -inf                     if SS_tot = 0 and SS_res > 0
/// ```
///
/// The justification for this piecewise approach can be split up as follows
///
/// Using L'Hospital's rule, we can observe the following behavior as the variance of `y` approaches zero:
/// - If `SS_res == 0`, then as `SS_tot` approaches zero, the ratio `SS_res / SS_tot` remains 0, so R² remains 1
/// - If `SS_res > 0`, then as `SS_tot` approaches zero, the ratio `SS_res / SS_tot` approaches infinity, so R² approaches -inf
///
/// In the second case, even if we assume a dataset of size `n+1` would in fact have `SS_tot > 0`, the model was unable to explain the shape of the sample of size `n`,
/// and thus, for the sample `n`, is immeasurably worse than a model that simply predicts the mean due to introducing noise where none existed.
/// Since such a model will always be outperformed by a model where `SS_res == 0`, we accept that `R² == -inf`
///
/// For the first case, the argument can be made that there is insufficient evidence to accept the model, since a set of `n + 1` items may introduce previously unseen variance.
/// However, since the model is able to perfectly predict the observed data, from an engineering standpoint it could be harmful to reject the case where a true dataset is in fact
/// constant, and explainable by a constant polynomial (degree 0).
///
/// Further, if you take a theoretical dataset with a variance approaching epsilon, the R² approaches 1, and if you expand n towards infinity with 0 variance, the threshold of
/// insufficient data remains - one would need to reject a model explaining an infinite sized constant dataset, which seems undesirable.
///
/// Thus we accept the case where `SS_tot == 0` and `SS_res == 0` as R² == 1, since we do not have sufficient evidence to reject the model,
/// but do have sufficient evidence to reject any model that does not perfectly fit the data (since it would have R² == -inf)
///
/// </div>
///
/// # Parameters
/// - `y`: The actual (observed) values.
/// - `y_fit`: The predicted values from the model.
///
/// # Returns
/// The proportion of variance explained by the model.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::r_squared;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let r2 = r_squared(y.into_iter(), y_fit.into_iter());
/// ```
pub fn r_squared<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
) -> Option<T> {
    let values = FittedValue::from_iters(y, y_fit);
    RSquaredAccumulator::new_from_iter(values).total()
}

/// Uses huber loss to compute a robust R-squared value.
/// - More robust to outliers than traditional R².
/// - Values closer to 1 indicate a better fit.
///
/// <div class="warning">
///
/// **Technical Details**
///
/// ```math
/// R²_robust = { 1 - (Σ huber_loss(y_i - y_fit_i, delta)) / (Σ (y_i - y_fit_i)²)   if TSS > 0
///             { 1                                                                 if TSS = 0 and TSE = 0
///             { -inf                                                              if TSS = 0 and TSE > 0
/// where
///   huber_loss(r, delta) = { 0.5 * r²                    if |r| ≤ delta
///                          { delta * (|r| - 0.5 * delta) if |r| > delta
///  delta = 1.345 * MAD
///  MAD = median( |y_i - y_fit_i| )
///  where
///    y_i = observed values, y_fit_i = predicted values
/// ```
///
/// See [`r_squared`] for the traditional R² calculation, and for the justification of the piecewise approach when TSS == 0
/// </div>
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: The actual (observed) values.
/// - `y_fit`: The predicted values from the model.
///
/// # Returns
/// The robust R² as a `T`.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::robust_r_squared;
/// let y = vec![1.0, 2.0, 3.0];
/// let y_fit = vec![1.1, 1.9, 3.05];
/// let r2_robust = robust_r_squared(y.into_iter(), y_fit.into_iter());
/// ```
pub fn robust_r_squared<T: Value>(y: &[T], y_fit: &[T]) -> Option<T> {
    let residuals: Vec<_> = y
        .iter()
        .zip(y_fit.iter())
        .map(|(&y, &y_fit)| y - y_fit)
        .collect();
    let mad = median_absolute_deviation(residuals.clone())?;
    let mut acc = HuberLogLikelihoodAccumulator::create(HuberLogLikelihoodConfig {
        median_absolute_deviation: mad,
        huber_const: None,
    });

    //
    // Get:
    //    TSE - Σ huber_loss(y_i - y_fit_i, delta))
    //    TSS - Σ (y_i - y_fit_i)²
    let mut tss = T::zero();
    for r in residuals {
        acc.add(r);
        tss += Value::powi(r, 2);
    }

    let tse = acc.sum_log_likelihood.total();

    // Handle the cases where TSS is zero (i.e., no variance in the observed data)
    if tss == T::zero() {
        if tse == T::zero() {
            return Some(T::one()); // Perfect fit
        }
        return Some(T::neg_infinity()); // No fit
    }

    Some(T::one() - (tse / tss))
}

/// Accumulator for calculating the R-squared value in a single pass through the data
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RSquaredAccumulator<T: Value> {
    pub count: T,
    pub sum: KahanAccumulator<T>,
    pub sum_sq: KahanAccumulator<T>,
    pub sum_res_sq: KahanAccumulator<T>,
}
impl<T: Value> Default for RSquaredAccumulator<T> {
    fn default() -> Self {
        Self {
            count: T::zero(),
            sum: KahanAccumulator::default(),
            sum_sq: KahanAccumulator::default(),
            sum_res_sq: KahanAccumulator::default(),
        }
    }
}
impl<T: Value> DefaultAccumulatorFactory for RSquaredAccumulator<T> {}
impl<T: Value> Accumulator<T, FittedValue<T>> for RSquaredAccumulator<T> {
    fn add(&mut self, value: FittedValue<T>) {
        self.count += T::one();
        self.sum.add(value.y);
        self.sum_sq.add(value.y * value.y);
        self.sum_res_sq.add(Value::powi(value.residual(), 2));
    }

    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.sum.merge(other.sum);
        self.sum_sq.merge(other.sum_sq);
        self.sum_res_sq.merge(other.sum_res_sq);
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn total(&self) -> Option<T> {
        if self.count == T::zero() {
            return None;
        }

        let mean = self.sum.total() / self.count;
        let tss = self.sum_sq.total() - self.count * mean * mean;
        if tss.is_zero() {
            let residual_sum = self.sum_res_sq.total();
            if residual_sum.is_zero() {
                // Perfect fit, R² = 1
                return Some(T::one());
            }

            // No variance in observed data but model has error, R² is undefined (could be considered 0 or negative infinity)
            return Some(T::neg_infinity());
        }

        let r2 = T::one() - (self.sum_res_sq.total() / tss);
        Some(r2)
    }
}
