use crate::{
    statistics::{root_mean_squared_error, UncertainValue},
    value::Value,
};

/// Strategy for selecting the number of folds (k) in k-fold cross-validation.
///
/// This determines how the data is split for training and validation during model evaluation.
/// Different strategies balance bias and variance in the error estimates.
///
/// Where:
/// - Bias: Error due to overly simplistic models (underfitting). This is how far off average predictions are from actual values.
/// - Variance: Error due to overly complex models (overfitting). This is how much predictions vary for different training sets.
///
/// - `MinimizeVariance`: Uses fewer folds (e.g., k=5) to reduce variance in error estimates, at the cost of higher bias.
/// - `MinimizeBias`: Uses more folds (e.g., k=10) to reduce bias in error estimates, at the cost of higher variance.
/// - `LeaveOneOut`: Leave-One-Out cross-validation (LOOCV), where each data point is used once as a validation set.
/// - `Balanced`: A compromise between bias and variance (e.g., k=7).
///
/// When to use each strategy:
/// - `MinimizeBias`: When the dataset is small and you want to avoid underfitting. Prevents a model from being too simple to capture data patterns.
/// - `MinimizeVariance`: When the dataset is large and you want to avoid overfitting. Helps ensure the model generalizes well to unseen data.
/// - `LeaveOneOut`: When the dataset is very small and you want to maximize training data for each fold, at the cost of high computational expense.
/// - `Balanced`: When you want a good trade-off between bias and variance, suitable for moderately sized datasets or when unsure.
/// - `Custom`: Specify your own number of folds (k) based on domain knowledge or specific requirements. Use with caution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvStrategy {
    /// Uses fewer folds (e.g., k=5) to reduce variance in error estimates, at the cost of higher bias.
    ///
    /// When to use: When the dataset is large and you want to avoid overfitting. Helps ensure the model generalizes well to unseen data.
    ///
    /// When using this strategy, the data is split into 5 folds.
    MinimizeVariance,

    /// Uses more folds (e.g., k=10) to reduce bias in error estimates, at the cost of higher variance.
    ///
    /// When to use: When the dataset is small and you want to avoid underfitting. Prevents a model from being too simple to capture data patterns.
    ///
    /// When using this strategy, the data is split into 10 folds.
    MinimizeBias,

    /// Leave-One-Out cross-validation (LOOCV), where each data point is used once as a validation set.
    ///
    /// When to use: When the dataset is very small and you want to maximize training data for each fold, at the cost of high computational expense.
    ///
    /// When using this strategy, the number of folds equals the number of data points.
    LeaveOneOut,

    /// A compromise between bias and variance (e.g., k=7).
    ///
    /// When to use: When you want a good trade-off between bias and variance, suitable for moderately sized datasets or when unsure.
    ///
    /// When using this strategy, the data is split into 7 folds.
    Balanced,

    /// Specify your own number of folds (k) based on domain knowledge or specific requirements. Use with caution.
    #[allow(missing_docs)]
    Custom { k: usize },
}
impl CvStrategy {
    /// Returns the number of folds (k) associated with the cross-validation strategy.
    #[must_use]
    pub fn k(self, n: usize) -> usize {
        match self {
            CvStrategy::MinimizeVariance => 5,
            CvStrategy::MinimizeBias => 10,
            CvStrategy::LeaveOneOut => n,
            CvStrategy::Balanced => 7,
            CvStrategy::Custom { k } => k,
        }
    }
}
impl From<usize> for CvStrategy {
    fn from(k: usize) -> Self {
        CvStrategy::Custom { k }
    }
}

/// Splits the data into k folds for cross-validation based on the specified strategy.
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `data`: A slice of tuples containing the data points (x, y).
/// - `strategy`: The cross-validation strategy to use. Determines the number of folds (k).
///
/// # Returns
/// A vector containing k folds, each fold is a vector of data points (x, y).
///
/// # Example
/// ```rust
/// # use polyfit::statistics::{cross_validation_split, CvStrategy};
/// let data = vec![(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)];
/// let folds = cross_validation_split(&data, CvStrategy::Balanced);
/// ```
pub fn cross_validation_split<I: Clone>(data: &[I], strategy: CvStrategy) -> Vec<Vec<I>> {
    let n = data.len();
    let k = strategy.k(n);
    let fold_size = n / k;
    let mut folds: Vec<Vec<I>> = Vec::with_capacity(k);

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { start + fold_size };
        folds.push(data[start..end].to_vec());
    }

    folds
}

/// Computes the Root Mean Square Error (RMSE) for the given data and model predictions, by splitting the data into folds.
///
/// This gives a more robust estimate of the model's performance when data changes.
///
/// Will use k-fold cross-validation based on the specified strategy to calculate the RMSE for each fold,
/// and then returns the mean and standard deviation of the RMSEs across all folds.
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
///
/// # Parameters
/// - `y`: An iterator of observed values (ground truth).
/// - `y_fit`: An iterator of predicted values from the model.
/// - `strategy`: The cross-validation strategy to use for splitting the data into folds.
///
/// # Returns
/// An `UncertainValue` containing the mean RMSE and its standard deviation across the folds, or `None` if the calculation fails due to empty data.
pub fn folded_rmse<T: Value>(
    y: impl Iterator<Item = T>,
    y_fit: impl Iterator<Item = T>,
    strategy: CvStrategy,
) -> Option<UncertainValue<T>> {
    let data: Vec<(T, T)> = y.zip(y_fit).collect();
    folded_metric(&data, strategy, &root_mean_squared_error)
}

/// A trait for computing a metric that can be folded across multiple folds of data.
///
/// Basically a way to abstract over different metrics (e.g., RMSE, MAE) that can be computed on training data in each fold of cross-validation.
trait FoldableMetric<T: Value, I1, I2> {
    fn compute(&self, y: I1, y_fit: I2) -> Option<T>;
}
impl<T: Value, F, I1, I2> FoldableMetric<T, I1, I2> for F
where
    I1: Iterator<Item = T>,
    I2: Iterator<Item = T>,
    F: Fn(I1, I2) -> Option<T>,
{
    #[inline(always)]
    fn compute(&self, y: I1, y_fit: I2) -> Option<T> {
        self(y, y_fit)
    }
}

/// A helper function to compute a foldable metric (like RMSE) across multiple folds of data
///
/// Returns an `UncertainValue` containing the mean and standard deviation of the metric across the folds, or `None` if the calculation fails.
fn folded_metric<T: Value, F>(
    data: &[(T, T)],
    strategy: CvStrategy,
    f: &F,
) -> Option<UncertainValue<T>>
where
    F: FoldableMetric<T, std::vec::IntoIter<T>, std::vec::IntoIter<T>>,
{
    let folds = cross_validation_split(data, strategy);

    // Now we try try k times, each time leaving out one fold for validation
    let mut metrics = Vec::with_capacity(folds.len());

    for i in 0..folds.len() {
        let training_set: Vec<(T, T)> = folds
            .iter()
            .enumerate()
            .filter_map(|(j, fold)| if j == i { None } else { Some(fold.clone()) })
            .flatten()
            .collect();

        // Calculate the metric on the training set
        let (train_y, train_y_fit): (Vec<T>, Vec<T>) = training_set.into_iter().unzip();
        let metric = f.compute(train_y.into_iter(), train_y_fit.into_iter())?;
        metrics.push(metric);
    }

    UncertainValue::new_from_values(metrics.into_iter())
}
