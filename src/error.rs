//! Error types for polynomial curve fitting
//!
//! This module defines the common errors encountered when creating or
//! evaluating polynomial fits, along with a convenient `Result` alias.

/// Errors that can occur during polynomial curve fitting.
///
/// This enum represents the common failure modes when constructing or
/// evaluating polynomial fits.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Cannot perform curve fitting because there is no data.
    #[error("No data available for fitting")]
    NoData,

    /// The specified basis cannot have the given number of coefficients.
    #[error("Specified basis cannot have exactly {0} coefficients")]
    InvalidNumberOfParameters(usize),

    /// Cannot compute polynomial fit because the design matrix is singular
    ///
    /// Usually, degree is too high, or there is not enough data.
    #[error(
        "Design matrix (X^T X) is not invertible; the data may be insufficient, collinear, or overfitted. [n: {n}, k: {k}]"
    )]
    SingularMatrix {
        /// Number of data points
        n: usize,
        /// Number of basis functions
        k: usize,
    },

    /// Autofit error - no valid models found
    ///
    /// Something is probably wrong with the data used. Try choosing a degree manually to test
    #[error("None of the models tested are valid")]
    NoModel,

    /// The requested polynomial degree is too high for the dataset.
    ///
    /// The degree must be less than the number of data points.
    #[error("Polynomial degree `{0}` is too high for the dataset")]
    DegreeTooHigh(usize),

    /// The input x-values are outside the valid range for this fit.
    ///
    /// Most bases (e.g., Chebyshev) only guarantee numerical stability
    /// within a specific domain. Use `as_polynomial` to ignore these bounds
    /// if you accept potential instability.
    #[error(
        "This fit is only stable within the x-value range {0}..{1}. Use call `as_polynomial` to ignore these bounds"
    )]
    DataRange(String, String),

    /// A numeric value could not be cast to the target type. This is usually a custom type much smaller than f64/f32
    #[error("Failed to cast value to target type")]
    CastFailed,

    /// Failed to solve the algebraic system during fitting.
    ///
    /// Contains a static string describing the solver error.
    #[error("Failed to solve: {0}")]
    Algebra(&'static str),
}

/// Result type for the polynomial curve fitting
pub type Result<T> = std::result::Result<T, Error>;
