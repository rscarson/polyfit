use crate::value::Value;

/// Standard Z-score confidence levels for fitted models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Confidence {
    /// 80% confidence level
    P80,

    /// 90% confidence level
    P90,

    /// 95% confidence level
    P95,

    /// 98% confidence level
    P98,

    /// 99% confidence level
    P99,

    /// 99.9% confidence level
    P999,

    /// Custom confidence level
    Custom(f64),
}

impl Confidence {
    // Ref: <https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf>
    #[allow(clippy::approx_constant)]
    #[rustfmt::skip]
    const T_TABLE: &[&[f64]] = &[
/* P/DF   0.800     0.900     0.950    0.980    0.990     0.999 */
/* 1 */ &[3.078,    6.314,    12.71,  31.82,    63.66,    636.62],
/* 2 */ &[1.886,    2.92,     4.303,  6.965,    9.925,    31.599],
/* 3 */ &[1.638,    2.353,    3.182,  4.541,    5.841,    12.924],
/* 4 */ &[1.533,    2.132,    2.776,  3.747,    4.604,    8.610],
/* 5 */ &[1.476,    2.015,    2.571,  3.365,    4.032,    6.869],
/* 6 */ &[1.440,    1.943,    2.447,  3.143,    3.707,    5.959],
/* 7 */ &[1.415,    1.895,    2.365,  2.998,    3.499,    5.408],
/* 8 */ &[1.397,    1.86,     2.306,  2.896,    3.355,    5.041],
/* 9 */ &[1.383,    1.833,    2.262,  2.821,    3.25,     4.781],
/*10 */ &[1.372,    1.812,    2.228,  2.764,    3.169,    4.587],
/*11 */ &[1.363,    1.796,    2.201,  2.718,    3.106,    4.437],
/*12 */ &[1.356,    1.782,    2.179,  2.681,    3.055,    4.318],
/*13 */ &[1.350,    1.771,    2.16,   2.65,     3.012,    4.221],
/*14 */ &[1.345,    1.761,    2.145,  2.624,    2.977,    4.140],
/*15 */ &[1.341,    1.753,    2.131,  2.602,    2.947,    4.073],
/*16 */ &[1.337,    1.746,    2.12,   2.583,    2.921,    4.015],
/*17 */ &[1.333,    1.74,     2.11,   2.567,    2.898,    3.965],
/*18 */ &[1.330,    1.734,    2.101,  2.552,    2.878,    3.922],
/*19 */ &[1.328,    1.729,    2.093,  2.539,    2.861,    3.883],
/*20 */ &[1.325,    1.725,    2.086,  2.528,    2.845,    3.850],
/*21 */ &[1.323,    1.721,    2.08,   2.518,    2.831,    3.819],
/*22 */ &[1.321,    1.717,    2.074,  2.508,    2.819,    3.792],
/*23 */ &[1.319,    1.714,    2.069,  2.5,      2.807,    3.768],
/*24 */ &[1.318,    1.711,    2.064,  2.492,    2.797,    3.745],
/*25 */ &[1.316,    1.708,    2.06,   2.485,    2.787,    3.725],
/*26 */ &[1.315,    1.706,    2.056,  2.479,    2.779,    3.707],
/*27 */ &[1.314,    1.703,    2.052,  2.473,    2.771,    3.690],
/*28 */ &[1.313,    1.701,    2.048,  2.467,    2.763,    3.674],
/*29 */ &[1.311,    1.699,    2.045,  2.462,    2.756,    3.659],
/*30 */ &[1.310,    1.697,    2.042,  2.457,    2.75,     3.646],
    ];

    /// Returns the confidence level as a percentage.
    #[must_use]
    pub fn percentage(&self) -> f64 {
        match self {
            Confidence::P80 => 0.8,
            Confidence::P90 => 0.9,
            Confidence::P95 => 0.95,
            Confidence::P98 => 0.98,
            Confidence::P99 => 0.99,
            Confidence::P999 => 0.999,
            Confidence::Custom(z) => *z,
        }
    }

    /// Returns the alpha level (1 - confidence level).
    #[must_use]
    pub fn alpha(&self) -> f64 {
        1.0 - self.percentage()
    }

    /// Returns the T-score associated with the confidence level and degrees of freedom.
    ///
    /// For very large degrees of freedom, the T-score approaches the Z-score.
    /// Also approximates using T-score for custom confidence levels.
    ///
    /// Ref: <https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf>
    #[must_use]
    pub fn t_score(self, df: usize) -> f64 {
        match df {
            0 => f64::INFINITY,
            i if i >= Self::T_TABLE.len() => self.z_score(),
            i => {
                let col = match self {
                    Confidence::P80 => 0,
                    Confidence::P90 => 1,
                    Confidence::P95 => 2,
                    Confidence::P98 => 3,
                    Confidence::P99 => 4,
                    Confidence::P999 => 5,
                    Confidence::Custom(_) => return self.z_score(),
                };

                Self::T_TABLE[i - 1][col]
            }
        }
    }

    /// Returns the Z-score associated with the confidence level.
    ///
    /// The Z-score a measure of how many standard deviations an element is from the mean in a standard normal distribution.
    ///
    /// Ref: <https://en.wikipedia.org/wiki/Normal_distribution>, 'Quantile function' section.
    #[must_use]
    pub fn z_score(self) -> f64 {
        match self {
            Confidence::P80 => 1.281,
            Confidence::P90 => 1.644,
            Confidence::P95 => 1.959,
            Confidence::P98 => 2.326,
            Confidence::P99 => 2.575,
            Confidence::P999 => 3.290,
            Confidence::Custom(z) => z,
        }
    }

    /// Converts the Z-score to the numeric type `T`.
    ///
    /// # Errors
    /// Returns an error if the confidence level cannot be cast to the required type.
    pub fn try_cast<T: Value>(self) -> crate::error::Result<T> {
        T::try_cast(self.z_score())
    }
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Confidence::P80 => write!(f, "80%"),
            Confidence::P90 => write!(f, "90%"),
            Confidence::P95 => write!(f, "95%"),
            Confidence::P98 => write!(f, "98%"),
            Confidence::P99 => write!(f, "99%"),
            Confidence::P999 => write!(f, "99.9%"),
            Confidence::Custom(z) => write!(f, "{z}σ"),
        }
    }
}

/// Specifies a tolerance level for numerical comparisons.
///
/// Can be either an absolute unit value, or a percentage of the variance or of the value itself.
///
/// The idea is to seperate the uncertainty in the model used (Confidence)
/// from the known uncertainty in the measurements (Tolerance).
///
/// It lets you encode domain knowledge about the expected accuracy of the data (Sensor specs, process variation, etc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tolerance<T: Value> {
    /// An absolute tolerance value.
    ///
    /// For example, if your sensor has a known error of ±0.5 dB, you would use `Tolerance::Absolute(0.5)`.
    Absolute(T),

    /// A percentage of the variance of the data set.
    ///
    /// For example, in vibration analysis of rotating machinery, engineers may allow a tolerance of ±10% of the signal variance
    /// to account for normal operational fluctuations. Use `Tolerance::Variance(0.1)`.
    Variance(T),

    /// A percentage of the value itself.
    ///
    /// For example, if your sensor has a known error of ±5% of the reading, you would use `Tolerance::Measurement(0.05)`.
    Measurement(T),
}

// A confidence band for a fitted model.
///
/// Represents a predicted range for model outputs at a given confidence level.
/// The band contains the central estimate (`value`) and the upper and lower bounds.
///
/// Created by a [`crate::CurveFit`]
///
/// # Type Parameters
/// - `T`: Numeric type that implements `Value` (e.g., `f64`, `f32`).
///
/// # Fields
/// - `level`: Confidence level (e.g., 95%) as a [`Confidence`] enum.
/// - `value`: Central predicted value of the model.
/// - `lower`: Lower bound of the confidence band.
/// - `upper`: Upper bound of the confidence band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceBand<T: Value> {
    pub(crate) level: Confidence,
    pub(crate) tolerance: Option<Tolerance<T>>,
    pub(crate) value: T,
    pub(crate) lower: T,
    pub(crate) upper: T,
}

impl<T: Value> ConfidenceBand<T> {
    /// Returns the tolerance used to compute the confidence band, if any.
    pub fn tolerance(&self) -> Option<Tolerance<T>> {
        self.tolerance
    }

    /// Returns the confidence level of the band.
    pub fn confidence(&self) -> Confidence {
        self.level
    }

    /// Returns the central predicted value of the model.
    pub fn value(&self) -> T {
        self.value
    }

    /// Returns the lower bound of the confidence band.
    pub fn min(&self) -> T {
        self.lower
    }

    /// Returns the upper bound of the confidence band.
    pub fn max(&self) -> T {
        self.upper
    }

    /// Returns the midpoint of the confidence band.
    pub fn center(&self) -> T {
        (self.lower + self.upper) / T::two()
    }

    /// Returns the width of the confidence band (upper - lower).
    pub fn width(&self) -> T {
        self.upper - self.lower
    }
}

impl<T: Value> std::fmt::Display for ConfidenceBand<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (min, y, max) = (self.min(), self.center(), self.max());
        let confidence = self.level;
        write!(f, "{y} ({min}, {max}) [confidence = {confidence}]")
    }
}
