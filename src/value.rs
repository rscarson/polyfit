//! Numeric types and iteration utilities for polynomial curves.
//!
//! This module defines the [`Value`] trait, which abstracts the numeric
//! types that can be used in polynomial fitting and evaluation, ensuring
//! compatibility with nalgebra, floating-point operations, and formatting.
//!
//! # Traits
//!
//! - [`Value`]: Extends `Float`, `Scalar`, and `ComplexField` to provide:
//!   - A canonical `two()` constant.
//!   - `try_cast` for safe type conversion with error handling.
//!   - `powi` for integer exponentiation.
//!
//! # Iterators
//!
//! - [`ValueRange`]: A floating-point range iterator with a specified step,
//!   useful for generating evaluation points for polynomials.
//!
//! # Example
//!
//! ```rust
//! use polyfit::value::{Value, ValueRange};
//!
//! // Create a range of f64 values from 0.0 to 1.0 in steps of 0.1
//! let range = ValueRange::new(0.0, 1.0, 0.1);
//! for x in range {
//!     println!("{x}");
//! }
//!
//! // Using Value trait methods
//! let two = f64::two();
//! let squared = two.powi(2);
//! ```
use std::ops::Range;

use crate::error::Error;

/// Numeric type for curves
pub trait Value:
    nalgebra::Scalar
    + nalgebra::ComplexField<RealField = Self>
    + num_traits::float::FloatCore
    + std::fmt::LowerExp
{
    /// Returns the value 2.0
    #[must_use]
    fn two() -> Self {
        Self::one() + Self::one()
    }

    /// Tries to cast a value to the target type
    ///
    /// # Errors
    /// Returns an error if the cast fails
    fn try_cast<U: num_traits::NumCast>(n: U) -> Result<Self, Error> {
        num_traits::cast(n).ok_or(Error::CastFailed)
    }

    /// Raises the value to the power of an integer
    #[must_use]
    fn powi(self, n: i32) -> Self {
        nalgebra::ComplexField::powi(self, n)
    }

    /// Get the absolute value for a numeric type
    #[must_use]
    fn abs(self) -> Self {
        nalgebra::ComplexField::abs(self)
    }

    /// Returns the absolute difference between two values.
    #[must_use]
    fn abs_sub(self, other: Self) -> Self {
        nalgebra::ComplexField::abs(self - other)
    }

    /// Check if the value is negative
    fn is_sign_negative(&self) -> bool {
        self < &Self::zero()
    }

    /// Returns the sign of the value as a numeric type
    ///
    /// This function returns -1 for negative values, 1 for positive values, and NaN for NaN values.
    #[must_use]
    fn f_signum(&self) -> Self {
        match self {
            _ if self.is_nan() => Self::nan(),
            _ if self.is_sign_negative() => -Self::one(),
            _ => Self::one(),
        }
    }
}

impl<T> Value for T where
    T: nalgebra::Scalar
        + nalgebra::ComplexField<RealField = Self>
        + num_traits::float::FloatCore
        + std::fmt::LowerExp
{
}

/// A stepped iterator over a range of floating-point numbers
pub struct ValueRange<T: Value> {
    start: T,
    end: T,
    step: T,
}
impl<T: Value> ValueRange<T> {
    /// Creates a new `ValueRange`
    pub fn new(start: T, end: T, step: T) -> Self {
        Self { start, end, step }
    }

    /// Creates a new `ValueRange` iterating step=1
    pub fn new_unit(start: T, end: T) -> Self {
        Self {
            start,
            end,
            step: T::one(),
        }
    }
}
impl<T: Value> Iterator for ValueRange<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let current = self.start;
            self.start += self.step;
            Some(current)
        } else {
            None
        }
    }
}

/// Extension trait for accessing the `x` and `y` coordinates of a type.
///
/// This trait is intended for any type that conceptually represents a 2D
/// coordinate or point. Implementations should provide accessors that return
/// the respective coordinate values.
///
/// # Associated Types
///
/// * `Output` – The type of the coordinate values returned by `x` and `y`.
///
/// # Examples
///
/// ```
/// # use polyfit::value::CoordExt;
/// let data = vec![(1.5, -2.0), (2.0, 3.0), (0.0, 1.0)];
/// println!("{:?}", data.y());
/// ```
pub trait CoordExt<T: Value> {
    /// Returns an iterator over the x-coordinates of this value.
    fn x_iter(&self) -> impl Iterator<Item = T>;

    /// Returns an iterator over the y-coordinates of this value.
    fn y_iter(&self) -> impl Iterator<Item = T>;

    /// Returns the x-coordinate of this value.
    fn x(&self) -> Vec<T> {
        self.x_iter().collect()
    }

    /// Returns the y-coordinate of this value.
    fn y(&self) -> Vec<T> {
        self.y_iter().collect()
    }

    /// Returns the range of x-coordinates of this value.
    fn x_range(&self) -> Option<Range<T>> {
        let x_min = self.x_iter().fold(None, |acc: Option<(T, T)>, x| {
            Some(match acc {
                Some((min, max)) => (min.min(x), max.max(x)),
                None => (x, x),
            })
        });
        x_min.map(|(start, end)| start..end)
    }

    /// Returns the range of y-coordinates of this value.
    fn y_range(&self) -> Option<Range<T>> {
        let y_min = self.y_iter().fold(None, |acc: Option<(T, T)>, y| {
            Some(match acc {
                Some((min, max)) => (min.min(y), max.max(y)),
                None => (y, y),
            })
        });
        y_min.map(|(start, end)| start..end)
    }

    /// Converts the coordinates of this value to `f64`.
    ///
    /// # Errors
    /// Returns an error if any of the coordinates cannot be converted to `f64`.
    fn as_f64(&self) -> crate::error::Result<Vec<(f64, f64)>> {
        self.x_iter()
            .zip(self.y_iter())
            .map(|(x, y)| {
                let x_f64 = f64::try_cast(x)?;
                let y_f64 = f64::try_cast(y)?;
                Ok((x_f64, y_f64))
            })
            .collect()
    }
}
impl<T: Value> CoordExt<T> for Vec<(T, T)> {
    fn x_iter(&self) -> impl Iterator<Item = T> {
        self.iter().map(|(x, _)| *x)
    }

    fn y_iter(&self) -> impl Iterator<Item = T> {
        self.iter().map(|(_, y)| *y)
    }
}
impl<T: Value> CoordExt<T> for &[(T, T)] {
    fn x_iter(&self) -> impl Iterator<Item = T> {
        self.iter().map(|(x, _)| *x)
    }

    fn y_iter(&self) -> impl Iterator<Item = T> {
        self.iter().map(|(_, y)| *y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_range() {
        let range = ValueRange::new(0.0, 1.0, 0.1);
        let values: Vec<_> = range.collect();
        assert_eq!(values.len(), 11);
    }
}
