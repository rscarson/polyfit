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
use std::ops::{Range, RangeInclusive};

use crate::error::Error;

/// Numeric type for curves
pub trait Value:
    nalgebra::Scalar
    + nalgebra::ComplexField<RealField = Self>
    + nalgebra::RealField
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

    /// Converts the value to `usize`
    fn as_usize(&self) -> Option<usize> {
        num_traits::cast(*self)
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
            _ if nalgebra::RealField::is_sign_negative(self) => -Self::one(),
            _ => Self::one(),
        }
    }

    /// Computes the factorial of a non-negative integer `n`.
    #[must_use]
    fn factorial(n: usize) -> Self {
        if n == 0 || n == 1 {
            Self::one()
        } else {
            let mut result = Self::one();
            for i in 2..=n {
                let i = Self::try_cast(i).unwrap_or(Self::infinity());
                result *= i;
            }
            result
        }
    }

    /// Converts a `usize` to the target numeric type.
    ///
    /// Results in `infinity` if the value is out of range.
    #[must_use]
    fn from_positive_int(n: usize) -> Self {
        Self::try_cast(n).unwrap_or(Self::infinity())
    }
}

impl<T> Value for T where
    T: nalgebra::Scalar
        + nalgebra::ComplexField<RealField = Self>
        + nalgebra::RealField
        + num_traits::float::FloatCore
        + std::fmt::LowerExp
{
}

/// Iterator over a range of floating-point values with a specified step.
///
/// This iterator yields values starting from `start` up to and including `end`,
/// incrementing by `step` on each iteration.
pub struct SteppedValues<T: Value> {
    range: RangeInclusive<T>,
    step: T,
    index: T,
}
impl<T: Value> SteppedValues<T> {
    /// Creates a new iterator over stepped values in a range
    ///
    /// Will yield values starting from `range.start` up to and including `range.end`
    pub fn new(range: RangeInclusive<T>, step: T) -> Self {
        Self {
            range,
            step,
            index: T::zero(),
        }
    }

    /// Creates a new iterator over stepped values in a range with a step of 1.0
    ///
    /// Will yield values starting from `range.start` up to and including `range.end`
    pub fn new_unit(range: RangeInclusive<T>) -> Self {
        Self::new(range, T::one())
    }

    /// Returns the number of steps remaining in the iterator
    pub fn len(&self) -> usize {
        let value = *self.range.start() + self.index * self.step;
        let remaining = *self.range.end() - value;
        let steps = remaining / self.step;
        steps.as_usize().unwrap_or(0)
    }

    /// Returns true if the iterator is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl<T: Value> Iterator for SteppedValues<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let value = *self.range.start() + self.index * self.step;
        if value <= *self.range.end() {
            self.index += T::one();
            Some(value)
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
                Some((min, max)) => (
                    nalgebra::RealField::min(min, x),
                    nalgebra::RealField::max(max, x),
                ),
                None => (x, x),
            })
        });
        x_min.map(|(start, end)| start..end)
    }

    /// Returns the range of y-coordinates of this value.
    fn y_range(&self) -> Option<Range<T>> {
        let y_min = self.y_iter().fold(None, |acc: Option<(T, T)>, y| {
            Some(match acc {
                Some((min, max)) => (
                    nalgebra::RealField::min(min, y),
                    nalgebra::RealField::max(max, y),
                ),
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

/// Trait for infallible integer casting with clamping.
pub trait IntClampedCast:
    num_traits::Num + num_traits::NumCast + num_traits::Bounded + Copy + PartialOrd + Ord
{
    /// Clamps a value to the range of the target type and casts it.
    fn clamped_cast<T: num_traits::PrimInt>(self) -> T {
        //
        // Simple case: self is in range of T
        if let Some(v) = num_traits::cast(self) {
            return v;
        }

        let min = match num_traits::cast::<T, Self>(T::min_value()) {
            Some(v) => v,              // Self can go lower than T - clamp to min
            None => Self::min_value(), // Self cannot go lower than T
        };

        let max = match num_traits::cast::<T, Self>(T::max_value()) {
            Some(v) => v,              // Self can go higher than T - clamp to max
            None => Self::max_value(), // Self cannot go higher than T
        };

        let clamped = self.clamp(min, max);
        num_traits::cast(clamped).expect("clamped value should be in range")
    }
}
impl<T: num_traits::PrimInt> IntClampedCast for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_range() {
        let range = SteppedValues::new(0.0..=1.0, 0.1);
        let values: Vec<_> = range.collect();
        assert_eq!(values.len(), 11);
    }

    #[test]
    fn clamped_cast_edge_cases() {
        // i8 -> i8 (trivial)
        assert_eq!(0i8.clamped_cast::<i8>(), 0);
        assert_eq!(127i8.clamped_cast::<i8>(), 127);
        assert_eq!((-128i8).clamped_cast::<i8>(), -128);

        // i8 -> u8 (negative clamps to 0)
        assert_eq!(0i8.clamped_cast::<u8>(), 0);
        assert_eq!(127i8.clamped_cast::<u8>(), 127);
        assert_eq!((-1i8).clamped_cast::<u8>(), 0);
        assert_eq!((-128i8).clamped_cast::<u8>(), 0);

        // u8 -> i8 (overflow clamps to 127)
        assert_eq!(0u8.clamped_cast::<i8>(), 0);
        assert_eq!(127u8.clamped_cast::<i8>(), 127);
        assert_eq!(128u8.clamped_cast::<i8>(), 127);
        assert_eq!(255u8.clamped_cast::<i8>(), 127);

        // i16 -> i8 (underflow/overflow)
        assert_eq!(0i16.clamped_cast::<i8>(), 0);
        assert_eq!(127i16.clamped_cast::<i8>(), 127);
        assert_eq!(128i16.clamped_cast::<i8>(), 127);
        assert_eq!((-1i16).clamped_cast::<i8>(), -1);
        assert_eq!((-128i16).clamped_cast::<i8>(), -128);
        assert_eq!((-129i16).clamped_cast::<i8>(), -128);
        assert_eq!(32767i16.clamped_cast::<i8>(), 127);
        assert_eq!((-32768i16).clamped_cast::<i8>(), -128);

        // i16 -> u8
        assert_eq!(0i16.clamped_cast::<u8>(), 0);
        assert_eq!(255i16.clamped_cast::<u8>(), 255);
        assert_eq!(256i16.clamped_cast::<u8>(), 255);
        assert_eq!((-1i16).clamped_cast::<u8>(), 0);
        assert_eq!((-32768i16).clamped_cast::<u8>(), 0);

        // u16 -> i8
        assert_eq!(0u16.clamped_cast::<i8>(), 0);
        assert_eq!(127u16.clamped_cast::<i8>(), 127);
        assert_eq!(128u16.clamped_cast::<i8>(), 127);
        assert_eq!(255u16.clamped_cast::<i8>(), 127);
        assert_eq!(65535u16.clamped_cast::<i8>(), 127);

        // i32 -> i16
        assert_eq!(0i32.clamped_cast::<i16>(), 0);
        assert_eq!(32767i32.clamped_cast::<i16>(), 32767);
        assert_eq!(32768i32.clamped_cast::<i16>(), 32767);
        assert_eq!((-32768i32).clamped_cast::<i16>(), -32768);
        assert_eq!((-32769i32).clamped_cast::<i16>(), -32768);

        // i32 -> u16
        assert_eq!(0i32.clamped_cast::<u16>(), 0);
        assert_eq!(65535i32.clamped_cast::<u16>(), 65535);
        assert_eq!(65536i32.clamped_cast::<u16>(), 65535);
        assert_eq!((-1i32).clamped_cast::<u16>(), 0);
        assert_eq!((-32768i32).clamped_cast::<u16>(), 0);

        // u32 -> i16
        assert_eq!(0u32.clamped_cast::<i16>(), 0);
        assert_eq!(32767u32.clamped_cast::<i16>(), 32767);
        assert_eq!(32768u32.clamped_cast::<i16>(), 32767);
        assert_eq!(65535u32.clamped_cast::<i16>(), 32767);
        assert_eq!(u32::MAX.clamped_cast::<i16>(), 32767);

        // u32 -> u16
        assert_eq!(0u32.clamped_cast::<u16>(), 0);
        assert_eq!(65535u32.clamped_cast::<u16>(), 65535);
        assert_eq!(65536u32.clamped_cast::<u16>(), 65535);
        assert_eq!(u32::MAX.clamped_cast::<u16>(), 65535);

        // i64 -> i8
        assert_eq!(i64::MIN.clamped_cast::<i8>(), -128);
        assert_eq!(i64::MAX.clamped_cast::<i8>(), 127);
        assert_eq!((-129i64).clamped_cast::<i8>(), -128);
        assert_eq!(128i64.clamped_cast::<i8>(), 127);

        // u64 -> i8
        assert_eq!(0u64.clamped_cast::<i8>(), 0);
        assert_eq!(255u64.clamped_cast::<i8>(), 127);
        assert_eq!(u64::MAX.clamped_cast::<i8>(), 127);

        // i128 -> u8
        assert_eq!(0i128.clamped_cast::<u8>(), 0);
        assert_eq!(255i128.clamped_cast::<u8>(), 255);
        assert_eq!(256i128.clamped_cast::<u8>(), 255);
        assert_eq!((-1i128).clamped_cast::<u8>(), 0);
        assert_eq!(i128::MIN.clamped_cast::<u8>(), 0);

        // u128 -> i8
        assert_eq!(0u128.clamped_cast::<i8>(), 0);
        assert_eq!(127u128.clamped_cast::<i8>(), 127);
        assert_eq!(128u128.clamped_cast::<i8>(), 127);
        assert_eq!(u128::MAX.clamped_cast::<i8>(), 127);
    }
}
