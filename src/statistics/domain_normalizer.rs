use std::ops::RangeInclusive;

use crate::value::{IntClampedCast, Value};

/// Normalizes values from one range to another.
///
/// Destination range use infinity to indicate unbounded ranges.
/// - `(f64::NEG_INFINITY, f64::INFINITY)` means no normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainNormalizer<T: Value> {
    src_range: (T, T),
    dst_range: (T, T),
    shift: T,
    scale: T,
}
impl<T: Value> Default for DomainNormalizer<T> {
    fn default() -> Self {
        DomainNormalizer {
            src_range: (T::zero(), T::one()),
            dst_range: (T::zero(), T::one()),
            shift: T::zero(),
            scale: T::one(),
        }
    }
}
impl<T: Value> DomainNormalizer<T> {
    /// Creates a new `DomainNormalizer` for the given source and destination ranges.
    pub fn new(src_range: (T, T), dst_range: (T, T)) -> Self {
        let (src_min, src_max) = src_range;
        let (dst_min, dst_max) = dst_range;

        if dst_min == T::neg_infinity() && dst_max == T::infinity() {
            return DomainNormalizer {
                src_range,
                dst_range,
                shift: T::zero(),
                scale: T::one(),
            };
        }

        if dst_min == T::neg_infinity() {
            // We have a maximum only
            // Adjust x by - src_max, then add dst_max
            return DomainNormalizer {
                src_range,
                dst_range,
                shift: -src_max + dst_max,
                scale: T::one(),
            };
        }

        if dst_max == T::infinity() {
            // We have a minimum only
            // Adjust x by - src_min, then add dst_min
            return DomainNormalizer {
                src_range,
                dst_range,
                shift: -src_min + dst_min,
                scale: T::one(),
            };
        }

        let scale = (dst_max - dst_min) / (src_max - src_min);
        let shift = dst_min - scale * src_min;

        DomainNormalizer {
            src_range,
            dst_range,
            shift,
            scale,
        }
    }

    /// Creates a new `DomainNormalizer` from an inclusive source range and a destination range.
    pub fn from_range(src_range: RangeInclusive<T>, dst_range: (T, T)) -> Self {
        let (min, max) = src_range.into_inner();
        Self::new((min, max), dst_range)
    }

    /// Creates a new `DomainNormalizer` from an iterator of source values and a destination range.
    pub fn from_data(src: impl Iterator<Item = T>, dst_range: (T, T)) -> Option<Self> {
        let range = src.fold(None, |acc: Option<(T, T)>, x| {
            Some(match acc {
                None => (x, x),
                Some((min, max)) => (
                    nalgebra::RealField::min(min, x),
                    nalgebra::RealField::max(max, x),
                ),
            })
        })?;
        Some(Self::new(range, dst_range))
    }

    /// Shift value applied during normalization.
    pub fn shift(&self) -> T {
        self.shift
    }

    /// Scale value applied during normalization.
    pub fn scale(&self) -> T {
        self.scale
    }

    /// Returns the source range of the normalizer.
    pub fn src_range(&self) -> (T, T) {
        self.src_range
    }

    /// Returns the destination range of the normalizer.
    pub fn dst_range(&self) -> (T, T) {
        self.dst_range
    }

    /// Normalizes a value from the source range to the destination range.
    #[inline(always)]
    pub fn normalize(&self, x: T) -> T {
        self.scale * x + self.shift
    }

    /// Denormalizes a value from the destination range back to the source range.
    pub fn denormalize(&self, x: T) -> T {
        (x - self.shift) / self.scale
    }

    /// Denormalizes a complex value from the destination range back to the source range.
    pub fn denormalize_complex(&self, z: nalgebra::Complex<T>) -> nalgebra::Complex<T> {
        let s = self.scale();
        let sh = self.shift();
        nalgebra::Complex::new(
            z.re / s + sh, // real part
            z.im / s,      // imaginary part
        )
    }

    /// Denormalizes a slice of polynomial coefficients from the destination range back to the source range.
    ///
    /// The coefficients are assumed to be in ascending order (constant term first).
    #[must_use]
    pub fn denormalize_coefs(&self, coefs: &[T]) -> Vec<T> {
        let (x_min, x_max) = self.src_range();
        let (d_min, d_max) = self.dst_range();
        let (alpha, beta) = if d_min == T::neg_infinity() && d_max == T::infinity() {
            (T::one(), T::zero()) // no change
        } else if d_min == T::neg_infinity() {
            // We have a maximum only
            // Adjust x by - src_max, then add dst_max
            let beta = d_max - x_max;
            (T::one(), beta)
        } else if d_max == T::infinity() {
            // We have a minimum only... shift only
            let beta = d_min - x_min;
            (T::one(), beta)
        } else {
            let alpha = (d_max - d_min) / (x_max - x_min);
            let beta = d_min - alpha * x_min;
            (alpha, beta)
        };

        let mut unnorm = vec![T::zero(); coefs.len()];
        for (i, &c) in coefs.iter().enumerate() {
            for j in 0..=i {
                let binom = T::factorial(i) / (T::factorial(j) * T::factorial(i - j));
                unnorm[j] += c
                    * binom
                    * Value::powi(alpha, j.clamped_cast())
                    * Value::powi(beta, (i - j).clamped_cast());
            }
        }
        unnorm
    }
}
impl<T: Value> std::fmt::Display for DomainNormalizer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (src_min, src_max) = self.src_range;
        let (dst_min, dst_max) = self.dst_range;

        let dst_min = if dst_min == T::neg_infinity() {
            "-∞".to_string()
        } else {
            dst_min.to_string()
        };

        let dst_max = if dst_max == T::infinity() {
            "∞".to_string()
        } else if Value::abs_sub(dst_max, T::pi()) < T::epsilon() {
            "π".to_string()
        } else if Value::abs_sub(dst_max, T::two_pi()) < T::epsilon() {
            "2π".to_string()
        } else {
            dst_max.to_string()
        };

        if self.shift == T::zero() && self.scale == T::one() {
            return write!(f, "T[ {dst_min}..{dst_max} ]");
        }

        write!(f, "T[ {src_min}..{src_max} -> {dst_min}..{dst_max} ]")
    }
}
