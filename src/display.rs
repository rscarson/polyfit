//! Utilities for displaying and formatting polynomials
//!
//! This module provides tools to convert polynomial objects into human-readable
//! strings, handle coefficients formatting, and add superscript exponents.
//!
//! # Key Concepts
//! - **[`PolynomialDisplay`]**: Trait to define how a polynomial basis renders terms.
//! - **[`Term`]**: Represents a single polynomial term with a sign and body.
//! - **[`Sign`]**: Tracks whether a term is positive or negative.
//!
//! # Using Display
//! 1. Implement [`PolynomialDisplay`] for your basis to control term formatting.
//! 2. Use `format_polynomial` to render the full polynomial as `"y = ..."`.
//!
//! # Helpers
//! - [`format_coefficient`]: Formats a numeric coefficient, skipping zeros.
//! - [`exponentiate`]: Converts a number to a Unicode superscript for exponents.
#![allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]

use crate::value::Value;

pub mod unicode;

/// Default precision for formatting used by the provided implementations of [`PolynomialDisplay`]
pub const DEFAULT_PRECISION: usize = 2;

/// Default range in which scientific notation is not used
#[must_use]
pub fn default_fixed_range<T: Value>() -> Option<std::ops::Range<T>> {
    const RANGE: std::ops::Range<f64> = 1e-3..1e3;
    let s = T::try_cast(RANGE.start).ok()?;
    let e = T::try_cast(RANGE.end).ok()?;
    Some(s..e)
}

/// Trait for formatting and displaying polynomial expressions.
///
/// This trait abstracts how polynomials are turned into human-readable
/// strings. Implementors control how individual terms are displayed,
/// while a default implementation assembles them into a full equation.
///
/// # Provided behavior
/// - [`format_term`] is required: defines how to render a single term
///   (e.g., `3x²`, `-x`, or `7`).
/// - [`format_polynomial`] is provided: writes the full polynomial as
///   `"y = ..."` into any [`std::fmt::Write`] buffer.
pub trait PolynomialDisplay<T: Value> {
    /// Formats a single polynomial term for display.
    ///
    /// Implementors define how each term should be written, which usually involves:
    /// - Determining the sign of the coefficient.
    /// - Formatting the coefficient itself (e.g., with fixed precision).
    /// - Choosing the representation of the basis function at the given degree
    ///   (e.g., `"x^2"` for monomials, `"T₂(x)"` for Chebyshev).
    ///
    /// Returning `None` indicates the term should be skipped
    /// (typically when the coefficient is zero).
    ///
    /// # Helpers
    /// There are a few functions provided to assist in formatting. See the `display` module:
    /// [`exponentiate`], [`format_coefficient`], and [`DEFAULT_PRECISION`].
    ///
    /// # Example
    /// A Chebyshev implementation might render terms as `coef·Tₙ(x)`:
    ///
    /// ```rust
    /// # use polyfit::display::{format_variable, format_coefficient, DEFAULT_PRECISION, PolynomialDisplay, Term, Sign};
    ///
    /// pub struct MyBasis;
    /// impl<T: polyfit::value::Value> PolynomialDisplay<T> for MyBasis {
    ///     fn format_term(&self, degree: i32, coef: T) -> Option<Term> {
    ///         let sign = Sign::from_coef(coef);
    ///
    ///         // Turns "T" into "Tₙ(x)"
    ///         let func = format_variable("T", Some(&degree.to_string()), 1);
    ///         let base = format!("{func}(x)");
    ///
    ///         // Formats the coefficient as SCI if numerically appropriate, and skips if zero.
    ///         let coef = format_coefficient(coef, degree, DEFAULT_PRECISION)?;
    ///
    ///         let body = format!("{coef}{base}");
    ///         Some(Term::new(sign, body))
    ///     }
    /// }
    /// ```
    fn format_term(&self, degree: i32, coef: T) -> Option<Term>;

    /// Formats the scaling formula for the polynomial.
    ///
    /// This represents domain scaling for the polynomial.
    fn format_scaling_formula(&self) -> Option<String> {
        None
    }

    /// Writes the full polynomial expression into the provided buffer.
    ///
    /// This method assembles a human-readable polynomial string, using
    /// [`format_term`] for each nonzero coefficient. The output is prefixed
    /// with `"y = "` and terms are separated by spaces, with proper signs
    /// inserted.
    ///
    /// # Coefficients
    /// - `coefficients[i]` corresponds to the coefficient for `x^(degree - i)`.
    /// - Zero coefficients are skipped automatically.
    ///
    /// # Behavior
    /// - The first nonzero term is written without a leading `+`.
    /// - Subsequent terms are prepended with `+` or `-` depending on the sign.
    ///
    /// # Parameters
    /// - `buffer`: A mutable [`std::fmt::Write`] buffer to write the string into.
    /// - `coefficients`: Slice of polynomial coefficients.
    ///
    /// # Errors
    /// Returns an error if writing to `buffer` fails.
    fn format_polynomial<B: std::fmt::Write>(
        &self,
        buffer: &mut B,
        coefficients: &[T],
    ) -> std::fmt::Result {
        let degree = coefficients.len() - 1;
        let mut terms = Vec::new();

        for (i, &coef) in coefficients.iter().rev().enumerate() {
            let degree_ = degree - i;
            if let Some(term) = self.format_term(degree_ as i32, coef) {
                terms.push(term);
            }
        }

        // Scaling first
        if let Some(scaling) = self.format_scaling_formula() {
            write!(buffer, "{scaling}, ")?;
        }

        write!(buffer, "y(x) = ")?;
        if terms.is_empty() {
            write!(buffer, "0")?;
            return Ok(());
        }

        // Extract the first term to avoid leading '+'
        let term_n = terms.remove(0);
        if term_n.sign == Sign::Negative {
            write!(buffer, "{}", term_n.sign.char())?;
        }
        write!(buffer, "{}", term_n.body)?;

        for term in terms {
            let sign = term.sign.char();
            let body = term.body;

            write!(buffer, " {sign} {body}")?;
        }

        Ok(())
    }
}

/// Represents the sign of a polynomial term.
///
/// Used when formatting polynomial expressions to determine how a term
/// should be connected to the rest of the polynomial (e.g., with `+` or `-`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// Positive sign (`+` when displayed).
    Positive,

    /// Negative sign (`-` when displayed).
    Negative,
}

impl Sign {
    /// Determines the sign from a numeric coefficient.
    ///
    /// # Example
    /// ```
    /// # use polyfit::display::Sign;
    /// assert_eq!(Sign::from_coef(3.0), Sign::Positive);
    /// assert_eq!(Sign::from_coef(-2.0), Sign::Negative);
    /// ```
    pub fn from_coef<T: Value>(coef: T) -> Self {
        if coef.is_sign_negative() {
            Self::Negative
        } else {
            Self::Positive
        }
    }

    /// Returns the character representation of the sign.
    ///
    /// `+` for `Positive`, `-` for `Negative`.
    ///
    /// # Example
    /// ```
    /// # use polyfit::display::Sign;
    /// assert_eq!(Sign::Positive.char(), '+');
    /// assert_eq!(Sign::Negative.char(), '-');
    /// ```
    #[must_use]
    pub fn char(&self) -> char {
        match self {
            Sign::Positive => '+',
            Sign::Negative => '-',
        }
    }
}

/// Represents a single term of a polynomial for display purposes.
///
/// A `Term` combines the **sign** and the **formatted body** of a polynomial
/// component (e.g., `"2x²"`, `"-3.14"`, `"x"`). Terms are typically produced
/// by [`PolynomialDisplay::format_term`] and assembled into a full polynomial
/// string by [`format_polynomial`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Term {
    /// The sign of the term (positive or negative).
    pub sign: Sign,

    /// The body of the term (e.g., `"2x²"`, `"3.14"`, `"x"`).
    ///
    /// Helper functions [`format_coefficient`] and [`exponentiate`] can be
    /// used to construct this string consistently.
    pub body: String,
}

impl Term {
    /// Creates a new polynomial term with the given sign and body.
    ///
    /// # Parameters
    /// - `sign`: The sign of the term (`Sign::Positive` or `Sign::Negative`).
    /// - `body`: The textual representation of the term.
    ///
    /// # Example
    /// ```
    /// # use polyfit::display::{Term, Sign};
    /// let term = Term::new(Sign::Negative, "3x²".to_string());
    /// assert_eq!(term.sign, Sign::Negative);
    /// assert_eq!(term.body, "3x²");
    /// ```
    #[must_use]
    pub fn new(sign: Sign, body: String) -> Self {
        Self { sign, body }
    }
}

/// Formats a numeric coefficient for display in a polynomial term.
///
/// - Returns `None` if the coefficient is zero or effectively zero (≤ epsilon).
/// - Formats as a decimal if the absolute value is between `1e-3` and `1e3`.
/// - Formats in scientific notation otherwise.
///
/// # Parameters
/// - `coef`: The numeric coefficient to format.
/// - `degree`: The degree of the polynomial term.
/// - `precision`: Number of digits after the decimal point.
///
/// # Returns
/// - `Some(String)` with the formatted coefficient, or `None` if the value is zero.
///
/// # Example
/// ```
/// # use polyfit::display::format_coefficient;
/// assert_eq!(format_coefficient(0.0, 1, 2), None);
/// assert_eq!(format_coefficient(2.5, 1, 2), Some("2.50".to_string()));
/// assert_eq!(format_coefficient(1e5, 1, 2), Some("1.00e5".to_string()));
/// ```
pub fn format_coefficient<T: Value + std::fmt::LowerExp>(
    coef: T,
    degree: i32,
    precision: usize,
) -> Option<String> {
    let abs = Value::abs(coef);

    if coef.is_zero() || abs <= T::epsilon() {
        return None;
    }

    if coef.abs_sub(T::one()) <= T::epsilon() && degree != 0 {
        return Some(String::new());
    }

    let sci_cutoff = default_fixed_range();
    Some(unicode::float(abs, sci_cutoff, precision))
}

/// Formats the variable part of a polynomial term for display purposes.
///
/// This is for formatting the variable name, subscript, and exponent.
///
/// # Behavior
/// - If `exp == 0`, returns an empty string (`""`).
/// - If `exp == 1`, returns the base string unchanged.
/// - Otherwise, appends the Unicode superscript version of `exp` to `base`.
///
/// # Examples
/// ```
/// # use polyfit::display::format_variable;
/// assert_eq!(format_variable("x".into(), None, 0), "");
/// assert_eq!(format_variable("x".into(), None, 1), "x");
/// assert_eq!(format_variable("x".into(), Some("1".into()), 2), "x₁²");
/// ```
#[must_use]
pub fn format_variable(base: &str, subscript: Option<&str>, exp: i32) -> String {
    match exp {
        0 => String::new(),
        1 => base.to_string(),
        _ => {
            let lbl = unicode::subscript(subscript.unwrap_or_default());
            let sup = unicode::superscript(&exp.to_string());
            format!("{base}{lbl}{sup}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBasis;

    impl PolynomialDisplay<f64> for DummyBasis {
        fn format_term(&self, degree: i32, coef: f64) -> Option<Term> {
            let sign = Sign::from_coef(coef);
            let coef_str = format_coefficient(coef, 1, DEFAULT_PRECISION)?;
            let body = if degree == 0 {
                coef_str
            } else {
                format!("{coef_str}{}", format_variable("x", None, degree))
            };
            Some(Term::new(sign, body))
        }
    }

    #[test]
    fn test_sign_from_coef() {
        assert_eq!(Sign::from_coef(1.0), Sign::Positive);
        assert_eq!(Sign::from_coef(-1.0), Sign::Negative);
        assert_eq!(Sign::from_coef(0.0), Sign::Positive);
    }

    #[test]
    fn test_sign_char() {
        assert_eq!(Sign::Positive.char(), '+');
        assert_eq!(Sign::Negative.char(), '-');
    }

    #[test]
    fn test_term_new() {
        let t = Term::new(Sign::Negative, "3x²".to_string());
        assert_eq!(t.sign, Sign::Negative);
        assert_eq!(t.body, "3x²");
    }

    #[test]
    fn test_format_coefficient_decimal() {
        assert_eq!(format_coefficient(2.5, 1, 2), Some("2.50".to_string()));
        assert_eq!(format_coefficient(-2.5, 1, 2), Some("2.50".to_string()));
    }

    #[test]
    fn test_format_coefficient_zero() {
        assert_eq!(format_coefficient(0.0, 1, 2), None);
        assert_eq!(format_coefficient(1e-20, 1, 2), None);
    }

    #[test]
    fn test_format_coefficient_scientific() {
        assert_eq!(format_coefficient(1e5, 1, 2), Some("1.00e5".to_string()));
        assert_eq!(format_coefficient(1e-5, 2, 2), Some("1.00e-5".to_string()));
    }

    #[test]
    fn test_format_variable() {
        assert_eq!(format_variable("x", None, 0), "");
        assert_eq!(format_variable("x", None, 1), "x");
        assert_eq!(format_variable("x", Some("1"), 2), "x₁²");
        assert_eq!(format_variable("T", None, 3), "T³");
        assert_eq!(format_variable("x", None, -2), "x⁻²");
    }

    #[test]
    fn test_format_polynomial_basic() {
        let basis = DummyBasis;
        let mut buf = String::new();
        basis
            .format_polynomial(&mut buf, &[2.0, -3.0, 0.0, 4.0])
            .unwrap();
        // 2x³ - 3x² + 4
        assert_eq!(buf, "y(x) = 4.00x³ - 3.00x + 2.00");
    }

    #[test]
    fn test_format_polynomial_all_zero() {
        let basis = DummyBasis;
        let mut buf = String::new();
        basis.format_polynomial(&mut buf, &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(buf, "y(x) = 0");
    }

    #[test]
    fn test_format_polynomial_leading_negative() {
        let basis = DummyBasis;
        let mut buf = String::new();
        basis.format_polynomial(&mut buf, &[-1.0, 2.0]).unwrap();
        assert_eq!(buf, "y(x) = 2.00x - 1.00");
    }

    #[test]
    fn test_format_polynomial_single_term() {
        let basis = DummyBasis;
        let mut buf = String::new();
        basis.format_polynomial(&mut buf, &[0.0, 0.0, 5.0]).unwrap();
        assert_eq!(buf, "y(x) = 5.00x²");
    }
}
