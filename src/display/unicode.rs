//! Unicode formatting utilities
use std::ops::Range;

use crate::value::Value;

/// Format a floating point number as a string
///
/// # Parameters
/// - `n`: The number to format
/// - `fixed_range`: An optional range specifying the values that will not be formatted in scientific notation
/// - `precision`: The number of decimal places to include
pub fn float<T: Value + std::fmt::LowerExp>(
    n: T,
    fixed_range: Option<Range<T>>,
    precision: usize,
) -> String {
    match fixed_range {
        Some(range) if range.contains(&n) => format!("{n:.precision$}"),
        _ => format!("{n:.precision$e}"),
    }
}

/// Convert a string into a superscript string, ignoring invalid characters
pub fn superscript(s: &str) -> String {
    s.chars().filter_map(to_superscript).collect()
}

/// Convert a string into a subscript string, ignoring invalid characters
pub fn subscript(s: &str) -> String {
    s.chars().filter_map(to_subscript).collect()
}

/// Greek symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum Greek {
    /// α
    LowerAlpha = 'α' as u64,

    /// Α
    UpperAlpha = 'Α' as u64,

    /// β
    LowerBeta = 'β' as u64,

    /// Β
    UpperBeta = 'Β' as u64,

    /// γ
    LowerGamma = 'γ' as u64,

    /// Γ
    UpperGamma = 'Γ' as u64,

    /// δ
    LowerDelta = 'δ' as u64,

    /// Δ
    UpperDelta = 'Δ' as u64,

    /// ε
    LowerEpsilon = 'ε' as u64,

    /// Ε
    UpperEpsilon = 'Ε' as u64,

    /// ζ
    LowerZeta = 'ζ' as u64,

    /// Ζ
    UpperZeta = 'Ζ' as u64,
}
impl std::fmt::Display for Greek {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = char::from_u32(*self as u32).ok_or(std::fmt::Error)?;
        write!(f, "{c}")
    }
}

fn to_superscript(c: char) -> Option<char> {
    match c {
        '0' => Some('⁰'),
        '1' => Some('¹'),
        '2' => Some('²'),
        '3' => Some('³'),
        '4' => Some('⁴'),
        '5' => Some('⁵'),
        '6' => Some('⁶'),
        '7' => Some('⁷'),
        '8' => Some('⁸'),
        '9' => Some('⁹'),
        '+' => Some('⁺'),
        '-' => Some('⁻'),
        '=' => Some('⁼'),
        '(' => Some('⁽'),
        ')' => Some('⁾'),
        'a' | 'A' => Some('ᵃ'),
        'b' | 'B' => Some('ᵇ'),
        'c' | 'C' => Some('ᶜ'),
        'd' | 'D' => Some('ᵈ'),
        'e' | 'E' => Some('ᵉ'),
        'f' | 'F' => Some('ᶠ'),
        'g' | 'G' => Some('ᵍ'),
        'h' | 'H' => Some('ʰ'),
        'i' | 'I' => Some('ⁱ'),
        'j' | 'J' => Some('ʲ'),
        'k' | 'K' => Some('ᵏ'),
        'l' | 'L' => Some('ˡ'),
        'm' | 'M' => Some('ᵐ'),
        'n' | 'N' => Some('ⁿ'),
        'o' | 'O' => Some('ᵒ'),
        'p' | 'P' => Some('ᵖ'),
        'r' | 'R' => Some('ʳ'),
        's' | 'S' => Some('ˢ'),
        't' | 'T' => Some('ᵗ'),
        'u' | 'U' => Some('ᵘ'),
        'v' | 'V' => Some('ᵛ'),
        'w' | 'W' => Some('ʷ'),
        'x' | 'X' => Some('ˣ'),
        'y' | 'Y' => Some('ʸ'),
        'z' | 'Z' => Some('ᶻ'),
        _ => None,
    }
}

fn to_subscript(c: char) -> Option<char> {
    match c {
        '0' => Some('₀'),
        '1' => Some('₁'),
        '2' => Some('₂'),
        '3' => Some('₃'),
        '4' => Some('₄'),
        '5' => Some('₅'),
        '6' => Some('₆'),
        '7' => Some('₇'),
        '8' => Some('₈'),
        '9' => Some('₉'),
        '+' => Some('₊'),
        '-' => Some('₋'),
        '=' => Some('₌'),
        '(' => Some('₍'),
        ')' => Some('₎'),
        'a' | 'A' => Some('ₐ'),
        'e' | 'E' => Some('ₑ'),
        'h' | 'H' => Some('ₕ'),
        'i' | 'I' => Some('ᵢ'),
        'j' | 'J' => Some('ⱼ'),
        'k' | 'K' => Some('ₖ'),
        'l' | 'L' => Some('ₗ'),
        'm' | 'M' => Some('ₘ'),
        'n' | 'N' => Some('ₙ'),
        'o' | 'O' => Some('ₒ'),
        'p' | 'P' => Some('ₚ'),
        'r' | 'R' => Some('ᵣ'),
        's' | 'S' => Some('ₛ'),
        't' | 'T' => Some('ₜ'),
        'u' | 'U' => Some('ᵤ'),
        'v' | 'V' => Some('ᵥ'),
        'x' | 'X' => Some('ₓ'),
        _ => None,
    }
}
