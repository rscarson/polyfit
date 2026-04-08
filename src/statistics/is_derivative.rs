use std::ops::RangeInclusive;

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    value::{FloatClampedCast, SteppedValues, Value},
    Polynomial,
};

/// Error information when a derivative check fails. See [`is_derivative`]
pub struct DerivationError<T: Value> {
    /// The x value where the derivative check failed.
    pub x: T,

    /// The finite difference approximation of the derivative at `x`
    /// - `(f(x + h) - f(x - h)) / (2h)`
    pub finite_diff: T,

    /// The value of the claimed derivative polynomial at `x`
    /// - `f'(x)`
    pub derivative: T,

    /// The absolute difference between the finite difference and the derivative
    /// - `|finite_diff - derivative|`
    pub diff: T,

    /// The relative tolerance used for the comparison
    /// - `sqrt(ε) * max(|derivative|, 1)`
    pub rel_tol: T,
}
impl<T: Value> std::fmt::Display for DerivationError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Derivative check failed at x = {}: finite difference = {}, derivative = {}, |diff| = {}, rel_tol = {}",
            self.x, self.finite_diff, self.derivative, self.diff, self.rel_tol
        )
    }
}

/// Checks if `f_prime` is the derivative of polynomial `f`.
///
/// Uses a numerical approach to verify the derivative relationship.
/// - Evaluates both polynomials at several points and compares the results.
/// - Uses central difference to approximate the derivative of `f`.
///
/// # Type Parameters
/// - `T`: A numeric type implementing the `Value` trait.
/// - `B`: Basis type for the original polynomial.
/// - `B2`: Basis type for the derivative polynomial.
///
/// # Parameters
/// - `f`: The original polynomial.
/// - `f_prime`: The polynomial claimed to be the derivative of `f`.
/// - `normalizer`: The domain normalizer used for scaling.
/// - `domain`: The range over which to check the derivative relationship.
///
/// # Returns
/// - `Ok(())` if `f_prime` is verified as the derivative of `f` within tolerance.
/// - `Err(DerivationError)` if the check fails, containing details of the failure
///
/// # Errors
/// Returns `Err` if the derivative check fails at any point in the specified domain.
pub fn is_derivative<T: Value, B, B2>(
    f: &Polynomial<B, T>,
    f_prime: &Polynomial<B2, T>,
    domain: &RangeInclusive<T>,
) -> Result<(), DerivationError<T>>
where
    B: Basis<T> + PolynomialDisplay<T>,
    B2: Basis<T> + PolynomialDisplay<T>,
{
    let range = *domain.end() - *domain.start();
    let one_hundred = 100.0.clamped_cast::<T>();
    let steps = Value::clamp(
        one_hundred * (range),
        one_hundred,
        10_000.0.clamped_cast::<T>(),
    );

    let step = (range) / steps;

    let tol = T::epsilon().sqrt();
    for x in SteppedValues::new(domain.clone(), step) {
        let h = T::epsilon().sqrt() * Value::max(T::one(), Value::abs(x));

        let xhp = x + h;
        let xhm = x - h;
        if xhp > *domain.end() || xhm < *domain.start() {
            continue;
        }

        let finite_diff = (f.y(xhp) - f.y(xhm)) / (T::two() * h);
        let derivative = f_prime.y(x);
        // let derivative = derivative * scale;

        let rel_tol = tol.sqrt() * Value::max(Value::abs(derivative), T::one());
        let diff = Value::abs(finite_diff - derivative);
        if diff > rel_tol {
            return Err(DerivationError {
                x,
                finite_diff,
                derivative,
                diff,
                rel_tol,
            });
        }
    }

    // Sanity check - over any tiny interval, the y position of the derivative should be the difference in y position of the original function
    let dx = 1e-6.clamped_cast::<T>();
    let x1 = *domain.start();
    let x2 = *domain.start() + dx;
    if x2 <= *domain.end() {
        let delta_y = f.y(x2) - f.y(x1);
        let derivative_y = f_prime.y(x1);
        let rel_tol = tol.sqrt() * Value::max(Value::abs(derivative_y), T::one());
        let diff = Value::abs(delta_y / dx - derivative_y);
        if diff > rel_tol {
            return Err(DerivationError {
                x: x1,
                finite_diff: delta_y / dx,
                derivative: derivative_y,
                diff,
                rel_tol,
            });
        }
    }

    Ok(())
}
