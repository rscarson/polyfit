/// Asserts that the fitted curve is a good representation of the canonical curve.
/// Compares the fit's r² value against a threshold to ensure it closely follows the expected curve.
///
/// Useful if you know the underlying function but want to validate the fitting process.
///
/// If the test fails, a plot showing both curves will be generated in `<target/test_output>`
///
/// The plot will also include the original source data, as well as 99% confidence error bars for the fit.
/// See [`crate::CurveFit::r_squared`] for more details.
///
/// Threshold for r² match can be specified, or defaults to 0.9.
///
/// If you add noise to the data you generated, the noise strength should correspond to the r² threshold
///
/// # Example plot
/// ![Failure Plot](https://github.com/rscarson/polyfit/blob/main/.github/assets/example_fail.png)
///
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::{DegreeBound, ScoringMethod, Tolerance}, function, transforms::ApplyNoise, assert_fits};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..1000.0, 1.0).apply_normal_noise(Tolerance::Relative(0.1), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).expect("Failed to create model");
/// assert_fits!(&test, &fit, 0.9);
/// ```
#[macro_export]
macro_rules! assert_fits {
    ($canonical:expr, $fit:expr, $r2:literal) => {{
        let fit = &$fit;
        let poly = &$canonical;
        let threshold = $r2;

        let r2 = fit.r_squared_against(poly);

        if r2 < threshold {
            // Create a failure plot
            #[cfg(feature = "plotting")]
            $crate::plot!(
                fit,
                functions = [poly],
                title = &format!("Polynomial Fit (R² = {r2:.4})")
            );

            // And finally, assert to end the test
            panic!("R² = {r2} is below {threshold}");
        }
    }};

    ($canonical:expr, $fit:expr) => {
        $crate::assert_fits!($canonical, $fit, 0.9)
    };
}

/// Macro for asserting that a fitted curve meets a minimum R² threshold in tests.
/// This is a measure of how well the curve explains how wiggly the data is.
///
/// General case of [`crate::assert_fits`] that does not require a known function.
///
/// See [`crate::CurveFit::r_squared`] for more details.
///
/// # Forms
/// - `assert_r_squared!(fit, 0.95)`
///
///   Asserts that the `fit` has R² ≥ 0.95. Generates a test label based on the
///   file and line number. Produces a failure plot if the assertion fails.
///
/// - `assert_r_squared!(canonical, fit)`
///
///   Compares `fit` against a canonical dataset with a default R² threshold of 0.9.
///
/// # Notes
/// - Automatically handles test labeling and failure plotting.
/// - Panics if the R² is below the threshold, ending the test.
///
/// # Example
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::{DegreeBound, ScoringMethod, Tolerance}, function, transforms::ApplyNoise, assert_r_squared};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..1000.0, 1.0).apply_normal_noise(Tolerance::Relative(0.1), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).expect("Failed to create model");
/// assert_r_squared!(fit, 0.95);
/// ```
#[macro_export]
macro_rules! assert_r_squared {
    ($fit:expr, $r2:literal) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            let ref fit = $fit;
            let threshold = $r2;
            let r2 = fit.r_squared(fit.data());

            if r2 <= threshold {
                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(fit, title = &format!("Polynomial Fit (R² = {r2:.4})"));

                // And finally, assert to end the test
                panic!("R² = {r2} is below {threshold}");
            }
        }
    };

    ($fit:expr) => {
        $crate::assert_r_squared!($fit, 0.9)
    };
}

/// Asserts that the residuals (the differences between the observed and predicted values) of a fit are normally distributed.
///
/// Think of it like making sure the errors are random, not based on some undiscovered pattern.
///
/// See [`crate::statistics::residual_normality`] for more details.
/// - Results will be between 0.0 and 1.0, with values closer to 1.0 indicating a better fit.
///
/// # Parameters
/// - `$fit`: A reference to the `CurveFit` object whose residuals will be tested.
/// - `$tolerance`: Minimum p-value for normality. Defaults to `0.05` if omitted.
///
/// # Behavior
/// - Computes mean, standard deviation, skewness, and excess kurtosis of residuals.
/// - If p-value < `$tolerance`, the macro will:
///   1. Optionally generate a failure plot (if the `plotting` feature is enabled).
///   2. Panic with a clear error message indicating skew/kurtosis values.
///
/// # Example
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::{DegreeBound, ScoringMethod, Tolerance}, function, transforms::ApplyNoise, assert_residuals_normal};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..1000.0, 1.0).apply_normal_noise(Tolerance::Relative(0.1), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).expect("Failed to create model");
///
/// // Uses default tolerance 0.05
/// assert_residuals_normal!(fit);
///
/// // Uses custom tolerance
/// assert_residuals_normal!(fit, 0.01);
/// ```
#[macro_export]
macro_rules! assert_residuals_normal {
    ($fit:expr, $tolerance:literal) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            use $crate::value::CoordExt;

            let ref fit = $fit;
            let tolerance = $tolerance;
            let residuals = fit.residuals();
            let residuals: Vec<_> = residuals.y();
            let p_value = $crate::statistics::residual_normality(&residuals);

            if p_value < tolerance {
                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(fit, title = &format!("Residuals not normally distributed"));

                let (skewness, kurtosis) = $crate::statistics::skewness_and_kurtosis(&residuals);
                panic!(
                    "Residuals not normal - p={p_value:.2} - skew={skewness:.4}, kurt={kurtosis:.4}, tol={tolerance}"
                );
            }
        }
    };

    ($fit:expr) => {
        $crate::assert_residuals_normal!($fit, 0.05)
    };
}

/// Asserts that the spread of the residuals (the differences between the observed and predicted values) of a fit is below a certain threshold.
///
/// This ensures that no residual is too large, i.e., the fit is sufficiently close to the data.
/// Unlike `assert_residuals_normal`, this checks **magnitude**, not distribution shape.
///
/// # Parameters
/// - `fit`: The curve fit to test.
/// - `max`: Maximum allowed spread of residuals.
///
/// # Panics
/// Panics if the spread exceeds `max`. If the `plotting` feature is enabled,
/// a failure plot is generated.
///
/// # Example
/// ```rust
/// # use polyfit::{MonomialFit, assert_residual_spread};
/// let fit = MonomialFit::new(&[(0.0, 0.0), (1.0, 1.0)], 1).unwrap();
/// assert_residual_spread!(fit, 0.01);
/// ```
#[macro_export]
macro_rules! assert_residual_spread {
    ($fit:expr, $max:expr) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            use $crate::value::CoordExt;

            let ref fit = $fit;
            let tolerance = $max;

            let residuals = fit.residuals();
            let spread = $crate::statistics::spread(residuals.y_iter());
            if spread > tolerance {
                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(fit, title = &format!("Abnormal Residuals"));

                panic!("Residual spread too large - spread={spread:.4}, tol={tolerance}");
            }
        }
    };
}

/// Asserts that the residuals (the differences between the observed and predicted values) of a fit are normally distributed.
/// Think of it like making sure the errors are random, not based on some undiscovered pattern.
///
/// See [`crate::statistics::residual_normality`] for more details.
/// - Results will be between 0.0 and 1.0, with values closer to 1.0 indicating a better fit.
///
/// # Parameters
/// - `$fit`: `CurveFit` object to test.
///
/// # Panics
/// Panics if derivative changes sign anywhere in the x-range.
///
/// # Example
/// ```rust
/// # use polyfit::{MonomialFit, assert_monotone};
/// let fit = MonomialFit::new(&[(0.0, 0.0), (1.0, 1.0)], 1).unwrap();
/// assert_monotone!(fit);
/// ```
#[macro_export]
macro_rules! assert_monotone {
    ($fit:expr) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            let ref fit = $fit;
            let violations = fit
                .monotonicity_violations()
                .expect("Failed to check monotonicity");
            if let Some(first) = violations.first() {
                #[cfg(feature = "plotting")]
                $crate::plot!(fit, title = &format!("Monotonicity Violation at x={first}"));

                panic!("Fit is not monotonic - derivative changes sign at x={first}");
            }
        }
    };
}

/// Asserts that evaluating a polynomial at a given `x` matches the expected `y` value
/// within floating-point epsilon tolerance.
///
/// Useful for spot-checking specific predictions of the model.
///
/// # Arguments
///
/// * `$function` - The polynomial under test (must implement `AsRef<Polynomial<...>>`).
/// * `$x` - The input `x` value where the polynomial is evaluated.
/// * `$expected` - The expected result of the polynomial evaluation.
///
/// # Panics
///
/// Panics if the evaluated value differs from the expected value
/// by more than machine epsilon for the type `T`.
///
/// # Example
/// ```
/// use polyfit::{function, assert_y};
///
/// function!(test(x) = 8.0 + 7.0 x^1 + 6.0 x^2);
///
/// // 8 + 7*2 + 6*4 = 8 + 14 + 24 = 46
/// assert_y!(test, 2.0, 46.0);
/// ```
#[macro_export]
macro_rules! assert_y {
    ($function:expr, $x:expr, $expected:expr) => {{
        let function = &$function;
        let function: &$crate::Polynomial<_, _> = function.as_ref();
        let x = $x;
        let expected = $expected;

        $crate::assert_close!(function.y(x), expected, "y({x}) != {expected}");
    }};
}

/// Asserts that two floating-point values are approximately equal within a small tolerance (epsilon).
///
/// This is useful for comparing computed values where exact equality is not expected due to rounding errors.
/// - Uses the machine epsilon for the floating-point type as the tolerance.
/// - `assert_eq!` equivalent for floats.
///
/// # Parameters
/// - `$a`: First value.
/// - `$b`: Second value.
/// - `$msg`: Custom failure message.
///
/// # Panics
/// Panics if the absolute difference `|a - b|` exceeds `::epsilon()` for the type `T`.
///
/// # Examples
/// ```
/// # use polyfit::assert_close;
/// assert_close!(1.0 + 1e-16, 1.0, "Nearly equal");
/// ```
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr $(, $msg:expr $(, $($args:tt),*)?)?) => {{
        fn assert_close<T: $crate::value::Value>(a: T, b: T, msg: &str) {
            assert!(
                a == b || $crate::value::Value::abs(a - b) <= T::epsilon(),
                "{msg}: {a} != {b}"
            );
        }

        #[allow(unused_mut, unused_assignments)] let mut msg = "Values not close".to_string();
        $( msg = format!($msg, $($($args)?)?); )?

        assert_close($a, $b, &msg);
    }};
}

/// Asserts that two slices of floating-point values are approximately equal element-wise within a small tolerance (epsilon).
///
/// This is useful for comparing arrays of computed values where exact equality is not expected due to rounding errors.
/// - Uses the machine epsilon for the floating-point type as the tolerance.
/// - Element-wise [`crate::assert_close`].
///
/// # Parameters
/// - `$src`: Source slice (implements `iter()`).
/// - `$dst`: Destination slice (same length as `$src`).
/// - `$msg`: *(optional)* Custom failure message. Defaults to `"{len} elements"`.
///   Supports formatting arguments just like `format!`.
///
/// # Panics
/// - If the lengths differ.
/// - If any pair of elements differ by more than `T::epsilon()`.
///
/// # Examples
/// ```
/// # use polyfit::assert_all_close;
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0 + 1e-16, 2.0, 3.0];
///
/// assert_all_close!(a, b); // OK
/// assert_all_close!(a, b, "Vectors must match"); // Custom message
/// ```
#[macro_export]
macro_rules! assert_all_close {
    ($src:expr, $dst:expr  $(, $msg:expr $(, $($args:tt),*)?)?) => {
        #[allow(unused_assignments, unused_mut)]
        let mut msg = format!("{} elements", $src.len());
        $(
            msg = format!($msg, $($($args)?)?);
        )?

        assert_eq!($src.len(), $dst.len(), "{msg} - length mismatch");

        for (i, (s, d)) in $src.iter().zip($dst.iter()).enumerate() {
            $crate::assert_close!(*s, *d, "{msg} - src[{i}]");
        }
    };
}
#[cfg(test)]
mod tests {
    use crate::{
        function,
        statistics::{DegreeBound, ScoringMethod, Tolerance},
        transforms::ApplyNoise,
        MonomialFit,
    };

    #[test]
    fn test_assert_y_macro() {
        // 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        assert_y!(poly, 2.0, 17.0);
    }

    #[test]
    fn test_assert_close_macro() {
        assert_close!(1.0 + 1e-16, 1.0, "Values should be close");
    }

    #[test]
    fn test_assert_all_close_macro() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0 + 1e-16, 2.0, 3.0];
        assert_all_close!(a, b, "Vectors must match");
    }

    #[test]
    fn test_assert_fits_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..1000.0, 1.0)
            .apply_normal_noise(Tolerance::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert_fits!(&poly, &fit, 0.99);
    }

    #[test]
    fn test_assert_r_squared_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..1000.0, 1.0)
            .apply_normal_noise(Tolerance::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert_r_squared!(&fit, 0.98);
    }

    #[test]
    fn test_assert_residuals_normal_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..1000.0, 1.0)
            .apply_normal_noise(Tolerance::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert_residuals_normal!(&fit);
    }

    #[test]
    fn test_assert_residual_spread_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..1000.0, 1.0)
            .apply_normal_noise(Tolerance::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert_residual_spread!(&fit, 80000.0);
    }

    #[test]
    fn test_assert_monotone_macro() {
        function!(mono(x) = 1.0 + 2.0 x^1); // strictly increasing
        let data = mono.solve_range(0.0..1000.0, 1.0);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
        assert_monotone!(&fit);
    }
}
