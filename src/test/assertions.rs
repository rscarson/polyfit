/// Asserts that the fitted curve is a good representation of a canonical curve.
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
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::DegreeBound, score::Aic, function, transforms::{Strength, ApplyNoise}, assert_fits};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..=1000.0, 1.0).apply_normal_noise(Strength::Relative(0.1), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic).expect("Failed to create model");
/// assert_fits!(&test, &fit, 0.9);
/// ```
#[macro_export]
macro_rules! assert_fits {
    ($canonical:expr, $fit:expr, $r2:expr $(, $msg:literal $(, $($args:tt),*)?)?) => {{
        let fit = &$fit;
        let poly = &$canonical;
        let threshold = $r2;

        let r2 = fit.r_squared_against(poly);

        if r2 < threshold || !$crate::value::Value::is_finite(r2) {
            #[allow(unused)] use std::fmt::Write;
            let mut msg = format!("Fit does not meet R² threshold: {r2} < {threshold}");

            // Print any seeds used in the test thread so far
            #[cfg(feature = "transforms")]
            {
                let seeds = $crate::transforms::SeedSource::all_seeds();
                if !seeds.is_empty() {
                    write!(msg, "\nSeeds used in this test thread: {:?}", seeds).ok();
                }
            }

            // Create a failure plot
            #[cfg(feature = "plotting")]
            {
                let filename = $crate::plot!(
                    [fit, poly],
                    {
                        title: format!("Polynomial Fit (R² = {r2:.4})"),
                        silent: true
                    },
                    prefix = "assert_fits"
                );
                write!(msg, "\nFailure plot saved to: {}", filename.display()).ok();
            }

            $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

            // And finally, panic to end the test
            panic!("{msg}");
        }
    }};

    ($canonical:expr, $fit:expr) => {
        $crate::assert_fits!(
            $canonical,
            $fit,
            $crate::value::Value::try_cast(0.9)
                .expect("Failed to cast 0.9 for assert_fits! threshold")
        )
    };
}

/// Macro for asserting that a fitted curve meets a minimum R² threshold in tests.
/// This is a measure of how well the curve explains how wiggly the data is.
///
/// General case of [`crate::assert_fits`] that does not require a known function.
///
/// See [`crate::CurveFit::r_squared`] for more details.
///
/// Will generate a failure plot in `<target/test_output>` if the assertion fails.
///
/// # Syntax
///
/// `assert_r_squared!(<CurveFit>, <threshold> [, msg = <custom message>])`
///
/// - `CurveFit`: The fitted curve to test.
/// - `threshold`: Minimum acceptable R² value (between 0.0 and 1.0). Defaults to `0.9` if omitted.
/// - `msg`: *(optional)* Custom message to include on failure, supports formatting arguments.
///
/// # Notes
/// - Automatically handles test labeling and failure plotting.
/// - Panics if the R² is below the threshold, ending the test.
///
/// # Example
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::DegreeBound, score::Aic, function, transforms::{ApplyNoise, Strength}, assert_r_squared};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..=1000.0, 1.0).apply_normal_noise(Strength::Relative(0.1), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic).expect("Failed to create model");
/// assert_r_squared!(fit, 0.95);
/// ```
#[macro_export]
macro_rules! assert_r_squared {
    ($fit:expr $(, msg = $msg:literal $(, $($args:tt),*)?)?) => {
        $crate::assert_r_squared!(
            $fit,
            $crate::value::Value::try_cast(0.9)
                .expect("Failed to cast 0.9 for assert_r_squared! threshold")
            $(, msg = $msg $(, $($args),*)?)?
        )
    };

    ($fit:expr, $r2:expr $(, msg = $msg:literal $(, $($args:tt),*)?)?) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            let ref fit = $fit;
            let threshold = $r2;
            #[allow(unused_mut)] let mut r2 = fit.r_squared(None);

            if r2 <= threshold {
                // Print any seeds used in the test thread so far
                #[cfg(feature = "transforms")]
                {
                    let seeds = $crate::transforms::SeedSource::all_seeds();
                    if !seeds.is_empty() {
                        eprintln!("Seeds used in this test thread: {:?}", seeds);
                    }
                }

                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(
                    fit,
                    { title: format!("Polynomial Fit (R² = {r2:.4})") },
                    prefix = "assert_r_squared"
                );

                #[allow(unused_mut, unused_assignments)] let mut msg = format!("R² = {r2} is below {threshold}");
                $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

                // And finally, assert to end the test
                panic!("{msg}");
            }
        }
    };
}

/// Macro for asserting that a fitted curve meets a minimum R² threshold in tests.
/// This is a measure of how well the curve explains how wiggly the data is.
///
/// See [`crate::CurveFit::adjusted_r_squared`] for more details.
///
/// Will generate a failure plot in `<target/test_output>` if the assertion fails.
///
/// # Syntax
///
/// `assert_r_squared!(<CurveFit>, <threshold> [, msg = <custom message>])`
///
/// - `CurveFit`: The fitted curve to test.
/// - `threshold`: Minimum acceptable R² value (between 0.0 and 1.0). Defaults to `0.9` if omitted.
/// - `msg`: *(optional)* Custom message to include on failure, supports formatting arguments.
///
/// # Notes
/// - Automatically handles test labeling and failure plotting.
/// - Panics if the R² is below the threshold, ending the test.
///
/// # Example
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::DegreeBound, score::Aic, function, transforms::{ApplyNoise, Strength}, assert_adjusted_r_squared};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..=1000.0, 1.0).apply_normal_noise(Strength::Relative(0.1), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic).expect("Failed to create model");
/// assert_adjusted_r_squared!(fit, 0.95);
/// ```
#[macro_export]
macro_rules! assert_adjusted_r_squared {
    ($fit:expr $(, msg = $msg:literal $(, $($args:tt),*)?)?) => {
        $crate::assert_adjusted_r_squared!(
            $fit,
            $crate::value::Value::try_cast(0.9)
                .expect("Failed to cast 0.9 for assert_adjusted_r_squared! threshold")
            $(, msg = $msg $(, $($args),*)?)?
        )
    };

    ($fit:expr, $r2:expr $(, msg = $msg:literal $(, $($args:tt),*)?)?) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            let ref fit = $fit;
            let threshold = $r2;
            #[allow(unused_mut)] let mut r2 = fit.adjusted_r_squared(None);

            if r2 <= threshold {
                // Print any seeds used in the test thread so far
                #[cfg(feature = "transforms")]
                {
                    let seeds = $crate::transforms::SeedSource::all_seeds();
                    if !seeds.is_empty() {
                        eprintln!("Seeds used in this test thread: {:?}", seeds);
                    }
                }

                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(
                    fit,
                    { title: format!("Polynomial Fit (R² = {r2:.4})") },
                    prefix = "assert_adjusted_r_squared"
                );

                #[allow(unused_mut, unused_assignments)] let mut msg = format!("R² = {r2} is below {threshold}");
                $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

                // And finally, assert to end the test
                panic!("{msg}");
            }
        }
    };
}

/// Macro for asserting that a fitted curve meets a minimum R² threshold in tests.
/// This is a measure of how well the curve explains how wiggly the data is.
///
/// See [`crate::CurveFit::robust_r_squared`] for more details.
///
/// Will generate a failure plot in `<target/test_output>` if the assertion fails.
///
/// # Syntax
///
/// `assert_r_squared!(<CurveFit>, <threshold> [, msg = <custom message>])`
///
/// - `CurveFit`: The fitted curve to test.
/// - `threshold`: Minimum acceptable R² value (between 0.0 and 1.0). Defaults to `0.9` if omitted.
/// - `msg`: *(optional)* Custom message to include on failure, supports formatting arguments.
///
/// # Notes
/// - Automatically handles test labeling and failure plotting.
/// - Panics if the R² is below the threshold, ending the test.
///
/// # Example
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::DegreeBound, score::Aic, function, transforms::{ApplyNoise, Strength}, assert_robust_r_squared};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3);
/// let data = test.solve_range(0.0..=1000.0, 1.0).apply_normal_noise(Strength::Relative(1.5), None);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic).expect("Failed to create model");
/// assert_robust_r_squared!(fit, 0.6);
/// ```
#[macro_export]
macro_rules! assert_robust_r_squared {
    ($fit:expr $(, msg = $msg:literal $(, $($args:tt),*)?)?) => {
        $crate::assert_robust_r_squared!(
            $fit,
            $crate::value::Value::try_cast(0.9)
                .expect("Failed to cast 0.9 for assert_robust_r_squared! threshold")
            $(, msg = $msg $(, $($args),*)?)?
        )
    };

    ($fit:expr, $r2:expr $(, msg = $msg:literal $(, $($args:tt),*)?)?) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            let ref fit = $fit;
            let threshold = $r2;
            #[allow(unused_mut)] let mut r2 = fit.robust_r_squared(None);

            if r2 <= threshold {
                // Print any seeds used in the test thread so far
                #[cfg(feature = "transforms")]
                {
                    let seeds = $crate::transforms::SeedSource::all_seeds();
                    if !seeds.is_empty() {
                        eprintln!("Seeds used in this test thread: {:?}", seeds);
                    }
                }

                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(
                    fit,
                    { title: format!("Polynomial Fit (R² = {r2:.4})") },
                    prefix = "assert_robust_r_squared"
                );

                #[allow(unused_mut, unused_assignments)] let mut msg = format!("R² = {r2} is below {threshold}");
                $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

                // And finally, assert to end the test
                panic!("{msg}");
            }
        }
    };
}

/// Asserts that the residuals (the differences between the observed and predicted values) of a fit are normally distributed.
///
/// This means the errors are likely random, not based on some undiscovered pattern.
///
/// See [`crate::statistics::residual_normality`] for more details.
/// - Results will be between 0.0 and 1.0, with values closer to 1.0 indicating a better fit.
///
/// # Parameters
/// - `$fit`: A reference to the `CurveFit` object whose residuals will be tested.
/// - `$tolerance`: Minimum p-value for normality. Defaults to `0.1` if omitted.
/// - `strict`: *(optional)* If true, uses unfiltered residuals. Defaults to false.
///   Use strict mode if you want to include all residuals, even those close to zero.
///   Normally, small residuals are due to floating-point noise and are filtered out.
///
/// # Behavior
/// - Computes mean, standard deviation, skewness, and excess kurtosis of residuals.
/// - If p-value < `$tolerance`, the macro will:
///   1. Optionally generate a failure plot (if the `plotting` feature is enabled).
///   2. Panic with a clear error message indicating skew/kurtosis values.
///
/// # Example
/// ```rust
/// # use polyfit::{ChebyshevFit, MonomialPolynomial, statistics::{DegreeBound, Tolerance}, score::Aic, function, transforms::ApplyNoise, assert_residuals_normal};
/// function!(test(x) = 20.0 + 3.0 x^1 + 2.0 x^2 + 4.0 x^3 );
/// let data = test.solve_range(0.0..=1000.0, 1.0);
///
/// let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic).expect("Failed to create model");
///
/// // Uses default tolerance 0.1
/// assert_residuals_normal!(fit);
///
/// // Strict mode uses unfiltered residuals - even floating point noise can cause failure
/// assert_residuals_normal!(fit, 0.00, strict = true);
/// ```
#[macro_export]
macro_rules! assert_residuals_normal {
    ($fit:expr $(, strict = $strict:expr)?) => {
        $crate::assert_residuals_normal!(
            $fit,
            $crate::value::Value::try_cast(0.1).expect("Failed to cast 0.1 for threshold in assert_residuals_normal!")
            $(, strict = $strict)?)
    };

    ($fit:expr, $tolerance:expr $(, strict = $strict:expr)? $(, $msg:literal $(, $($args:tt),*)?)?) => {
        {
            use $crate::value::CoordExt;

            #[allow(clippy::toplevel_ref_arg)]
            let ref fit = $fit;
            let tolerance = $tolerance;

            let strict = false $( || $strict )?;

            let residuals = if strict {
                fit.residuals()
            } else {
                fit.filtered_residuals()
            };

            let residuals_y: Vec<_> = residuals.y();
            let p_value = $crate::statistics::residual_normality(&residuals_y);

            if p_value < tolerance {
                // Print any seeds used in the test thread so far
                #[cfg(feature = "transforms")]
                {
                    let seeds = $crate::transforms::SeedSource::all_seeds();
                    if !seeds.is_empty() {
                        eprintln!("Seeds used in this test thread: {:?}", seeds);
                    }
                }

                // Create a failure plot
                #[cfg(feature = "plotting")]
                {
                    fn get_cutoff<T: $crate::value::Value>(residuals: &[(T, T)], p: T) -> Option<T> {
                        let mut sorted_residuals = residuals.to_vec();
                        sorted_residuals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                        let p = T::one() - ($crate::value::Value::clamp(p, T::zero(), T::one()));
                        let n = residuals.len();
                        let index = $crate::value::Value::ceil(T::from_usize(n)? * p).to_usize()? - 1;
                        sorted_residuals.get(index).map(|(_, y)| y.abs())
                    }

                    // residual trendline
                    let fit = $crate::ChebyshevFit::new_auto(
                        &residuals,
                        $crate::statistics::DegreeBound::Relaxed,
                        &$crate::score::Aic,
                    ).expect("Failed to create residual trendline fit");
                    let fit = fit.as_monomial().expect("Failed to convert residual trendline to monomial");

                    let title = format!("Residuals not normally distributed (p={:.2})", p_value);
                    let caption = format!("Residuals ({})", fit.equation());

                    // 1-p percentile
                    let cutoff = get_cutoff(&residuals, tolerance);
                    if let Some(cutoff) = cutoff {
                        let cutoff_lineu = residuals.iter().map(|(x, _)| (*x, cutoff)).collect::<Vec<_>>();
                        let cutoff_textu = format!("Upper Cutoff (p={:.2})", p_value);

                        let cutoff_linel = residuals.iter().map(|(x, _)| (*x, -cutoff)).collect::<Vec<_>>();
                        let cutoff_textl = format!("Lower Cutoff (p={:.2})", p_value);

                        $crate::plot!([
                            (&residuals, caption.as_str()),
                            fit,
                            (&cutoff_lineu, cutoff_textu.as_str()),
                            (&cutoff_linel, cutoff_textl.as_str())
                        ], { title: title }, prefix = "assert_residuals_normal");
                    } else {
                        $crate::plot!([(&residuals, caption.as_str()), fit], { title: title }, prefix = "assert_residuals_normal");

                    }
                }

                let (skewness, kurtosis) = $crate::statistics::skewness_and_kurtosis(&residuals_y);

                #[allow(unused_mut, unused_assignments)] let mut msg = format!(
                    "Residuals not normal - p={p_value:.2} - skew={skewness:.4}, kurt={kurtosis:.4}, tol={tolerance}"
                );
                $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

                panic!("{msg}");
            }
        }
    };
}

/// Asserts that at least a certain proportion of residuals (the differences between the observed and predicted values) of a fit are below a certain threshold.
///
/// This ensures that residuals are not too large, i.e., the fit is sufficiently close to the data.
/// Unlike `assert_residuals_normal`, this checks **magnitude**, not distribution shape.
///
/// # Parameters
/// - `fit`: The curve fit to test.
/// - `max`: Maximum allowed residual magnitude.
/// - `tolerance`: Proportion of residuals that must be below `max`. Defaults to `0.95` if omitted.
///
/// # Panics
/// Panics if the proportion of residuals below `max` is less than `tolerance`. If the `plotting` feature is enabled,
/// a failure plot is generated.
///
/// # Example
/// ```rust
/// # use polyfit::{MonomialFit, assert_max_residual};
/// let fit = MonomialFit::new(&[(0.0, 0.0), (1.0, 1.0)], 1).unwrap();
/// assert_max_residual!(fit, 0.01);
/// ```
#[macro_export]
macro_rules! assert_max_residual {
    ($fit:expr, $max:expr $(, $msg:literal $(, $($args:tt),*)?)?) => {
        $crate::assert_max_residual!($fit, $max,
            $crate::value::Value::try_cast(0.95)
                .expect("Failed to cast 0.95 for assert_max_residual! tolerance")
            $(, $msg $(, $($args),*)? )?
        )
    };

    ($fit:expr, $max:expr, $tolerance:expr $(, $msg:literal $(, $($args:tt),*)?)?) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            fn get_cutoff<T: $crate::value::Value>(residuals: &[(T, T)], p: T) -> Option<T> {
                let mut sorted_residuals = residuals.to_vec();
                sorted_residuals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let p = $crate::value::Value::clamp(p, T::zero(), T::one() - T::epsilon());
                let n = residuals.len();
                let index = $crate::value::Value::ceil(T::from_usize(n)? * p).to_usize()? - 1;
                sorted_residuals.get(index).map(|(_, y)| y).cloned()
            }

            let ref fit = $fit;
            let max = $max;
            let tolerance = $tolerance;

            let residuals = fit.residuals().iter().map(|(x, y)| (*x, $crate::value::Value::abs(*y))).collect::<Vec<_>>();
            let cutoff = get_cutoff(&residuals, tolerance).unwrap_or_else(|| {
                panic!("Failed to compute residual cutoff for assert_max_residual!");
            });

            if cutoff > max {
                // Print any seeds used in the test thread so far
                #[cfg(feature = "transforms")]
                {
                    let seeds = $crate::transforms::SeedSource::all_seeds();
                    if !seeds.is_empty() {
                        eprintln!("Seeds used in this test thread: {:?}", seeds);
                    }
                }

                // Create a failure plot
                #[cfg(feature = "plotting")]
                $crate::plot!(
                    fit,
                    { title: format!("Abnormal Residuals") },
                    prefix = "assert_max_residual"
                );

                #[allow(unused_mut, unused_assignments)] let mut msg = format!(
                    "Residuals above threshold - max={cutoff:.4}/{max:.4}, tol={tolerance}"
                );
                $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

                panic!("{msg}");
            }
        }
    };
}

/// Asserts that the derivative of a fitted curve does not change sign over its x-range, indicating monotonicity.
/// This means the function always increases or always decreases.
///
/// # Parameters
/// - `$fit`: `CurveFit` or `Polynomal` object to test.
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
    ($fit:expr $(, $msg:literal $(, $($args:tt),*)?)?) => {
        #[allow(clippy::toplevel_ref_arg)]
        {
            let ref fit = $fit;
            let violations = fit
                .monotonicity_violations()
                .expect("Failed to check monotonicity");
            if let Some(first) = violations.first() {
                // Print any seeds used in the test thread so far
                #[cfg(feature = "transforms")]
                {
                    let seeds = $crate::transforms::SeedSource::all_seeds();
                    if !seeds.is_empty() {
                        eprintln!("Seeds used in this test thread: {:?}", seeds);
                    }
                }

                #[cfg(feature = "plotting")]
                $crate::plot!(
                    fit,
                    { title: format!("Monotonicity Violation at x={first}") },
                    prefix = "assert_monotone"
                );

                #[allow(unused_mut, unused_assignments)] let mut msg = format!("Fit is not monotonic - derivative changes sign at x={first}");
                $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

                panic!("{msg}");
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
    ($function:expr, $x:expr, $expected:expr $(, $msg:literal $(, $($args:tt),*)?)?) => {{
        let function = &$function;
        let function: &$crate::Polynomial<_, _> = function.as_ref();
        let x = $x;
        let expected = $expected;

        #[allow(unused_mut, unused_assignments)] let mut msg = format!("y({x}) != {expected}");
        $( msg = format!("{msg}: {}", format!($msg, $($($args)?)?)); )?

        $crate::assert_close!(function.y(x), expected, "{msg}");
    }};
}

/// Asserts that one polynomial is the derivative of another over a specified domain.
///
/// This macro checks that the derivative of `f` matches `f_prime` at multiple points within the given domain.
///
/// If the assertion fails, a plot showing both functions will be generated in `<target/test_output>`.
///
/// # Arguments
/// * `$f` - The original polynomial function.
/// * `$f_prime` - The polynomial function that should be the derivative of `$f`.
/// * `$norm` - The domain normalizer used for both polynomials. Use `DomainNormalizer::default()` if none.
/// * `$domain` - The range of x-values over which to check the derivative relationship. (inclusive range)
///
/// # Panics
/// Panics if the derivative relationship does not hold within a reasonable tolerance.
#[macro_export]
macro_rules! assert_is_derivative {
    ($f:expr, $f_prime:expr, $norm:expr, $domain:expr $(, f_lbl = $f_lbl:literal)? $(, fprime_lbl = $fprime_lbl:literal)?) => {
        if let Err(e) = $crate::statistics::is_derivative(&$f, &$f_prime, $norm, &$domain) {
            #[cfg(feature = "plotting")]
            {
                $crate::plot!([$f, $f_prime], {
                    x_range: Some(*$domain.start()..*$domain.end()),
                });
            }

            #[allow(unused_mut, unused_assignments)] let mut f_lbl = "f(x)"; $(f_lbl = $f_lbl;)?
            #[allow(unused_mut, unused_assignments)] let mut fprime_lbl = "f'(x)"; $(fprime_lbl = $fprime_lbl;)?

            eprintln!("{f_lbl}={}", $f);
            eprintln!("{fprime_lbl}={}", $f_prime);
            panic!("{e}");
        }
    };
}

/// Asserts that two floating-point values are approximately equal within a small tolerance (epsilon).
///
/// Also works for complex numbers.
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
    ($a:expr, $b:expr $(, $msg:literal $(, $($args:tt),*)?)?) => { #[allow(clippy::float_cmp)] {
        #[allow(unused_imports)] use $crate::nalgebra::ComplexField;
        fn epsilon<C: $crate::nalgebra::ComplexField<RealField = T>, T: $crate::value::Value>(_: C) -> T {
            T::epsilon()
        }

        #[allow(unused_mut, unused_assignments)] let mut msg = "Values not close".to_string();
        $( msg = format!($msg, $($($args)?)?); )?

        let (a, b) = ($a, $b);
        assert!(
            a.imaginary() == b.imaginary() || $crate::value::Value::abs(a.imaginary() - b.imaginary()) <= epsilon($a),
            "{msg} - imaginary parts differ {} != {}", a.imaginary(), b.imaginary()
        );
        assert!(
            a.real() == b.real() || $crate::value::Value::abs(a.real() - b.real()) <= epsilon($a),
            "{msg}: {a} != {b}"
        );
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
    ($src:expr, $dst:expr  $(, $msg:literal $(, $($args:tt),*)?)?) => {
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
#[cfg(feature = "transforms")]
mod tests {
    use crate::{
        function,
        score::Aic,
        statistics::DegreeBound,
        transforms::{ApplyNoise, Strength},
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
            .solve_range(0.0..=1000.0, 1.0)
            .apply_normal_noise(Strength::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_fits!(&poly, &fit, 0.99);
    }

    #[test]
    fn test_assert_r_squared_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..=1000.0, 1.0)
            .apply_normal_noise(Strength::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_r_squared!(&fit, 0.98);
        assert_r_squared!(&fit, 0.98, msg = "test");
        assert_r_squared!(&fit, msg = "test");
        assert_r_squared!(&fit);
    }

    #[test]
    fn test_assert_residuals_normal_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..=1000.0, 1.0)
            .apply_normal_noise(Strength::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_residuals_normal!(&fit, 0.01);
    }

    #[test]
    fn test_assert_max_residual_macro() {
        function!(poly(x) = 1.0 + 2.0 x^1 + 3.0 x^2);
        let data = poly
            .solve_range(0.0..=1000.0, 1.0)
            .apply_normal_noise(Strength::Absolute(0.01), None);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_max_residual!(&fit, 80000.0);
    }

    #[test]
    fn test_assert_monotone_macro() {
        function!(mono(x) = 1.0 + 2.0 x^1); // strictly increasing
        let data = mono.solve_range(0.0..=1000.0, 1.0);
        let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
        assert_monotone!(&fit);
    }
}
