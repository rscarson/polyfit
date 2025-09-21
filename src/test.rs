//! A test-suite designed to bridge the gap between statistical rigour and engineering pragmatism.
//!
//! # Features
//!
//! ## General Purpose Macros
//!
//! ### [`crate::function!`]
//!
//! DSL for generating monomial polynomials. Great for generating synthetic data sets!
//! ```rust
//! polyfit::function!(const f(x) = 5 x^4 - 4 x^3 + 2.5);
//! let data = f.solve_range(0.0..=100.0, 1.0);
//! ```
//!
//! ### [`crate::basis_select!`]
//! Automatically fits a dataset against every supported polynomial base and reports the best fits.
//! - The best 3 models are printed to the console, and if the `plotting` feature is enabled, they are plotted too!
//! ```rust
//! # use polyfit::statistics::{DegreeBound, ScoringMethod, Tolerance};
//! # use polyfit::transforms::ApplyNoise;
//! # use polyfit::{function, basis_select};
//! function!(test(x) = 2.0 x^3 + 3.0 x^2 - 4.0 x + 5.0);
//! let data = test
//!     .solve_range(0.0..=100.0, 1.0)
//!     .apply_normal_noise(Tolerance::Relative(0.1), None);
//! basis_select!(&data, DegreeBound::Relaxed, ScoringMethod::AIC);
//! ```
//!
//! ### [`crate::plot!`]
//! Macro to plot a polynomial fit or function if the `plotting` feature is enabled.
//! - Supports custom titles, and axis ranges.
//! - Can plot multiple functions on the same graph.
//! - Includes residuals, confidence intervals, and source data for fits
//! ```rust
//! # let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 0.5), (3.0, 4.0)];
//! # use polyfit::{MonomialFit};
//! # use polyfit::statistics::{DegreeBound, ScoringMethod};
//! let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, ScoringMethod::AIC).unwrap();
//! polyfit::plot!(fit, title = "My Fit", x_range = 0.0..3.0, y_range = 0.0..5.0);
//! ```
//!
//! ## Fit quality assertions
//! These are designed to be used in unit tests to validate fit quality.
//! - If `plotting` feature is enabled, the fit is plotted on failure to help diagnose issues.
//!
//! ### [`crate::assert_close`]
//! Asserts that two floating-point values are approximately equal within a small tolerance (epsilon).
//! This is useful for comparing computed values where exact equality is not expected due to rounding errors.
//! - Uses the machine epsilon for the floating-point type as the tolerance.
//! - `assert_eq!` equivalent for floats.
//!
//! ### [`crate::assert_all_close`]
//! Asserts that two slices of floating-point values are approximately equal element-wise within a small tolerance (epsilon).
//! This is useful for comparing arrays of computed values where exact equality is not expected due to rounding errors.
//! - Uses the machine epsilon for the floating-point type as the tolerance.
//! - Element-wise [`crate::assert_close`].
//!
//! ### [`crate::assert_y`]
//! Asserts that a fit produces an expected 'y' value at a given 'x' input.
//! Useful for spot-checking specific predictions of the model.
//! - Compares the fit's output against an expected value within a small tolerance (epsilon).
//!
//! ### [`crate::assert_fits`]
//! Asserts that a fit matches a known polynomial function.
//! Compares the fit's r² value against a threshold to ensure it closely follows the expected curve.
//! Useful if you know the underlying function but want to validate the fitting process.
//! See [`crate::CurveFit::r_squared`] for more details.
//!
//! ### [`crate::assert_r_squared`]
//! General case of [`crate::assert_fits`] that does not require a known function.
//! Asserts that the fit's r² value relative to the source data is above a certain threshold.
//! This is a measure of how well the curve explains how wiggly the data is.
//! See [`crate::CurveFit::r_squared`] for more details.
//!
//! ### [`crate::assert_monotone`]
//! Asserts that a fit is monotonic (either strictly increasing or strictly decreasing) over the range of the source data.
//! This is useful for validating models where the output should consistently rise or fall with the input.
//! See [`crate::CurveFit::monotonicity_violations`] for more details.
//!
//! ### [`crate::assert_residuals_normal`]
//! Asserts that the residuals (the differences between the observed and predicted values) of a fit are normally distributed.
//! Think of it like making sure the errors are random, not based on some undiscovered pattern.
//! See [`crate::statistics::residual_normality`] for more details.
//! - Results will be between 0.0 and 1.0, with values closer to 1.0 indicating a better fit.
//!
//! ### [`crate::assert_residual_spread`]
//! Asserts that the spread of the residuals (the differences between the observed and predicted values) of a fit is below a certain threshold.
//! This helps to ensure that the fit is not only accurate, but also consistent.
//! - This is an absolute measure, unlike [`crate::assert_residuals_normal`] which is a relative measure.

mod assertions;
mod basis_assertions;

/// Macro to generate a monomial polynomial function.
///
/// This is good for using as a data source for testing
/// - Terms can be listed in any order
/// - Same-power terms are summed
/// - Missing terms are 0
///
/// The only major limitation is that it needs a space between the coefficient and the variable:
/// - `20.0 x^3` is valid, but `20.0x^3` is not.
///
/// Syntax:
/// ```text
/// function!(
///     [const | static]?
///     [<name>(<x>) = ]?
///     [ [+]? <coef> [ x [ ^ <deg> ]? ]? ]+
/// )
/// ```
///
/// # Example
/// ```
/// # use polyfit::function;
/// function!(test(x) = 20.0 x^3 + 3.0 x^2 - 2.0 x + 4.0); // Normal let-binding
/// function!(const test2(x) = 20.0 x^3 + 3.0 x^2 - 2.0 x + 4.0); // const! can live outside functions
/// function!(static test3(x) = 20.0 x^3 + 3.0 x^2 - 2.0 x + 4.0); // I added static for some reason
/// let test4 = function!(20.0 x^3 + 3.0 x^2 - 2.0); // No auto bindings
///
/// // Evaluate at x = 5
/// println!("{}", test.y(5.0));
///
/// // You could visualize it too:
/// // polyfit::plot_function!(test, x_range = 0.0..1000.0);
/// ```
#[macro_export]
macro_rules! function {
    ($( $(+)? $c0:literal $(x $( ^ $d0:literal )?)? )+) => { {
        const LEN: usize = {
            let mut degree = 0; $(
                let d2 = 1 $(+ 1 $(* $d0 as usize)?)?;
                if d2 > degree { degree = d2; }
            )+
            degree
        };

        const COEFS: [f64; LEN] = {
            let mut coefs = [0.0; LEN];
            // coef alone is degree 0, 1 if just x, or the power if specified
            $( coefs[ 0 $(+ 1 $(* $d0 as usize)?)? ] += $c0 as f64; )+
            coefs
        };

        $crate::MonomialPolynomial::borrowed(&COEFS)
    }};

    ($name:ident (x) = $($rest:tt)+ ) => {
        let $name: $crate::MonomialPolynomial = $crate::function!($($rest)+);
    };

    (const $name:ident (x) = $($rest:tt)+ ) => {
        const $name: $crate::MonomialPolynomial<'static> = $crate::function!($($rest)+);
    };

    (static $name:ident (x) = $($rest:tt)+ ) => {
        static $name: $crate::MonomialPolynomial<'static> = $crate::function!($($rest)+);
    };
}

/// Automatically fits a dataset against multiple polynomial bases and reports the best fits.
///
/// # Syntax
/// ```text
/// basis_select!(data, options = [Basis1<T>, Basis2<T>, …]);
/// basis_select!(data); // Uses default [MonomialBasis<f64>, ChebyshevBasis<f64>]
/// ```
///
/// # Behavior
/// - Tries to construct a `CurveFit<Basis>` for each basis in the provided list.
/// - Uses `CurveFit::new_auto` with for each basis with the provided `DegreeBound` and `ScoringMethod`.
/// - For each successful fit, computes:
///   - r² value against the source data
///   - p-value for residual normality test
///   - A combined rating (0.75 * r² + 0.25 * p-value) for overall quality.
///   - A star rating out of 5 based on the combined rating.
///   - An equation string representation.
///
/// - Displays a summary table of the top 3 fits
/// - For each of the top 3 fits, prints:
///   - A plot if the `plotting` feature is enabled
///   - The equation string
///
/// # Parameters
/// - `$data`: A slice of `(x, y)` points or any type accepted by `CurveFit`.
/// - `$degree_bound`: The degree bound to use for fitting (see [`crate::statistics::DegreeBound`]).
/// - `$method`: The scoring method to use for fitting (see [`crate::statistics::ScoringMethod`]).
/// - `options`: Optional. List of basis types to compare. Default is all supported bases.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::{DegreeBound, ScoringMethod, Tolerance};
/// # use polyfit::transforms::ApplyNoise;
/// # use polyfit::{function, basis_select};
/// function!(test(x) = 2.0 x^3 + 3.0 x^2 - 4.0 x + 5.0);
/// let data = test
///     .solve_range(0.0..=100.0, 1.0)
///     .apply_normal_noise(Tolerance::Relative(0.1), None);
/// basis_select!(&data, DegreeBound::Relaxed, ScoringMethod::AIC);
/// ```
///
/// The example above will output something like:
/// ```text
/// [ Evaluating 100 data points against 3 basis options ]
///
///      Basis      |     R²     | Residuals Normality  | Rating
/// --------------- | ---------- | -------------------- | ----------
/// ChebyshevBasis  | 99.10%     | 62.07%               | 90% ☆☆★★★
///  MonomialBasis  | 99.10%     | 62.07%               | 90% ☆☆★★★
///  FourierBasis   | 82.41%     | 0.00%                | 62% ☆☆☆☆★
///
/// ChebyshevBasis: xₛ = 2(x - a) / (b - a) - 1, a=0.00e0, b=99.00, y(x) = 5.38e4T₃(xₛ) + 3.65e5T₂(xₛ) + 9.09e5T₁(xₛ) + 6.12e5
/// Wrote plot to target\plot_output\chebyshevbasis_src_test.rs_line_307_1757796134.png
///
/// MonomialBasis: y(x) = 1.77x³ + 34.28x² - 1.33e3x + 1.45e4
/// Wrote plot to target\plot_output\monomialbasis_src_test.rs_line_307_1757796135.png
///
/// FourierBasis: xₛ = 2(x - a) / (b - a) - 1, a=0.00e0, b=99.00, y(x) = 1.72e4cos(2π4xₛ) - 1.42e5sin(2π4xₛ) + 4.27e4cos(2π3xₛ) - 1.98e5sin(2π3xₛ) + 6.79e4cos(2π2xₛ) - 2.94e5sin(2π2xₛ) + 2.98e5cos(2πxₛ) - 5.30e5sin(2πxₛ) + 4.92e5
/// Wrote plot to target\plot_output\fourierbasis_src_test.rs_line_307_1757796135.png
/// ```
#[macro_export]
macro_rules! basis_select {
    ($data:expr, $degree_bound:expr, $method:expr) => {{
        use $crate::basis::*;
        $crate::basis_select!($data, $degree_bound, $method, options = [
            MonomialBasis<f64> = "Monomial",
            ChebyshevBasis<f64> = "Chebyshev",
            FourierBasis<f64> = "Fourier",
            LegendreBasis<f64> = "Legendre",
            PhysicistsHermiteBasis<f64> = "Physicists' Hermite",
            ProbabilistsHermiteBasis<f64> = "Probabilists' Hermite",
            LaguerreBasis<f64> = "Laguerre",
        ])
    }};

    ($data:expr, $degree_bound:expr, $method:expr, options = [ $( $basis:path $( = $name:literal)? ),+ $(,)? ]) => {{
        use $crate::value::CoordExt;
        struct FitProps {
            model_score: f64,
            rating: f64,
            plot_fn: Box<dyn Fn()>,
            name: &'static str,
            r2: f64,
            p_value: f64,
            stars: usize,
            equation: String,
        }

        let num_basis = 0 $( + { let _ = stringify!($basis); 1 } )+;
        let count = $data.len();

        println!("[ Evaluating {count} data points against {num_basis} basis options ]\n");
        if count < 100 {
            println!("[ WARNING - SMALL DATASET ]");
            println!("[ Results may be misleading for small datasets (<100 points) ]\n");
        }

        let mut options = vec![];
        $(
            if let Ok(fit) = $crate::CurveFit::<$basis>::new_auto($data, $degree_bound, $method) {
                #[allow(unused_mut, unused_assignments)] let mut name = stringify!($basis); $( name = $name; )?
                let equation = fit.equation();

                let model_score = fit.model_score($method);
                let residuals = fit.residuals().y();
                let r2 = fit.r_squared(fit.data());
                let p_value = $crate::statistics::residual_normality(&residuals);
                let rating = 0.75 * r2 + 0.25 * p_value;

                //
                // Get a star rating out of 5 based on rating
                let stars = match rating {
                    r if r >= 0.95 => 5,
                    r if r >= 0.9 => 4,
                    r if r >= 0.8 => 3,
                    r if r >= 0.7 => 2,
                    r if r >= 0.6 => 1,
                    _ => 0,
                };

                #[allow(unused_mut, unused_assignments)] let mut plot_fn: Box<dyn Fn()> = Box::new(|| ());

                #[cfg(feature = "plotting")]
                {
                    let prefix = name.to_lowercase().replace([' ', '\'', '"', '<', '>', ':', ';', ',', '.'], "_");

                    plot_fn = Box::new(move || $crate::plot!(fit, title = name.to_string(), prefix = prefix));
                }

                options.push(FitProps {
                    model_score,
                    rating,
                    plot_fn,
                    name,
                    r2,
                    p_value,
                    stars,
                    equation,
                });
            }
        )+

        // Sort by f.model_score, descending
        options.sort_by(|p1, p2| p1.model_score.total_cmp(&p2.model_score));

        //
        // Subtract the best model score from all model scores to get relative scores
        // Then we will use that to calculate a probability-like value for correctness of the model vs all others
        let best_score = options.first().map_or(0.0, |p| p.model_score);
        let likelihoods: Vec<_> = options.iter().map(|o| (-0.5 * (o.model_score - best_score)).exp()).collect();
        let sum_likelihoods: f64 = likelihoods.iter().sum();
        for (o, l) in options.iter_mut().zip(likelihoods) {
            o.model_score = l / sum_likelihoods;
        }

        let best_3: Vec<_> = options.iter().take(3).collect();

        //
        // Small table first
        let (basis, score, r2, norm, rating) = ("Basis", "Score Weight", "R²", "Residuals Normality", "Rating");
        let sep = || { println!("--|-{:-^30}-|-{:-^12}-|-{:-^10}-|-{:-^20}-|-{:-^10}", "", "", "", "", ""); };
        println!("# | {basis:^30} | {score:^12} | {r2:^10} | {norm:^20} | {rating}");
        sep();
        for (i, props) in options.iter().enumerate() {
            let name = props.name;
            let score = props.model_score * 100.0;
            let r2 = props.r2 * 100.0;
            let norm = props.p_value * 100.0;
            let rating = props.rating * 100.0;
            let stars = "☆".repeat(5 - props.stars) + &"★".repeat(props.stars);

            let score = format!("{score:.4}%");
            let r2 = format!("{r2:.4}%");
            let norm = format!("{norm:.4}%");
            let rating = format!("{rating:.0}%");

            let n = i + 1;
            println!(
                "{n} | {name:^30} | {score:^12} | {r2:<10} | {norm:<20} | {rating} {stars}",
            );

            // Separator after best 3
            if i == 2 {
                sep();
            }
        }

        //
        // Instructions
        println!();
        println!("[ How to interpret the results ]");
        println!("[ Results may be misleading for small datasets (<100 points) ]");
        println!(" - Score Weight: Relative likelihood of being the best model among the options tested.");
        println!(" - R²: Proportion of variance in the data explained by the model (useless for small datasets).");
        println!(" - Residuals Normality: How closely the residuals follow a normal distribution (useless for small datasets).");
        println!(" - Rating: Combined score (0.75 * R² + 0.25 * Residuals Normality) to give an overall quality measure.");
        println!(" - Stars: A simple star rating out of 5 based on the Rating score.");

        for props in &best_3 {
            println!();
            println!("{}: {}", props.name, props.equation);

            #[cfg(feature = "plotting")]
            (props.plot_fn)();
        }
    }};
}

#[cfg(test)]
#[test]
fn test() {
    use crate::statistics::{DegreeBound, ScoringMethod, Tolerance};
    use crate::transforms::ApplyNoise;

    function!(test(x) = 2.0 x^3 + 3.0 x^2 - 4.0 x + 5.0);
    let data = test
        .solve_range(0.0..=1000.0, 1.0)
        .apply_normal_noise(Tolerance::Relative(0.3), None);
    basis_select!(&data, DegreeBound::Relaxed, ScoringMethod::AIC);
}
