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
//! # use polyfit::statistics::DegreeBound;
//! # use polyfit::score::Aic;
//! # use polyfit::transforms::{ApplyNoise, Strength};
//! # use polyfit::{function, basis_select};
//! function!(test(x) = 2.0 x^3 + 3.0 x^2 - 4.0 x + 5.0);
//! let data = test
//!     .solve_range(0.0..=100.0, 1.0)
//!     .apply_normal_noise(Strength::Relative(0.1), None);
//! basis_select!(&data, DegreeBound::Relaxed, &Aic);
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
//! # use polyfit::statistics::{DegreeBound};
//! # use polyfit::score::Aic;
//! let fit = MonomialFit::new_auto(&data, DegreeBound::Relaxed, &Aic).unwrap();
//! # #[cfg(feature = "plotting")]
//! # {
//! polyfit::plot!(fit, { title: "My Fit".to_string(), x_range: Some(0.0..3.0), y_range: Some(0.0..5.0) });
//! # }
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
//! ### [`crate::assert_max_residual`]
//! Asserts that all residuals (the differences between the observed and predicted values) of a fit are below a certain threshold.
//! This helps to ensure that the fit is not only accurate, but also consistent.
//! - This is an absolute measure, unlike [`crate::assert_residuals_normal`] which is a relative measure.

#[macro_use]
#[cfg(test)]
pub mod basis_assertions;

mod assertions;

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
/// - Uses `CurveFit::new_auto` with for each basis with the provided `DegreeBound` and scoring method ([`crate::score`]).
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
/// - `$method`: The scoring method to use for fitting (see [`crate::score`]).
/// - `options`: Optional. List of basis types to compare. Default is all supported bases.
///
/// # Example
/// ```rust
/// # use polyfit::statistics::DegreeBound;
/// # use polyfit::transforms::{ApplyNoise, Strength};
/// # use polyfit::score::Aic;
/// # use polyfit::{function, basis_select};
/// function!(test(x) = 2.0 x^3 + 3.0 x^2 - 4.0 x + 5.0);
/// let data = test
///     .solve_range(0.0..=100.0, 1.0)
///     .apply_normal_noise(Strength::Relative(0.1), None);
/// basis_select!(&data, DegreeBound::Relaxed, &Aic);
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
            ChebyshevBasis<f64> = "Chebyshev",
            FourierBasis<f64> = "Fourier",
            LegendreBasis<f64> = "Legendre",
            PhysicistsHermiteBasis<f64> = "Physicists' Hermite",
            ProbabilistsHermiteBasis<f64> = "Probabilists' Hermite",
            LaguerreBasis<f64> = "Laguerre",
            LogarithmicBasis<f64> = "Logarithmic",
        ])
    }};

    ($data:expr, $degree_bound:expr, $method:expr, options = [ $( $basis:path $( = $name:literal)? ),+ $(,)? ]) => {{
        use $crate::value::CoordExt;

        #[cfg(feature = "plotting")]
        use $crate::plotting::AsPlottingElement;

        struct FitProps {
            model_score: f64,
            rating: f64,

            #[cfg(feature = "plotting")]
            plot_e: $crate::plotting::PlottingElement<f64>,

            name: &'static str,
            r2: f64,
            max_r2: f64,
            p_value: f64,
            stars: f64,
            equation: String,
            parameters: usize,
        }

        let num_basis = 0 $( + { let _ = stringify!($basis); 1 } )+;
        let count = $data.len();

        println!("[ Evaluating {count} data points against {num_basis} basis options ]\n");
        if count < 100 {
            println!("[ WARNING - SMALL DATASET ]");
            println!("[ Results may be misleading for small datasets (<100 points) ]\n");
        }

        let mut all_normals_zero = true;
        let mut options = vec![];
        let mut min_params = usize::MAX;
        $(
            if let Ok(fit) = $crate::CurveFit::<$basis>::new_auto($data, $degree_bound, $method) {
                #[allow(unused_mut, unused_assignments)] let mut name = stringify!($basis); $( name = $name; )?
                let equation = fit.equation();

                let model_score = fit.model_score($method);
                let residuals = fit.filtered_residuals().y();
                let r2 = fit.r_squared(None);
                let robust_r2 = fit.robust_r_squared(None);
                let max_r2 = $crate::value::Value::max(r2, robust_r2);
                let p_value = $crate::statistics::residual_normality(&residuals);
                
                let rating = (0.75 * max_r2 + 0.25 * p_value).clamp(0.0, 1.0);
                let parameters = fit.coefficients().len();

                if parameters < min_params {
                    min_params = parameters;
                }

                if p_value > f64::EPSILON {
                    all_normals_zero = false;
                }

                #[cfg(feature = "plotting")]
                let plot_e = fit.as_plotting_element(&[], $crate::statistics::Confidence::P95, None);

                options.push(FitProps {
                    model_score,
                    rating,

                    #[cfg(feature = "plotting")]
                    plot_e,

                    name,
                    r2,
                    max_r2,
                    p_value,
                    stars: 0.0, // to be filled later
                    equation,
                    parameters,
                });
            }
        )+

        // Give a small penalty to models with more parameters - we divide the score by `params/min_params`
        let normalizer = $crate::statistics::DomainNormalizer::new((0.3, 1.0), (0.0, 5.0));
        for o in options.iter_mut() {
            let param_penalty = o.parameters as f64 / min_params as f64;
            o.rating /= param_penalty;

            //
            // Get a star rating out of 5 based on rating
            o.stars = normalizer.normalize(o.rating).clamp(0.0, 5.0);
        }

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

        // let best_3: Vec<_> = options.iter().take(3).collect();

        //
        // Small table first
        let (basis, parameters, score, r2, norm, rating) = ("Basis", "Params", "Score Weight", "Fit Quality", "Normality", "Rating");
        let sep = || { println!("--|-{:-^30}-|-{:-^6}-|-{:-^12}-|-{:-^11}-|-{:-^9}-|-{:-^10}", "", "", "", "", "", ""); };
        println!("# | {basis:^30} | {parameters:^6} | {score:^12} | {r2:^11} | {norm:^9} | {rating}");
        sep();
        for (i, props) in options.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let whole_stars = props.stars.round() as usize;

            let name = props.name;
            let parameters = props.parameters;
            let score = props.model_score * 100.0;
            let r2 = props.max_r2 * 100.0;
            let norm = props.p_value * 100.0;
            let rating = props.rating * 100.0;

            let vis_stars = "☆".repeat(5 - whole_stars) + &"★".repeat(whole_stars);

            let score = format!("{score:.2}%");
            let r2 = format!("{r2:.2}%");
            let rating = format!("{rating:.0}%");

            let norm = if all_normals_zero {
                "-----".to_string()
            } else {
                format!("{norm:.2}%")
            };

            let n = i + 1;
            println!(
                "{n} | {name:>30} | {parameters:>6} | {score:>12} | {r2:>11} | {norm:>9} | {rating} {vis_stars}",
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
        println!(" - Params: Number of parameters (coefficients) in the fitted model. Less means simpler model, less risk of overfitting.");
        println!(" - Score Weight: Relative likelihood of being the best model among the options tested, based on the scoring method used.");
        println!(" - Fit Quality: Proportion of variance in the data explained by the model (uses huber loss weighted r2).");
        println!(" - Normality: How closely the residuals follow a normal distribution (useless for small datasets).");
        println!(" - Rating: Combined score (0.75 * Fit Quality + 0.25 * Normality) to give an overall quality measure.");
        println!(" - Stars: A simple star rating out of 5 based on the Rating score. Not scientific.");
        println!(" - The best 3 models are shown below with their equations and plots (if enabled).");

        for props in &options {
            println!();
            println!("{}: {}", props.name, props.equation);
            println!("Fit R²: {:.4}, Residuals Normality p-value: {:.4}", props.r2, props.p_value);

            #[cfg(feature = "plotting")]
            {
                let prefix = props.name.to_lowercase().replace([' ', '\'', '"', '<', '>', ':', ';', ',', '.'], "_");
                $crate::plot!(props.plot_e, {
                    title: props.name.to_string()
                }, prefix = prefix);
            }
        }
    }};
}

#[cfg(test)]
#[cfg(feature = "transforms")]
mod tests {
    #[test]
    fn test_bselect() {
        use crate::score::Aic;
        use crate::statistics::DegreeBound;
        use crate::transforms::{ApplyNoise, Strength};

        function!(test(x) = 2.0 x^3 + 3.0 x^2 - 4.0 x + 5.0);
        let data = test
            .solve_range(0.0..=1000.0, 1.0)
            .apply_normal_noise(Strength::Relative(0.3), None);
        basis_select!(&data, DegreeBound::Relaxed, &Aic);
    }
}
