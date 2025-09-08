//! A test-suite designed to simplify making sure the math worked
//!
//! - The [`function`] macro is great for generating synthetic data sets:
//! ```rust
//! polyfit::function!(const f(x) = 5 x^4 - 4 x^3 + 2.5);
//! let data = f.solve_range(0.0..100.0, 1.0);
//! ```
//!
//! - The [`Noise`] trait can help you add a variety of noises to your data:
//!   -
//!

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
        #[allow(non_upper_case_globals)]
        const $name: $crate::MonomialPolynomial = $crate::function!($($rest)+);
    };

    (static $name:ident (x) = $($rest:tt)+ ) => {
        #[allow(non_upper_case_globals)]
        static $name: $crate::MonomialPolynomial = $crate::function!($($rest)+);
    };
}

/// Automatically fits a dataset against multiple polynomial bases and reports the best fits.
///
/// # Syntax
/// ```ignore
/// basis_select!(data, options = [Basis1<T>, Basis2<T>, …]);
/// basis_select!(data); // Uses default [MonomialBasis<f64>, ChebyshevBasis<f64>]
/// ```
///
/// # Behavior
/// - Tries to construct a `CurveFit<Basis>` for each basis in the provided list.
/// - Uses `CurveFit::new_auto` with:
///   - `DegreeBound::Relaxed`
///   - `ScoringMethod::AIC`
/// - For each successful fit, collects:
///   - **AIC score**
///   - **basis name** (via `stringify!`)
///   - **R²** coefficient of determination
///   - **equation** string
/// - Sorts candidate fits by AIC score (ascending = better).
/// - Prints the top 3 models in the format:
///   ```text
///   Best fitting models:
///   - `ChebyshevBasis<f64>`: R² = 0.998, Equation: ...
///   - `MonomialBasis<f64>`: R² = 0.990, Equation: ...
///   ```
/// - If the `plotting` feature is enabled, plots each reported fit.
///
/// # Parameters
/// - `$data`: A slice of `(x, y)` points or any type accepted by `CurveFit`.
/// - `options`: Optional. List of basis types to compare.
///   - Defaults to `[MonomialBasis<f64>, ChebyshevBasis<f64>]`.
///
/// # Notes
/// - Fits that fail to converge are skipped silently.
/// - AIC is used for sorting, but R² and the explicit equation are reported for transparency.
/// - Only the top 3 fits are printed, but the full candidate list is available inside the macro scope.
/// - Models are created fresh for evaluation and are not returned to the caller.
#[macro_export]
macro_rules! basis_select {
    ($data:expr, options = [ $( $basis:path ),+ ]) => {
        let mut options = vec![];
        $(
            if let Ok(fit) = $crate::CurveFit::<$basis>::new_auto(&$data, $crate::statistics::DegreeBound::Relaxed, $crate::statistics::ScoringMethod::AIC) {
                let score = fit.model_score($crate::statistics::ScoringMethod::AIC);
            let name = stringify!($basis);
            let r2 = fit.r_squared(fit.data());
            let equation = fit.equation();
            options.push((score, name, r2, equation));
            }
        )+

        // Sort by f.model_score, descending
        options.sort_by(|(s1, _, _, _), (s2, _, _, _)| s1.total_cmp(&s2));

        println!("Best fitting models:");
        for (_, name, r2, equation) in options.iter().take(3) {
            println!("- `{}`: R² = {}, Equation: {}", name, r2, equation);

            #[cfg(feature = "plotting")]
            polyfit::plot!(f, title = name.to_string());
        }
    };

    ($data:expr) => {
        $crate::basis_select!($data, options = [ $crate::basis::MonomialBasis<f64>, $crate::basis::ChebyshevBasis<f64> ])
    }
}
