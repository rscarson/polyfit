//! ## Easy Polynomial Fitting for Rust
//! **Because you don't need to be able to build a powerdrill to use one safely**
//!
//! [![Homepage](https://img.shields.io/badge/homepage-grey?logo=astro&link=https%3A%2F%2Fpolyfit.richardcarson.ca%2F)](https://polyfit.richardcarson.ca/)
//! [![Crates.io](https://img.shields.io/crates/v/polyfit.svg)](https://crates.io/crates/polyfit/)
//! [![Build Status](https://github.com/rscarson/polyfit/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/rscarson/polyfit/actions?query=branch%3Amaster)
//! [![docs.rs](https://img.shields.io/docsrs/polyfit)](https://docs.rs/polyfit/latest/polyfit/)
//!
//! Statistics is hard, and linear regression is made entirely out of footguns;
//! Curve fitting might be simple in theory, but there sure is a LOT of theory.
//!
//! This library is designed for developers who need to make use of the plotting or predictive powers of a curve fit without needing to worry about Huber loss,
//! D'Agostino, or what on earth a kurtosis is.
//!
//! ## Built for developers, not statisticians
//! The crate is designed to be easy to use for developers who need to make use of curve fitting without needing to understand all the underlying statistics.
//! - Sensible defaults are provided for all parameters
//! - The most common use-cases are covered by simple functions and macros
//! - All documentation includes examples, explanations of parameters, and assumes zero prior knowledge of statistics
//! - API is designed to guide towards best practices, while being flexible enough to allow advanced users to customize behavior as needed
//! - Includes a suite of testing tools to bridge the gap between statistical rigour and engineering pragmatism
//!
//! ## Features & Capabilities
//!
//! ### Polynomial Fitting
//!
//! [ [More Information](<https://polyfit.richardcarson.ca/api/#fitting>) ] [ [Example](<https://polyfit.richardcarson.ca/tutorials/#getting-started>) ]
//!
//! As a curve fitting library, Polyfit does curve fitting. What's more interesting is that I do it a way that avoids most common footguns to make it intuitive and safe even with zero knowledge of statistics.
//!
//! - Functions and fits generic over floating point type
//! - Human readable polynomial equation display
//! - Cross-validation based scoring for very noisy data
//! - Robust degree selection strategies that avoid overfitting
//! - Built in outlier detection strategies
//! - Lots of metrics, examples, and explanations to help you understand your data
//!
//! **Human Readable Equation Display**
//!
//! The crate includes a set of tools to help you display polynomials in a human readable format
//! - See the [`display`](crate::display) module for more details
//! - All built-in basis options implement [`display::PolynomialDisplay`](crate::display::PolynomialDisplay) to give you a nicely formatted equation like:
//!  - `y(x) = 1.81x³ + 26.87x² - 1.00e3x + 9.03e3`
//!
//!
//! **11 Choices of Polynomial Basis**
//! [ [How do I Choose ](<https://polyfit.richardcarson.ca/tutorials/#basis-selection>) ] [ [How Do I Check I Chose Well](<https://polyfit.richardcarson.ca/testing#validating-your-choice-of-basis>) ]
//!
//! A polynomial basis is a type of mathematical function used to represent polynomials.
//!
//! The crate includes a variety of polynomial bases to choose from, each with their own strengths and weaknesses.
//! [`Polynomial`](https://docs.rs/polyfit/latest/polyfit/struct.Polynomial.html) and [`CurveFit`](https://docs.rs/polyfit/latest/polyfit/struct.CurveFit.html) are generic over the basis, so you can easily swap them out to see which works best for your data.
//!
//! The choice of polynomial basis has massive implications for the shape of the curve, the numerical stability of the fit, and how well it fits!
//! - Use the links above to help you choose the best basis for your data
//! - This table gives hints at which basis to choose based on the characteristics of your data:
//!
//! I also include [`basis_select!`](https://docs.rs/polyfit/latest/polyfit/macro.basis_select.html), a macro that will help you choose the best basis for your data.
//! - It does an automatic fit for each basis I support, and scores them using the method of your choice.
//! - It will show and plot out the best 3
//! - Use it a few times will real data and see which basis seems to consistently come out on top for your use-case
//!
//! | Basis Name | Handles Curves Well | Repeating Patterns | Extremes / Outliers | Growth/Decay | Best Data Shape / Domain |
//! | ---------- | ------------------- | ------------------ | ------------------- | ------------ | ------------------------ |
//! | Monomial   | Poor                | No                 | No                  | Poor         | Any simple trend         |
//! | Chebyshev  | Good                | No                 | No                  | Poor         | Smooth curves, bounded   |
//! | Legendre   | Fair                | No                 | No                  | Poor         | Smooth curves, bounded   |
//! | Hermite    | Good                | No                 | Yes                 | Yes          | Bell-shaped, any range   |
//! | Laguerre   | Good                | No                 | Yes                 | Yes          | Decaying, positive-only  |
//! | Fourier    | Fair                | Yes                | No                  | Poor         | Periodic signals         |
//! | Logarithmic| Fair                | No                 | Yes                 | Yes          | Logarithmic growth       |
//!
//!
//! ### Plotting & Data Visualization
//!
//! [ [More Information](<https://polyfit.richardcarson.ca/api/#plotting>) ] [ [Example](<https://polyfit.richardcarson.ca/tutorials/#fits-for-generating-graphs>) ]
//!
//! After I built this library I realized that debugging curve fits is a pain without visualization. So I added some plotting utilities, and made my whole test suite generate plots on failure.
//!
//! Use the [`plot!`](https://docs.rs/polyfit/latest/polyfit/macro.plot.html) macro to create plots from data and fits.
//! - You can plot data, functions, and multiple things at once, and theres a lot of customization available
//!
//! ### Symbolic Calculus
//!
//! [ [More Information](<https://polyfit.richardcarson.ca/api/#calculus>) ] [ [Example](<https://polyfit.richardcarson.ca/tutorials/#using-calculus>) ]
//!
//! Polyfit supports indefinite integration and differentiation in many bases, as well as root finding for all bases:
//!
//! *Note: Exact root finding is a faster method only available for some bases*
//!
//! | Basis                     | Exact Root Finding | Derivative      | Integral (Indefinite) | As Monomial |
//! |---------------------------|--------------------|-----------------|-----------------------|-------------|
//! | **Monomial**              | Yes                | Yes             | Yes                   | Yes         |
//! | **Chebyshev (1st form)**  | Yes                | Yes             | No                    | Yes         |
//! | **Chebyshev (2nd form)**  | Yes                | Yes             | Yes                   | No          |
//! | **Chebyshev (3rd form)**  | Yes                | No              | Yes                   | No          |
//! | **Legendre**              | No                 | Yes             | No                    | Yes         |
//! | **Laguerre**              | No                 | Yes             | No                    | Yes         |
//! | **Hermite (Both kinds)**  | No                 | Yes             | No                    | Yes         |
//! | **Fourier (sin/cos)**     | No                 | Yes             | Yes                   | No          |
//! | **Exponential (e^{λx})**  | No                 | Yes             | Yes                   | No          |
//! | **Logarithmic (ln^n x)**  | No                 | No              | No                    | No          |
//!
//! **Monomial Conversions for Calculus**
//! Most bases without calculus support implement [`IntoMonomialBasis`](https://docs.rs/polyfit/latest/polyfit/basis/trait.IntoMonomialBasis.html), which allows them to be converted into a monomial for calculus operations, or for the recognizable formula (for example `y(x) = 3x³ + 2x² + 1`).
//!
//! The exception is logarithmic series, which cannot be converted into monomials; For those, use [`Polynomial::project`](https://docs.rs/polyfit/latest/polyfit/struct.Polynomial.html#method.project), which can be a good way to approximate over certain ranges.
//!
//! **Derivatives for Trend Analysis**
//! The derivative describes the rate of change at specific points in a dataset; For example the derivative of position with respect to time is velocity, and the derivative of velocity with respect to time is acceleration.
//!
//! - Find local minimums and maximums
//! - Discover if a function changes direction
//! - Analyze rates of change in your data
//!
//! **Integrals for Total Values**
//! The integral represents the accumulation of quantities, such as total distance traveled or total growth over time.
//!
//! - Calculate total accumulated values
//! - Calculate area under curves
//!
//! ### Analysis & Mathematics
//!
//! [ [More Information](<https://polyfit.richardcarson.ca/api/#analysis>) ] [ [Example](<https://polyfit.richardcarson.ca/tutorials/#understanding-your-data>) ]
//!
//! - A suite of statistical tests to validate fit quality - [`statistics`](crate::statistics) module
//!     - r², residual normality, homoscedasticity, autocorrelation, and more
//!     - Comprehensive documentation on what all those words mean and when to use each test
//!
//! - Orthogonal-basis-specific tools:
//!     - Spectral energy filtering for advanced smoothing and de-noising ([`Polynomial::spectral_energy_filter`](https://docs.rs/polyfit/latest/polyfit/struct.Polynomial.html#method.spectral_energy_filter))
//!       - Smoothness metric for regularization and model selection ([`Polynomial::smoothness`](https://docs.rs/polyfit/latest/polyfit/struct.Polynomial.html#method.smoothness))
//!     - Coefficient energy breakdown for understanding curve shape ([`Polynomial::coefficient_energies`](https://docs.rs/polyfit/latest/polyfit/struct.Polynomial.html#method.coefficient_energies))
//!       - Orthogonal projection for converting functions between bases ([`Polynomial::project_orthogonal`](https://docs.rs/polyfit/latest/polyfit/struct.Polynomial.html#method.project_orthogonal))
//!
//! ### Testing & Validation
//!
//! [ [More Information](<https://polyfit.richardcarson.ca/api/#testing>) ] [ [Example](<https://polyfit.richardcarson.ca/testing/>) ]
//!
//! Being able to use a fitting library without stats knowledge isn't very useful if you can't test and validate your fits.
//!
//! To that end I built in a testing framework to make it easy to validate your fits and make sure everything is working as expected.
//!
//! - Generate synthetic datasets with configurable noise and transforms
//! - Macros that generate a plot of your fit and data on test failure for easy debugging
//! - Built-in assertions for common fit properties and more complex statistical tests
//! - Sane auto-defaults so you don't need to guess parameters
//! - Seed management! Any random transforms log seeds used so you can reproduce failures easily
//! - I even added [`transforms::SeedSource::from_seeds`](crate::transforms::SeedSource::from_seeds) to replay tests with the same random data every time after a failure
//!
//! ### Data Manipulation
//!
//! [ [More Information](<https://polyfit.richardcarson.ca/api/#transforms>) ] [ [Example](<https://polyfit.richardcarson.ca/testing/#simulating-noisy-data-for-testing>) ] [ [Another Example](<https://polyfit.richardcarson.ca/tutorials/#scaling-and-normalization-transforms>) ]
//!
//! I built in an optional [`transforms`](crate::transforms) feature with 3 different classes of transforms:
//!
//! **Normalization Transforms**
//!
//! Choose from Domain, Clipping, Mean Subtraction, or Z-score normalization to preprocess your data before fitting.
//! - Great for improving numerical stability and fit quality, especially for high degree polynomials or large ranges of data.
//!
//! **Noise Transforms**
//!
//! Generate synthetic noisy datasets by adding one of several types of configurable noise
//! - Each option has a plot showing its characteristics and the effect of each parameter.
//!
//! **Scaling Transforms**
//!
//! Transform data by applying scaling functions. Can improve fit quality or be used for data conversions
//! - Shift
//! - Linear Scale
//! - Quadratic Scale
//! - Cubic Scale
//! - Exponential Scale
//! - Logarithmic Scale
//! - Or polynomial scale where you just give a function or fit to apply
//!
//! ---
//!
//! ## Crate Features
//! The parts of the library that require extra dependencies are behind feature flags, so you can keep your binary size down if you don't need them:
//!
//! ### `plotting`
//! - Active by default
//! - All testing macros in the [`test`](crate::test) module will generate plots on failure
//! - Enables the [`plotting`](crate::plotting) module for generating plots of fits and polynomials
//! - Includes the [`plot!`](https://docs.rs/polyfit/latest/polyfit/macro.plot.html) macro for easy plotting
//! - Includes the [`plotting::Plot`](crate::plotting::Plot) type for more customized plots
//!
//! ### `transforms`
//! - Active by default
//! - Enables the [`transforms`](crate::transforms) module for data transformation tools
//!     - Add noise to data (Gaussian, Poisson, Impulse, Correlated, Salt & Pepper)
//!     - Scale data (Shift, Linear, Quadratic, Cubic)
//!     - Normalize data (Domain, Clip, Mean, zscore)
//!
//! ### `parallel`
//! - Inactive by default
//! - Enables parallel processing using `rayon`
//!     - [`CurveFit::new_auto`](https://docs.rs/polyfit/latest/polyfit/struct.CurveFit.html#method.new_auto) uses parallel processing to speed up fitting when this feature is enabled
//!     - All curve fits can use a faster solver for large enough datasets when this feature is enabled
//!     - May significantly speed up fitting for very large datasets on multi-core systems
//!     - **Don't worry your data is tested to make sure the data set is stable enough to benefit from parallelization before using it**
//!
//! ---
//!
//! ## Basic Example
//!
//! The simplest use-case is to find a mathematical function to help approximate a set of data:
//!
//! ```rust
//! # use polyfit::{error::Error, ChebyshevFit, assert_r_squared, assert_residuals_normal, statistics::DegreeBound, score::Aic};
//! # fn main() -> Result<(), Error> {
//! //
//! // Load data from a file
//! let data = include_str!("../examples/sample_data.json");
//! let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();
//!
//! //
//! // Now we can create a curve fit to this data
//! //
//! // `ChebyshevFit` is a type alias for `CurveFit<ChebyshevBasis>`
//! // `ChebyshevBasis` is a robust choice for many types of data; It's more numerically stable and performs better at higher degrees
//! // It is one of several bases available, each with their own strengths and weaknesses. `basis_select!` can help you choose the best one for your data
//! let fit = ChebyshevFit::new_auto(
//!     &data,                 // The data to fit to
//!     DegreeBound::Relaxed,  // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
//!     &Aic                   // The method used to score the fit quality (See [`crate::score`])
//! )?;
//!
//! //
//! // Of course if we don't test our code, how do we know it works - please don't assume I remembered all the edge cases
//! // If these assertion fail, a plot will be generated showing you exactly what went wrong!
//! //
//!
//! // Here we test using r_squared, which is a measure of how well the curve explains how wiggly the data is
//! // An r_squared of 1.0 is a perfect fit
//! assert_r_squared!(fit);
//!
//! // Here we check that the residuals (the difference between the data and the fit) are normally distributed
//! // Think of it like making sure the errors are random, not based on some undiscovered pattern
//! assert_residuals_normal!(fit);
//! # Ok(())
//! # }
//! ```
//!
//! ---
//!
//! ## Performance
//! The crate is designed to be fast and efficient, and includes benchmarks to help test that performance is linear with respect to
//! the number of data points and polynomial degree.
//!
//! Key Characteristics:
//! - A 3rd degree fit (1,000 points, Chebyshev basis) takes about 23µs in my benchmarks
//! - Going up to 100 million points takes about 1.18s with parallelization enabled on my machine (8 cores @ 2.2GHz, 32GB RAM)
//! - Scales linearly with degree, and better than linearly with number of data points (with parallelization enabled)
//!
//! Auto-fit is also reasonably fast; `new_auto` needs to build all candidate models but can do so in parallel:
//! - 1,000 points, Chebyshev basis, and 9 candidate degrees takes about 330µs
//! - If you disable parallelization, it takes about 600µs
//!
//! There are also performance differences between bases
//!
//! Below are median times for fitting a 3rd degree polynomial to 1,000 data points using different bases:
//!
//! | Basis       | Median Time (µs)| Notes / Reason                               |
//! |-------------|-----------------|----------------------------------------------|
//! | Chebyshev   | 24.5            | Fastest; stable matrix and recurrence        |
//! | Legendre    | 29.7            | Comparable; stable polynomial basis          |
//! | Hermite     | 30.9            | Comparable; stable polynomial basis          |
//! | Laguerre    | 31.4            | Comparable; stable polynomial basis          |
//! | Monomial    | 54.0            | Poor conditioning; slower                    |
//! | Fourier     | 57.0            | Trigonometric ops; slower                    |
//! | Logarithmic | 57.0            | Logarithmic ops; slower                      |

//!
//! The benchmarks actually use my library to test that the scaling is linear - which I think is a pretty cool use-case:
//! ```rust
//! # use polyfit::{error::Error, MonomialFit, assert_r_squared};
//! # fn main() -> Result<(), Error> {
//! # let data: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]; // Example data
//! let linear_fit = MonomialFit::new(&data, 1)?; // Create a linear fit (degree=1)
//! polyfit::assert_r_squared!(linear_fit);       // Assert that the linear fit explains the data well (r² > 0.9)
//!
//! // If the assertion fails, a plot will be generated showing you exactly what went wrong
//! # Ok(())
//! # }
//! ```
//!
//! Raw benchmark results:
//! ```text
//! Benchmarking fit vs n (Chebyshev, Degree=3)
//! fit_vs_n/n=100                  [3.3817 µs 3.4070 µs 3.4363 µs]
//! fit_vs_n/n=1_000                [23.791 µs 23.926 µs 24.098 µs]
//! fit_vs_n/n=10_000               [302.99 µs 304.40 µs 306.01 µs]
//! fit_vs_n/n=100_000              [4.5086 ms 4.5224 ms 4.5376 ms]
//! fit_vs_n/n=1_000_000            [12.471 ms 12.592 ms 12.725 ms]
//! fit_vs_n/n=10_000_000           [115.49 ms 116.76 ms 118.07 ms]
//! fit_vs_n/n=100_000_000          [1.1768 s 1.1838 s 1.1908 s]
//!
//! Benchmarking fit vs degree (Chebyshev, n=1000)
//! fit_vs_degree/Degree=1          [11.587 µs 11.691 µs 11.802 µs]
//! fit_vs_degree/Degree=2          [18.109 µs 18.306 µs 18.505 µs]
//! fit_vs_degree/Degree=3          [24.672 µs 24.954 µs 25.269 µs]
//! fit_vs_degree/Degree=4          [33.074 µs 33.206 µs 33.368 µs]
//! fit_vs_degree/Degree=5          [44.399 µs 44.887 µs 45.401 µs]
//! fit_vs_degree/Degree=10         [126.07 µs 127.26 µs 128.62 µs]
//! fit_vs_degree/Degree=20         [420.20 µs 423.44 µs 426.93 µs]
//!
//! Benchmarking fit vs basis (Degree=3, n=1000)
//! fit_vs_basis/Monomial           [53.513 µs 53.980 µs 54.450 µs]
//! fit_vs_basis/Chebyshev          [24.307 µs 24.504 µs 24.710 µs]
//! fit_vs_basis/Legendre           [29.325 µs 29.716 µs 30.160 µs]
//! fit_vs_basis/Hermite            [30.496 µs 30.872 µs 31.321 µs]
//! fit_vs_basis/Laguerre           [31.146 µs 31.428 µs 31.734 µs]
//! fit_vs_basis/Fourier            [56.421 µs 56.985 µs 57.612 µs]
//!
//! Benchmarking auto fit vs basis (n=1000, Candidates=9)
//! auto_fit_vs_basis/Monomial      [497.33 µs 500.27 µs 503.30 µs]
//! auto_fit_vs_basis/Chebyshev     [327.75 µs 329.67 µs 331.76 µs]
//! auto_fit_vs_basis/Legendre      [331.41 µs 330.56 µs 339.52 µs]
//! auto_fit_vs_basis/Hermite       [337.65 µs 339.90 µs 342.44 µs]
//! auto_fit_vs_basis/Laguerre      [428.36 µs 431.26 µs 434.09 µs]
//! auto_fit_vs_basis/Fourier       [710.46 µs 713.07 µs 715.85 µs]
//! auto_fit_vs_basis/Logarithmic   [525.36 µs 528.21 µs 531.10 µs]
//! ```
//!
//! For transparency I ran the same benchmarks in numpy (`benches/numpy_bench.py`):
//! - Comparing rust to python is not exactly fair, but it gives a rough idea of how things compare
//! - Numpy is highly optimized C code under the hood, so it should be an ok comparison
//! - If you want a different library benchmarked, please open an issue or PR
//! ```text
//! Fit vs n (degree=3, Chebyshev):
//! fit_vs_n/n=100 [186.56µs]
//! fit_vs_n/n=1000 [203.37µs]
//! fit_vs_n/n=10000 [803.47µs]
//! fit_vs_n/n=100000 [8672.95µs]
//! fit_vs_n/n=1000000 [92372.89µs]
//! fit_vs_n/n=10000000 [924808.74µs]
//! fit_vs_n/n=100000000 [10059782.74µs]
//!
//! Fit vs degree (n=1000, Chebyshev):
//! fit_vs_degree/degree=1 [177.74µs]
//! fit_vs_degree/degree=2 [141.38µs]
//! fit_vs_degree/degree=3 [190.26µs]
//! fit_vs_degree/degree=4 [221.97µs]
//! fit_vs_degree/degree=5 [222.56µs]
//! fit_vs_degree/degree=10 [501.99µs]
//! fit_vs_degree/degree=20 [2053.02µs]
//! ```
//!
//! ---
//!
//! ## More Examples
//!
//! Oh no! I have some data but I need to try and predict some other value!
//!
//! ```rust
//! # use polyfit::{error::Error, MonomialFit, statistics::{DegreeBound, Confidence, Tolerance}, score::Aic, transforms::Strength};
//! # use polyfit::transforms::ApplyNoise;
//! # fn main() -> Result<(), Error> {
//!
//! //
//! // I don't have any real data, so I'm still going to make some up!
//! // `function!` is a macro that makes it easy to define polynomials for testing
//! // `apply_poisson_noise` is part of the `transforms` module, which provides a set of tools for manipulating data
//! polyfit::function!(f(x) = 2 x^2 + 3 x - 5);
//! let synthetic_data = f.solve_range(0.0..=100.0, 1.0).apply_poisson_noise(Strength::Absolute(0.1), None);
//!
//! //
//! // Now we can create a curve fit to this data
//! // Monomials don't like high degrees, so we will use a more conservative degree bound
//! // `Monomials` are the simplest polynomial basis, and the one most people are familiar with, it looks like `1x^2 + 2x + 3`
//! let fit = MonomialFit::new_auto(&synthetic_data, DegreeBound::Conservative, &Aic).expect("Failed to create fit");
//!
//! //
//! // Now we can make some predictions!
//! //
//!
//! let bad_and_silly_prediction = fit.y(150.0); // This is outside the range of the data, so it is probably nonsense
//!                                              // Fits are usually only valid within the range of the data they were created from
//!                                              // Violating that is called extrapolation, which we generally want to avoid
//!                                              // This will return an error!
//!
//! let bad_prediction_probably = fit.as_polynomial().y(150.0); // This is outside the range of the data, but you asked for it specifically        
//!                                                             // Unlike a CurveFit, a Polynomial is just a mathematical function - no seatbelts
//!                                                             // This will return a value, but it is probably nonsense
//!
//! let good_prediction = fit.y(50.0); // This is within the range of the data, so it is probably reasonable
//!                                    // This will return Ok(value)
//!
//! //
//! // Maybe we need to make sure our predictions are good enough
//! // A covariance matrix sounds terrifying. Because it is.
//! // Which is why I do it for you - covariance matrices measure uncertainty about your fit
//! // They can be used to calculate confidence intervals for predictions
//! // They can also be used to find outliers in your data
//! let covariance = fit.covariance().expect("Failed to calculate covariance");
//! let confidence_band = covariance.confidence_band(
//!     50.0,                          // Confidence band for x=50
//!     Confidence::P95,               // Find the range where we expect 95% of points to fall within
//!     Some(Tolerance::Variance(0.1)) // Tolerate some extra noise in the data (10% of variance of the data, in this case)
//! ).unwrap(); // 95% confidence band
//! println!("I am 95% confident that the true value at x=50.0 is between {} and {}", confidence_band.min(), confidence_band.max());
//!
//! # Ok(())
//! # }
//! ```
//!
//! -----
//!
//! Oh dear! I sure do wish I could find which pieces of data are outliers!
//!
//! ```rust
//! # use polyfit::{error::Error, MonomialFit, statistics::{DegreeBound, Confidence, Tolerance}, score::Aic, transforms::Strength};
//! # use polyfit::transforms::ApplyNoise;
//! # fn main() -> Result<(), Error> {
//!
//! //
//! // I still don't have any real data, so I'm going to make some up! Again!
//! polyfit::function!(f(x) = 2 x^2 + 3 x - 5);
//! let synthetic_data = f.solve_range(0.0..=100.0, 1.0).apply_poisson_noise(Strength::Absolute(0.1), None);
//!
//! //
//! // Let's add some outliers
//! // Salt and pepper noise is a simple way to do this; She's good n' spiky
//! // We will get nice big jumps of +/- 50 in 5% of the data points
//! let synthetic_data_with_outliers = synthetic_data.apply_salt_pepper_noise(0.05, Strength::Absolute(-50.0), Strength::Absolute(50.0), None);
//!
//! //
//! // Now we can create a curve fit to this data, like before
//! let fit = MonomialFit::new_auto(&synthetic_data_with_outliers, DegreeBound::Conservative, &Aic).expect("Failed to create fit");
//!
//! //
//! // Now we can find the outliers!
//! // These are the points that fall outside the 95% confidence interval
//! // This means that they are outside the range where we expect 95% of the data to fall
//! // The `Some(0.1)` means we tolerate some noise in the data, so we don't flag points that are just a little bit off
//! // The noise tolerance is a fraction of the variance of the data (10% in this case)
//! //
//! // If we had a sensor that specified a tolerance of ±5 units, we could use `Some(Tolerance::Absolute(5.0))` instead
//! // If we had a sensor that specified a tolerance of 10% of the reading, we could use `Some(Tolerance::Measurement(0.1))` instead
//! let outliers = fit.covariance().unwrap().outliers(Confidence::P95, Some(Tolerance::Variance(0.1))).unwrap();
//!
//! # Ok(())
//! # }
//! ```
//!
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::needless_range_loop)] //   The worst clippy lint
#![allow(clippy::cast_precision_loss)] //   I don't care about this one
#![allow(clippy::similar_names)] //         Clippy does not get to decide what names are similar
#![allow(clippy::inline_always)] //         I know it doesn't do anything but it makes me feel better
#![allow(clippy::manual_is_multiple_of)] // I hate this one deep in my soul
#![cfg_attr(docsrs, feature(doc_cfg))]

#[macro_use]
pub mod test;

#[cfg(feature = "plotting")]
#[cfg_attr(docsrs, doc(cfg(feature = "plotting")))]
pub mod plotting;

#[cfg(feature = "transforms")]
#[cfg_attr(docsrs, doc(cfg(feature = "transforms")))]
pub mod transforms;

pub mod basis;
pub mod display;
pub mod error;
pub mod score;
pub mod statistics;
pub mod value;

mod fit;
mod polynomial;

pub use fit::*;
pub use polynomial::Polynomial;

pub use basis::fourier::FourierPolynomial;
pub use basis::monomial::MonomialPolynomial;

pub use nalgebra;
