<!-- cargo-rdme start -->

# Polyfit;  Because you don't need to be able to build a powerdrill to use one safely

[![Crates.io](https://img.shields.io/crates/v/polyfit.svg)](https://crates.io/crates/polyfit/)
[![Build Status](https://github.com/caliangroup/polyfit/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/caliangroup/polyfit/actions?query=branch%3Amaster)
[![docs.rs](https://img.shields.io/docsrs/polyfit)](https://docs.rs/polyfit/latest/polyfit/)

Statistics is hard, and linear regression is made entirely out of footguns;
Curve fitting might be simple in theory, but there sure is a LOT of theory.

This library is designed for developers who need to make use of the plotting or predictive powers of a curve fit without needing to worry about Huber loss,
D'Agostino, or what on earth a kurtosis is

I provide a set a tools designed to help you:
- Select the right kind (basis) of polynomial for your data
- Automatically determine the optimal degree of the polynomial
- Make predictions and get confidence values based on it
- Write easy to understand tests to confirm function
  - Tests even plot the data and functions for you on failure! (`plotting` feature)

-----

## Features

### Built for developers, not statisticians
The crate is designed to be easy to use for developers who need to make use of curve fitting without needing to understand all the underlying statistics.
- Sensible defaults are provided for all parameters
- The most common use-cases are covered by simple functions and macros
- All documentation includes examples, explanations of parameters, and assumes zero prior knowledge of statistics
- API is designed to guide towards best practices, while being flexible enough to allow advanced users to customize behavior as needed
- Includes a suite of testing tools to bridge the gap between statistical rigour and engineering pragmatism

### Hot-swappable Polynomial Basis
A polynomial basis is a sum of solving a function, called a basis function, multiplied by a coefficient:
- `y(x) = c1*b1(x) + c2*b2(x) + ... + cn*bn(x)`

The basis determines what the basis functions `b1`, `b2`, ..., `bn` are. The choice of basis has massive implications for the shape of the curve,
the numerical stability of the fit, and how well it fits!

The crate includes a variety of polynomial bases to choose from, each with their own strengths and weaknesses.
[`Polynomial`] and [`CurveFit`] are generic over the basis, so you can easily swap them out to see which works best for your data:
- Monomial - Simple and intuitive, but can be numerically unstable for high degrees or wide x-ranges
- Chebyshev - More numerically stable than monomials, and can provide better fits for certain types of data
- Legendre - Orthogonal polynomials that can provide good fits for certain types of data
- Hermite - Useful for data that is normally distributed
- Laguerre - Useful for data that is exponentially distributed
- Exponential - Useful for data that grows or decays exponentially
- Fourier - Useful for periodic data

I also include [`basis_select!`], a macro that will help you choose the best basis for your data.
- It does an automatic fit for each basis I support, and scores them using the method of your choice.
- It will show and plot out the best 3
- Use it a few times will real data and see which basis seems to consistently come out on top for your use-case

### Calculus Support
All built-in bases support differentiation and integration, including built-in methods for definite integrals, and finding critical points.
- Many basis options implement calculus directly
- A few bases implement `IntoMonomialBasis`, which allows them to be converted into a monomial basis for calculus operations

### Testing Library
The crate includes a set of macros designed to make it easy to write unit tests to validate fit quality.
There are a variety of assertions available, from simple r² checks to ensuring that residuals are normally distributed.

If the `plotting` feature is enabled, any failed assertion will generate a plot showing you exactly what went wrong.

See [`test`] for more details on all included tests

### Plotting
If the `plotting` feature is enabled, you can use the [`plot!`] macro to generate plots of your fits and polynomials.
- Plots are saved as PNG files
- Fits include source data, confidence bands, and residuals
- If enabled, failed assertions in the testing library will automatically generate plots showing what went wrong
- ![Example plot](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/example_fail.png)

See [`plot`] for more details

### Transforms
If the `transforms` feature is enabled, you can use the tools in the [`transforms`] module to manipulate your data.
- Add noise for testing purposes
- Scale, shift, or normalize data

### Human Readable Display
The crate includes a set of tools to help you display polynomials in a human readable format
- See the [`display`] module for more details
- All built-in basis options implement `PolynomialDisplay` to give you a nicely formatted equation like:
 - `y(x) = 1.81x³ + 26.87x² - 1.00e3x + 9.03e3`

-----

The simplest use-case is to find a mathematical function to help approximate a set of data:

```rust

//
// Behold some definitely real data
let data = &[
    (0.0, -5.0), (1.0, 0.0), (2.0, 3.0), (3.0, 10.0), (4.0, 19.0), (5.0, 30.0),
    (6.0, 43.0), (7.0, 58.0), (8.0, 75.0), (9.0, 94.0), (10.0, 115.0),
    (11.0, 138.0), (12.0, 163.0), (13.0, 190.0), (14.0, 219.0), (15.0, 250.0),
    (16.0, 283.0), (17.0, 318.0), (18.0, 355.0), (19.0, 394.0), (20.0, 435.0),
    (21.0, 478.0), (22.0, 523.0), (23.0, 570.0), (24.0, 619.0), (25.0, 670.0),
    (26.0, 723.0), (27.0, 778.0), (28.0, 835.0), (29.0, 894.0), (30.0, 955.0),
];

//
// Now we can create a curve fit to this data
//
// `MonomialFit` is a type alias for `CurveFit<MonomialBasis>`
// `MonomialBasis` is the simplest polynomial basis, and the one most people are familiar with, it looks like `1x^2 + 2x + 3`
// It is one of several bases available, each with their own strengths and weaknesses. `basis_select!` can help you choose the best one for your data
let fit = MonomialFit::new_auto(
    data,                            // The data to fit to
    DegreeBound::Relaxed,            // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
    ScoringMethod::AIC               // The method used to score the fit quality (See [`statistics::ScoringMethod`])
).expect("Failed to create fit");

//
// Of course if we don't test our code, how do we know it works - please don't assume I remembered all the edge cases
// If these assertion fail, a plot will be generated showing you exactly what went wrong!
//

// Here we test using r_squared, which is a measure of how well the curve explains how wiggly the data is
// An r_squared of 1.0 is a perfect fit
assert_r_squared!(fit);

// Here we check that the residuals (the difference between the data and the fit) are normally distributed
// Think of it like making sure the errors are random, not based on some undiscovered pattern
assert_residuals_normal!(fit);
```

-----

Oh no! I have some data but I need to try and predict some other value!

```rust

//
// I don't have any real data, so I'm still going to make some up!
// `function!` is a macro that makes it easy to define polynomials for testing
// `apply_poisson_noise` is part of the `transforms` module, which provides a set of tools for manipulating data
polyfit::function!(f(x) = 2 x^2 + 3 x - 5);
let synthetic_data = f.solve_range(0.0..100.0, 1.0).apply_poisson_noise(0.1, None);

//
// Now we can create a curve fit to this data
let fit = MonomialFit::new_auto(
    &synthetic_data,                 // The data to fit to
    DegreeBound::Relaxed,            // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
    ScoringMethod::AIC               // The method used to score the fit quality (See [`statistics::ScoringMethod`])
).expect("Failed to create fit");

//
// Now we can make some predictions!
//

let bad_and_silly_prediction = fit.y(150.0); // This is outside the range of the data, so it is probably nonsense
                                             // Fits are usually only valid within the range of the data they were created from
                                             // Violating that is called extrapolation, which we generally want to avoid
                                             // This will return an error!

let bad_prediction_probably = fit.as_polynomial().y(150.0); // This is outside the range of the data, but you asked for it specifically        
                                                            // Unlike a CurveFit, a Polynomial is just a mathematical function - no seatbelts
                                                            // This will return a value, but it is probably nonsense

let good_prediction = fit.y(50.0); // This is within the range of the data, so it is probably reasonable
                                   // This will return Ok(value)

//
// Maybe we need to make sure our predictions are good enough
// A covariance matrix sounds terrifying. Because it is.
// Which is why I do it for you - covariance matrices measure uncertainty about your fit
// They can be used to calculate confidence intervals for predictions
// They can also be used to find outliers in your data
let covariance = fit.covariance().expect("Failed to calculate covariance");
let confidence_band = covariance.confidence_band(50.0, Confidence::P95).unwrap(); // 95% confidence band
println!("I am 95% confident that the true value at x=50.0 is between {} and {}", confidence_band.min(), confidence_band.max());
```

-----

Oh dear! I sure do wish I could find which pieces of data are outliers!

```rust

//
// I still don't have any real data, so I'm going to make some up! Again!
polyfit::function!(f(x) = 2 x^2 + 3 x - 5);
let synthetic_data = f.solve_range(0.0..100.0, 1.0).apply_poisson_noise(0.1, None);

//
// Let's add some outliers
// Salt and pepper noise is a simple way to do this; She's good n' spiky
// We will get nice big jumps of +/- 50 in 5% of the data points
let synthetic_data_with_outliers = synthetic_data.apply_salt_pepper_noise(0.05, -50.0, 50.0, None);

//
// Now we can create a curve fit to this data, like before
let fit = MonomialFit::new_auto(
    &synthetic_data_with_outliers,  // The data to fit to
    DegreeBound::Relaxed,            // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
    ScoringMethod::AIC               // The method used to score the fit quality (See [`statistics::ScoringMethod`])
).expect("Failed to create fit");

//
// Now we can find the outliers!
// These are the points that fall outside the 95% confidence interval
// This means that they are outside the range where we expect 95% of the data to fall
let outliers = fit.covariance().unwrap().outliers(Confidence::P95);
```

<!-- cargo-rdme end -->
