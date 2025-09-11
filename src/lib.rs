//! # Polyfit
//! ## I learned linear algebra so you don't have to
//!
//! Statistics is hard, and linear regression is made entirely out of footguns; Curve fitting might be simple in theory, but there sure is a LOT of the theory.
//!
//! This library is designed for developers who need to make use of the plotting or predictive powers of a curve fit without needing to worry about Huber loss,
//! D'Agostino, or what on earth a kurtosis is
//!
//! I provide a set a tools designed to help you:
//! - Select the right kind (basis) of polynomial for your data
//! - Automatically determine the optimal degree of the polynomial
//! - Make predictions and get confidence values based on it
//! - Write easy to understand tests to confirm function
//!   - Tests even plot the data and functions for you on failure! (`plotting` feature)
//!
//! This crate provides tools for fitting, evaluating, and displaying polynomials
//! using multiple basis types (e.g., monomial, Chebyshev), along with testing
//! utilities and statistical scoring for automatic model selection.
//!
//! It is primarily designed for use by people who need to leverage polynomial fitting
//! without delving into the underlying mathematics.
//!
//! The simplest use-case is to find a mathematical function to help approximate a set of data:
//! ```rust
//! # use polyfit::{MonomialFit, test::Noise, statistics::{DegreeBound, ScoringMethod}, assert_r_squared};
//!
//! polyfit::function!(f(x) = 2 x^2 + 3 x - 5);
//! let synthetic_data = f.solve_range(0.0..100.0, 1.0).poisson_noise(0.1).unwrap();
//!
//! let fit = MonomialFit::new_auto(&synthetic_data, DegreeBound::Relaxed, ScoringMethod::AIC).expect("Failed to create fit");
//!
//! // If the assertion fails, a plot will be generated showing you exactly what went wrong!
//! assert_r_squared!(fit);
//! ```
//!
//! # Core Concepts
//! - A [`Polynomial`] is a a mathematical function returning a value `y` for a given input `x`.
//!     - It is considered `canonical`, or correct, for any set of input values
//! - A [`CurveFit`] is a model that approximates a polynomial function based on a set of data points.
//!     - It is used to find the best-fitting polynomial for a given dataset.
//!     - It is only valid within the `x` range of the provided data points.
//!     - It has a number of methods built in for evaluating the fit quality and making predictions.
//! - A [`basis::Basis`] is the method used to represent a polynomial:
//!     - [`MonomialPolynomial`] or [`MonomialFit`] is the simplest case `y = Ax^2 + Bx + C`. This is the `default` basis.
//!         - It is unsuitable for very large degree polynomials - try [`ChebyshevFit`] instead.
//!     - [`ChebyshevFit`] is a more advanced basis that can represent higher degree polynomials more efficiently.
//!         - It normalizes the input data to improve numerical stability and accuracy.
//! - The **degree** of a polynomial is the highest power of the variable `x` in the polynomial expression.
//!     - It is how wiggly the line is
//!     - Use [`CurveFit::new_auto`] to choose this for you
//!         - [`statistics::ScoringMethod::AIC`] is a good default choice. It is more lenient and may select a higher degree
//!         - [`statistics::ScoringMethod::BIC`] is a more conservative choice. It penalizes higher degrees more heavily.
//!
//! # Implementation Details
//!
//! This crate is implemented in Rust and makes use of the `nalgebra` library for linear algebra operations.
//! It is designed to be efficient and easy to use, with a focus on providing a high-level interface for polynomial fitting.
//!
//! # Testing utilities
//!
//! This crate includes a set of testing utilities to facilitate the development and validation of polynomial fitting models. See [`test`].
//!
//! This includes a set of macros that can plot your functions and curve fits for visual confirmation of what goes wrong.
//!
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::needless_range_loop)] // The worst clippy lint
#![allow(clippy::cast_precision_loss)] // I don't care about this one
#![allow(clippy::similar_names)] //       Clippy does not get to decide what names are similar
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod test;

#[cfg(feature = "plotting")]
#[cfg_attr(docsrs, doc(cfg(feature = "plotting")))]
pub mod plot;

#[cfg(feature = "transforms")]
#[cfg_attr(docsrs, doc(cfg(feature = "transforms")))]
pub mod transforms;

pub mod basis;
pub mod display;
pub mod error;
pub mod statistics;
pub mod value;

mod fit;
mod polynomial;

pub use fit::*;
pub use polynomial::{MonomialPolynomial, Polynomial};

pub use nalgebra;
