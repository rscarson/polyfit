//!
//! An example of using `basis_select!` to choose the best basis for fitting data.
//! This example loads some sample data from a JSON file, evaluates several basis options,
//! and fits a polynomial using the best basis according to AIC score.
//!
//! A basis is the set of functions used to build the polynomial. Different bases have different strengths and weaknesses.
//! - Stability: Some bases (like Chebyshev) are more numerically stable for large ranges of x-values.
//! - Fit Quality: Some bases (like Fourier) can fit certain types of data better.
//! - Outliers: Some bases are more robust to outliers in the data, like Logarithmic or Laguerre.
//! - Performance: Some bases are faster to compute than others, like Chebyshev or Legendre.
//!
use polyfit::{
    assert_r_squared, basis_select, error::Error, score::Aic, statistics::DegreeBound,
    LogarithmicFit,
};

fn main() -> Result<(), Error> {
    //
    // Let's load our noisy sample data again
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();

    //
    // Because this is the first time we've used this data, let's run `basis_select!` on it
    // Normally you'd do this from a #[test] function or the built in binary `basis_select` in this crate
    basis_select!(&data, DegreeBound::Relaxed, &Aic);

    //
    // When you run this, you'll see (amonst other output) something like:
    // [ Evaluating 100 data points against 7 basis options ]
    //
    // # |             Basis              | Score Weight | Fit Quality | Normality | Rating
    // --|--------------------------------|--------------|-------------|-----------|-----------
    // 1 |                      Chebyshev |       20.00% |      67.92% |    47.88% | 63% ☆☆☆☆★
    // 2 |                       Legendre |       20.00% |      67.92% |    47.88% | 63% ☆☆☆☆★
    // 3 |          Probabilists' Hermite |       20.00% |      67.92% |    47.88% | 63% ☆☆☆☆★
    // --|--------------------------------|--------------|-------------|-----------|-----------
    // 4 |                       Laguerre |       20.00% |      67.92% |    47.88% | 63% ☆☆☆☆★
    // 5 |            Physicists' Hermite |       20.00% |      67.92% |    47.88% | 63% ☆☆☆☆★
    // 6 |                    Logarithmic |        0.00% |      67.51% |    65.50% | 67% ☆☆☆★★
    // 7 |                        Fourier |        0.00% |      86.77% |     0.00% | 65% ☆☆☆★★

    //
    // It's a confusing table but the important part is the ranking on the left.
    // Here we can see that Chebyshev, and a few others are tied for first place.
    // This makes sense because I used Chebyshev to generate the data!
    //
    //           Adjusted R² (How well the model fits the data)                 Likelihood that the errors are random, and not due to an unerlying pattern
    //   Likihood of being the best based on AIC               \               /            Combined ranking for fit quality, and normality of residuals
    //                                          \               |             |            /
    // # |             Basis              | Score Weight | Fit Quality | Normality | Rating
    // --|--------------------------------|--------------|-------------|-----------|-----------
    // 1 |                      Chebyshev |       20.00% |      67.92% |    47.88% | 63% ☆☆☆☆★
    //
    // But 2 other things stand out here:
    // - Although they produces worse scores with AIC, fourier has a better fit quality (R²) than Chebyshev1
    // - Logarithmic has a better normality score than Chebyshev, meaning the errors are more normally distributed
    //   That means the logarithmic fit is less likely to be overfitting the data, so it might generalize better to new data
    //
    // That being said, ideally you'd collect more data and run this a few times to see if the results are consistent.
    // Today we'll use Logarithmic because I want to show you how to use a different basis.
    //
    // I happen to know the data is a bit noisy, so I'll use k-fold cross validation to help avoid overfitting.
    // 5-fold regression means I split the data into 5 parts, fit to 4/5 of it, and test on the remaining 1/5.
    // This is repeated 5 times, each time with a different 1/5 held out for testing.
    let fit = LogarithmicFit::new_kfold_cross_validated(&data, 5, DegreeBound::Relaxed, &Aic)?;

    //
    // And of course don't forget to test!
    // Here we assert that the fit has an R² of at least 90%
    // If this fails, and the `plotting` feature is enabled, a plot will be generated to show you what went wrong.
    println!("\n\n");
    assert_r_squared!(fit);
    println!("Fitted Logarithmic Polynomial:\n  {fit}");
    Ok(())
}
