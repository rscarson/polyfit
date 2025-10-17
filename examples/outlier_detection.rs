//!
//! This example shows how to detect outliers in data using a polynomial fit.
//!
use polyfit::{
    error::Error,
    plot,
    plotting::PlottingElement,
    score::Aic,
    statistics::{Confidence, DegreeBound, Tolerance},
    ChebyshevFit,
};

fn main() -> Result<(), Error> {
    //
    // Load data from the sample file
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();

    //
    // Chebyshev is a good general purpose basis for data you don't know much about
    // It is orthogonal, which helps with numerical stability and avoiding overfitting
    let fit = ChebyshevFit::new_auto(
        &data,                // The data to fit to
        DegreeBound::Relaxed, // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
        &Aic,                 // How to score the fits (See [`crate::score`])
    )?;

    //
    // Get the outliers - a covariance object can give us outliers based on a confidence band
    //
    // For this example let's say we used a sensor that has a known error of +- 10%
    // We could specify Tolerance::Measurement(0.1) here to account for this
    //
    // In this case I happen to know the data has around ~10% noise by variance, so I'll use that
    let covariance = fit.covariance()?;
    let outliers = covariance.outliers(Confidence::P99, Some(Tolerance::Variance(0.1)))?;

    //
    // Let's plot the outliers
    let points = PlottingElement::from_outliers(outliers.into_iter());
    plot!([fit, points], {
        // Make sure the generated error bars match the outlier detection parameters
        confidence: Confidence::P99,
        noise_tolerance: Some(Tolerance::Variance(0.1)),
    });

    Ok(())
}
