//!
//! This example shows how to detect outliers in data using a polynomial fit.
//!
use polyfit::{
    plot, plot_filename,
    plotting::{
        plotters::{Plot, Root, Split},
        PlotOptions, PlottingElement,
    },
    score::Aic,
    statistics::{Confidence, DegreeBound, Tolerance},
    ChebyshevFit,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    // The main way to control outlier detection is via the confidence band used
    // The confidence level is a measure of how much we trust the fit to represent the data
    // A higher confidence level means we want more insurance against a fit not representing the data well!
    //
    // The larger the confidence level, the wider the band, and the fewer outliers will be detected
    let filename = plot_filename!(Some("outlier_detection_confidence_bands"));
    let root = Root::new_split(&filename, (1280, 480), Split::Horizontal(2));
    Plot::new(
        &root[0],
        PlotOptions::default()
            .with_confidence(Confidence::P80)
            .with_title("Narrow Confidence Band (80%)"),
        &fit,
    )?
    .finish()?;
    Plot::new(
        &root[1],
        PlotOptions::default()
            .with_confidence(Confidence::P999)
            .with_title("Wide Confidence Band (99.9%)"),
        &fit,
    )?
    .finish()?;

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
