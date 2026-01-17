//! A simple example of fitting a polynomial to data.
//!
//! This example loads some sample data from a JSON file, fits a Chebyshev polynomial to it,
//! prunes insignificant terms, and prints the resulting polynomial.
//!
use polyfit::{
    error::Error,
    score::Aic,
    statistics::{Confidence, DegreeBound},
    transforms::{ApplyNoise, Strength},
    ChebyshevFit,
};

fn main() -> Result<(), Error> {
    //
    // Load data from the sample file
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();
    let data = data.apply_normal_noise(Strength::Relative(0.5), None);

    let wigglyboi = polyfit::basis::FourierBasis::new_polynomial(
        (0.0, 100.0),
        &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
    )
    .unwrap();
    let wigglydats = wigglyboi
        .solve_range(0.0..=100.0, 0.1)
        .apply_normal_noise(Strength::Relative(0.1), None);
    polyfit::basis_select!(&wigglydats, DegreeBound::Relaxed, &Aic);
    let wigglyfit = polyfit::FourierFit::new_auto(&wigglydats, DegreeBound::Relaxed, &Aic)?;
    let uncertain_value = wigglyfit.folded_rmse(polyfit::statistics::CvStrategy::MinimizeBias);

    println!(
        "Folded RMSE: {}",
        uncertain_value.confidence_band(Confidence::P95)
    );
    polyfit::plot_residuals!(wigglyfit, prefix = "wiggly example");

    //
    // Let's use a Chebyshev basis to fit this data! There might be a big range of X values,
    // so Chebyshev polynomials will help keep things stable.
    let mut fit = ChebyshevFit::new_auto(
        &data,                // The data to fit to
        DegreeBound::Relaxed, // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
        &Aic,                 // How to score the fits (See [`crate::score`])
    )?;

    //
    // Sometimes a least-squares fit will include terms that don't really contribute much.
    // We can prune these insignificant terms to get a simpler polynomial.
    // Here we prune terms we are 95% confident are insignificant.
    fit.prune_insignificant(Confidence::P95)?;

    //
    // Now we can print out the polynomial and some stats about the fit
    let r_squared = fit.r_squared(None);
    if r_squared < 0.9 {
        eprintln!("Warning: Low R² - noisy data or poor fit: {}", r_squared);
    }

    //
    // I want to print the equation, but Chebyshev polynomials are ugly - let's convert to monomial form
    let poly = fit.as_monomial()?;
    println!("Fitted Polynomial: {poly}");

    //
    // If I was using this fit for plotting (that's why I wrote the crate, actually),
    // I can generate points from the fit easily:
    let _fitted_points = fit.solve_range(0.0..=99.0, 1.0)?;

    //
    // Or I can evaluate the fit at specific points:
    let _fitted_points = fit.solve([42.0, 76.0, 89.9])?;

    //
    // If you have the `plotting` feature enabled, you can plot the fit and data to a PNG file:
    // The file will be created in the `target/plot_output/` directory with a unique name.
    #[cfg(feature = "plotting")]
    polyfit::plot!(fit, prefix = "example");

    Ok(())
}

/// So this is actually how I generated the sample data file used in all the examples.
/// If you want to reshuffle the data or generate your own, you can run this function.
#[allow(dead_code)]
#[cfg(feature = "transforms")]
fn gen_sample_data() {
    use polyfit::function;
    use polyfit::transforms::{ApplyNoise, Strength};
    function!(y(x) = 5.3 x^2 + 3.0 x + 1.0);
    let data = y
        .solve_range(0.0..=100.0, 1.0)
        .apply_normal_noise(Strength::Relative(0.1), None)
        .apply_poisson_noise(Strength::Absolute(0.05), None);

    // data to json
    let json = serde_json::to_string(&data).unwrap();
    std::fs::write("examples/sample_data.json", json).unwrap();
}
