use polyfit::{
    error::Error,
    statistics::{Confidence, DegreeBound, ScoringMethod, Tolerance},
    ChebyshevFit,
};

fn main() -> Result<(), Error> {
    //
    // Load data from the sample file
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();

    //
    // Let's use a Chebyshev basis to fit this data! There might be a big range of X values,
    // so Chebyshev polynomials will help keep things stable.
    let mut fit = ChebyshevFit::new_auto(
        &data,                // The data to fit to
        DegreeBound::Relaxed, // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
        ScoringMethod::AIC,   // How to score the fits (See [`statistics::ScoringMethod`])
    )?;

    //
    // Sometimes a least-squares fit will include terms that don't really contribute much.
    // We can prune these insignificant terms to get a simpler polynomial.
    // Here we prune terms we are 95% confident are insignificant.
    fit.prune_insignificant(Confidence::P95)?;

    //
    // Now we can print out the polynomial and some stats about the fit
    let r_squared = fit.r_squared(fit.data());
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
    // Or maybe you want to detect outliers?
    //
    // This will get a range, called a confidence band, where we expect 99% of points to fall within.
    // Any points outside this band are potential outliers.
    // But - the data may have noise, so we can specify a noise tolerance to avoid flagging points
    // In this case I happen to know the data has about 10% noise, so I set a tolerance of 0.1
    // Mathematically this treats noise_tolerance as a fraction of the standard deviation of the data to tolerate.
    let covariance = fit.covariance()?;
    for (x, y, confidence_band) in
        covariance.outliers(Confidence::P99, Some(Tolerance::Relative(0.1)))?
    {
        println!(
            "x={x} may be an outlier. y={y} outside confidence range: {} - {}",
            confidence_band.min(),
            confidence_band.max()
        );
    }

    //
    // If you have the `plotting` feature enabled, you can plot the fit and data to a PNG file:
    // The file will be created in the `target/plot_output/` directory with a unique name.
    #[cfg(feature = "plotting")]
    polyfit::plot!(
        fit,
        prefix = "example",
        confidence = Confidence::P99,
        noise_tolerance = Tolerance::Relative(0.1)
    );

    Ok(())
}

#[allow(dead_code)]
#[cfg(feature = "transforms")]
fn gen_sample_data() {
    use polyfit::function;
    use polyfit::transforms::ApplyNoise;
    function!(y(x) = 5.3 x^2 + 3.0 x + 1.0);
    let data = y
        .solve_range(0.0..=100.0, 1.0)
        .apply_normal_noise(Tolerance::Relative(0.1), None)
        .apply_poisson_noise(0.05, None);

    // data to json
    let json = serde_json::to_string(&data).unwrap();
    std::fs::write("examples/sample_data.json", json).unwrap();
}
