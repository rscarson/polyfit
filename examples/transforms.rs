//!
//! Example showing data transformations.
//!
use polyfit::{
    error::Error, plot_filename, plotting::plotters, transforms::ApplyNormalization, MonomialFit,
};

fn main() -> Result<(), Error> {
    //
    // We'll make some plots to show the effect of different transformations on data.
    // Let's start with somewhere to put em
    let filename = plot_filename!(Some("transforms"));
    let roots = plotters::Root::new_split(&filename, (800, 1800), plotters::Split::Vertical(3));

    //
    // Let's grab our sample data again
    // This time we'll just plot the raw data first
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();
    plotters::plot_data(&roots[0], &data, 0.0..100.0, "Original Data").unwrap();

    //
    // To start let's see how good a fit we can get with no transformations
    // For the purpose of the demonstration I used the least stable basis and
    // massively overfit the data to exaggerate the effect
    let fit = MonomialFit::new(&data, 15)?;
    let r2 = fit.r_squared(None);
    println!("R² with no transformations: {r2:.4}");

    //
    // Let's do a transformation called regularization.
    // This will subtract make the mean = 0  and standard deviation = 1
    // This can help with numerical stability when fitting.
    // It's also useful if you want to compare fits on different datasets.
    //
    // This is sometimes called "Z-score normalization" or "standardization",
    // since it makes the data have a "standard" normal distribution.
    let regularized = data.apply_z_score_normalization();
    plotters::plot_data(&roots[1], &regularized, 0.0..100.0, "Z-score Normalization").unwrap();

    //
    // Let's also scale the x values to [0, 1] range.
    // This can help with numerical stability when fitting.
    let scaled = regularized.apply_domain_normalization(0.0, 1.0);
    plotters::plot_data(&roots[2], &scaled, 0.0..1.0, "Domain Normalization [0, 1]").unwrap();

    // Now let's see how good a fit we can get with these transformations
    let fit = MonomialFit::new(&scaled, 15)?;
    let r2_transformed = fit.r_squared(None);
    println!("R² with transformations: {r2_transformed:.4}");

    // R² in practice should be identical under these transformations
    // (R² is invariant under linear transformations)
    // But in many cases, including here, overfitting, numerical instability, and
    // badly behaved data can prevent a good qualitative fit.
    //
    // So here there actually should be a noticeable improvement:
    assert!(r2_transformed > r2);
    println!("Improvement in R²: {:.4}%", (r2_transformed - r2) * 100.0);

    Ok(())
}
