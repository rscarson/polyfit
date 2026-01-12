//! Validate your choice of basis using polyfit's basis selection and residual analysis tools.
//! This example creates some sample data from a complex Fourier function with noise,
//! uses `basis_select!` to determine the best basis for fitting the data,
//! fits the data using that basis,
//! and analyzes the residuals to ensure a good fit.
use polyfit::{FourierFit, basis_select, error::Error, plot_residuals, score::Aic, statistics::{Confidence, CvStrategy, DegreeBound}, transforms::{ApplyNoise, Strength}};

fn main() -> Result<(), Error> {
    // Create some sample data
    // This is a nice big complex fourier function with some noise added
    let sample_fn = polyfit::basis::FourierBasis::new_polynomial((0.0, 100.0), &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]).unwrap();
    let data = sample_fn.solve_range(0.0..=100.0, 0.1).apply_normal_noise(Strength::Relative(0.1), None);

    //
    // First let's try basis_select! to pick the best basis for this data
    basis_select!(&data, DegreeBound::Relaxed, &Aic);

    // You'll get output topped by a table like this one:
    // # |             Basis              | Params | Score Weight | Fit Quality | Normality | Rating
    // --|--------------------------------|--------|--------------|-------------|-----------|-----------
    // 1 |                        Fourier |      9 |      100.00% |      98.98% |    91.99% | 76% ☆☆★★★
    // 2 |                       Laguerre |     11 |        0.00% |      69.75% |     0.00% | 33% ☆☆☆☆☆
    // 3 |                       Legendre |     11 |        0.00% |      70.75% |     0.00% | 34% ☆☆☆☆☆
    // --|--------------------------------|--------|--------------|-------------|-----------|-----------
    // 4 |                      Chebyshev |     11 |        0.00% |      70.75% |     0.00% | 34% ☆☆☆☆☆
    // 5 |                    Logarithmic |     11 |        0.00% |      67.63% |     0.00% | 32% ☆☆☆☆☆
    // 6 |          Probabilists' Hermite |      7 |        0.00% |      65.73% |     0.00% | 49% ☆☆☆☆★
    // 7 |            Physicists' Hermite |     11 |        0.00% |      68.45% |     0.00% | 33% ☆☆☆☆☆
    //
    // The testing page on the homepage has more details on interpreting this output, but here the winner is clear:
    // Fourier has a low parameter count, excellent fit quality, and good normality of residuals.
    //
    // You can also see more data below, including a plot you should review to ensure the fit looks good:
    // Fourier: xₛ = T[ 0..100 -> 0..2π ], y(x) = 129.07·cos(4xₛ) + 63.64·sin(4xₛ) + 31.58·cos(3xₛ) + 16.88·sin(3xₛ) + 7.69·cos(2xₛ) + 3.83·sin(2xₛ) + 2.20·cos(xₛ) + 1.50·sin(xₛ) + 0.31
    // Fit R²: 0.9898, Residuals Normality p-value: 0.9199
    // Wrote plot to target\plot_output\fourier_examples_validating_your_choice_of_basis.rs_line_11.png    

    let fit = FourierFit::new_auto(
        &data,
        DegreeBound::Relaxed,
        &Aic,
    )?;

    // We should also check the residuals plot to ensure there are no obvious patterns in the residuals.
    // Look for:
    // - Small residuals compared to the data range
    // - Blue trendline fairly flat and centered around zero, no visible patterns in the residuals
    // - Residuals evenly distributed above and below zero
    plot_residuals!(fit);

    // Finally we can compute the folded RMSE to get an idea of the uncertainty in our predictions
    // `MinimizeBias` will prefer better average performance over optimizing best-case performance
    let uncertain_value = fit.folded_rmse(CvStrategy::MinimizeBias);
    println!("Folded RMSE: {}", uncertain_value.confidence_band(Confidence::P95));

    // That value is small compared to the range of the data (~±150), so we can be confident in our fit!
    // And the range is small so the model will generalize well to new data within the same range.
    Ok(())
}