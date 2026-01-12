//!
//! This example demonstrates the benefits of using an orthogonal basis for polynomial fitting.
//! We generate a noisy periodic dataset and fit it using both Chebyshev (orthogonal)
//!
//! An orthogonal basis is a set of functions where each pair of different functions is orthogonal under some inner product.
//! But that's boring - what does it mean in practice?
//!
//! 1. **Numerical Stability**: Orthogonal bases reduce numerical instability, meaning that calculations are less likely to be affected by rounding errors.
//! 2. **Reduced Overfitting**: Orthogonal bases help in reducing overfitting, as each basis function captures unique aspects of the data without redundancy.
//! 3. **Improved Interpretability**: Coefficients in an orthogonal basis can be interpreted independently, making it easier to understand the contribution of each basis function.
//!
use polyfit::{
    basis::{ChebyshevBasis, FourierBasis},
    error::Error,
    plot,
    score::Aic,
    statistics::DegreeBound,
    transforms::{ApplyNoise, Strength},
    ChebyshevFit, FourierFit,
};

fn main() -> Result<(), Error> {
    const SEED: u64 = 42;

    //
    // Let's generate some REAL gnarly data
    // We will add some gaussian (normal) noise (in the Fourier curve, this looks like high frequency noise - tiny wiggles)
    // and some Poisson noise (this looks like outliers - occasional big jumps)
    let f = FourierBasis::new_polynomial((0.0, 1000.0), &[6.0, 0.5, 0.1, 0.01, 2.7, 8.9, 1.2])?;
    let clean_data = f.solve_range(0.0..=1000.0, 1.0);
    let data = clean_data
        .clone()
        .apply_normal_noise(Strength::Relative(5.0), Some(SEED))
        .apply_poisson_noise(5.0, true, Some(SEED));

    //
    // Let's do a chebyshev fit first, just to see how bad it is
    // Chebyshev is Orthogonal - this means it is less likely to overfit and it is more numerically stable
    // This means it should do ok here
    let cheb_fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic)?;
    println!("Chebyshev fit:\n{cheb_fit}\n");
    println!(
        "R² of Chebyshev fit: {}\n",
        cheb_fit.as_polynomial().r_squared(&clean_data)
    );

    //
    // Now instead let's try something else
    // Fourier is very good at capturing periodic signals, so it should do better here
    let fourier_fit = FourierFit::new_auto(&data, DegreeBound::Relaxed, &Aic)?;
    println!("Fourier fit:\n{fourier_fit}\n");
    println!(
        "R² of Fourier fit: {}\n",
        fourier_fit.as_polynomial().r_squared(&clean_data)
    );

    //
    // But we can't get a monomial representation of Fourier basis functions
    // So we can't easily compare it to other fits. But we can project it into
    // a basis that can! Orthogonal projection is far more stable than regular
    // polynomial fitting, so we can use a higher degree here without worrying
    // about overfitting or numerical instability.
    // We will use 3(degree of fourier fit) as our degree here; this is enough
    // to capture the same complexity as the Fourier fit, but not enough to
    // start overfitting.
    let cheb_proj = fourier_fit
        .as_polynomial()
        .project_orthogonal::<ChebyshevBasis>(0.0..=1000.0, 3 * fourier_fit.degree())?;
    println!(
        "Projected Fourier fit to Chebyshev basis (degree {}):\n{cheb_proj}\n",
        3 * fourier_fit.degree()
    );
    println!(
        "R² of projected Chebyshev fit: {}\n",
        cheb_proj.r_squared(&clean_data)
    );

    // Finally, let's denoise the projected fit using a spectral energy filter
    // This is a cool thing orthogonal bases let us do - we can look at the `energy`
    // of each coefficient (how much it contributes to the overall signal, vs what is likely noise)
    // and filter out the ones that are likely noise.
    let mut denoised = cheb_proj.clone();
    denoised.spectral_energy_filter()?;

    //
    // When run you'll see these values:
    // R² of Chebyshev fit: 0.0045
    // R² of Fourier fit : 0.2304
    // R² of projected Chebyshev fit: 0.2771
    //
    // The intensity of the noise prevented chebyshev from fitting well
    // The Fourier fit did better, but the projection did best of all!
    // This is because fundamentally Chebyshev and Fourier are both trigonometric
    // making them naturally good at representing periodic signals.

    // Now let's plot them all to see how they look

    let original_signal = (&clean_data[100..=900], "Original Signal");
    let chebyshev_fit = (
        &cheb_fit.solve_range(100.0..=900.0, 1.0)?,
        "Direct Chebyshev Fit (9 candidate models, AICc with Huber loss)",
    );
    let fourier_fit = (
        &fourier_fit.as_polynomial().solve_range(100.0..=900.0, 1.0),
        "Stage 1 Fourier Fit (9 candidate models, AICc with Huber loss)",
    );
    let projected_fit = (
        &cheb_proj.solve_range(100.0..=900.0, 1.0),
        "Re-Projected Fit",
    );

    let denoised_fit = (&denoised.solve_range(100.0..=900.0, 1.0), "Denoised Fit");

    plot!(
        [
            original_signal,
            chebyshev_fit,
            fourier_fit,
            projected_fit,
            denoised_fit
        ],
        prefix = "whats_an_orthogonal"
    );

    Ok(())
}
