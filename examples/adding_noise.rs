//!
//! It can be hard to test fitting and analysis functions without some sample data.
//!
//! Here I show how you can generate some data with noise and events that simulates
//! real-world data, like RF signals.
//!
//! Note this isn't a robust simulation of real-world data, just something to generate
//! data for testing.
//!
use polyfit::{
    basis::FourierBasis,
    error::Error,
    plot,
    score::Aic,
    statistics::DegreeBound,
    transforms::{ApplyNoise, Strength},
    ChebyshevFit,
};

fn main() -> Result<(), Error> {
    //
    // Let's start with a fourier function, which is periodic
    // That makes it a good candidate for generating some pseudo-rf data
    let function = FourierBasis::new_polynomial((0.0, 100.0), &[0.0, 5.0, 3.5])?;
    let data = function.solve_range(0.0..=100.0, 1.0);

    //
    // Now let's add a couple of layers of background noise
    // We will use correlated normal noise, and vary the correlation length a bit
    let with_bg_noise = data
        .apply_correlated_noise(Strength::Absolute(20.0), 0.3, None)
        .apply_correlated_noise(Strength::Absolute(40.0), 0.15, None)
        .apply_correlated_noise(Strength::Absolute(80.0), 0.9, None);

    //
    // Now we need some events in the data - poisson spikes are a good model for this
    let with_events = with_bg_noise
        .apply_poisson_noise(0.1, None)
        .apply_poisson_noise(0.5, None);

    //
    // Now some salt and pepper noise to simulate random glitches
    // This will randomly replace about 5% of the data with -50 and 50
    let with_events = with_events.apply_salt_pepper_noise(0.05, -50.0, 50.0, None);

    //
    // Now we have some pseudo-rf data with background noise and events
    // We can use it for testing our fitting and analysis functions
    let fit = ChebyshevFit::new_auto(&with_events, DegreeBound::Relaxed, &Aic)?;
    plot!(fit);

    Ok(())
}
