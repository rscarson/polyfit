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
    FourierFit, basis::FourierBasis, error::Error, plot, score::Aic, statistics::DegreeBound, transforms::{ApplyNoise, Strength}
};

fn main() -> Result<(), Error> {
    //
    // Let's start with a fourier function, which is periodic
    // That makes it a good candidate for generating some pseudo-rf data
    let function = FourierBasis::new_polynomial((0.0, 100.0), &[0.0, 5.0, 3.5])?;
    println!("Base function: {function}");
    let data = function.solve_range(0.0..=100.0, 1.0);

    plot!(function, { title: "Base function without noise".to_string(), x_range: Some(0.0..100.0) });

    //
    // Now let's add a couple of layers of background noise
    // We will use correlated normal noise, and vary the correlation length a bit
    let with_bg_noise = data.apply_normal_noise(Strength::Absolute(1.5), None);
    let with_bg_noise = with_bg_noise.apply_correlated_noise(Strength::Absolute(1.0), 0.75, None);
    plot!(with_bg_noise, { title: "With background noise".to_string() });

    //
    // Now we need some events in the data - poisson spikes are a good model for this
    let with_events = with_bg_noise.apply_poisson_noise(0.3, true, None);
    plot!(with_events, { title: "With events".to_string() });

    //
    // Now some salt and pepper noise to simulate random glitches
    // This will randomly replace about 5% of the data with -50 and 50
    let with_glitches = with_events.apply_salt_pepper_noise(0.01, -50.0, 50.0, None);
    plot!(with_glitches, { title: "With glitches".to_string() });

    //
    // Now we have some pseudo-rf data with background noise and events
    // We can use it for testing our fitting and analysis functions
    let fit = FourierFit::new_auto(&with_glitches, DegreeBound::Relaxed, &Aic)?;
    plot!([fit, function]);

    Ok(())
}
