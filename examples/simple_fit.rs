use polyfit::{
    error::Error,
    statistics::{DegreeBound, ScoringMethod},
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
    let fit = ChebyshevFit::new_auto(
        &data,                // The data to fit to
        DegreeBound::Relaxed, // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
        ScoringMethod::AIC,   // How to score the fits (See [`statistics::ScoringMethod`])
    )?;

    Ok(())
}
