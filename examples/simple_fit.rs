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

#[allow(dead_code)]
#[cfg(feature = "transforms")]
fn gen_sample_data() {
    use polyfit::function;
    use polyfit::transforms::ApplyNoise;
    function!(y(x) = 5.3 x^4 - 2.1 x^3 + 0.5 x^2 + 3.0 x + 1.0);
    let data = y.solve_range(0.0..100.0, 1.0).apply_normal_noise(0.1, None);

    // data to json
    let json = serde_json::to_string(&data).unwrap();
    std::fs::write("examples/sample_data.json", json).unwrap();
}
