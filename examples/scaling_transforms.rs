//!
//! This example demonstrates how to use scaling transforms on data.
//!
//! I also show how to use transformations without the convenience traits.
//!
use polyfit::{
    error::Error,
    function, plot,
    transforms::{ApplyScale, ScaleTransform, Transform},
};

fn main() -> Result<(), Error> {
    //
    // Let's grab our sample data again
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();

    //
    // Let's suppose these are sensor readings of temperature over time (celsius / seconds)
    // And the sensor voltage follows a quadratic relation:
    // V = 0.01T² + 0.5T
    // We can use a polynomial fit to model this relationship
    let voltage_data = data.apply_polynomial_scale(&function!(0.01 x^2 + 0.5 x));

    // Let's convert to megavolts (MV) to keep things numerically stable
    // Because I've reused this data for a lot of examples and the voltage values are quite large
    let mut voltage_data = voltage_data.apply_linear_scale(1e-6);

    // Let's also convert time to minutes
    // The trait applies to y, so let's apply it manually
    let transformation = ScaleTransform::Linear(1.0 / 60.0);
    transformation.apply(voltage_data.iter_mut().map(|(x, _)| x));

    plot!(voltage_data);

    Ok(())
}
