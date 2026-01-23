//!
//! So I actually use my plotting feature to generate the logo for the crate
//! This is a tiny 50x50 pixel image of a fancy Fourier curve with its first and second derivatives
//!
use polyfit::{basis::FourierBasis, error::Error, plot};

fn main() -> Result<(), Error> {
    let fancy_curve = FourierBasis::new_polynomial((0.0, 100.0), &[1.5, 3.0, 4.5, 6.0, 7.5])?;
    let dx = fancy_curve.derivative()?;
    let ddx = dx.derivative()?;

    plot!([ddx, dx, fancy_curve], {
        title: "".to_string(),
        size: (50, 50),
        x_range: Some(0.0..100.0),

        hide_legend: true,
        x_axis_labels: Some(0),
        y_axis_labels: Some(0),
    }, prefix = "logo");

    plot!([ddx, dx], {
        title: "".to_string(),
        size: (16, 16),
        margins: Some(1),
        x_range: Some(0.0..100.0),

        hide_legend: true,
        x_axis_labels: Some(0),
        y_axis_labels: Some(0),
    }, prefix = "icon");

    Ok(())
}
