//!
//! This example demonstrates how to use scaling transforms on data.
//!
//! I also show how to use transformations without the convenience traits.
//!
use polyfit::{
    error::Error,
    function, plot, plot_filename,
    plotting::{
        plotters::{Plot, Root, Split},
        PlotOptions, PlottingElement,
    },
    transforms::{ApplyScale, ScaleTransform, Transform},
};

fn main() -> Result<(), Error> {
    //
    // Let's grab our sample data again
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();

    //
    // For example if we pretend this is temperature sensor data
    // The sensor outputs temp in Fahrenheit but we want Celcius
    // because we realized it's the superior unit of measurement
    //
    // We can use a polynomial to model this relationship (5/9)x - 32(5/9)
    // or approximately 0.5556x - 17.7778
    function!(celcius(f) = 0.5556 f - 17.7778);
    let celcius_data = data.clone().apply_polynomial_scale(&celcius);

    // Let's visualize the difference
    let orig_data = PlottingElement::from_data(data.iter().copied(), Some("°F Data".to_string()));
    let tx_data = PlottingElement::from_data(celcius_data.into_iter(), Some("°C Data".to_string()));
    plot!([tx_data, orig_data]);

    //
    // Now instead let's say this is volts / seconds from some experiment
    // Let's convert to kilovolts to keep things numerically stable
    // Because I've reused this data for a lot of examples and the values are quite large
    let mut voltage_data = data.clone().apply_linear_scale(1e-3);

    // Let's also convert time to minutes
    // The trait applies to y, so let's apply it manually
    let transformation = ScaleTransform::Linear(1.0 / 60.0);
    transformation.apply(voltage_data.iter_mut().map(|(x, _)| x));

    //
    // Let's plot these side-by-side
    // First we need to create a split root
    let mut options = PlotOptions {
        title: "Voltage over Time".to_string(),
        size: (1280, 480),
        ..Default::default()
    };
    let filename = plot_filename!(Some("voltage_time_comparison"));
    let root = Root::new_split(&filename, options.size, Split::Horizontal(2));

    //
    // Plot the left side
    options.x_label = Some("Time (min)".to_string());
    options.y_label = Some("Voltage (kV)".to_string());
    let plot = Plot::new(&root[0], options.clone(), &voltage_data).unwrap();
    plot.finish().unwrap();

    //
    // And now the right side
    options.x_label = Some("Time (s)".to_string());
    options.y_label = Some("Voltage (V)".to_string());
    let plot = Plot::new(&root[1], options, &data).unwrap();
    plot.finish().unwrap();

    Ok(())
}
