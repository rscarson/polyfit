use polyfit::{
    basis::CriticalPoint, error::Error, plot, score::Aic, statistics::DegreeBound, FourierFit,
};

fn main() -> Result<(), Error> {
    //
    // Let's get a fit to some data
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();
    let data = &data[..30]; // To simplify let's just use the first 30 points

    let fit = FourierFit::new_auto(data, DegreeBound::Relaxed, &Aic)?;
    println!("Fourier fit:\n{fit}\n");

    // The derivative function is the rate of change of the original function
    // Think of velocity being the derivative of position with respect to time
    let dx = fit.as_polynomial().derivative()?;
    println!("Derivative of the fit:\n{dx}\n");
    plot!([fit, dx], { title: "Fourier Fit and Derivative".to_string() });

    //
    // For the sake of this example, let's say this is a measurement of power, in watts, over time, in seconds.
    // Now let's say we want to know how quickly the power is changing at a specific moment in time.
    // The derivative of the function gives us the rate of change at any point, so lets get that
    let rate_of_change_at_15s = dx.y(15.0);
    println!(
        "Rate of change of power at 15 seconds: {:.2} kilowatts/second",
        rate_of_change_at_15s / 1000.0
    );

    //
    // We can also use calculus to find the total energy used in the first minute (60 seconds).
    // This is just the area under the curve - so we need to integrate the function from 0 to 60.
    // The result will be in watt-seconds, or joules.
    let energy_joules = fit.area_under_curve(0.0, 20.0, None)?;
    println!(
        "Total energy used in the first 20 seconds: {:.2} kilojoules",
        energy_joules / 1000.0
    );

    //
    // It would also be to find the critical points - where the curve changes from concave up to concave down, or vice versa.
    // this can tell us when the rate of change of power is increasing or decreasing.
    let critical_points = fit.critical_points()?;
    println!("\nCritical points:");
    for point in &critical_points {
        if point.coords().1 < 0.0 {
            // Ignore negative power readings - they don't make sense in this context
            continue;
        }

        match point {
            CriticalPoint::Minima(x, y) => {
                println!(
                    "  Local minimum at time {x:.2} seconds, power {y:.2} kilowatts",
                    y = *y / 1000.0
                );
            }
            CriticalPoint::Maxima(x, y) => {
                println!(
                    "  Local maximum at time {x:.2} seconds, power {y:.2} kilowatts",
                    y = *y / 1000.0
                );
            }
            CriticalPoint::Inflection(x, y) => {
                println!(
                    "  Inflection point at time {x:.2} seconds, power {y:.2} kilowatts",
                    y = *y / 1000.0
                );
            }
        }
    }

    //
    // For stuff like this, it's really helpful to have a plot to visualize it all instead of debugging numbers
    // That's why I built plotting right into this library
    //
    // We can use the plot! macro to quickly generate a plot
    // And we'll include the critical points as markers
    let crit_markers = CriticalPoint::as_plotting_element(&critical_points);
    plot!([
        fit,
        dx,
        crit_markers
    ], {
        title: "Fourier Fit with Calculus".to_string(),
        x_label: Some("Time (seconds)".to_string()),
        y_label: Some("Power (watts)".to_string()),
    });

    Ok(())
}
