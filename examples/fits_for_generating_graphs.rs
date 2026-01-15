//!
//! This example shows a few ways to use polyfit for plotting beyond the basic examples used elsewhere.
use polyfit::{
    error::Error,
    plot, plot_filename, plot_residuals,
    plotting::{self, PlottingElement},
    score::Aic,
    statistics::DegreeBound,
    ChebyshevFit,
};

fn main() -> Result<(), Error> {
    //
    // Load data from the sample file
    let data = include_str!("sample_data.json");
    let data: Vec<(f64, f64)> = serde_json::from_str(data).unwrap();

    //
    // Chebyshev is a good general purpose basis for data you don't know much about
    // It is orthogonal, which helps with numerical stability and avoiding overfitting
    let fit = ChebyshevFit::new_auto(
        &data,                // The data to fit to
        DegreeBound::Relaxed, // How picky we are about the degree of the polynomial (See [`statistics::DegreeBound`])
        &Aic,                 // How to score the fits (See [`crate::score`])
    )?;

    //
    // So far in other examples I've shown the `plot!(fit)` and `plot!([fit, ...])`
    // It takes in some options - see [`crate::plotting::PlotOptions`] for the full set
    //
    // The options are prefix are optional
    plot!(fit, {
        title: "Chebyshev Fit to Sample Data".to_string(),
        x_label: Some("X Axis".to_string()),
        y_label: Some("Y Axis".to_string()),
    }, prefix = "filename_prefix");

    //
    // But this is just a macro around the underlying plotting functions
    // You can use those directly if you want more control
    // You can even implement your own backend if you want to use a different plotting library or a UI framework
    //
    // Probably have this live in a different function and don't unwrap normally
    let path = plot_filename!(Some("optional_prefix"));
    let options = plotting::PlotOptions::<_>::default();
    let root = plotting::plotters::Root::new(&path, options.size);
    let mut plot =
        plotting::Plot::<plotting::plotters::Backend, _>::new(&root, options, &fit).unwrap();

    //
    // You can add lots of stuff to a plot:
    plot.with_element(
        // Label markers at specific points
        &PlottingElement::Markers(vec![(25.0, 20_000.0, Some("OoogaBooga".to_string()))]),
    )
    .unwrap();
    plot.with_element(
        // Other data sets
        &PlottingElement::Data(
            vec![(10.0, 20.0), (20.0, 5.0), (30.0, 15.0)],
            Some("My neat data".to_string()),
        ),
    )
    .unwrap();

    // Finally, finish the plot and write it to disk
    plot.finish().unwrap();
    drop(root); // For good luck

    //
    // Besides the one macro I also included `plot_residuals!` for convenience
    // This will plot the residuals of the fit to the data
    // It takes the same options as `plot!`
    //
    // Residuals are just the difference between the data points and the fit at those points
    // It's a wrongness plot
    plot_residuals!(fit);

    //
    // The last part I want to show is

    Ok(())
}
