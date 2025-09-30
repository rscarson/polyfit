//! Debug utilities for plotting fits and curves
//!
//! Mainly used through the [`crate::plot!`] macro.
//! - The asserts built-in will use this on failure if the `plotting` feature is active!
//!
//! You can also use the [`Plot`] struct directly for more control (I also expose [`plotters`] directly)
//!
//! The [`crate::plot_filename!`] macro can be used to generate a unique filename for each plot.
//! - This is how the asserts get a filename on failure
pub use plotters;

mod core;
pub use core::*;

mod element;
pub use element::*;

mod builder;
pub use builder::PlotBuilder;

mod palette;
pub use palette::ColorSource;

/// Plots a `CurveFit` or `Polynomial` plus one or more polynomials or fits to a PNG file.
///
/// At the moment, converts all input types to `f64` for plotting purposes.
///
/// # Syntax
/// ```text
/// plot!(
///     fit                                    // Required: a `CurveFit` or `PlottingElement`
///     , functions = [element1, element2, … ] // Optional: one or more additional `CurveFit` or polynomials
///     , title = "My Plot"                    // Optional: custom title (default: "Graph Output")
///     , x_range = (start, end)               // Optional: x-axis range (default: from function's x_range)
///     , confidence = Confidence::P95         // Optional: confidence level (default: P95)
///     , size = (width, height)               // Optional: image size in pixels (default: (640, 480))
/// );
/// ```
///
/// # Examples
/// ```ignore
/// // Simple plot of a single fit
/// plot!(fit);
///
/// // Plot with multiple polynomials, custom title, and P99 confidence
/// plot!(fit, functions = [poly1, poly2], title = "R² Comparison", confidence = Confidence::P99);
///
/// // Plot with custom size
/// plot!(fit, title = "Scaled Plot", size = (1024, 768));
/// ```
///
/// # Behavior
/// - Automatically generates a filename based on the source file, line number, and timestamp.
/// - Ensures the target directory exists before writing the PNG file.
/// - Prints the path of the generated file to stdout.
///
/// # Notes
/// - Uses the plotters backend for rendering.
/// - Converts all input types to `f64` for plotting.
/// - Confidence levels default to `P95` unless overridden.
/// - Image size defaults to `(640, 480)` pixels unless specified.
#[macro_export]
macro_rules! plot {
    (
        $function:expr
        $(, functions = [ $($extra_function:expr),* ] )?
        $(, title = $title:expr)?
        $(, prefix = $prefix:expr)?
        $(, x_range = $x_range:expr)?
        $(, y_range = $y_range:expr)?
        $(, confidence = $confidence:expr)?
        $(, noise_tolerance = $noise_tolerance:expr)?
        $(, size = ($width:expr, $height:expr))?
        $(, )?
    ) => {{
        use $crate::plot::AsPlottingElement;

        #[allow(unused)] let mut size = (640, 480); $( size = ($width, $height); )?
        #[allow(unused)] let mut confidence = $crate::statistics::Confidence::P95; $( confidence = $confidence; )?
        #[allow(unused)] let mut noise_tolerance = None; $( noise_tolerance = Some($noise_tolerance); )?
        #[allow(unused)] let mut xs = vec![]; $( xs = $crate::value::SteppedValues::new_unit($x_range.start..=$x_range.end).collect(); )?

        let mut svg_buffer = String::new();
        let prime = $function.as_plotting_element(&xs, confidence, noise_tolerance);
        let prime_xs = prime.x_values();
        #[allow(unused)] let mut builder = $crate::plot::PlotBuilder::<$crate::plot::PlottersBackend<_>, _>::from_element(&mut svg_buffer, size, prime).expect("Failed to create plot builder");

        #[allow(unused)]
        if xs.is_empty() {
            xs = prime_xs;
        }

        $( $({
                let element = $extra_function.as_plotting_element(&xs, confidence, noise_tolerance);
                builder.add_element(element);
        })+ )?

        $( builder.title($title); )?
        $( builder.x_range($x_range); )?
        $( builder.y_range($y_range); )?

        builder.build().expect("Failed to build plot");

        #[allow(unused)] let mut prefix: Option<String> = None; $( prefix = Some($prefix.to_string()); )?
        let path = $crate::plot_filename!(prefix);
        $crate::plot::svg2png(&svg_buffer, &path).expect("Failed to write PNG");
        println!("Wrote plot to {}", path.display());
    }};
}

/// Generate a filename for a plot: `target/plot_output/{file}_line_{line}_{datetime}.png`
///
/// Creates the necessary directories if they don't exist.
///
/// `plot_filename!(Some("prefix"))` will prepend `prefix_` to the filename.
#[macro_export]
macro_rules! plot_filename {
    ($prefix:expr) => {{
        let prefix: Option<String> = $prefix.map(|s| s.to_string());
        let prefix = match prefix {
            Some(p) if !p.is_empty() => format!("{p}_"),
            _ => String::new(),
        };

        let file = file!().replace(['/', '\\'], "_");
        let line = line!();
        //  let datetime = ::std::time::SystemTime::now()
        //       .duration_since(::std::time::UNIX_EPOCH)
        //       .unwrap_or_default()
        //       .as_secs();

        let target_dir = ::std::env::var("TARGET_DIR").unwrap_or_else(|_| "target".into());
        let plots_dir = ::std::path::Path::new(&target_dir).join("plot_output");
        let _ = std::fs::create_dir_all(&plots_dir);

        let filename = format!("{prefix}{file}_line_{line}.png");

        plots_dir.join(filename)
    }};
}
