//! Debug utilities for plotting fits and curves
pub use plotters;

mod core;
pub use core::*;

mod element;
pub use element::*;

mod palette;
use palette::Palettes;

/// Plots a `CurveFit` or `Polynomial` plus one or more polynomials or fits to a PNG file.
///
/// At the moment, converts all input types to `f64` for plotting purposes.
///
/// # Syntax
/// ```ignore
/// plot!(
///     fit,                     // Required: a `CurveFit` or `PlottingElement`
///     functions = [element1, element2, … ]   // Optional: one or more additional `CurveFit` or polynomials
///     , title = "My Plot"      // Optional: custom title (default: "Graph Output")
///     , x_range = (start, end) // Optional: x-axis range (default: from function's x_range)
///     , confidence = Confidence::P95 // Optional: confidence level (default: P95)
///     , size = (width, height) // Optional: image size in pixels (default: (640, 480))
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
/// - The macro requires that `PlottingElement` implements `From` for the types passed in.
/// - Confidence levels default to `P95` unless overridden.
/// - Image size defaults to `(640, 480)` pixels unless specified.
#[macro_export]
macro_rules! plot {
    (
        $function:expr
        $(, functions = [ $($element:expr),+ ] )?
        $(, title = $title:expr)?
        $(, x_range = $x_range:expr)?
        $(, y_range = $y_range:expr)?
        $(, confidence = $confidence:expr)?
        $(, size = ($width:expr, $height:expr))?
    ) => { #[allow(unused)] {
        use $crate::value::CoordExt;

        //
        // Prep arguments
        let mut size = (640, 480); $( size = ($width, $height); )?
        let mut confidence = $crate::statistics::Confidence::P95; $( confidence = $confidence; )?
        let mut title = "Graph Output".to_string(); $( title = $title.to_string(); )?
        let function = $crate::plot::PlottingElement::from($function);
        let x_range = function.x_range(); $( x_range = Some($x_range); )?

        //
        // Prep the backend
        let path = $crate::plot_filename!();
        let backend = $crate::plot::plotters::backend::BitMapBackend::new(&path, size);
        let root = $crate::plot::plotters::prelude::IntoDrawingArea::into_drawing_area(backend);
        root.fill(&$crate::plot::plotters::prelude::WHITE).expect("Failed to fill drawing area");

        let x_range = x_range.expect("x_range must be provided either via a CurveFit or the `x_range` argument");
        let range = $crate::value::ValueRange::new_unit(x_range.start, x_range.end);
        let data = function.solve(range);
        let x = data.x();
        let mut y_range = data.y_range().expect("Range was empty!"); $( y_range = $y_range; )?

        //
        // Build plot
        //

        let mut plot = $crate::plot::Plot::new(&root, &title, x_range, y_range).expect("Failed to create plot");
        plot = plot.with_element(&function, confidence, &x).expect("Failed to add main element to plot");
        $(
            $(
                let e = $crate::plot::PlottingElement::from($element);
                plot = plot.with_element(&e, confidence, &x).expect("Failed to add element to plot");
            )*
        )?

        plot.build().expect("Failed to build plot");
        println!("Wrote plot to {}", path.display());
    }};
}

/// Generate a filename for a plot: `target/plot_output/{file}_line_{line}_{datetime}.png`
#[macro_export]
macro_rules! plot_filename {
    () => {{
        let file = file!().replace(['/', '\\'], "_");
        let line = line!();
        let datetime = ::std::time::SystemTime::now()
            .duration_since(::std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let plots_dir = $crate::plot::Plot::plots_dir();
        let _ = std::fs::create_dir_all(&plots_dir);

        let filename = format!("{file}_line_{line}_{datetime}.png");
        plots_dir.join(filename)
    }};
}
