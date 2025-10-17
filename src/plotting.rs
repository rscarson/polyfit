//! Debug utilities for plotting fits and curves
//!
//! Mainly used through the [`crate::plot!`] macro.
//! - The asserts built-in will use this on failure if the `plotting` feature is active!
//!
//! You can also use the [`Plot`] struct directly for more control (I also expose [`plotters`] directly)
//!
//! The [`crate::plot_filename!`] macro can be used to generate a unique filename for each plot.
//! - This is how the asserts get a filename on failure

mod backend;

pub use backend::*;

mod element;
pub use element::*;

mod palette;
pub use palette::ColorSource;

/// Options for plotting
#[derive(Debug, Clone)]
pub struct PlotOptions<T>
where
    T: crate::value::Value,
{
    /// Caption for the plot
    pub title: String,

    /// X-axis label
    pub x_label: Option<String>,

    /// Y-axis label
    pub y_label: Option<String>,

    /// Size of the output image in pixels
    pub size: (u32, u32),

    /// X-axis range
    pub x_range: Option<std::ops::Range<T>>,

    /// Y-axis range
    pub y_range: Option<std::ops::Range<T>>,

    /// Confidence level for the error bands on fits
    pub confidence: crate::statistics::Confidence,

    /// Noise tolerance for the error bands on fits
    pub noise_tolerance: Option<crate::statistics::Tolerance<T>>,

    /// Whether to show the legend
    pub hide_legend: bool,

    /// Number of labels to show on the x-axis (if supported by the backend)
    pub x_axis_labels: Option<usize>,

    /// Number of labels to show on the y-axis (if supported by the backend)
    pub y_axis_labels: Option<usize>,
}
impl<T: crate::value::Value> Default for PlotOptions<T> {
    fn default() -> Self {
        Self {
            title: "Graph Output".into(),
            x_label: None,
            y_label: None,
            size: (640, 480),
            x_range: None,
            y_range: None,
            confidence: crate::statistics::Confidence::P95,
            noise_tolerance: None,

            hide_legend: false,
            x_axis_labels: None,
            y_axis_labels: None,
        }
    }
}

/// Helper trait so I could escape macro generics hell
pub trait WithTypeFrom<T: crate::value::Value> {
    /// Get default plot options for this type
    fn options_with_type_from(&self) -> PlotOptions<T>;

    /// Create a new plot with this as the primary function
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn plot_with_type_from<P: PlotBackend>(
        &self,
        root: &P::Root,
        options: PlotOptions<T>,
    ) -> Result<Plot<P, T>, P::Error>;

    /// Adds this element to the given plot
    ///
    /// # Errors
    /// Returns an error if the element cannot be added.
    fn add_to_plot_with_type_from<P: PlotBackend>(
        &self,
        plot: &mut Plot<P, T>,
    ) -> Result<(), P::Error>
    where
        Self: Sized;
}
impl<E, T> WithTypeFrom<T> for E
where
    E: AsPlottingElement<T>,
    T: crate::value::Value,
{
    fn options_with_type_from(&self) -> PlotOptions<T> {
        PlotOptions::default()
    }

    fn plot_with_type_from<P: PlotBackend>(
        &self,
        root: &P::Root,
        options: PlotOptions<T>,
    ) -> Result<Plot<P, T>, P::Error> {
        Plot::new(root, options, self)
    }

    fn add_to_plot_with_type_from<P: PlotBackend>(
        &self,
        plot: &mut Plot<P, T>,
    ) -> Result<(), P::Error>
    where
        Self: Sized,
    {
        plot.with_element(self)?;
        Ok(())
    }
}

/// A plot of one or more elements (fits, functions, etc) using a given backend.
#[allow(clippy::struct_field_names)]
pub struct Plot<P, T>
where
    P: PlotBackend,
    T: crate::value::Value,
{
    plot: P,
    options: PlotOptions<T>,
    xs: Vec<T>,
}

impl<P, T> Plot<P, T>
where
    P: PlotBackend,
    T: crate::value::Value,
{
    /// Create a new plot with the given root, options, and primary function.
    /// The primary function is used to determine the axis ranges if they are not specified in the options.
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    pub fn new(
        root: &P::Root,
        options: PlotOptions<T>,
        function: &impl AsPlottingElement<T>,
    ) -> Result<Self, P::Error> {
        // Bootstrap range for the first element
        let mut xs = options.x_range.as_ref().map_or_else(Vec::new, |r| {
            crate::value::SteppedValues::new_unit(r.start..=r.end).collect()
        });

        let prime = function.as_plotting_element(&xs, options.confidence, options.noise_tolerance);
        let prime_xs = prime.x_values();
        if xs.is_empty() {
            xs = prime_xs;
        }

        // Get real ranges from the prime element
        let (x_range, y_range) = (prime.x_range(), prime.y_range());

        //
        // Range overrides
        let x_range = options.x_range.clone().unwrap_or(x_range);
        let y_range = options.y_range.clone().unwrap_or(y_range);

        let mut plot = P::new_plot(
            root,
            &options.title,
            options.x_label.clone(),
            options.y_label.clone(),
            x_range,
            y_range,
            options.hide_legend,
            options.x_axis_labels,
            options.y_axis_labels,
        )?;
        plot.add_element(&prime)?;
        Ok(Self { plot, options, xs })
    }

    /// Add another plotting element to this plot.
    ///
    /// The element will be converted into a plotting element using the same
    /// x-values, confidence, and noise tolerance as the initial function.
    ///
    /// # Errors
    /// Returns an error if the element cannot be added.
    pub fn with_element(
        &mut self,
        element: &impl AsPlottingElement<T>,
    ) -> Result<&mut Self, P::Error> {
        let element = element.as_plotting_element(
            &self.xs,
            self.options.confidence,
            self.options.noise_tolerance,
        );
        self.plot.add_element(&element)?;
        Ok(self)
    }

    /// Finalize the plot and write it to the output.
    ///
    /// # Errors
    /// Returns an error if the plot cannot be finalized.
    pub fn finish(self) -> Result<(), P::Error> {
        self.plot.finalize()
    }
}

/// Plot a `CurveFit`, `Polynomial` or set of points to a PNG file.
///
/// Generates a filename based on the source file, line number, and timestamp.
/// - Creates the necessary directories if they don't exist.
/// - Prints the path of the generated file to stdout.
/// - If prefix is specified, it is prepended to the filename.
///
/// # Examples
/// ```ignore
/// plot!(fit);
/// plot!([fit, poly1, poly2]);
/// plot!(fit, PlotOptions<_> { title: "My Plot", ..Default::default() });
/// plot!([fit, poly1], PlotOptions<_> { title: "My Plot", ..Default::default() }, prefix = "custom");
#[macro_export]
macro_rules! plot {
    ([$prime:expr $(, $($fit:expr),+ $(,)? )? ], { $( $name:ident : $value:expr ),* $(,)? } $( , prefix = $prefix:expr )?) => {{
        use $crate::plotting::WithTypeFrom;

        let prime = &$prime;
        #[allow(unused_mut)] let mut options = prime.options_with_type_from();
        $( options.$name = $value; )*

        #[allow(unused)] let mut prefix: Option<String> = None; $( prefix = Some($prefix.to_string()); )?
        let path = $crate::plot_filename!(prefix);

        let root = $crate::plotting::plotters::Root::new(&path, options.size);

        #[allow(unused_mut)] let mut plot = prime.plot_with_type_from::<$crate::plotting::plotters::Backend>(&root, options).expect("Failed to create plot");

        $(
            $(
                let fit = &$fit;
                fit.add_to_plot_with_type_from(&mut plot).expect("Failed to add plotting element");
            )+
        )?

        plot.finish().expect("Failed to finalize plot");
        println!("Wrote plot to {}", path.display());
    }};

    ([$prime:expr $(, $($fit:expr),+ $(,)? )? ] $( , prefix = $prefix:expr )?) => {
        $crate::plot!([$prime $(, $($fit),+ )? ], {} $(, prefix = $prefix)?);
    };

    ($prime:expr, { $( $name:ident : $value:expr ),* $(,)? } $( , prefix = $prefix:expr )?) => {
        $crate::plot!([$prime], { $( $name: $value ),*} $(, prefix = $prefix)?);
    };

    ($prime:expr$( , prefix = $prefix:expr )?) => {
        $crate::plot!([$prime] $(, prefix = $prefix)?);
    };
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

        let target_dir = ::std::env::var("TARGET_DIR").unwrap_or_else(|_| "target".into());
        let plots_dir = ::std::path::Path::new(&target_dir).join("plot_output");
        let _ = std::fs::create_dir_all(&plots_dir);

        let filename = format!("{prefix}{file}_line_{line}.png");

        plots_dir.join(filename)
    }};
}
