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

    /// Whether to suppress all output (for backends that print to stdout)
    pub silent: bool,

    /// Whether to show the legend
    pub hide_legend: bool,

    /// Margins around the plot area (if supported by the backend)
    /// The backend may choose a default value if `None` is provided
    pub margins: Option<i32>,

    /// Number of labels to show on the x-axis (if supported by the backend)
    pub x_axis_labels: Option<usize>,

    /// Number of labels to show on the y-axis (if supported by the backend)
    pub y_axis_labels: Option<usize>,
}
impl<T: crate::value::Value> PlotOptions<T> {
    /// Default size for plots
    pub const DEFAULT_SIZE: (u32, u32) = (640, 480);

    /// Default title for plots
    pub const DEFAULT_TITLE: &'static str = "Graph Output";

    /// Set the title of the plot
    #[must_use]
    pub fn with_title(self, title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..self
        }
    }

    /// Set the x-axis label
    #[must_use]
    pub fn with_x_label(mut self, x_label: impl Into<String>) -> Self {
        self.x_label = Some(x_label.into());
        self
    }

    /// Set the y-axis label
    #[must_use]
    pub fn with_y_label(mut self, y_label: impl Into<String>) -> Self {
        self.y_label = Some(y_label.into());
        self
    }

    /// Set the size of the output image in pixels
    #[must_use]
    pub fn with_size(mut self, size: (u32, u32)) -> Self {
        self.size = size;
        self
    }

    /// Set the x-axis range
    #[must_use]
    pub fn with_x_range(mut self, x_range: std::ops::Range<T>) -> Self {
        self.x_range = Some(x_range);
        self
    }

    /// Set the y-axis range
    #[must_use]
    pub fn with_y_range(mut self, y_range: std::ops::Range<T>) -> Self {
        self.y_range = Some(y_range);
        self
    }

    /// Set the noise tolerance for the error bands on fits
    #[must_use]
    pub fn with_noise_tolerance(
        mut self,
        noise_tolerance: crate::statistics::Tolerance<T>,
    ) -> Self {
        self.noise_tolerance = Some(noise_tolerance);
        self
    }

    /// Suppress all output (for backends that print to stdout)
    #[must_use]
    pub fn silent(mut self, silent: bool) -> Self {
        self.silent = silent;
        self
    }

    /// Hide or show the legend
    #[must_use]
    pub fn hide_legend(mut self, hide: bool) -> Self {
        self.hide_legend = hide;
        self
    }

    /// Set the margins around the plot area
    #[must_use]
    pub fn with_margins(mut self, margins: i32) -> Self {
        self.margins = Some(margins);
        self
    }

    /// Set the number of labels to show on the x-axis
    #[must_use]
    pub fn with_x_axis_labels(mut self, count: usize) -> Self {
        self.x_axis_labels = Some(count);
        self
    }

    /// Set the number of labels to show on the y-axis
    #[must_use]
    pub fn with_y_axis_labels(mut self, count: usize) -> Self {
        self.y_axis_labels = Some(count);
        self
    }

    /// Set the confidence level for the plot
    #[must_use]
    pub fn with_confidence(self, confidence: crate::statistics::Confidence) -> Self {
        Self { confidence, ..self }
    }
}
impl<T: crate::value::Value> Default for PlotOptions<T> {
    fn default() -> Self {
        Self {
            title: Self::DEFAULT_TITLE.into(),
            x_label: None,
            y_label: None,
            size: Self::DEFAULT_SIZE,
            x_range: None,
            y_range: None,
            confidence: crate::statistics::Confidence::P95,
            noise_tolerance: None,

            silent: false,
            hide_legend: false,
            margins: None,
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
            options.margins,
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

        let silent = options.silent;
        #[allow(unused_mut)] let mut plot = prime.plot_with_type_from::<$crate::plotting::plotters::Backend>(&root, options).expect("Failed to create plot");

        $(
            $(
                let fit = &$fit;
                fit.add_to_plot_with_type_from(&mut plot).expect("Failed to add plotting element");
            )+
        )?

        plot.finish().expect("Failed to finalize plot");
        if !silent {
            println!("Wrote plot to {}", path.display());
        }

        drop(root);
        path
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

/// Plot the residuals of a `CurveFit` to a PNG file.
///
/// Generates a filename based on the source file, line number, and timestamp.
/// - Creates the necessary directories if they don't exist.
/// - Prints the path of the generated file to stdout.
/// - If prefix is specified, it is prepended to the filename.
///
/// # Examples
/// ```ignore
/// plot_residuals!(fit);
/// plot_residuals!(fit, prefix = "custom");
/// plot_residuals!(fit, PlotOptions<_> { title: "My Plot", ..Default::default() });
/// plot_residuals!(fit, PlotOptions<_> { title: "My Plot", ..Default::default() }, prefix = "custom");
#[macro_export]
macro_rules! plot_residuals {
    ($fit:expr, { $( $name:ident : $value:expr ),* $(,)? } $( , prefix = $prefix:expr )?) => {{
        use $crate::plotting::WithTypeFrom;
        use $crate::value::CoordExt;

        let fit = &$fit;
        let y_range = fit.y_range();
        let residuals = fit.residuals();

        let p = $crate::statistics::residual_normality(&residuals.y());

        #[allow(unused_mut)] let mut options = fit.options_with_type_from();
        options.title = format!("Residuals for fit (normality p = {:.3})", p);
        options.y_range = Some(*y_range.start()..*y_range.end()); // keep same y-range as fit for visual consistency
        $( options.$name = $value; )*

        // residual trendline
        let residual_fit = $crate::ChebyshevFit::new_auto(
            &residuals,
            $crate::statistics::DegreeBound::Relaxed,
            &$crate::score::Aic,
        ).and_then(|f| f.as_monomial());

        #[allow(unused)] let mut prefix: Option<String> = None; $( prefix = Some($prefix.to_string()); )?
        let path = $crate::plot_filename!(prefix);

        let root = $crate::plotting::plotters::Root::new(&path, options.size);

        let silent = options.silent;
        #[allow(unused_mut)] let mut plot = (&residuals, "Residuals").plot_with_type_from::<$crate::plotting::plotters::Backend>(&root, options).expect("Failed to create plot");

        if let Ok(monomial) = residual_fit {
            let data = monomial.solve_range(fit.x_range(), 1.0);
            let equation = monomial.to_string();
            let label = format!("Residual Trendline ({equation})");
            (&data, label.as_str()).add_to_plot_with_type_from(&mut plot).expect("Failed to add residual trendline");
        }

        fit.add_to_plot_with_type_from(&mut plot).expect("Failed to add fit to plot");

        plot.finish().expect("Failed to finalize plot");
        if !silent {
            println!("Wrote plot to {}", path.display());
        }

        drop(root);
        path
    }};

    ($fit:expr$( , prefix = $prefix:expr )?) => {
        $crate::plot_residuals!($fit, {} $(, prefix = $prefix)?);
    };
}

/// Get a filename inside the directory where plots are saved: `target/plot_output/`
pub fn plot_directory(filename: impl AsRef<std::path::Path>) -> std::path::PathBuf {
    let target_dir = std::env::var("TARGET_DIR").unwrap_or_else(|_| "target".into());
    let plots_dir = std::path::Path::new(&target_dir).join("plot_output");
    let _ = std::fs::create_dir_all(&plots_dir);
    plots_dir.join(filename)
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

        let filename = format!("{prefix}{file}_line_{line}.png");
        $crate::plotting::plot_directory(filename)
    }};
}
