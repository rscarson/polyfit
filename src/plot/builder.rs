use std::ops::Range;

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    plot::{AsPlottingElement, PlotBackend, PlottingElement},
    statistics::{Confidence, Tolerance},
    value::Value,
    CurveFit, Polynomial,
};

/// Allows building plots with multiple elements, including functions, curve fits, or raw data.
///
/// You might prefer the [`crate::plot!`] macro
///
/// # Example
/// ```rust
/// use polyfit::{plot_filename, function, Polynomial, value::SteppedValues, plot::{svg2png, PlotBuilder, PlottersBackend}};
///
/// // Create a polynomial
/// function!(f(x) = 2 x^3 - 3 x^2 + 1 x - 5);
///    
/// // Plot it
/// let mut backing = String::new(); // Use a string buffer for SVG output
/// let mut builder = PlotBuilder::<PlottersBackend<f64>, _>::new(&mut backing, (800, 600)).unwrap();
///    
/// builder
///    .title("Test Plot")
///    .add_polynomial(&f, &SteppedValues::new_unit(0.0..=100.0).collect::<Vec<_>>());
/// builder.build().unwrap();
///    
/// // Convert to PNG
/// let path = plot_filename!(Some("builder_ex_"));
/// svg2png(&backing, &path).expect("Failed to write PNG");
/// ```
pub struct PlotBuilder<P, T>
where
    P: PlotBackend<T>,
    T: Value,
{
    root: P::Root,
    elements: Vec<PlottingElement<T>>,
    title: String,

    x_range: Option<Range<T>>,
    y_range: Option<Range<T>>,
}
impl<P, T> PlotBuilder<P, T>
where
    P: PlotBackend<T>,
    T: Value,
{
    /// Create a new plot builder
    ///
    /// Defaults:
    /// - Title: "Graph Output"
    /// - Confidence: 95%
    /// - No noise tolerance
    /// - Axis ranges determined by first element added
    ///
    /// # Errors
    /// Returns an error if the root cannot be created.
    pub fn new(backing: P::RootBacking, size: (u32, u32)) -> Result<Self, P::Error> {
        let root = P::new_root(backing, size)?;
        Ok(Self::from_root(root))
    }
    /// Create a new plot builder on the given root
    ///
    /// Defaults:
    /// - Title: "Graph Output"
    /// - Confidence: 95%
    /// - No noise tolerance
    /// - Axis ranges determined by first element added
    #[must_use]
    pub fn from_root(root: P::Root) -> Self {
        Self {
            root,
            elements: Vec::new(),
            title: "Graph Output".to_string(),

            x_range: None,
            y_range: None,
        }
    }

    /// Create a new plot builder with a single element on the given root
    ///
    /// # Errors
    /// Returns an error if the root cannot be created.
    pub fn from_element(
        backing: P::RootBacking,
        size: (u32, u32),
        element: PlottingElement<T>,
    ) -> Result<Self, P::Error> {
        let root = P::new_root(backing, size)?;
        Ok(Self {
            root,
            elements: vec![element],
            title: "Graph Output".to_string(),

            x_range: None,
            y_range: None,
        })
    }

    /// Add raw data to the plot
    pub fn add_data(&mut self, data: &[(T, T)]) -> &mut Self {
        self.elements.push(PlottingElement::from_data(data));
        self
    }

    /// Add a curve fit to the plot
    pub fn add_fit<B: Basis<T> + PolynomialDisplay<T>>(
        &mut self,
        fit: &CurveFit<B, T>,
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> &mut Self {
        self.elements.push(PlottingElement::from_curve_fit(
            fit,
            confidence,
            noise_tolerance,
        ));
        self
    }

    /// Add a canonical polynomial to the plot
    pub fn add_polynomial<B: Basis<T> + PolynomialDisplay<T>>(
        &mut self,
        poly: &Polynomial<B, T>,
        xs: &[T],
    ) -> &mut Self {
        self.elements
            .push(PlottingElement::from_polynomial(poly, xs));
        self
    }

    /// Add a plotting element to the plot
    ///
    /// Can be used on:
    /// - [`crate::Polynomial`]
    /// - [`crate::CurveFit`]
    /// - `Vec<(T, T)>`
    pub fn add<E>(
        &mut self,
        element: &E,
        xs: &[T],
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> &mut Self
    where
        E: AsPlottingElement<T>,
    {
        let plotting_element = element.as_plotting_element(xs, confidence, noise_tolerance);
        self.elements.push(plotting_element);
        self
    }

    /// Add a plotting element to the plot
    ///
    /// Can be used on:
    /// - [`crate::Polynomial`]
    /// - [`crate::CurveFit`]
    /// - `Vec<(T, T)>`
    pub fn add_element(&mut self, element: PlottingElement<T>) -> &mut Self {
        self.elements.push(element);
        self
    }

    /// Set the title of the plot
    pub fn title(&mut self, title: impl Into<String>) -> &mut Self {
        self.title = title.into();
        self
    }

    /// Set the x-axis range for the plot
    pub fn x_range(&mut self, range: Range<T>) -> &mut Self {
        self.x_range = Some(range);
        self
    }

    /// Set the y-axis range for the plot
    pub fn y_range(&mut self, range: Range<T>) -> &mut Self {
        self.y_range = Some(range);
        self
    }

    /// Build the plot on the given root
    ///
    /// # Errors
    /// Returns an error if the plot cannot be built.
    pub fn build(self) -> Result<(), P::Error> {
        //
        // Get ranges
        let range_result = self.elements.first().map(|e| (e.x_range(), e.y_range()));
        let (x_range, y_range) = range_result.unwrap_or((T::zero()..T::one(), T::zero()..T::one()));

        //
        // Range overrides
        let x_range = self.x_range.unwrap_or(x_range);
        let y_range = self.y_range.unwrap_or(y_range);

        //
        // Create plot, and populate elements
        let mut plot = P::new_plot(&self.root, &self.title, x_range, y_range)?;
        for element in self.elements {
            plot.add_element(&element)?;
        }

        //
        // Render
        plot.finalize()?;
        Ok(())
    }
}
