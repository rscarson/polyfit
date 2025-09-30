use std::{ops::Range, path::Path};

use plotters::{
    coord::{types::RangedCoordf64, Shift},
    prelude::*,
};
use resvg::usvg;

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    plot::{palette::ColorSource, PlottingElement},
    statistics::{Confidence, ConfidenceBand, Tolerance},
    value::{CoordExt, Value},
    CurveFit, Polynomial,
};

/// Error occurring during plotting
#[derive(Debug, thiserror::Error)]
pub enum PlottingError<'root> {
    /// Error drawing the plot
    #[error("Error drawing plot: {0}")]
    Draw(#[from] DrawingAreaErrorKind<<SVGBackend<'root> as DrawingBackend>::ErrorType>),

    /// Error casting a value
    #[error("A value could not be represented as f64")]
    Cast,

    /// Error parsing SVG
    #[error("Rendering error: {0}")]
    SvgParse(#[from] usvg::Error),

    /// Error encoding PNG
    #[error("PNG encoding error: {0}")]
    PngEncode(String),
}

/// Type alias for the root drawing area.
pub type PlotRoot<'root> = DrawingArea<SVGBackend<'root>, Shift>;

/// Trait for plot backends
pub trait PlotBackend<T: Value = f64> {
    /// Error type for the plot backend
    type Error: std::error::Error;

    /// Root type for the plot backend
    type Root;

    /// Backing type for the root drawing area
    /// For example in `SVGBackend` this is just a `&mut String`
    type RootBacking;

    /// Color type for the plot backend
    type Color: Clone;

    /// Get the next color in the palette
    fn next_color(&mut self) -> Self::Color;

    /// Set the alpha (opacity) of a color
    fn color_with_alpha(color: &Self::Color, alpha: f64) -> Self::Color;

    /// Create a new root drawing area with the given backing and size
    ///
    /// # Errors
    /// Returns an error if the root cannot be created.
    fn new_root(backing: Self::RootBacking, size: (u32, u32)) -> Result<Self::Root, Self::Error>;

    /// Create a new plot with the given title and ranges on the given root
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn new_plot(
        root: &Self::Root,
        title: &str,
        x_range: Range<T>,
        y_range: Range<T>,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Add a line to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn add_line(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        color: Self::Color,
    ) -> Result<(), Self::Error>;

    /// Add a dashed to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn add_dashed_line(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        sizing: (u32, u32),
        color: Self::Color,
    ) -> Result<(), Self::Error>;

    /// Add a confidence band to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn add_confidence(
        &mut self,
        data: &[(T, ConfidenceBand<T>)],
        color: Self::Color,
    ) -> Result<(), Self::Error>;

    /// Finalize the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn finalize(self) -> Result<(), Self::Error>;

    /// Add a plotting element to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn add_element(&mut self, element: &PlottingElement<T>) -> Result<(), Self::Error> {
        match element {
            PlottingElement::Fit(data, bands, equation) => {
                let fit_color = self.next_color();
                self.add_dashed_line(data, "Source Data", 1, (1, 2), fit_color)?;

                let solution = bands
                    .iter()
                    .map(|(x, band)| (*x, band.value()))
                    .collect::<Vec<_>>();
                let color = self.next_color();
                self.add_line(&solution, equation, 1, color.clone())?;

                let confidence_color = Self::color_with_alpha(&color, 0.3);
                self.add_confidence(bands, confidence_color)
            }

            PlottingElement::Canonical(data, equation) => {
                let color = self.next_color();
                self.add_line(data, equation, 1, color)
            }

            PlottingElement::Data(data) => {
                let color = self.next_color();
                self.add_line(data, "Data", 1, color)
            }
        }
    }

    /// Add a polynomial to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn add_polynomial<B: Basis<T> + PolynomialDisplay<T>>(
        &mut self,
        function: &Polynomial<B, T>,
        x: &[T],
    ) -> Result<(), Self::Error> {
        let element = PlottingElement::from_polynomial(function, x);
        self.add_element(&element)
    }

    /// Add a curve fit to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn add_fit<B: Basis<T> + PolynomialDisplay<T>>(
        &mut self,
        fit: &CurveFit<B, T>,
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> Result<(), Self::Error> {
        self.add_element(&PlottingElement::from_curve_fit(
            fit,
            confidence,
            noise_tolerance,
        ))
    }

    /// Add raw data to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn add_data(&mut self, data: &[(T, T)]) -> Result<(), Self::Error> {
        self.add_element(&PlottingElement::from_data(data))
    }
}

/// Plotters backend for plotting
pub struct PlottersBackend<'root, T: Value> {
    _marker: std::marker::PhantomData<T>,
    context: ChartContext<'root, SVGBackend<'root>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    y_range: Range<f64>,
    palette: ColorSource<RGBAColor>,
}
impl<'root, T: Value> PlotBackend<T> for PlottersBackend<'root, T> {
    type Error = PlottingError<'root>;
    type Color = RGBAColor;
    type Root = PlotRoot<'root>;
    type RootBacking = &'root mut String;

    fn next_color(&mut self) -> Self::Color {
        self.palette.next_color()
    }

    fn color_with_alpha(color: &Self::Color, alpha: f64) -> Self::Color {
        let mut color = *color;
        color.3 = alpha;
        color
    }

    fn new_root(backing: Self::RootBacking, size: (u32, u32)) -> Result<Self::Root, Self::Error> {
        let backend = SVGBackend::with_string(backing, size);
        let root = IntoDrawingArea::into_drawing_area(backend);
        root.fill(&WHITE).expect("Failed to fill drawing area");
        Ok(root)
    }

    fn new_plot(
        root: &Self::Root,
        title: &str,
        x_range: Range<T>,
        y_range: Range<T>,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        //
        // T(Range) -> f64(Range)
        let x_range: Range<f64> = cast(x_range.start)?..cast(x_range.end)?;
        let y_range: Range<f64> = cast(y_range.start)?..cast(y_range.end)?;

        let context = ChartBuilder::on(root)
            .caption(title, ("sans-serif", 24).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range.clone())?;

        let palette = ColorSource::new(vec![
            RED.into(),
            BLUE.into(),
            GREEN.into(),
            MAGENTA.into(),
            CYAN.into(),
        ]);

        Ok(Self {
            _marker: std::marker::PhantomData,
            context,
            y_range,
            palette,
        })
    }

    fn add_line(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        color: Self::Color,
    ) -> Result<(), Self::Error> {
        let data = data.as_f64().map_err(|_| PlottingError::Cast)?;
        let data = data.y_clipped(&self.y_range);

        let style = ShapeStyle::from(color).stroke_width(width);
        self.context
            .draw_series(LineSeries::new(data, style))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        Ok(())
    }

    fn add_dashed_line(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        sizing: (u32, u32),
        color: Self::Color,
    ) -> Result<(), Self::Error> {
        let data = data.as_f64().map_err(|_| PlottingError::Cast)?;
        let data = data.y_clipped(&self.y_range);

        let style = ShapeStyle::from(color).stroke_width(width);
        self.context
            .draw_series(DashedLineSeries::new(data, sizing.0, sizing.1, style))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        Ok(())
    }

    fn add_confidence(
        &mut self,
        data: &[(T, ConfidenceBand<T>)],
        color: Self::Color,
    ) -> Result<(), Self::Error> {
        let style = ShapeStyle::from(color).stroke_width(1);
        let series: Vec<_> = data
            .iter()
            .map(|(x, band)| {
                let x = x.to_f64().ok_or(PlottingError::Cast)?;
                let min = band
                    .min()
                    .to_f64()
                    .ok_or(PlottingError::Cast)?
                    .clamp(self.y_range.start, self.y_range.end);
                let value = band
                    .value()
                    .to_f64()
                    .ok_or(PlottingError::Cast)?
                    .clamp(self.y_range.start, self.y_range.end);
                let max = band
                    .max()
                    .to_f64()
                    .ok_or(PlottingError::Cast)?
                    .clamp(self.y_range.start, self.y_range.end);
                Ok(ErrorBar::new_vertical(x, min, value, max, style, 1))
            })
            .collect::<Result<_, Self::Error>>()?;
        self.context.draw_series(series)?;
        Ok(())
    }

    fn finalize(mut self) -> Result<(), Self::Error> {
        //
        // Mesh and axes
        self.context
            .configure_mesh()
            .x_label_formatter(&|v| format!("{v:.2e}"))
            .y_label_formatter(&|v| format!("{v:.2e}"))
            .draw()?;
        self.context
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::LowerRight)
            .draw()?;

        self.context.plotting_area().present()?;
        Ok(())
    }
}

fn cast<'root, T: Value>(value: T) -> Result<f64, PlottingError<'root>> {
    num_traits::cast(value).ok_or(PlottingError::Cast)
}

/// Convert an SVG string to a PNG file at the given path
///
/// # Errors
/// Returns an error if the SVG cannot be parsed or the PNG cannot be written.
pub fn svg2png(svg: &str, path: impl AsRef<Path>) -> std::io::Result<()> {
    let mut opt = usvg::Options::default();
    opt.fontdb_mut().load_system_fonts();

    let now = std::time::SystemTime::now();
    eprintln!("Parsing SVG..., len={}", svg.len());
    let rtree = usvg::Tree::from_str(svg, &opt).map_err(std::io::Error::other)?;
    let pixmap_size = rtree.size().to_int_size();
    eprintln!(
        "Parsing SVG ({}x{}) took {:?}",
        pixmap_size.width(),
        pixmap_size.height(),
        now.elapsed().unwrap()
    );

    let mut pixmap = resvg::tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height())
        .ok_or(std::io::Error::other("Failed to create pixmap"))?;
    resvg::render(&rtree, usvg::Transform::default(), &mut pixmap.as_mut());

    pixmap.save_png(path).map_err(std::io::Error::other)?;
    Ok(())
}
