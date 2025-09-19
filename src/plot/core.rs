use std::{ops::Range, path::Path};

use plotters::{
    coord::{types::RangedCoordf64, Shift},
    prelude::*,
};
use resvg::usvg;

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    plot::{Palettes, PlottingElement},
    statistics::{Confidence, Tolerance},
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

/// Debug plot for curves and fits
pub struct Plot<'root> {
    chart: ChartContext<'root, SVGBackend<'root>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    y_range: Range<f64>,
    palettes: Palettes,
}
impl<'root> Plot<'root> {
    /// Create a new plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    pub fn new<T: Value>(
        root: &PlotRoot<'root>,
        title: &str,
        x_range: Range<T>,
        y_range: Range<T>,
    ) -> Result<Self, PlottingError<'root>> {
        let palettes = Palettes::default();

        //
        // T(Range) -> f64(Range)
        let x_range: Range<f64> = cast(x_range.start)?..cast(x_range.end)?;
        let y_range: Range<f64> = cast(y_range.start)?..cast(y_range.end)?;

        let chart = ChartBuilder::on(root)
            .caption(title, ("sans-serif", 24).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(x_range, y_range.clone())?;

        Ok(Plot {
            chart,
            y_range,
            palettes,
        })
    }

    /// Create a plot from a function
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    pub fn from_canonical<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
        root: &PlotRoot<'root>,
        title: &str,
        function: &Polynomial<B, T>,
        x_range: Range<T>,
    ) -> Result<Self, PlottingError<'root>> {
        let solution = function.solve_range(x_range.start..=x_range.end, T::one());
        let x = solution.x();
        let y = solution.y();

        let min_y = y
            .iter()
            .copied()
            .fold(T::infinity(), <T as nalgebra::RealField>::min);
        let max_y = y
            .iter()
            .copied()
            .fold(T::neg_infinity(), <T as nalgebra::RealField>::max);
        let y_range = min_y..max_y;

        //
        // T(Range) -> f64(Range)
        let x_range: Range<f64> = cast(x_range.start)?..cast(x_range.end)?;
        let y_range: Range<f64> = cast(y_range.start)?..cast(y_range.end)?;

        Self::new(root, title, x_range, y_range)?.with_canonical(function, &x)
    }

    /// Create a plot from a curve fit
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    pub fn from_fit<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
        root: &PlotRoot<'root>,
        title: &str,
        fit: &CurveFit<B, T>,
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
    ) -> Result<Self, PlottingError<'root>> {
        let x_range = fit.x_range();
        let y_range = fit.y_range();

        //
        // T(Range) -> f64(Range)
        let x_range: Range<f64> = cast(*x_range.start())?..cast(*x_range.end())?;
        let y_range: Range<f64> = cast(*y_range.start())?..cast(*y_range.end())?;

        Self::new(root, title, x_range, y_range)?.with_fit(fit, confidence, noise_tolerance)
    }

    /// Clip the y-values of a sample to the plot's y-range
    pub fn clip_y(&self, sample: &mut [(f64, f64)]) {
        for c in sample {
            c.1 = c.1.clamp(self.y_range.start, self.y_range.end);
        }
    }

    /// Add a plotting element to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    pub fn with_element<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
        mut self,
        element: &PlottingElement<B, T>,
        confidence: Confidence,
        noise_tolerance: Option<Tolerance<T>>,
        x: &[T],
    ) -> Result<Self, PlottingError<'root>> {
        match element {
            PlottingElement::Fit(fit) => self.with_fit(fit, confidence, noise_tolerance),
            PlottingElement::Canonical(canonical) => self.with_canonical(canonical, x),
            PlottingElement::Data(data) => {
                let palette = self.palettes.next();
                let data = data.as_f64().map_err(|_| PlottingError::Cast)?;
                self.with_line(&data, "Data", 1, palette.data)
            }
        }
    }

    /// Add a curve fit to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    pub fn with_fit<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
        mut self,
        fit: &CurveFit<B, T>,
        confidence: Confidence,
        tolerance: Option<Tolerance<T>>,
    ) -> Result<Self, PlottingError<'root>> {
        let palette = self.palettes.next();
        let data = fit.data().as_f64().map_err(|_| PlottingError::Cast)?;
        let solution = fit.solution().as_f64().map_err(|_| PlottingError::Cast)?;

        //
        // Confidence bands
        let covariance = fit.covariance().map_err(|_| PlottingError::Cast)?;
        let confidence = covariance
            .solution_confidence(confidence, tolerance)
            .map_err(|_| PlottingError::Cast)?;
        let bands = confidence
            .into_iter()
            .map(|band| {
                let (low, high) = (band.lower, band.upper);
                Some((cast(low).ok()?, cast(high).ok()?))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(PlottingError::Cast)?;

        //
        // Residuals
        let mut residuals = fit
            .residuals()
            .y_iter()
            .zip(&solution)
            .map(|(r, s)| {
                let r = cast(r).ok()?;
                Some((s.0, r))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(PlottingError::Cast)?;
        self.clip_y(&mut residuals);

        let equation = fit.equation();
        self.with_line(&data, &format!("Source Data ({equation})"), 1, palette.data)?
            .with_confidence(&solution, &bands, palette.fit_error)?
            .with_line(&solution, &equation, 1, palette.fit)?
            .with_line(
                &residuals,
                &format!("Residuals ({equation})"),
                1,
                palette.fit_residual,
            )
    }

    /// Add a polynomial to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    pub fn with_canonical<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
        mut self,
        func: &Polynomial<B, T>,
        x: &[T],
    ) -> Result<Self, PlottingError<'root>> {
        let palette = self.palettes.next();
        let data = func
            .solve(x.iter().copied())
            .as_f64()
            .map_err(|_| PlottingError::Cast)?;
        self.with_line(&data, &func.equation(), 3, palette.canonical)
    }

    /// Add a line to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    pub fn with_line(
        mut self,
        data: &[(f64, f64)],
        label: &str,
        width: u32,
        color: impl Into<ShapeStyle>,
    ) -> Result<Self, PlottingError<'root>> {
        let mut data = data.to_vec();
        self.clip_y(&mut data);

        let style = color.into().stroke_width(width);
        self.chart
            .draw_series(LineSeries::new(data, style))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));

        Ok(self)
    }

    /// Add a confidence band to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    pub fn with_confidence(
        mut self,
        data: &[(f64, f64)],
        bands: &[(f64, f64)],
        color: impl Into<ShapeStyle>,
    ) -> Result<Self, PlottingError<'root>> {
        let style = color.into().stroke_width(1);

        let mut data = data.to_vec();
        self.clip_y(&mut data);

        let mut bands = bands.to_vec();
        self.clip_y(&mut bands);

        let series = data
            .into_iter()
            .zip(bands)
            .map(|((x, y), (low, high))| ErrorBar::new_vertical(x, low, y, high, style, 1));
        self.chart.draw_series(series)?;

        Ok(self)
    }

    /// Build the final plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    pub fn build(mut self) -> Result<(), PlottingError<'root>> {
        //
        // Mesh and axes
        self.chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{v:.2e}"))
            .y_label_formatter(&|v| format!("{v:.2e}"))
            .draw()?;
        self.chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .position(SeriesLabelPosition::UpperLeft)
            .draw()?;

        self.chart.plotting_area().present()?;
        Ok(())
    }

    /// Build a PNG file from the SVG data
    ///
    /// # Errors
    /// Returns an error if the PNG cannot be created.
    pub fn build_png(svg: &str, target: &Path) -> Result<(), PlottingError<'root>> {
        let mut opt = usvg::Options::default();
        opt.fontdb_mut().load_system_fonts();

        let rtree = usvg::Tree::from_str(svg, &opt)?;
        let pixmap_size = rtree.size().to_int_size();

        let mut pixmap = resvg::tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height())
            .ok_or(PlottingError::Cast)?;
        resvg::render(&rtree, usvg::Transform::default(), &mut pixmap.as_mut());

        pixmap
            .save_png(target)
            .map_err(|e| PlottingError::PngEncode(e.to_string()))?;
        Ok(())
    }
}

fn cast<'root, T: Value>(value: T) -> Result<f64, PlottingError<'root>> {
    num_traits::cast(value).ok_or(PlottingError::Cast)
}
