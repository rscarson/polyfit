//! Plotting backend using the `plotters` crate
//!
//! Everything is coeerced to `f64` for plotting purposes.
//!
//! Uses the bitmap backend to create PNG files.
//!
//! sans-serif font is included for use in plots.
//! - Copyright 2010-2024 by Bitstream, Inc.
use std::{ops::Range, path::Path};

use plotters::{
    coord::{types::RangedCoordf64, ReverseCoordTranslate, Shift},
    prelude::*,
};

use crate::{
    plotting::{palette::ColorSource, PlotBackend},
    statistics::ConfidenceBand,
    value::{CoordExt, IntClampedCast, Value},
};

const FONT_BYTES: &[u8] = include_bytes!("DejaVuSans.ttf");
const MAX_LBL_WIDTH: usize = 120;

/// Register the built-in font with plotters
///
/// # Panics
/// Panics if the built-in font was corrupted somehow
pub fn register_font() {
    plotters::style::register_font("sans-serif", FontStyle::Normal, FONT_BYTES)
        .ok()
        .expect("Failed to register font; font.ttf corrupted?");
}

/// Plot a set of data points onto the given root
///
/// Convenience function for quick plotting of data
///
/// # Parameters
/// - `root`: The drawing area root to plot onto
/// - `data`: The data points to plot
/// - `x_range`: The range of x values to display
/// - `caption`: The caption for the plot
///
/// # Errors
/// Returns an error if the plot could not be created or drawn
pub fn plot_data<'a, T: Value>(
    root: &Root<'a>,
    data: &[(T, T)],
    x_range: Range<T>,
    caption: &str,
) -> Result<(), Error<'a>> {
    crate::plotting::Plot::<Backend, _>::new(
        root,
        crate::plotting::PlotOptions {
            title: caption.to_string(),
            x_range: Some(x_range),
            ..Default::default()
        },
        &data,
    )?
    .finish()
}

/// How to split a drawing area
#[derive(Debug, Clone, Copy)]
pub enum Split {
    /// Split the area into the given number of horizontal pieces
    Horizontal(usize),
    /// Split the area into the given number of vertical pieces
    Vertical(usize),
}

/// A drawing area root for plotters
pub struct Root<'a>(DrawingArea<BitMapBackend<'a>, Shift>);
impl<'a> AsRef<DrawingArea<BitMapBackend<'a>, Shift>> for Root<'a> {
    fn as_ref(&self) -> &DrawingArea<BitMapBackend<'a>, Shift> {
        &self.0
    }
}
impl<'a> Root<'a> {
    /// Create a new drawing area root for plotters
    ///
    /// Will create a PNG file at the given path with the given size
    ///
    /// # Panics
    /// Panics if the built-in font was corrupted somehow
    #[must_use]
    pub fn new(path: &'a Path, size: (u32, u32)) -> Self {
        register_font();

        let backend = BitMapBackend::new(path, size);
        let root = IntoDrawingArea::into_drawing_area(backend);
        root.fill(&WHITE).expect("Failed to fill drawing area");
        Self(root)
    }

    /// Create multiple drawing area roots for plotters, splitting the area into pieces
    ///
    /// Will create a PNG file at the given path with the given size,
    ///
    /// # Panics
    /// Panics if the built-in font was corrupted somehow
    #[must_use]
    pub fn new_split(path: &'a Path, size: (u32, u32), split: Split) -> Vec<Self> {
        register_font();

        let pieces = match split {
            Split::Horizontal(p) => (1, p),
            Split::Vertical(p) => (p, 1),
        };

        let backend = BitMapBackend::new(path, size);
        let root = IntoDrawingArea::into_drawing_area(backend);
        let areas = root.split_evenly(pieces);
        areas
            .into_iter()
            .map(|area| {
                area.fill(&WHITE).expect("Failed to fill drawing area");
                Self(area)
            })
            .collect()
    }
}

/// Plotters backend for plotting
pub struct Backend<'root> {
    context: ChartContext<'root, BitMapBackend<'root>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    y_range: Range<f64>,
    palette: ColorSource<RGBAColor>,

    x_label: Option<String>,
    y_label: Option<String>,

    hide_legend: bool,
    x_axis_labels: Option<usize>,
    y_axis_labels: Option<usize>,
}
impl<'root> PlotBackend for Backend<'root> {
    type Error = Error<'root>;
    type Color = RGBAColor;
    type Root = Root<'root>;

    fn next_color(&mut self) -> Self::Color {
        self.palette.next_color()
    }

    fn color_with_alpha(color: &Self::Color, alpha: f64) -> Self::Color {
        let mut color = *color;
        color.3 = alpha;
        color
    }

    fn new_plot<T: Value>(
        root: &Self::Root,
        title: &str,
        x_label: Option<String>,
        y_label: Option<String>,
        x_range: Range<T>,
        y_range: Range<T>,
        hide_legend: bool,
        margins: Option<i32>,
        x_axis_labels: Option<usize>,
        y_axis_labels: Option<usize>,
    ) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        //
        // T(Range) -> f64(Range)
        let x_range: Range<f64> = cast(x_range.start)?..cast(x_range.end)?;
        let y_range: Range<f64> = cast(y_range.start)?..cast(y_range.end)?;

        let mut context = ChartBuilder::on(root.as_ref());
        context
            .margin(margins.unwrap_or(5))
            .x_label_area_size(30)
            .y_label_area_size(60);

        if !title.is_empty() {
            context.caption(title, (FontFamily::SansSerif, 16).into_font());
        }

        if let Some(0) = x_axis_labels {
            context.x_label_area_size(1);
        }

        if let Some(0) = y_axis_labels {
            context.y_label_area_size(1);
        }

        let context = context.build_cartesian_2d(x_range, y_range.clone())?;

        let palette = ColorSource::new(vec![
            RED.into(),
            BLUE.into(),
            GREEN.into(),
            MAGENTA.into(),
            CYAN.into(),
            YELLOW.into(),
            BLACK.into(),
            RGBColor(255, 165, 0).into(), // Orange
            RGBColor(128, 0, 128).into(), // Purple
        ]);

        Ok(Self {
            context,
            y_range,
            palette,

            x_label,
            y_label,

            hide_legend,
            x_axis_labels,
            y_axis_labels,
        })
    }

    fn add_line<T: Value>(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        color: Self::Color,
    ) -> Result<(), Self::Error> {
        let data = data.as_f64().map_err(|_| Error::Cast)?;
        let data = data.y_clipped(&self.y_range);

        //
        // Shorten label and add [...] if too long
        let label = if label.len() > MAX_LBL_WIDTH {
            let mut s: String = label.chars().take(MAX_LBL_WIDTH - 3).collect();
            s.push_str("...");
            s
        } else {
            label.to_string()
        };

        let style = ShapeStyle::from(color).stroke_width(width);
        self.context
            .draw_series(LineSeries::new(data, style))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        Ok(())
    }

    fn add_marker<T: Value>(&mut self, x: T, y: T, label: Option<&str>) -> Result<(), Self::Error> {
        let x = cast(x)?;
        let y = cast(y)?.clamp(self.y_range.start, self.y_range.end);

        let style = ShapeStyle::from(&BLACK).filled();

        //
        // Draw a large cross
        let shape = Cross::new((x, y), 5, style);
        self.context.draw_series(std::iter::once(shape))?;

        if let Some(label) = label {
            //
            // Shorten label and add [...] if too long
            let label = if label.len() > MAX_LBL_WIDTH {
                let mut s: String = label.chars().take(MAX_LBL_WIDTH - 3).collect();
                s.push_str("...");
                s
            } else {
                label.to_string()
            };

            let text_style = ("sans-serif", 10).into_font().color(&BLACK);
            let width = self
                .context
                .plotting_area()
                .estimate_text_size(&label, &text_style)?
                .0;
            let end_pix = self.context.plotting_area().get_pixel_range().0;

            // X-distance should be ~5% of the total x-range
            let mut x_dist = (self.context.as_coord_spec().get_x_range().end
                - self.context.as_coord_spec().get_x_range().start)
                * 0.05;

            // However if the label would go off the right edge, move it to the left side
            let (text_end_pix, y_pix) = self.context.as_coord_spec().translate(&(x, y));
            if text_end_pix + width.clamped_cast::<i32>() > end_pix.end {
                let c_width = self
                    .context
                    .as_coord_spec()
                    .reverse_translate((width.clamped_cast(), y_pix))
                    .unwrap_or_default()
                    .0;
                x_dist = -(x_dist + 2.0 * c_width);
            }

            self.context.draw_series(std::iter::once(Text::new(
                label,
                (x + x_dist, y - 1.0),
                text_style,
            )))?;
        }

        Ok(())
    }

    fn add_dashed_line<T: Value>(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        sizing: (u32, u32),
        color: Self::Color,
    ) -> Result<(), Self::Error> {
        let data = data.as_f64().map_err(|_| Error::Cast)?;
        let data = data.y_clipped(&self.y_range);

        //
        // Shorten label and add [...] if too long
        let label = if label.len() > MAX_LBL_WIDTH {
            let mut s: String = label.chars().take(MAX_LBL_WIDTH - 3).collect();
            s.push_str("...");
            s
        } else {
            label.to_string()
        };

        let style = ShapeStyle::from(color).stroke_width(width);
        self.context
            .draw_series(DashedLineSeries::new(data, sizing.0, sizing.1, style))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
        Ok(())
    }

    fn add_confidence<T: Value>(
        &mut self,
        data: &[(T, ConfidenceBand<T>)],
        color: Self::Color,
    ) -> Result<(), Self::Error> {
        let style = ShapeStyle::from(color).stroke_width(1);
        let series: Vec<_> = data
            .iter()
            .map(|(x, band)| {
                let x = x.to_f64().ok_or(Error::Cast)?;
                let min = band
                    .min()
                    .to_f64()
                    .ok_or(Error::Cast)?
                    .clamp(self.y_range.start, self.y_range.end);
                let value = band
                    .value()
                    .to_f64()
                    .ok_or(Error::Cast)?
                    .clamp(self.y_range.start, self.y_range.end);
                let max = band
                    .max()
                    .to_f64()
                    .ok_or(Error::Cast)?
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
        let mut context = self.context.configure_mesh();

        context
            .label_style((FontFamily::SansSerif, 12))
            .x_label_formatter(&|v| {
                if (1e-3..1e3).contains(v) {
                    format!("{v:.2}")
                } else {
                    format!("{v:.2e}")
                }
            })
            .y_label_formatter(&|v| {
                if (1e-3..1e3).contains(v) {
                    format!("{v:.2}")
                } else {
                    format!("{v:.2e}")
                }
            });

        if let Some(x_label) = &self.x_label {
            context.x_desc(x_label);
        }

        if let Some(y_label) = &self.y_label {
            context.y_desc(y_label);
        }

        if let Some(x_labels) = self.x_axis_labels {
            context.x_labels(x_labels);
        }

        if let Some(y_labels) = self.y_axis_labels {
            context.y_labels(y_labels);
        }

        context.draw()?;

        if !self.hide_legend {
            //
            // Legend
            self.context
                .configure_series_labels()
                .label_font((FontFamily::SansSerif, 10))
                .background_style(WHITE.mix(0.5))
                .border_style(BLACK)
                .position(SeriesLabelPosition::LowerRight)
                .draw()?;
        }

        self.context.plotting_area().present()?;
        Ok(())
    }
}

fn cast<'root, T: Value>(value: T) -> Result<f64, Error<'root>> {
    num_traits::cast(value).ok_or(Error::Cast)
}

/// Error occurring during plotting
#[derive(Debug, thiserror::Error)]
pub enum Error<'root> {
    /// Error drawing the plot
    #[error("Error drawing plot: {0}")]
    Draw(#[from] DrawingAreaErrorKind<<BitMapBackend<'root> as DrawingBackend>::ErrorType>),

    /// Error casting a value
    #[error("A value could not be represented as f64")]
    Cast,
}
