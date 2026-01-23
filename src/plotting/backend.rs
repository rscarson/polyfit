use std::ops::Range;

use crate::{
    basis::Basis,
    display::PolynomialDisplay,
    plotting::PlottingElement,
    statistics::{Confidence, ConfidenceBand, Tolerance},
    value::Value,
    CurveFit, Polynomial,
};

pub mod plotters;

/// Trait for plot backends
pub trait PlotBackend {
    /// Error type for the plot backend
    type Error: std::error::Error;

    /// Root type for the plot backend
    type Root;

    /// Color type for the plot backend
    type Color: Clone;

    /// Get the next color in the palette
    fn next_color(&mut self) -> Self::Color;

    /// Set the alpha (opacity) of a color
    fn color_with_alpha(color: &Self::Color, alpha: f64) -> Self::Color;

    /// Create a new plot with the given title and ranges on the given root
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    #[allow(clippy::too_many_arguments)]
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
        Self: Sized;

    /// Add a line to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn add_line<T: Value>(
        &mut self,
        data: &[(T, T)],
        label: &str,
        width: u32,
        color: Self::Color,
    ) -> Result<(), Self::Error>;

    /// Add a marker to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn add_marker<T: Value>(&mut self, x: T, y: T, label: Option<&str>) -> Result<(), Self::Error>;

    /// Add a dashed line to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be modified.
    fn add_dashed_line<T: Value>(
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
    fn add_confidence<T: Value>(
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
    fn add_element<T: Value>(&mut self, element: &PlottingElement<T>) -> Result<(), Self::Error> {
        match element {
            PlottingElement::Fit(data, bands, equation) => {
                let fit_color = self.next_color();
                // For sanity sake, sample evenly to get ~100 points max
                let step_size = (data.len() / 100).max(1);
                let data: Vec<(T, T)> = data.iter().step_by(step_size).copied().collect();
                self.add_dashed_line(&data, "Source Data", 1, (1, 2), fit_color)?;

                let solution = bands
                    .iter()
                    .map(|(x, band)| (*x, band.value()))
                    .collect::<Vec<_>>();
                let color = self.next_color();

                self.add_line(&solution, equation, 1, color.clone())?;

                let confidence_color = Self::color_with_alpha(&color, 0.3);
                // Sample the bands the same way as the data
                let bands: Vec<(T, ConfidenceBand<T>)> =
                    bands.iter().step_by(step_size).copied().collect();
                self.add_confidence(&bands, confidence_color)
            }

            PlottingElement::Canonical(data, equation) => {
                let color = self.next_color();
                self.add_line(data, equation, 1, color)
            }

            PlottingElement::Data(data, label) => {
                let color = self.next_color();
                // For sanity sake, sample evenly to get ~100 points max
                let step_size = (data.len() / 100).max(1);
                let data: Vec<(T, T)> = data.iter().step_by(step_size).copied().collect();
                self.add_line(&data, label.as_deref().unwrap_or("Data"), 1, color)
            }

            PlottingElement::Markers(points) => {
                for (x, y, label) in points {
                    self.add_marker(*x, *y, label.as_ref().map(String::as_str))?;
                }
                Ok(())
            }
        }
    }

    /// Add a polynomial to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn add_polynomial<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
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
    fn add_fit<T: Value, B: Basis<T> + PolynomialDisplay<T>>(
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
    fn add_data<T: Value>(
        &mut self,
        data: &[(T, T)],
        label: Option<String>,
    ) -> Result<(), Self::Error> {
        self.add_element(&PlottingElement::from_data(data.iter().copied(), label))
    }

    /// Add a set of markers to the plot
    ///
    /// # Errors
    /// Returns an error if the plot cannot be created.
    fn add_markers<T: Value>(
        &mut self,
        points: &[(T, T, Option<String>)],
    ) -> Result<(), Self::Error> {
        self.add_element(&PlottingElement::Markers(points.to_vec()))
    }
}
