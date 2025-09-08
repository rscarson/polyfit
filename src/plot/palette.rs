use plotters::prelude::*;

pub struct Palettes {
    palettes: Vec<PlotPalette>,
    index: usize,
}
impl Palettes {
    pub fn next(&mut self) -> PlotPalette {
        let palette = self.palettes[self.index];
        self.index = (self.index + 1) % self.palettes.len();
        palette
    }
}
impl Default for Palettes {
    fn default() -> Self {
        use plotters::prelude::full_palette::*;

        Self {
            palettes: vec![
                PlotPalette::new(RED, BLUE, GREEN),
                PlotPalette::new(MAGENTA, CYAN, YELLOW),
                PlotPalette::new(ORANGE, PURPLE, TEAL),
                PlotPalette::new(BROWN, PINK, GREY),
            ],
            index: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlotPalette {
    pub fit: RGBColor,
    pub fit_error: RGBAColor,
    pub fit_residual: RGBAColor,

    pub canonical: RGBColor,
    pub data: RGBColor,
}
impl PlotPalette {
    pub const fn new(fit: RGBColor, canonical: RGBColor, data: RGBColor) -> Self {
        let fit_error = RGBAColor(fit.0, fit.1, fit.2, 0.3);
        let fit_residual = RGBAColor(fit.0, fit.1, fit.2, 0.6);

        Self {
            fit,
            fit_error,
            fit_residual,
            canonical,
            data,
        }
    }
}
