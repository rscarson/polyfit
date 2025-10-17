/// A simple cycling palette of colors
pub struct ColorSource<C: Clone>(Vec<C>);
impl<C: Clone> ColorSource<C> {
    /// Create a new palette from the given colors
    #[must_use]
    pub fn new(colors: Vec<C>) -> Self {
        Self(colors)
    }

    /// Get the next color in the palette, cycling back to the start if necessary
    #[must_use]
    pub fn next_color(&mut self) -> C {
        let color = self.0.remove(0);
        self.0.push(color.clone());
        color
    }
}
