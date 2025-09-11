use crate::{
    basis::Basis, display::PolynomialDisplay, transforms::Transform, value::Value, Polynomial,
};

/// Types of scaling transformations for data
pub enum ScaleTransform<T: Value> {
    /// Adds a fixed offset to every element of a dataset.
    ///
    /// This is useful for translating a signal up or down without changing its shape.
    ///
    /// ![Shift example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/shift_example.png)
    ///
    /// # Parameters
    ///
    /// - `shift`: The value to add to each element. Positive shifts move data up,
    ///   negative shifts move it down.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = x + shift
    /// > ```
    /// >
    /// > - Mean: μ_y = μ_x + shift  
    /// > - Variance: σ²_y = σ²_x
    Shift(T),

    /// Multiplies every element of a dataset by a fixed factor.
    ///
    /// Useful for scaling a signal up or down without changing its shape.
    ///
    /// ![Scale example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/linear_example.png)
    ///
    /// # Parameters
    ///
    /// - `factor`: The multiplier applied to each element.  
    ///   - `factor > 1` → enlarges values  
    ///   - `factor = 1` → leaves values unchanged  
    ///   - `factor < 0` → flips the sign
    ///
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = x * factor
    /// > ```
    /// >
    /// > - Mean: μ_y = μ_x * factor  
    /// > - Variance: σ²_y = σ²_x * factor²
    Linear(T),

    /// Applies a quadratic scaling to each element of a dataset.
    ///
    /// Each element is squared and then multiplied by the specified factor.
    /// Useful for emphasizing larger values or modeling parabolic effects.
    ///
    /// ![Quadratic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/quadratic_example.png)
    ///
    /// # Parameters
    ///
    /// - `factor`: Multiplier applied after squaring each element.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = factor * x^2
    /// > ```
    /// >
    /// > - Mean and variance are transformed non-linearly:
    /// >   Larger absolute values grow faster than smaller ones.
    Quadratic(T),

    /// Applies a cubic scaling to each element of a dataset.
    ///
    /// Each element is cubed and then multiplied by the specified factor.
    /// Useful for emphasizing extremes and modeling cubic effects.
    ///
    /// ![Cubic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/cubic_example.png)
    ///
    /// # Parameters
    ///
    /// - `factor`: Multiplier applied after cubing each element.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = factor * x^3
    /// > ```
    /// >
    /// > - Mean and variance are transformed non-linearly:
    /// >
    /// >   Extreme values are amplified more than smaller ones.
    Cubic(T),
}
impl<T: Value> Transform<T> for ScaleTransform<T> {
    fn apply<'a>(&self, data: impl Iterator<Item = &'a mut T>) {
        match self {
            ScaleTransform::Shift(amount) => {
                for value in data {
                    *value += *amount;
                }
            }
            ScaleTransform::Linear(slope) => {
                for value in data {
                    *value *= *slope;
                }
            }
            ScaleTransform::Quadratic(coef) => {
                for value in data {
                    *value = *value * *value * *coef;
                }
            }
            ScaleTransform::Cubic(coef) => {
                for value in data {
                    *value = *value * *value * *value * *coef;
                }
            }
        }
    }
}

/// Trait for applying scaling transformations to data.
pub trait ApplyScale<T: Value> {
    /// Adds a fixed offset to every element of a dataset.
    ///
    /// This is useful for translating a signal up or down without changing its shape.
    ///
    /// ![Shift example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/shift_example.png)
    ///
    /// # Parameters
    ///
    /// - `shift`: The value to add to each element. Positive shifts move data up,
    ///   negative shifts move it down.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = x + shift
    /// > ```
    /// >
    /// > - Mean: μ_y = μ_x + shift  
    /// > - Variance: σ²_y = σ²_x
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_shift_scale(2.0);
    /// ```
    #[must_use]
    fn apply_shift_scale(self, amount: T) -> Self;

    /// Multiplies every element of a dataset by a fixed factor.
    ///
    /// Useful for scaling a signal up or down without changing its shape.
    ///
    /// ![Scale example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/linear_example.png)
    ///
    /// # Parameters
    ///
    /// - `factor`: The multiplier applied to each element.  
    ///   - `factor > 1` → enlarges values  
    ///   - `factor = 1` → leaves values unchanged  
    ///   - `factor < 0` → flips the sign
    ///
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = x * factor
    /// > ```
    /// >
    /// > - Mean: μ_y = μ_x * factor  
    /// > - Variance: σ²_y = σ²_x * factor²
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_linear_scale(2.0);
    /// ```
    #[must_use]
    fn apply_linear_scale(self, factor: T) -> Self;

    /// Applies a quadratic scaling to each element of a dataset.
    ///
    /// Each element is squared and then multiplied by the specified factor.
    /// Useful for emphasizing larger values or modeling parabolic effects.
    ///
    /// ![Quadratic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/quadratic_example.png)
    ///
    /// # Parameters
    ///
    /// - `factor`: Multiplier applied after squaring each element.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```math
    /// > y = factor * x^2
    /// > ```
    /// >
    /// > - Mean and variance are transformed non-linearly:
    /// >   Larger absolute values grow faster than smaller ones.
    ///
    /// # Example
    /// ```
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_quadratic_scale(2.0);
    /// ```
    #[must_use]
    fn apply_quadratic_scale(self, coef: T) -> Self;

    /// Applies a cubic scaling to each element of a dataset.
    ///
    /// Each element is cubed and then multiplied by the specified factor.
    /// Useful for emphasizing extremes and modeling cubic effects.
    ///
    /// ![Cubic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/cubic_example.png)
    ///
    /// # Parameters
    ///
    /// - `factor`: Multiplier applied after cubing each element.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```ignore
    /// > y = factor * x^3
    /// > ```
    /// >
    /// > - Mean and variance are transformed non-linearly:
    /// >
    /// >   Extreme values are amplified more than smaller ones.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_cubic_scale(2.0);
    /// ```
    #[must_use]
    fn apply_cubic_scale(self, coef: T) -> Self;

    /// Applies a polynomial series as a transformation to each element of a dataset.
    ///
    /// The value of each element is replaced by the polynomial evaluated at that element.
    /// Useful for modeling non-linear relationships or custom transformations.
    ///
    /// # Parameters
    ///
    /// - `polynomial`: Reference to a `Polynomial` object, which defines the coefficients
    ///   and basis to use for the transformation.
    ///
    /// > # Technical Details
    /// >
    /// > Element-wise operation:
    /// >
    /// > ```ignore
    /// > y = P(x)
    /// > ```
    /// >
    /// > where `P` is the provided polynomial. Both mean and variance of the dataset
    /// > may change non-linearly depending on the polynomial’s degree and coefficients.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::function;
    /// # use polyfit::transforms::ApplyScale;
    /// function!(y(x) = 2 x^2 - 3x + 4);
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_polynomial_scale(&y);
    /// ```
    #[must_use]
    fn apply_polynomial_scale<B: Basis<T> + PolynomialDisplay<T>>(
        self,
        polynomial: &Polynomial<B, T>,
    ) -> Self;
}
impl<T: Value> ApplyScale<T> for Vec<(T, T)> {
    fn apply_shift_scale(mut self, amount: T) -> Self {
        ScaleTransform::Shift(amount).apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_linear_scale(mut self, factor: T) -> Self {
        ScaleTransform::Linear(factor).apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_quadratic_scale(mut self, coef: T) -> Self {
        ScaleTransform::Quadratic(coef).apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_cubic_scale(mut self, coef: T) -> Self {
        ScaleTransform::Cubic(coef).apply(self.iter_mut().map(|(_, y)| y));
        self
    }

    fn apply_polynomial_scale<B: Basis<T> + PolynomialDisplay<T>>(
        mut self,
        polynomial: &Polynomial<B, T>,
    ) -> Self {
        // Apply the polynomial scale transformation
        for (_, y) in &mut self {
            *y = polynomial.y(*y);
        }

        self
    }
}
