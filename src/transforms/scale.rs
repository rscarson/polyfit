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
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + shift
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `shift`: The value to add to each element. Positive shifts move data up,
    ///   negative shifts move it down.
    Shift(T),

    /// Multiplies every element of a dataset by a fixed factor.
    ///
    /// Useful for scaling a signal up or down without changing its shape.
    ///
    /// ![Scale example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/linear_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x * factor
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `factor`: The multiplier applied to each element.  
    ///   - `factor > 1` → enlarges values  
    ///   - `factor = 1` → leaves values unchanged  
    ///   - `factor < 0` → flips the sign
    Linear(T),

    /// Applies a quadratic scaling to each element of a dataset.
    ///
    /// Each element is squared and then multiplied by the specified factor.
    /// Useful for emphasizing larger values or modeling parabolic effects.
    ///
    /// ![Quadratic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/quadratic_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = factor * x^2
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `factor`: Multiplier applied after squaring each element.
    Quadratic(T),

    /// Applies a cubic scaling to each element of a dataset.
    ///
    /// Each element is cubed and then multiplied by the specified factor.
    /// Useful for emphasizing extremes and modeling cubic effects.
    ///
    /// ![Cubic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/cubic_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = factor * x^3
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `factor`: Multiplier applied after cubing each element.
    Cubic(T),

    /// Applies an exponential scaling to each element of a dataset.
    ///
    /// Each element is raised to the specified degree and then multiplied by the specified factor.
    /// Useful for modeling exponential growth or decay.
    ///
    /// ![Exponential example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/exponential_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = factor * x^degree
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `degree`: The exponent to which each element is raised.
    /// - `factor`: The multiplier applied after exponentiation.
    Exponential(T, T),

    /// Applies a logarithmic scaling to each element of a dataset.
    ///
    /// Each element is transformed using the logarithm with the specified base
    /// and then multiplied by the specified factor.
    /// Useful for compressing wide-ranging data or modeling logarithmic relationships.
    ///
    /// ![Logarithmic example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/logarithmic_example.png)
    ///
    /// # Parameters
    /// - `base`: The base of the logarithm.
    /// - `factor`: The multiplier applied after logarithmic transformation.
    Logarithmic(T, T),
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
            ScaleTransform::Exponential(degree, coef) => {
                for value in data {
                    *value = value.powf(*degree) * *coef;
                }
            }
            ScaleTransform::Logarithmic(base, coef) => {
                for value in data {
                    *value = Value::max(*value, T::epsilon()).log(*base) * *coef;
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
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x + shift
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `shift`: The value to add to each element. Positive shifts move data up,
    ///   negative shifts move it down.
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
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = x * factor
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `factor`: The multiplier applied to each element.  
    ///   - `factor > 1` → enlarges values  
    ///   - `factor = 1` → leaves values unchanged  
    ///   - `factor < 0` → flips the sign
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
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = factor * x^2
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `factor`: Multiplier applied after squaring each element.
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
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = factor * x^3
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `factor`: Multiplier applied after cubing each element.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_cubic_scale(2.0);
    /// ```
    #[must_use]
    fn apply_cubic_scale(self, coef: T) -> Self;

    /// Applies an exponential scaling to each element of a dataset.
    ///
    /// Each element is raised to the specified degree and then multiplied by the specified factor.
    /// Useful for modeling exponential growth or decay.
    ///
    /// ![Exponential example](https://raw.githubusercontent.com/caliangroup/polyfit/refs/heads/master/.github/assets/exponential_example.png)
    ///
    /// <div class="warning">
    ///
    /// **Technical Details**
    ///
    /// ```math
    /// xₙ = factor * x^degree
    /// ```
    /// </div>
    ///
    /// # Parameters
    /// - `degree`: The exponent to which each element is raised.
    /// - `factor`: The multiplier applied after exponentiation.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_exponential_scale(2.0, 3.0);
    /// ```
    #[must_use]
    fn apply_exponential_scale(self, degree: T, factor: T) -> Self;

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
    /// # Example
    ///
    /// ```rust
    /// # use polyfit::function;
    /// # use polyfit::transforms::ApplyScale;
    /// function!(y(x) = 2 x^2 - 3 x + 4);
    /// let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_polynomial_scale(&y);
    /// ```
    #[must_use]
    fn apply_polynomial_scale<B: Basis<T> + PolynomialDisplay<T>>(
        self,
        polynomial: &Polynomial<B, T>,
    ) -> Self;

    /// Applies a logarithmic scaling to each element of a dataset.
    ///
    /// Each element is transformed using the logarithm with the specified base
    /// and then multiplied by the specified factor.
    /// Useful for compressing wide-ranging data or modeling logarithmic relationships.
    ///
    /// # Parameters
    /// - `base`: The base of the logarithm.
    /// - `factor`: The multiplier applied after logarithmic transformation.
    ///
    /// # Example
    /// ```rust
    /// # use polyfit::transforms::ApplyScale;
    /// let data = vec![(1.0, 10.0), (10.0, 100.0)].apply_logarithmic_scale(10.0, 2.0);
    /// ```
    #[must_use]
    fn apply_logarithmic_scale(self, base: T, factor: T) -> Self;
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

    fn apply_exponential_scale(mut self, degree: T, factor: T) -> Self {
        ScaleTransform::Exponential(degree, factor).apply(self.iter_mut().map(|(_, y)| y));
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

    fn apply_logarithmic_scale(mut self, base: T, factor: T) -> Self {
        ScaleTransform::Logarithmic(base, factor).apply(self.iter_mut().map(|(_, y)| y));
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::transforms::ApplyScale;

    #[test]
    fn test_shift_scale() {
        let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_shift_scale(2.0);
        assert_eq!(data, vec![(1.0, 4.0), (2.0, 5.0)]);
    }

    #[test]
    fn test_linear_scale() {
        let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_linear_scale(2.0);
        assert_eq!(data, vec![(1.0, 4.0), (2.0, 6.0)]);
    }

    #[test]
    fn test_quadratic_scale() {
        let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_quadratic_scale(2.0);
        assert_eq!(data, vec![(1.0, 8.0), (2.0, 18.0)]);
    }

    #[test]
    fn test_cubic_scale() {
        let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_cubic_scale(2.0);
        assert_eq!(data, vec![(1.0, 16.0), (2.0, 54.0)]);
    }

    #[test]
    fn test_polynomial_scale() {
        function!(y(x) = 2 x^2 - 3 x + 4);
        let data = vec![(1.0, 2.0), (2.0, 3.0)].apply_polynomial_scale(&y);
        assert_eq!(data, vec![(1.0, 6.0), (2.0, 13.0)]);
    }

    #[test]
    fn test_logarithmic_scale() {
        let data = vec![(1.0, 10.0), (10.0, 100.0)].apply_logarithmic_scale(10.0, 2.0);
        assert_eq!(data, vec![(1.0, 0.0), (10.0, 4.0)]);
    }
}
