use crate::{
    basis::Basis, display::PolynomialDisplay, transforms::Transform, value::Value, Polynomial,
};

/// Types of scaling transformations for data
pub enum ScaleTransform<T: Value> {
    Shift(T),
    Linear(T),
    Quadratic(T),
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
    /// Applies a shift transformation to the data.
    fn apply_shift_scale(&mut self, amount: T);

    /// Applies a linear scaling transformation to the data.
    fn apply_scale_scale(&mut self, factor: T);

    /// Applies a quadratic scaling transformation to the data.
    fn apply_quadratic_scale(&mut self, coef: T);

    /// Applies a cubic scaling transformation to the data.
    fn apply_cubic_scale(&mut self, coef: T);

    /// Applies a polynomial series as a transformation to the data.
    fn apply_polynomial_scale<B: Basis<T> + PolynomialDisplay<T>>(
        &mut self,
        polynomial: &Polynomial<B, T>,
    );
}
impl<T: Value> ApplyScale<T> for Vec<T> {
    fn apply_shift_scale(&mut self, amount: T) {
        ScaleTransform::Shift(amount).apply(self.iter_mut());
    }

    fn apply_scale_scale(&mut self, factor: T) {
        ScaleTransform::Linear(factor).apply(self.iter_mut());
    }

    fn apply_quadratic_scale(&mut self, coef: T) {
        ScaleTransform::Quadratic(coef).apply(self.iter_mut());
    }

    fn apply_cubic_scale(&mut self, coef: T) {
        ScaleTransform::Cubic(coef).apply(self.iter_mut());
    }

    fn apply_polynomial_scale<B: Basis<T> + PolynomialDisplay<T>>(
        &mut self,
        polynomial: &Polynomial<B, T>,
    ) {
        // Apply the polynomial scale transformation
        for value in self {
            *value = polynomial.y(*value);
        }
    }
}
