use polyfit::{
    basis_select, plot,
    score::{shape_constraint::*, Aic},
    statistics::{CvStrategy, DegreeBound},
    ChebyshevFit,
};

const CHILDREN_HEIGHT_DATA: &str = include_str!("childrens_height_data.json");

fn main() -> Result<(), polyfit::error::Error> {
    //
    // Here's some data on children's heights at different ages.
    // Because it's a finished curve, there's no noise to ignore - we want the simplest possible fit that's more or less exact.
    let data: Vec<(f64, f64)> = serde_json::from_str(CHILDREN_HEIGHT_DATA).unwrap();

    //
    // A good first step is to confirm how the data behaves in different bases. This can help us choose a good basis for fitting.
    // The `basis_select` function will fit the data in multiple bases and print out some scores for each
    basis_select!(&data, DegreeBound::Relaxed, &Aic);

    //
    // Chebyshev, Legendre, and Laguerre all perform similarly, but Chebyshev is a good default for data with a wide range of X values, so let's use that
    // Here is what it has to say about the Chebyshev fit it tried:
    // --
    // Chebyshev: xₛ = T[ 61..228 -> -1..1 ], y(x) = 0.57·T₅(xₛ) - 1.40·T₄(xₛ) - 2.60·T₃(xₛ) - 3.81·T₂(xₛ) + 35.24·T₁(xₛ) + 147.70
    // Fit R²: 0.9996, Residuals Normality p-value: 0.0045
    // Wrote plot to target\plot_output\chebyshev_examples_children_height_data.rs_line_19.png

    // Here we aren't doing a normal fit where we need to worry about overfitting - we care a lot more about getting the shape right than anything else
    // ShapeConstraint is a custom scoring method that penalizes curvature and non-monotonicity, which is exactly what we want for this data
    //
    let score = ShapeConstraint::new(SamplingStrategy::Total) // Sample all points - its not a big dataset and we want to get the shape right across the whole curve
        .with_curvature_penalty(PenaltyWeight::Medium) // We want to avoid unnecessary curvature, but we know there is some real curvature in the data
        .with_monotonic_penalty(PenaltyWeight::Large, MonotonicityDirection::Infer); // The data is monotonic, so we want to heavily penalize any non-monotonicity

    //
    // We are also going to use k-fold cross validation instead of a normal fit - this will help ensure we aren't overfitting or underfitting, and that the shape of the curve is good
    // Note the parameters we use here - they will make this fairly slow!
    // - `CvStrategy::LeaveOneOut` means we will do as many fits as there are data points, each time leaving out one point and testing the fit on that point.
    //   This is the most thorough cross-validation strategy but its the reason this is slow - for any data set bigger than a few hundred points - dont!
    // - `DegreeBound::Aggressive` means we will test a much wider range of polynomial degrees than normal - important since we don't care about overfitting for once
    let logfit = ChebyshevFit::new_kfold_cross_validated(
        data,
        CvStrategy::LeaveOneOut,
        DegreeBound::Aggressive,
        &score,
    )?;

    println!("Fitted Polynomial: {}", logfit);
    plot!(logfit);

    Ok(())
}
