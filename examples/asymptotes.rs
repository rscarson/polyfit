use polyfit::{
    plot, score::Aic, statistics::DegreeBound, transforms::ApplyNormalization, ChebyshevFit,
};

const CHILDREN_HEIGHT_DATA: &str = include_str!("childrens_height_data.json");

fn main() -> Result<(), polyfit::error::Error> {
    //
    // Here's some data on children's heights at different ages.
    let data: Vec<(f64, f64)> = serde_json::from_str(CHILDREN_HEIGHT_DATA).unwrap();

    //
    // The data has a property called asymptotic growth - the height increases rapidly in early years, then slows down and approaches a maximum as the child grows older.
    // This has the fun side effect of making the data very hard to fit with a polynomial, because polynomials don't have asymptotes and will try to "chase" the data up and down
    //
    // One solution, if you just want an exact fit and dont care about overfitting, can be seen in the 'childrens_height_data' example
    // Here we do something less silly and just fix the data
    //
    // The 'LogOffset' normalization transformation is designed for exactly this kind of data
    // It basically just changes the shape of the data to remove the asymptote, by applying a log transformation after shifting the data so the asymptote is at 0
    //
    // To demonstrate, first we fit the data without the transformation, and then with it. The first fit will be pretty bad, and the second will be much better.
    let fit = ChebyshevFit::new_auto(&data, DegreeBound::Relaxed, &Aic)?;
    println!(
        "root mean squared error = {}",
        fit.root_mean_squared_error()
    );
    println!("Fitted Polynomial: {}", fit);
    plot!(fit);

    println!("\n\nNow applying log offset normalization...\n\n");

    //
    // Now with the log offset normalization applied. This should give us a much better fit, because the transformation has removed the asymptote
    //
    // The fit should also be simpler, because the transformation has made the data more "polynomial-like"
    //
    // The None is a tuning parameter to shift the asymptote around to get a better fit
    // If you find the default transformation isn't giving you a good fit, you can try adjusting this parameter to shift the data more or less
    let data = data.apply_log_offset_normalization(None);
    let fit = ChebyshevFit::new_auto(data, DegreeBound::Relaxed, &Aic)?;
    println!(
        "root mean squared error = {}",
        fit.root_mean_squared_error()
    );
    println!("Fitted Polynomial: {}", fit);
    plot!(fit);

    Ok(())
}
