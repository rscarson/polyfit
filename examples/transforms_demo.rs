//!
//! This is the example that I used to generate the graphs for the documentation on transforms.

use polyfit::plotting;
use polyfit::transforms::{
    ApplyNoise, ApplyScale, NoiseTransform, NormalizationTransform, ScaleTransform, Strength,
    Transformable,
};

#[rustfmt::skip]
fn main() -> Result<(), String> {

    //
    // Noises
    //

    //
    // Uncorrelated gaussian noise
    let data = SourceSampleType::Growing(false).generate();
    let abs = NoiseTransform::CorrelatedGaussian { rho: 0.0, strength: Strength::Absolute(100.0), seed: None };
    let rel_small = NoiseTransform::CorrelatedGaussian { rho: 0.0, strength: Strength::Relative(0.1), seed: None };
    let rel_big = NoiseTransform::CorrelatedGaussian { rho: 0.0, strength: Strength::Relative(0.3), seed: None };
    generate_plot(
        data.clone(),
        vec![
            ("strength=Absolute(100),rho=0", data.transformed(&abs)),
            ("strength=Relative(0.1),rho=0", data.transformed(&rel_small)),
            ("strength=Relative(0.3),rho=0", data.transformed(&rel_big)),
        ],
        true,
        "Normal Noise",
        "normal_example.png",
    )?;

    //
    // Correlated gaussian noise
    let data = SourceSampleType::Growing(false).generate();
    let abs_non_cor = NoiseTransform::CorrelatedGaussian { rho: 0.0, strength: Strength::Absolute(100.0), seed: None };
    let rel_non_cor = NoiseTransform::CorrelatedGaussian { rho: 0.0, strength: Strength::Relative(0.1), seed: None };
    let rel_cor = NoiseTransform::CorrelatedGaussian { rho: 0.9, strength: Strength::Relative(0.2), seed: None };
    generate_plot(
        data.clone(),
        vec![
            ("strength=Absolute(100),rho=0", data.transformed(&abs_non_cor)),
            ("strength=Relative(0.1),rho=0", data.transformed(&rel_non_cor)),
            ("strength=Absolute(100),rho=0.9", data.transformed(&rel_cor)),
        ],
        true,
        "Correlated Gaussian Noise",
        "correlated_gaussian_example.png",
    )?;

    //
    // Uniform noise
    let data = SourceSampleType::Growing(false).generate();
    let low_heavy = NoiseTransform::Uniform { lower: Strength::Relative(1.0), upper: Strength::Relative(0.1), seed: None };
    let upp_heavy = NoiseTransform::Uniform { lower: Strength::Relative(0.1), upper: Strength::Relative(1.0), seed: None };
    let symm = NoiseTransform::Uniform { lower: Strength::Relative(0.1), upper: Strength::Relative(0.1), seed: None };
    generate_plot(
        data.clone(),
        vec![
            ("lower=0.1, upper=0.1", data.transformed(&symm)),
            ("lower=1.0, upper=0.1", data.transformed(&low_heavy)),
            ("lower=0.1, upper=1.0", data.transformed(&upp_heavy)),
        ],
        false,
        "Uniform Noise",
        "uniform_example.png",
    )?;

    //
    // Poisson noise
    let data = SourceSampleType::Flat(false).generate();
    let low_rel = NoiseTransform::Poisson { lambda: Strength::Relative(0.5), seed: None };
    let med_rel = NoiseTransform::Poisson { lambda: Strength::Relative(1.0), seed: None };
    let med_abs = NoiseTransform::Poisson { lambda: Strength::Absolute(1.0), seed: None };
    generate_plot(
        data.clone(),
        vec![
            ("lambda=Relative(1.0)", data.transformed(&low_rel)),
            ("lambda=Relative(2.0)", data.transformed(&med_rel)),
            ("lambda=Absolute(2.0)", data.transformed(&med_abs)),
        ],
        true,
        "Poisson Noise",
        "poisson_example.png",
    )?;

    // 
    // Salt and pepper noise
    let data = SourceSampleType::Growing(false).generate();
    let low_abs = NoiseTransform::Impulse { probability: 0.03, alpha: 0.0, beta: 0.0, min: Strength::Absolute(-50.0), max: Strength::Absolute(50.0), seed: None };
    let med_abs = NoiseTransform::Impulse { probability: 0.1, alpha: 0.0, beta: 0.0, min: Strength::Absolute(-50.0), max: Strength::Absolute(50.0), seed: None };
    let med_rel = NoiseTransform::Impulse { probability: 0.1, alpha: 0.0, beta: 0.0, min: Strength::Relative(-1.0), max: Strength::Relative(1.0), seed: None };
    generate_plot(
        data.clone(),
        vec![
            ("p=0.03, min=Abs(-50), max=Abs(50)", data.transformed(&low_abs)),
            ("p=0.1, min=Abs(-50), max=Abs(50)", data.transformed(&med_abs)),
            ("p=0.1, min=Rel(-1.0), max=Rel(1.0)", data.transformed(&med_rel)),
        ],
        true,
        "Salt and Pepper Noise",
        "salt_and_pepper_example.png",
    )?;

    //
    // Impulse noise
    let data = SourceSampleType::GrowingSlow(false).generate();
    let common_wide_abs = NoiseTransform::Impulse { probability: 0.1, alpha: 0.2, beta: 0.2, min: Strength::Absolute(-50.0), max: Strength::Absolute(50.0), seed: None };
    let rare_uniform_rel = NoiseTransform::Impulse { probability: 0.1, alpha: 1.0, beta: 1.0, min: Strength::Relative(-5.0), max: Strength::Relative(5.0), seed: None };
    let common_uniform_rel = NoiseTransform::Impulse { probability: 1.0, alpha: 1.0, beta: 1.0, min: Strength::Absolute(-25.0), max: Strength::Absolute(25.0), seed: None };
    let s_and_p = NoiseTransform::Impulse { probability: 0.1, alpha: 0.0, beta: 0.0, min: Strength::Absolute(-25.0), max: Strength::Absolute(25.0), seed: None };
    generate_plot(
        data.clone(),
        vec![
            ("p=0.1, N(0.2,0.2), min=Abs(-50), max=Abs(50)", data.transformed(&common_wide_abs)),
            ("Uniform: p=1.0, N(1.0,1.0), min=Rel(-5.0), max=Rel(5.0)", data.transformed(&rare_uniform_rel)),
            ("Uniform: p=1.0, N(1.0,1.0), min=Abs(-25), max=Abs(25)", data.transformed(&common_uniform_rel)),
            ("Salt & Pepper: p=0.1, N(0.0,0.0), min=Abs(-25), max=Abs(25)", data.transformed(&s_and_p)),
        ],
        true,
        "Impulse Noise",
        "impulse_example.png",
    )?;

    //
    // Normalization
    //

    //
    // Domain normalization
    let data = SourceSampleType::GrowingSlow(true).generate();
    let zero_to_one = NormalizationTransform::Domain { min: 0.0, max: 1.0 };
    let minus_one_to_one = NormalizationTransform::Domain { min: -1.0, max: 1.0 };
    generate_plot(
        data.clone(),
        vec![
            ("[0, 1]", data.transformed(&zero_to_one)),
            ("[-1, 1]", data.transformed(&minus_one_to_one)),
        ],
        false,
        "Domain Normalization",
        "domain_normalization_example.png",
    )?;

    //
    // Clip normalization
    let data = SourceSampleType::Growing(true).generate();
    let clip_40 = NormalizationTransform::Clip { min: 4000.0, max: 6000.0 };
    let clip_20 = NormalizationTransform::Clip { min: 2000.0, max: 8000.0 };
    generate_plot(
        data.clone(),
        vec![
            ("[4000, 6000]", data.transformed(&clip_40)),
            ("[2000, 8000]", data.transformed(&clip_20)),
        ],
        true,
        "Clip Normalization",
        "clip_normalization_example.png",
    )?;

    //
    // Mean subtraction normalization
    let data = SourceSampleType::GrowingSlow(true).generate();
    let mean_sub = NormalizationTransform::MeanSubtraction;
    generate_plot(
        data.clone(),
        vec![
            ("Mean Subtraction", data.transformed(&mean_sub)),
        ],
        false,
        "Mean Subtraction Normalization",
        "mean_subtraction_example.png",
    )?;

    //
    // Z-score normalization
    let data = SourceSampleType::GrowingSlow(true).generate();
    let z_score = NormalizationTransform::ZScore;
    generate_plot(
        data.clone(),
        vec![
            ("Z-Score", data.transformed(&z_score)),
        ],
        false,
        "Z-Score Normalization",
        "z_score_normalization_example.png",
    )?;

    //
    // Scale transforms
    //

    //
    // Shift 
    let data = SourceSampleType::Growing(false).generate();
    let shift_up = ScaleTransform::Shift(500.0);
    let shift_down = ScaleTransform::Shift(-500.0);
    generate_plot(
        data.clone(),
        vec![
            ("Shift +500", data.transformed(&shift_up)),
            ("Shift -500", data.transformed(&shift_down)),
        ],
        false,
        "Shift Transform",
        "shift_example.png",
    )?;

    //
    // Linear
    let data = SourceSampleType::Growing(false).generate();
    let linear_double = ScaleTransform::Linear(2.0);
    let linear_half = ScaleTransform::Linear(0.5);
    let linear_invert = ScaleTransform::Linear(-1.0);
    generate_plot(
        data.clone(),
        vec![
            ("Linear x2", data.transformed(&linear_double)),
            ("Linear x0.5", data.transformed(&linear_half)),
            ("Linear x-1", data.transformed(&linear_invert)),
        ],
        false,
        "Linear Transform",
        "linear_example.png",
    )?;

    //
    // Quadratic
    let data = SourceSampleType::GrowingSlow(false).generate();
    let half_coeff = ScaleTransform::Quadratic(0.5);
    let double_coeff = ScaleTransform::Quadratic(2.0);
    generate_plot(
        data.clone(),
        vec![
            ("Quadratic coef=0.5", data.transformed(&half_coeff)),
            ("Quadratic coef=2.0", data.transformed(&double_coeff)),
        ],
        false,
        "Quadratic Transform",
        "quadratic_example.png",
    )?;

    //
    // Cubic
    let data = SourceSampleType::GrowingSlow(false).generate();
    let half_coeff = ScaleTransform::Cubic(0.5);
    let double_coeff = ScaleTransform::Cubic(2.0);
    generate_plot(
        data.clone(),
        vec![
            ("Cubic coef=0.5", data.transformed(&half_coeff)),
            ("Cubic coef=2.0", data.transformed(&double_coeff)),
        ],
        false,
        "Cubic Transform",
        "cubic_example.png",
    )?;

    //
    // Exponential
    let data = SourceSampleType::GrowingSlow(false).generate();
    let half_coeff = ScaleTransform::Exponential(0.5, 0.8);
    let double_coeff = ScaleTransform::Exponential(2.0, 1.2);
    generate_plot(
        data.clone(),
        vec![
            ("Exponential exp=0.5, coef=0.8", data.transformed(&half_coeff)),
            ("Exponential exp=2.0, coef=1.2", data.transformed(&double_coeff)),
        ],
        false,
        "Exponential Transform",
        "exponential_example.png",
    )?;

    //
    // Logarithmic (base, factor)
    let data = SourceSampleType::GrowingSlow(false).generate().apply_shift_scale(10.0);
    let log_base_10 = ScaleTransform::Logarithmic(10.0, 2.0);
    let log_base_e = ScaleTransform::Logarithmic(std::f64::consts::E, 1.0);

    generate_plot(
        data.clone(),
        vec![
            ("Logarithmic base=10, factor=2.0", data.transformed(&log_base_10)),
            ("Logarithmic base=e, factor=1.0", data.transformed(&log_base_e)),
        ],
        false,
        "Logarithmic Transform",
        "logarithmic_example.png",
    )?;

    Ok(())
}

/// The type of sample data to generate
/// The boolean indicates the presence of normal noise
enum SourceSampleType {
    Flat(bool),
    Growing(bool),
    GrowingSlow(bool),
}
impl SourceSampleType {
    pub fn generate(self) -> Vec<(f64, f64)> {
        let mut data: Vec<(f64, f64)> = Vec::with_capacity(100);
        match self {
            SourceSampleType::Flat(_) => {
                for i in 0..100 {
                    data.push((i as f64, 10.0));
                }
            }
            SourceSampleType::Growing(_) => {
                for i in 0..100 {
                    data.push((i as f64, (i * i + 10) as f64));
                }
            }
            SourceSampleType::GrowingSlow(_) => {
                for i in 0..100 {
                    data.push((i as f64, (i as f64).sqrt()));
                }
            }
        }
        if self.has_noise() {
            data = data.apply_normal_noise(Strength::Relative(0.3), None);
        }
        data
    }

    pub fn has_noise(&self) -> bool {
        match self {
            SourceSampleType::Flat(noise) => *noise,
            SourceSampleType::Growing(noise) => *noise,
            SourceSampleType::GrowingSlow(noise) => *noise,
        }
    }
}

/// This is how I actually build the plots you see in the documentation for the transforms module
fn generate_plot(
    og_data: Vec<(f64, f64)>,
    mut samples: Vec<(&str, Vec<(f64, f64)>)>,
    shift: bool,
    title: &str,
    filename: &str,
) -> Result<(), String> {
    const SHIFT_BY: f64 = 0.5;

    //
    // Get the y range
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, y) in &og_data {
        if *y < y_min {
            y_min = *y;
        }
        if *y > y_max {
            y_max = *y;
        }
    }
    for (_, dataset) in &samples {
        for &(_, y) in dataset {
            if y < y_min {
                y_min = y;
            }
            if y > y_max {
                y_max = y;
            }
        }
    }

    // Optionally shift the y range a bit for better visibility
    if shift {
        let y_range = y_max - y_min;
        let y_shift = SHIFT_BY * y_range;
        y_max += SHIFT_BY * y_range * (samples.len() as f64);

        for (i, (_, dataset)) in samples.iter_mut().enumerate() {
            for &mut (_, ref mut y) in dataset {
                *y += y_shift * ((i + 1) as f64);
            }
        }
    }

    //
    // Now we can create the plot
    let filename = plotting::plot_directory(filename);
    let options = plotting::PlotOptions {
        title: title.to_string(),
        x_range: None,
        y_range: Some(y_min..y_max),
        ..Default::default()
    };
    let root = plotting::plotters::Root::new(&filename, options.size);
    let element =
        plotting::PlottingElement::Data(og_data.clone(), Some("Original Data".to_string()));
    let mut plot: plotting::Plot<plotting::plotters::Backend, _> =
        plotting::Plot::new(&root, options, &element)
            .map_err(|e: plotting::plotters::Error<'_>| e.to_string())?;

    //
    // Add each transformed dataset
    for (label, dataset) in samples {
        plot.with_element(&plotting::PlottingElement::Data(
            dataset,
            Some(label.to_string()),
        ))
        .map_err(|e: plotting::plotters::Error<'_>| e.to_string())?;
    }

    plot.finish()
        .map_err(|e: plotting::plotters::Error<'_>| e.to_string())?;
    drop(root);
    Ok(())
}
