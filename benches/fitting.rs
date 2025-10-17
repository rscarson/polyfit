use criterion::{criterion_group, criterion_main, Criterion};
use polyfit::{
    basis::{
        Basis, ChebyshevBasis, FourierBasis, LaguerreBasis, LegendreBasis, LogarithmicBasis,
        MonomialBasis, ProbabilistsHermiteBasis,
    },
    display::PolynomialDisplay,
    score::Aic,
    statistics::DegreeBound,
    CurveFit,
};
use std::hint::black_box;

fn gen_sample_data(n: f64) -> Vec<(f64, f64)> {
    let coefs = &[1.0, 3.0, 5.3];
    let y = MonomialBasis::new_polynomial(coefs).unwrap();
    y.solve_range(1.0..=n, 1.0)
}

fn auto_fit<B: Basis<f64> + PolynomialDisplay<f64>>(data: &[(f64, f64)]) -> CurveFit<'_, B, f64> {
    CurveFit::<B, f64>::new_auto(data, DegreeBound::Relaxed, &Aic).expect("Failed to fit data")
}

fn fit<B: Basis<f64> + PolynomialDisplay<f64>>(
    data: &[(f64, f64)],
    degree: usize,
) -> CurveFit<'_, B, f64> {
    CurveFit::<B, f64>::new(data, degree).expect("Failed to fit data")
}

fn criterion_benchmark(c: &mut Criterion) {
    //
    // First we test how the solver scales with data size (Cheb basis only)
    println!("Benchmarking fit vs n (Chebyshev, Degree=3)...");
    test_linear_criterion_group(
        c,
        "fit_vs_n",
        &[
            CriterionTestEntry::new("n=100", 1e2, gen_sample_data(1e2)),
            CriterionTestEntry::new("n=1_000", 1e3, gen_sample_data(1e3)),
            CriterionTestEntry::new("n=10_000", 1e4, gen_sample_data(1e4)),
            CriterionTestEntry::new("n=100_000", 1e5, gen_sample_data(1e5)),
            CriterionTestEntry::new("n=1_000_000", 1e6, gen_sample_data(1e6)),
            #[cfg(feature = "parallel")]
            CriterionTestEntry::new("n=10_000_000", 1e7, gen_sample_data(1e7)),
            #[cfg(feature = "parallel")]
            CriterionTestEntry::new("n=100_000_000", 1e8, gen_sample_data(1e8)),
        ],
        |b, data| b.iter(|| fit::<ChebyshevBasis>(black_box(data), 3)),
    );

    //
    // Now the same but scaling with degree (Cheb basis only)
    println!("Benchmarking fit vs degree (Chebyshev, n=1000)...");
    let samples = gen_sample_data(1e3);
    test_linear_criterion_group(
        c,
        "fit_vs_degree",
        &[
            CriterionTestEntry::new("Degree=1", 1.0, (1, &samples)),
            CriterionTestEntry::new("Degree=2", 2.0, (2, &samples)),
            CriterionTestEntry::new("Degree=3", 3.0, (3, &samples)),
            CriterionTestEntry::new("Degree=4", 4.0, (4, &samples)),
            CriterionTestEntry::new("Degree=5", 5.0, (5, &samples)),
            CriterionTestEntry::new("Degree=10", 10.0, (10, &samples)),
            CriterionTestEntry::new("Degree=20", 20.0, (20, &samples)),
        ],
        |b, (degree, data)| b.iter(|| fit::<ChebyshevBasis>(black_box(data), *degree)),
    );

    //
    // Now we compare different bases for the same data
    // First with single degree (3)
    println!("Benchmarking fit vs basis (Degree=3, n=1000)...");
    let samples = gen_sample_data(1e3);
    let mut group = c.benchmark_group("fit_vs_basis");
    group.bench_function("Monomial", |b| {
        b.iter(|| fit::<MonomialBasis>(black_box(&samples), 3))
    });
    group.bench_function("Chebyshev", |b| {
        b.iter(|| fit::<ChebyshevBasis>(black_box(&samples), 3))
    });
    group.bench_function("Legendre", |b| {
        b.iter(|| fit::<LegendreBasis>(black_box(&samples), 3))
    });
    group.bench_function("Hermite", |b| {
        b.iter(|| fit::<ProbabilistsHermiteBasis>(black_box(&samples), 3))
    });
    group.bench_function("Laguerre", |b| {
        b.iter(|| fit::<LaguerreBasis>(black_box(&samples), 3))
    });
    group.bench_function("Fourier", |b| {
        b.iter(|| fit::<FourierBasis>(black_box(&samples), 3))
    });
    group.bench_function("Logarithmic", |b| {
        b.iter(|| fit::<LogarithmicBasis>(black_box(&samples), 3))
    });
    group.finish();

    //
    // Now with auto degree selection
    let samples = gen_sample_data(1e3);
    let n_models = DegreeBound::Relaxed.max_degree(samples.len());
    println!("Benchmarking auto fit vs basis (n=1000, Candidates={n_models})...");
    let mut group = c.benchmark_group("auto_fit_vs_basis");
    group.bench_function("Monomial", |b| {
        b.iter(|| auto_fit::<MonomialBasis>(black_box(&samples)))
    });
    group.bench_function("Chebyshev", |b| {
        b.iter(|| auto_fit::<ChebyshevBasis>(black_box(&samples)))
    });
    group.bench_function("Legendre", |b| {
        b.iter(|| auto_fit::<LegendreBasis>(black_box(&samples)))
    });
    group.bench_function("Hermite", |b| {
        b.iter(|| auto_fit::<ProbabilistsHermiteBasis>(black_box(&samples)))
    });
    group.bench_function("Laguerre", |b| {
        b.iter(|| auto_fit::<LaguerreBasis>(black_box(&samples)))
    });
    group.bench_function("Fourier", |b| {
        b.iter(|| auto_fit::<FourierBasis>(black_box(&samples)))
    });
    group.bench_function("Logarithmic", |b| {
        b.iter(|| auto_fit::<LogarithmicBasis>(black_box(&samples)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn get_data_for_run<V>(group_id: &str, tests: &[CriterionTestEntry<V>]) -> Vec<(f64, f64)> {
    // Each test corresponds to a different x value in the series
    tests
        .iter()
        .map(|test| {
            let y = get_sample_for_run(group_id, &test.id);
            (test.x, y)
        })
        .collect()
}

fn get_sample_for_run(group_id: &str, test_id: &str) -> f64 {
    #[derive(serde::Deserialize)]
    struct CriterionSamples {
        iters: Vec<f64>,
        times: Vec<f64>,
    }

    let raw = std::fs::read_to_string(format!(
        "target/criterion/{group_id}/{test_id}/new/sample.json"
    ))
    .expect("Failed to read sample data");
    let samples: CriterionSamples =
        serde_json::from_str(&raw).expect("Failed to parse sample data");

    samples
        .iters
        .iter()
        .zip(samples.times.iter())
        .map(|(i, t)| t / i)
        .sum::<f64>()
        / (samples.iters.len() as f64)
}

struct CriterionTestEntry<V> {
    id: String,
    x: f64,
    values: V,
}
impl<V> CriterionTestEntry<V> {
    pub fn new(id: &str, x: impl TryInto<f64>, values: V) -> Self {
        Self {
            id: id.to_string(),
            x: x.try_into().ok().expect("Failed to convert x to f64"),
            values,
        }
    }
}

fn test_linear_criterion_group<F, V>(
    c: &mut Criterion,
    id: &str,
    samples: &[CriterionTestEntry<V>],
    runner: F,
) where
    for<'a, 'b, 'c> F: Fn(&'a mut criterion::Bencher<'b>, &'c V),
{
    //
    // First we test how the solver scales with data size (Cheb basis only)
    let mut group = c.benchmark_group(id);
    for sample in samples {
        group.bench_with_input(&sample.id, &sample.values, &runner);
    }
    group.finish();

    let data = get_data_for_run(id, samples);
    let linear_fit = fit::<MonomialBasis<_>>(&data, 1);
    polyfit::assert_r_squared!(linear_fit);

    #[cfg(feature = "plotting")]
    polyfit::plot!(linear_fit, prefix = id);
}
