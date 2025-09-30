use criterion::{criterion_group, criterion_main, Criterion};
use polyfit::{
    basis::{
        Basis, ChebyshevBasis, FourierBasis, LaguerreBasis, LegendreBasis, LogarithmicBasis,
        MonomialBasis, ProbabilistsHermiteBasis,
    },
    display::PolynomialDisplay,
    score::Aic,
    statistics::DegreeBound,
    value::Value,
    CurveFit,
};
use std::hint::black_box;

fn gen_sample_data<T: Value>(n: T) -> Vec<(T, T)> {
    let coefs = &[
        T::one(),
        T::from_f64(3.0).unwrap(),
        T::from_f64(5.3).unwrap(),
    ];
    let y = MonomialBasis::new_polynomial(coefs).unwrap();
    y.solve_range(T::one()..=n, T::one())
}

fn auto_fit<B: Basis<T> + PolynomialDisplay<T>, T: Value>(data: &[(T, T)]) -> CurveFit<'_, B, T> {
    CurveFit::<B, T>::new_auto(data, DegreeBound::Relaxed, &Aic).expect("Failed to fit data")
}

fn fit<B: Basis<T> + PolynomialDisplay<T>, T: Value>(
    data: &[(T, T)],
    degree: usize,
) -> CurveFit<'_, B, T> {
    CurveFit::<B, T>::new(data, degree).expect("Failed to fit data")
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
            CriterionTestEntry::new("n=10_000_000", 1e7, gen_sample_data(1e7)),
            CriterionTestEntry::new("n=100_000_000", 1e8, gen_sample_data(1e8)),
        ],
        |b, data| b.iter(|| fit::<ChebyshevBasis, _>(black_box(data), 3)),
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
        |b, (degree, data)| b.iter(|| fit::<ChebyshevBasis, _>(black_box(data), *degree)),
    );

    //
    // Now we compare different bases for the same data
    // First with single degree (3)
    println!("Benchmarking fit vs basis (Degree=3, n=1000)...");
    let samples = gen_sample_data(1e3);
    let mut group = c.benchmark_group("fit_vs_basis");
    group.bench_function("Monomial", |b| {
        b.iter(|| fit::<MonomialBasis, _>(black_box(&samples), 3))
    });
    group.bench_function("Chebyshev", |b| {
        b.iter(|| fit::<ChebyshevBasis, _>(black_box(&samples), 3))
    });
    group.bench_function("Legendre", |b| {
        b.iter(|| fit::<LegendreBasis, _>(black_box(&samples), 3))
    });
    group.bench_function("Hermite", |b| {
        b.iter(|| fit::<ProbabilistsHermiteBasis, _>(black_box(&samples), 3))
    });
    group.bench_function("Laguerre", |b| {
        b.iter(|| fit::<LaguerreBasis, _>(black_box(&samples), 3))
    });
    group.bench_function("Fourier", |b| {
        b.iter(|| fit::<FourierBasis, _>(black_box(&samples), 3))
    });
    group.bench_function("Logarithmic", |b| {
        b.iter(|| fit::<LogarithmicBasis, _>(black_box(&samples), 3))
    });
    group.finish();

    //
    // Now with auto degree selection
    let samples = gen_sample_data(1e3);
    let n_models = DegreeBound::Relaxed.max_degree(samples.len());
    println!("Benchmarking auto fit vs basis (n=1000, Candidates={n_models})...");
    let mut group = c.benchmark_group("auto_fit_vs_basis");
    group.bench_function("Monomial", |b| {
        b.iter(|| auto_fit::<MonomialBasis, _>(black_box(&samples)))
    });
    group.bench_function("Chebyshev", |b| {
        b.iter(|| auto_fit::<ChebyshevBasis, _>(black_box(&samples)))
    });
    group.bench_function("Legendre", |b| {
        b.iter(|| auto_fit::<LegendreBasis, _>(black_box(&samples)))
    });
    group.bench_function("Hermite", |b| {
        b.iter(|| auto_fit::<ProbabilistsHermiteBasis, _>(black_box(&samples)))
    });
    group.bench_function("Laguerre", |b| {
        b.iter(|| auto_fit::<LaguerreBasis, _>(black_box(&samples)))
    });
    group.bench_function("Fourier", |b| {
        b.iter(|| auto_fit::<FourierBasis, _>(black_box(&samples)))
    });
    group.bench_function("Logarithmic", |b| {
        b.iter(|| auto_fit::<LogarithmicBasis, _>(black_box(&samples)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn get_data_for_run<T: Value, V>(
    group_id: &str,
    tests: &[CriterionTestEntry<T, V>],
) -> Vec<(T, T)> {
    // Each test corresponds to a different x value in the series
    tests
        .iter()
        .map(|test| {
            let y = get_sample_for_run(group_id, &test.id);
            let y = T::from_f64(y).unwrap_or(T::nan());
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

struct CriterionTestEntry<T: Value, V> {
    id: String,
    x: T,
    values: V,
}
impl<T: Value, V> CriterionTestEntry<T, V> {
    pub fn new(id: &str, x: T, values: V) -> Self {
        Self {
            id: id.to_string(),
            x,
            values,
        }
    }
}

fn test_linear_criterion_group<T: Value, F, V>(
    c: &mut Criterion,
    id: &str,
    samples: &[CriterionTestEntry<T, V>],
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
    let linear_fit = fit::<MonomialBasis<T>, T>(&data, 1);
    polyfit::assert_r_squared!(linear_fit);
    polyfit::plot!(linear_fit, prefix = id);
}
