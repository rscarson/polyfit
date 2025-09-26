use criterion::{criterion_group, criterion_main, Criterion};
use polyfit::{
    basis::{
        Basis, ChebyshevBasis, FourierBasis, LaguerreBasis, LegendreBasis, MonomialBasis,
        ProbabilistsHermiteBasis,
    },
    display::PolynomialDisplay,
    score::Aic,
    statistics::DegreeBound,
    value::Value,
    CurveFit,
};
use std::hint::black_box;

fn gen_sample_data(n: f64) -> Vec<(f64, f64)> {
    use polyfit::function;
    function!(y(x) = 5.3 x^2 + 3.0 x + 1.0);
    y.solve_range(1.0..=n, 1.0)
}

fn auto_fit<B: Basis<T> + PolynomialDisplay<T>, T: Value>(data: &[(T, T)]) -> CurveFit<B, T> {
    CurveFit::<B, T>::new_auto(data, DegreeBound::Relaxed, &Aic).expect("Failed to fit data")
}

fn fit<B: Basis<T> + PolynomialDisplay<T>, T: Value>(
    data: &[(T, T)],
    degree: usize,
) -> CurveFit<B, T> {
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
            CriterionTestEntry::new("n=1000", 1e3, gen_sample_data(1e3)),
            CriterionTestEntry::new("n=10000", 1e4, gen_sample_data(1e4)),
            CriterionTestEntry::new("n=100000", 1e5, gen_sample_data(1e5)),
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
            CriterionTestEntry::new("Degree=1", 1, (1, &samples)),
            CriterionTestEntry::new("Degree=2", 2, (2, &samples)),
            CriterionTestEntry::new("Degree=3", 3, (3, &samples)),
            CriterionTestEntry::new("Degree=4", 4, (4, &samples)),
            CriterionTestEntry::new("Degree=5", 5, (5, &samples)),
        ],
        |b, (degree, data)| b.iter(|| fit::<ChebyshevBasis, _>(black_box(data), *degree)),
    );

    //
    // Now we compare different bases for the same data
    // First with single degree (3)
    println!("Benchmarking fit vs basis (Degree=3, n=1000)...");
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
    group.finish();

    //
    // Now with auto degree selection
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
    let linear_fit = fit::<MonomialBasis, _>(&data, 1);
    polyfit::assert_r_squared!(linear_fit);
}
