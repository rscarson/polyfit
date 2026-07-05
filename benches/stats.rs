use criterion::{criterion_group, criterion_main, Criterion};
use polyfit::{
    function,
    statistics::{mean, median, median_absolute_deviation},
    value::CoordExt,
};
use rand::seq::SliceRandom;

use std::hint::black_box;

fn bench_with_count(c: &mut Criterion, count: f64) {
    function!(const F(x) = 5.3 x^3 - 2.1 x^2 + 0.5 x - 1.0);
    let mut dataset = F.solve_range(0.0..=count, 1.0).y();
    /* shuffle */
    dataset.shuffle(&mut rand::rng());

    let mut group = c.benchmark_group(format!("bench_with_count_{}", count));

    //
    // First median, because it's the slowest
    // Need a slice of f64 for the median function
    println!("Benchmarking median of {} values...", dataset.len());
    group.bench_function("median", |b| {
        b.iter_with_setup(
            || dataset.clone(),
            |mut data| {
                let _ = black_box(median(black_box(&mut data)));
            },
        )
    });

    //
    // Mean because it's simple
    println!("Benchmarking mean of {} values...", dataset.len());
    group.bench_function("mean", |b| {
        b.iter_with_setup(
            || dataset.iter().copied(),
            |data| {
                let _ = black_box(mean(black_box(data)));
            },
        )
    });

    //
    // Median absolute deviation because it's the heaviest of the three, as it needs to compute the median twice
    println!(
        "Benchmarking median absolute deviation of {} values...",
        dataset.len()
    );
    group.bench_function("median_absolute_deviation", |b| {
        b.iter_with_setup(
            || dataset.clone(),
            |data| {
                let _ = black_box(median_absolute_deviation(black_box(data)));
            },
        )
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_with_count(c, 100.0);
    bench_with_count(c, 1_000.0);
    bench_with_count(c, 10_000.0);
    bench_with_count(c, 100_000.0);
    bench_with_count(c, 1_000_000.0);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
