use criterion::{criterion_group, criterion_main, Criterion};
use polyfit::{
    function,
    statistics::{mean, median, median_absolute_deviation},
    value::CoordExt,
};

use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    function!(const F(x) = 5.3 x^3 - 2.1 x^2 + 0.5 x - 1.0);
    let dataset = F.solve_range(0.0..=1000.0, 0.1);

    //
    // First median, because it's the slowest
    // Need a slice of f64 for the median function
    let mut data = dataset.y();
    println!("Benchmarking median of {} values...", data.len());
    c.bench_function("median", |b| {
        b.iter(|| {
            let _ = black_box(median(black_box(&mut data)));
        })
    });

    //
    // Mean because it's simple
    println!("Benchmarking mean of {} values...", dataset.len());
    c.bench_function("mean", |b| {
        b.iter(|| {
            let _ = black_box(mean(black_box(dataset.y_iter())));
        })
    });

    //
    // Median absolute deviation because it's the heaviest of the three, as it needs to compute the median twice
    let other_data = F.solve_range(500.0..=1500.0, 0.1);
    println!(
        "Benchmarking median absolute deviation of {} values...",
        other_data.len()
    );
    c.bench_function("median_absolute_deviation", |b| {
        b.iter(|| {
            let _ = black_box(median_absolute_deviation(
                black_box(dataset.y_iter()),
                black_box(other_data.y_iter()),
            ));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
