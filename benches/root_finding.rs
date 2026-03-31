use criterion::{criterion_group, criterion_main, Criterion};
use polyfit::{
    basis::{
        Basis, ChebyshevBasis, DifferentialBasis, FourierBasis, MonomialBasis, RootFindingBasis,
    },
    display::PolynomialDisplay,
    Polynomial,
};

use std::{hint::black_box, ops::RangeInclusive};

fn criterion_benchmark(c: &mut Criterion) {
    //
    // First, fourier - small degree, then medium, then large
    //

    let fourier_small = FourierBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, 3.5]).unwrap();
    bench_roots_of(fourier_small, 0.0..=100.0, c);

    let fourier_medium =
        FourierBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5]).unwrap();
    bench_roots_of(fourier_medium, 0.0..=100.0, c);

    let fourier_large = FourierBasis::new_polynomial(
        (0.0, 100.0),
        &[0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5],
    )
    .unwrap();
    bench_roots_of(fourier_large, 0.0..=100.0, c);

    //
    // Now Chebyshev
    //

    let chebyshev_small = ChebyshevBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, 3.5]).unwrap();
    bench_roots_of(chebyshev_small, 0.0..=100.0, c);

    let chebyshev_medium =
        ChebyshevBasis::new_polynomial((0.0, 100.0), &[0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5])
            .unwrap();
    bench_roots_of(chebyshev_medium, 0.0..=100.0, c);

    let chebyshev_large = ChebyshevBasis::new_polynomial(
        (0.0, 100.0),
        &[0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5],
    )
    .unwrap();
    bench_roots_of(chebyshev_large, 0.0..=100.0, c);

    //
    // Monomials
    //

    let monomial_small = MonomialBasis::new_polynomial(&[0.5, 2.0, 3.5]).unwrap();
    bench_roots_of(monomial_small, 0.0..=100.0, c);

    let monomial_medium =
        MonomialBasis::new_polynomial(&[0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5]).unwrap();
    bench_roots_of(monomial_medium, 0.0..=100.0, c);

    let monomial_large =
        MonomialBasis::new_polynomial(&[0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5, -1.5, 0.5, 2.0, 3.5])
            .unwrap();
    bench_roots_of(monomial_large, 0.0..=100.0, c);
}

fn bench_roots_of<B>(f: Polynomial<B, f64>, x_range: RangeInclusive<f64>, c: &mut Criterion)
where
    B: Basis<f64> + PolynomialDisplay<f64> + RootFindingBasis<f64> + DifferentialBasis<f64>,
{
    let mut group = c.benchmark_group(format!("Root finding for {f}"));

    //
    // First iteratively
    group.bench_function("Iterative root finding", |b| {
        b.iter(|| {
            let roots = f.iterative_roots(x_range.clone()).unwrap();
            black_box(roots);
        })
    });

    //
    // Then with the closed-form solution
    group.bench_function("Closed-form root finding", |b| {
        b.iter(|| {
            let roots = f.roots(x_range.clone()).unwrap();
            black_box(roots);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
