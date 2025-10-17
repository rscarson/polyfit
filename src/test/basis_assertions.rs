//! A set of macros and functions for testing polynomial bases.
use crate::{
    assert_close,
    basis::{Basis, OrthogonalBasis},
    value::Value,
};

/// Asserts that a basis correctly fills a matrix row with expected values.
///
/// Also verifies that the `start_index` parameter is respected by filling the row
/// multiple times with increasing offsets.
///
/// # Parameters
/// - `basis`: The basis instance to test (must implement [`crate::basis::Basis`]).
/// - `x`: The point at which to evaluate the basis functions.
/// - `expected`: Slice of expected values corresponding to each basis function.
///
/// # Panics
/// Panics if any filled matrix value deviates from the expected value.
pub fn assert_basis_matrix_row<B: Basis<T>, T: Value>(basis: &B, x: T, expected: &[T]) {
    let mut zeros = 0;
    let x = basis.normalize_x(x);
    while zeros <= expected.len() {
        let mut matrix = nalgebra::DMatrix::<T>::zeros(1, expected.len());
        basis.fill_matrix_row(zeros, x, matrix.row_mut(0));

        // Make sure the first zeros elements are 0
        for i in 0..zeros {
            assert_eq!(matrix[(0, i)], T::zero(), "Matrix col {i} should be zero");
        }

        basis.fill_matrix_row(0, x, matrix.row_mut(0));
        for i in 0..expected.len() {
            assert_close!(matrix[(0, i)], expected[i], "Matrix col {i}");
        }

        zeros += 1;
    }
}

/// Asserts that a basis evaluates to expected values at a given x.
///
/// # Parameters
/// - `basis`: The basis instance to test (must implement [`crate::basis::Basis`]).
/// - `x`: The point at which to evaluate the basis functions.
/// - `expected`: Slice of expected values corresponding to each basis function.
/// - `tol`: Tolerance for comparison.
///
/// # Panics
/// Panics if any basis function value deviates from the expected value by more than `tol`
pub fn assert_basis_functions_close<B: Basis<T>, T: Value>(
    basis: &B,
    x: T,
    expected: &[T],
    tol: T,
) {
    let mut actual = vec![T::zero(); expected.len()];
    for i in 0..expected.len() {
        actual[i] = basis.solve_function(i, x);
    }

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.abs_sub(e) > tol {
            eprintln!("Expected ∑{expected:?}");
            eprintln!("Got      ∑{actual:?}");
            panic!("Basis function {i} differs: {a:?} != {e:?} (tol {tol:?})");
        }
    }
}

/// Tests that a basis is orthogonal over a set of x values.
///
/// Constructs the Gram matrix using the basis's `gauss_matrix` method, and checks:
/// - Diagonal elements match the expected normalization within `tol`.
/// - Off-diagonal elements are close to zero within `tol`.
///
/// On failure, prints the Gram matrix for debugging.
///
/// # Parameters
/// - `basis`: The basis instance to test (must implement [`crate::basis::OrthogonalBasis`]).
/// - `functions`: Number of basis functions to test.
/// - `nodes`: Number of quadrature nodes to use for integration (must be >= `functions`).
/// - `tol`: Tolerance for orthogonality checks.
///
/// # Panics
/// Panics if the orthogonality conditions are not met.
pub fn assert_basis_orthogonal<B, T>(basis: &B, functions: usize, nodes: usize, tol: T)
where
    T: Value,
    B: OrthogonalBasis<T>,
{
    assert!(nodes >= functions, "need >= `functions` quadrature nodes");
    let gram_matrix = basis.gauss_matrix(functions, nodes);
    for i in 0..functions {
        for j in i..functions {
            let val = gram_matrix[(i, j)];
            if i == j {
                let expected = basis.gauss_normalization(i);
                let err = Value::abs(val - expected);
                assert!(
                    err <= tol,
                    "gram[{i},{j}] : {val:?} != {expected:?} ; {err:?} > {tol:?}\n{gram_matrix}"
                );
            } else {
                let abs_val = Value::abs(val);
                assert!(
                    abs_val <= tol,
                    "gram[{i},{j}] : {val:?} != 0 ; {abs_val:?} > {tol:?}\n{gram_matrix}"
                );
            }
        }
    }
}

/// Asserts that a basis normalizes input values correctly.
///
/// # Parameters
/// - basis: The basis instance to test (must implement [`crate::basis::Basis`]).
/// - `src_range`: Source range of input values (e.g., `(0.0, 1.0)`).
/// - `dst_range`: Expected normalized range (e.g., `(−1.0, 1.0)`).
///
/// # Panics
/// Panics if the normalized start or end values deviate from the expected range by more than `T::epsilon()`.
pub fn assert_basis_normalizes<B: Basis<T>, T: Value>(
    basis: &B,
    src_range: (T, T),
    dst_range: (T, T),
) {
    let min = basis.normalize_x(src_range.0);
    assert_close!(min, dst_range.0, "Min normalization failed");

    let max = basis.normalize_x(src_range.1);
    assert_close!(max, dst_range.1, "Max normalization failed");
}

/// Uses a numerical method to comfirm that f'(x) is the derivative of f(x), and that f''(x) is the derivative of f'(x).
macro_rules! test_derivation {
    ($f:expr, $norm:expr $(, with_reverse=$bool:literal)?) => {
        let norm = $norm;
        let f = &$f;

        let domain = $norm.src_range();
        let domain = domain.0..=domain.1;

        let f_prime = f.derivative().expect("Failed to compute first derivative");
        let f_double_prime = f_prime
            .derivative()
            .expect("Failed to compute second derivative");

        #[cfg(feature = "plotting")]
        {
            let critical_points = $f.critical_points(domain.clone()).expect("Failed to compute critical points");
            let crit_markers = $crate::basis::CriticalPoint::as_plotting_element(&critical_points);

            $crate::plot!([f, f_prime, crit_markers], {
                x_range: Some(*domain.start()..*domain.end()),
            });
        }

        $crate::assert_is_derivative!(f, f_prime, norm, domain);
        $crate::assert_is_derivative!(f_prime, f_double_prime, norm, domain, f_lbl = "f'(x)", fprime_lbl = "f''(x)");

        $(
            if $bool {
                let c0 = f.coefficients()[0];
                let c1 = f.coefficients()[1];

                let f_prime2 = f_double_prime.integral(Some(c1)).expect("Failed to integrate f''(x)");
                let f2 = f_prime2.integral(Some(c0)).expect("Failed to integrate f'(x)");

                $crate::assert_is_derivative!(f_prime2, f_double_prime, norm, &domain, f_lbl = "∫(f'')(x)", fprime_lbl = "f''(x)");
                $crate::assert_is_derivative!(f2, f_prime2, norm, &domain, f_lbl = "∫∫(f'')(x)", fprime_lbl = "∫(f'')(x)");
            }
        )?
    };
}

/// Uses a numerical method to comfirm that g(x) is the integral of f(x), and that h(x) is the integral of g(x).
macro_rules! test_integration {
    ($f:expr, $norm:expr $(, with_reverse=$bool:literal)?) => {
        let normalizer = $norm;
        let f = &$f;

        let domain = $norm.src_range();
        let domain = domain.0..=domain.1;

        let c0 = f.coefficients()[0];
        let c1 = f.coefficients()[1];

        let g = f.integral(Some(c1)).expect("Failed to compute first integral");
        let h = g.integral(Some(c0)).expect("Failed to compute second integral");

        $crate::assert_is_derivative!(g, f, normalizer, domain, fprime_lbl = "g(x)");
        $crate::assert_is_derivative!(
            h,
            g,
            normalizer,
            domain,
            f_lbl = "g(x)",
            fprime_lbl = "h(x)"
        );

        $(
            if $bool {
                let g2 = h.derivative().expect("Failed to compute first derivative");
                let f2 = g2.derivative().expect("Failed to compute second derivative");

                $crate::assert_is_derivative!(
                    h,
                    g2,
                    normalizer,
                    domain,
                    f_lbl = "h(x)",
                    fprime_lbl = "d(h(x))/dx"
                );

                $crate::assert_is_derivative!(
                    g2,
                    f2,
                    normalizer,
                    domain,
                    f_lbl = "d(h(x))/dx",
                    fprime_lbl = "d(d(h(x))/dx)/dx"
                );
            }
        )?
    };
}
