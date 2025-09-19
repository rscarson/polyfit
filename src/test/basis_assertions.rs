/// Macro for testing the construction of a polynomial basis.
///
/// This verifies that a basis correctly
/// populates a Vandermonde-style row and respects the `start_index` offset.
///
/// # Parameters
/// - `$basis`: The basis instance to test (must implement [`crate::basis::Basis`]).
/// - `$x`: The x-value at which to evaluate the basis.
/// - `$matrix_values`: Slice of expected basis function values at `x`.
///
/// # Panics
/// This macro will panic if the filled row does not match the expected values
/// or if the `start_index` behavior is incorrect.
///
/// # Example
/// ```rust
/// # use polyfit::{basis::MonomialBasis, test_basis_build};
/// test_basis_build!(MonomialBasis::default(), 0.5, &[1.0, 0.5, 0.25]);
/// ```
#[macro_export]
macro_rules! test_basis_build {
    ($basis:expr, $x:expr, $matrix_values:expr) => {{
        fn test_basis_build<B: $crate::basis::Basis<T>, T: $crate::value::Value>(
            basis: &B,
            x: T,
            matrix_values: &[T],
        ) {
            let mut zeros = 0;
            let x = basis.normalize_x(x);
            while zeros <= matrix_values.len() {
                let mut matrix = $crate::nalgebra::DMatrix::<T>::zeros(1, matrix_values.len());
                basis.fill_matrix_row(zeros, x, matrix.row_mut(0));

                // Make sure the first zeros elements are 0
                for i in 0..zeros {
                    assert_eq!(matrix[(0, i)], T::zero(), "Matrix col {i} should be zero");
                }

                basis.fill_matrix_row(0, x, matrix.row_mut(0));
                for i in 0..matrix_values.len() {
                    $crate::assert_close!(matrix[(0, i)], matrix_values[i], "Matrix col {i}");
                }

                zeros += 1;
            }
        }
        test_basis_build(&$basis, $x, $matrix_values);
    }};
}

/// Macro for asserting that a polynomial basis evaluates to expected values.
///
/// # Parameters
/// - `$basis`: The basis instance to test (must implement [`crate::basis::Basis`]).
/// - `$x`: The point at which to evaluate the basis functions.
/// - `$expected`: Slice of expected values corresponding to each basis function.
///
/// # Panics
/// Panics if any basis function does not match the expected value.
///
/// # Example
/// ```rust
/// # use polyfit::{basis::{Basis, MonomialBasis}, test_basis_functions};
/// let basis = MonomialBasis::default();
/// let expected = vec![1.0, 0.5, 0.25]; // expected basis function values at x=0.5
/// test_basis_functions!(basis, 0.5, &expected);
/// ```
#[macro_export]
macro_rules! test_basis_functions {
    ($basis:expr, $x:expr, $expected:expr) => {{
        fn test_basis_functions<B: $crate::basis::Basis<T>, T: $crate::value::Value>(
            basis: &B,
            x: T,
            expected: &[T],
        ) {
            for (i, &expected) in expected.iter().enumerate() {
                let value = basis.solve_function(i, x);
                $crate::assert_close!(value, expected, "B_{i}({x})");
            }
        }
        test_basis_functions(&$basis, $x, $expected);
    }};
}

/// Macro to test that a basis is orthogonal over a set of x values.
///
/// # Parameters
/// - `$basis`: The basis instance to test.
/// - `norm_fn`: A function or closure that takes a basis function index and returns its expected norm (integral of the square).
/// - `values`: Slice of x values at which to evaluate the basis functions.
/// - `weights`: Slice of weights corresponding to each x value for weighted quadrature.
/// - `n_funcs`: Number of basis functions to test.
/// - `eps`: Tolerance for orthogonality checks.
///
/// # Panics
/// Panics if any pair of basis functions are not orthogonal within the given tolerance.
///
/// See the implementation on Chebyshev or Legendre polynomials for examples.
#[macro_export]
macro_rules! test_basis_orthogonal {
    (
        $basis:expr, norm_fn = $norm_fn:expr,
        values = $xs:expr, weights = $weights:expr,
        n_funcs = $n_funcs:expr, eps = $tol:expr
    ) => {{
        fn test_orthogonality<T>(basis: impl Fn(usize, T) -> T, norm: impl Fn(usize) -> T, n_funcs: usize, xs: &[T], weights: &[T], tol: T)
        where
            T: $crate::value::Value,
        {
            assert_eq!(xs.len(), weights.len());
            assert!(weights.len() >= n_funcs, "need >= n_funcs quadrature nodes");

            for i in 0..n_funcs {
                for j in i..n_funcs {
                    // weighted quadrature: sum_i w_i * phi_i(x_i) * phi_j(x_i)
                    let mut sum: T = T::zero();
                    for (&x, &w) in xs.iter().zip(weights.iter()) {
                        sum += basis(i, x) * basis(j, x) * w;
                    }

                    if i == j {
                        // exact integral: ∫_{-1..1} P_n^2 = 2/(2n+1)
                        let expected = norm(i);
                        let err = $crate::value::Value::abs(sum - expected);
                        assert!(
                            err <= tol,
                            "Function {i} norm mismatch: got {sum:?} expected {expected:?} err={err:?} > {tol:?}"
                        );
                    } else {
                        let abs_sum = $crate::value::Value::abs(sum);
                        assert!(
                            abs_sum <= tol,
                            "Functions {i} and {j} not orthogonal: inner product = {sum:?} > {tol:?}"
                        );
                    }
                }
            }
        }

        test_orthogonality(
            |j, x| $basis.solve_function(j, x),
            $norm_fn,
            $n_funcs,
            &$xs,
            &$weights,
            $tol,
        );
    }};
}

/// Tests that a basis normalizes input values correctly.
///
/// Use `test_basis_normalizes!` instead of calling the function directly.
///
/// # Parameters
/// - `$basis`: The basis instance to test.
/// - `$src_range`: Source range of input values (e.g., `0.0..1.0`).
/// - `$dst_range`: Expected normalized range (e.g., `-1.0..1.0`).
///
/// # Panics
/// Panics if the normalized start or end values deviate from the expected range by more than `T::epsilon()`.
#[macro_export]
macro_rules! test_basis_normalizes {
    ($basis:expr, $src_range:expr, $dst_range:expr) => {{
        fn test_basis_normalizes<B: $crate::basis::Basis<T>, T: $crate::value::Value>(
            basis: &B,
            src_range: ::std::ops::Range<T>,
            dst_range: ::std::ops::Range<T>,
        ) {
            let min = basis.normalize_x(src_range.start);
            $crate::assert_close!(min, dst_range.start, "Min normalization failed");

            let max = basis.normalize_x(src_range.end);
            $crate::assert_close!(max, dst_range.end, "Max normalization failed");
        }

        test_basis_normalizes(&$basis, $src_range, $dst_range);
    }};
}
