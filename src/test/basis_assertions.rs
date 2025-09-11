/// Macro for testing the construction of a polynomial basis.
///
/// This macro wraps [`fn_test_basis_build`] and verifies that a basis correctly
/// populates a Vandermonde-style row and respects the `start_index` offset.
///
/// # Parameters
/// - `$basis`: The basis instance to test (must implement [`Basis<T>`]).
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
/// - `$basis`: The basis instance to test (must implement [`Basis<T>`]).
/// - `$x`: The point at which to evaluate the basis functions.
/// - `$expected`: Slice of expected values corresponding to each basis function.
///
/// # Panics
/// Panics if any basis function does not match the expected value.
///
/// # Example
/// ```rust
/// # use polyfit::{basis::{Basis, MonomialBasis}, test_basis_functions};
/// let basis = MonomialBasis::new(&[(0.0, 0.0), (1.0, 1.0)]);
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
/// - `$xs`: Slice of x values to evaluate the basis over.
///
/// # Panics
/// Panics if any pair of basis functions are not orthogonal within the given tolerance.
///
/// # Example
/// ```no_run
/// # use polyfit::{basis::{Basis, MonomialBasis}, value::CoordExt, test_basis_orthogonal};
/// let xs: Vec<(f64, f64)> = vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)];
/// let basis = MonomialBasis::new(&xs);
/// test_basis_orthogonal!(basis, &xs.x());
/// ```
#[macro_export]
macro_rules! test_basis_orthogonal {
    ($basis:expr, $xs:expr) => {{
        fn test_basis_orthogonal<B: $crate::basis::Basis<T>, T: $crate::value::Value>(
            basis: &B,
            xs: &[T],
        ) {
            let k = basis.k(xs.len() - 1);
            let max_val = xs
                .iter()
                .map(|&x| {
                    (0..k)
                        .map(|i| $crate::value::Value::abs(basis.solve_function(i, x)))
                        .fold(T::zero(), |a, b| nalgebra::RealField::max(a, b))
                })
                .fold(T::zero(), |a, b| nalgebra::RealField::max(a, b));

            let tol = T::epsilon() * T::try_cast(xs.len() * 10).unwrap_or(T::one()) * max_val;

            for i in 0..k {
                for j in i..k {
                    let mut sum = T::zero();
                    for &x in xs {
                        let val_i = basis.solve_function(i, x);
                        let val_j = basis.solve_function(j, x);
                        sum += val_i * val_j;
                    }

                    if i == j {
                        assert!(
                            $crate::value::Value::abs(sum) > tol,
                            "Basis function {i} has near-zero norm",
                        );
                    } else {
                        assert!(
                            $crate::value::Value::abs(sum) <= tol,
                            "Basis functions {i} and {j} are not orthogonal: inner product = {sum}"
                        );
                    }
                }
            }
        }
        test_basis_orthogonal(&$basis, $xs);
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
    ($basis:expr, $src_range:expr, $dst_range:expr) => {
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
    };
}
