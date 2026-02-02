import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import time

# Function to generate some test data
def f(x):
    return np.sin(x) + 0.1 * np.random.randn(*x.shape)

# Settings
points_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
degree_list = [1, 2, 3, 4, 5, 10, 20]
repeats = 10  # number of repetitions for median timing

def median_time(func, *args):
    times = []
    for _ in range(repeats):
        start = time.time()
        func(*args)
        elapsed = (time.time() - start) * 1e6  # µs
        times.append(elapsed)
    return np.median(times)

print("Fit vs n (degree=3, Chebyshev):")
for n in points_list:
    x = np.linspace(0, 10, n)
    y = f(x)
    median_elapsed = median_time(lambda xx, yy: Chebyshev.fit(xx, yy, deg=3).convert().coef, x, y)
    print(f"fit_vs_n/n={n} [{median_elapsed:.2f}µs]")

print("\nFit vs degree (n=1000, Chebyshev):")
x = np.linspace(0, 10, 1000)
y = f(x)
for deg in degree_list:
    median_elapsed = median_time(lambda xx, yy, d: Chebyshev.fit(xx, yy, deg=d).convert().coef, x, y, deg)
    print(f"fit_vs_degree/degree={deg} [{median_elapsed:.2f}µs]")
