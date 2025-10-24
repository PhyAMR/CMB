# Performance Analysis

This document presents the performance metrics of key simulation and analysis scripts in this project.

## `functions/test_simulation.py`

This script benchmarks the Monte Carlo simulation functions.

```
TTCl extraction took 0.0000 seconds.
correlation_func executed in 1.79 seconds
correlation_func took 1.7874 seconds.
Matrix loading for interval (-0.9999999999999999, 0.5) took 0.0003 seconds.
S12 calculation for interval (-0.9999999999999999, 0.5) took 0.0892 seconds.
xivar executed in 0.00 seconds
xivar calculation for interval (-0.9999999999999999, 0.5) took 0.0043 seconds.
Matrix loading for interval (0.866, 0.9999999999999999) took 0.0003 seconds.
S12 calculation for interval (0.866, 0.9999999999999999) took 0.0892 seconds.
xivar executed in 0.00 seconds
xivar calculation for interval (0.866, 0.9999999999999999) took 0.0043 seconds.
test_mc_calculations_performance took 1.9751 seconds to run.
dist_per_cl generation took 0.0016 seconds.
DataFrame creation took 0.0010 seconds.
MC_results function call took 195.9005 seconds.
test_mc_results_performance took 195.9035 seconds to run.

----------------------------------------------------------------------
Ran 2 tests in 197.885s

OK
```

## `functions/xiv.py`

This script tests the performance and accuracy of the `xivar` statistic calculation.

```
--- xivar Performance and Accuracy Test ---
xivar executed in 0.00 seconds
correlation_func executed in 4.95 seconds
Interval [cos(theta)]: [0.0, 0.5] (Theta: [60, 90] degrees)

Analytical Calculation:
  Result: -51.35532340896712
  Execution Time: 0.004341 seconds

Numerical Calculation:
  Result: -51.355323408817284
  Execution Time: 4.951445 seconds

Comparison:
  Absolute Difference: 1.4983214668973233e-10
  Relative Difference: 0.0000%
--- End of Test ---
```

## `functions/s12.py`

This script tests the performance and accuracy of the `S12` statistic calculation.

```
--- S12 Performance and Accuracy Test ---
correlation_func executed in 4.97 seconds
Interval [cos(theta)]: [-1, 0.5] (Theta: [60, 180] degrees)

Analytical Calculation:
  Result: 6780.273422211044
  Execution Time: 0.095593 seconds

Numerical Calculation:
  Result: 6780.374298614481
  Execution Time: 4.965659 seconds

Comparison:
  Absolute Difference: 0.10087640343681414
  Relative Difference: 0.0015%
--- End of Test ---
```