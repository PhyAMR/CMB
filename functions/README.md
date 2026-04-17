This Python package contains the modularized numerical building blocks for the CMB angular correlation analysis. It provides correlation-function and power-spectrum utilities, map helpers, Monte Carlo pipelines, and plotting/statistics tools.

## Purpose

Provide a well-organized, reusable Python package that scripts and notebooks import for all numerical work in the CMB analysis pipeline. This modular design enables clean separation of concerns and facilitates testing and maintenance.

## Module Descriptions

### Core Analysis Modules

- **`correlation_function.py`** — Compute the two-point angular correlation function C(θ) from power spectrum data (D_ell).
  - `correlation_func(D_ell, xvals)`: Main function to compute C(θ) from D_ell using Legendre polynomial expansion.
  - `correlation_func_err(error, xvals)`: Error propagation for correlation function (vectorized approach 1).
  - `correlation_func_err2(error, xvals)`: Error propagation for correlation function (vectorized approach 2).

- **`cosmo.py`** — Interface with CAMB and GetDist for theoretical cosmology calculations.
  - `compute_cl_cor_pl(parss, lmax, xvals)`: Compute power spectrum and correlation for standard ΛCDM.
  - `compute_cl_cor_dv(parss, lmax, xvals)`: Compute power spectrum and correlation with dark energy parameterization.
  - `expand_dict_values(dict1, dict2)`: Utility to expand scalar parameters to arrays matching the first dict's length.
  - `chain_calculations(parss, data_loader, intervals, c)`: Process a single MCMC sample to compute D_ell, correlation, and derived statistics (S12, xivar).
  - `chain_results(data_loader, intervals, roots, name, n)`: Load and process complete MCMC chains, compute statistics for all samples, and save results.

- **`data.py`** — Load and manage CMB power spectrum data.
  - `Data_loader` class: Central interface for experimental data.
    - `__init__(path, lmax, n_xvals, time_it)`: Initialize with Planck power spectrum data.
    - `get_correlation_function(force_recalc)`: Compute or retrieve the experimental correlation function.
    - `get_xivar(a, b)`: Compute xivar statistic over angular range [a, b].
    - `get_s12(a, b)`: Compute S12 statistic with error propagation.
    - `_load_matrix(a, b)`: Load precomputed Tmn matrix for S12 computation.
    - `experimental_values(intervals)`: Generate dict of all experimental statistics (C180, S12, xivar) for comparison.

- **`xiv.py`** — Compute the xivar (average correlation) statistic.
  - `xivar(D_ell, a, b)`: Compute xivar analytically using loop-based Legendre evaluation.
  - `xivar2(D_ell, a, b)`: Vectorized xivar computation using numpy arrays.
  - `xivar_err(D_ell_err, a, b)`: Error propagation for xivar.
  - `xivar_err2(D_ell_err, a, b)`: Alternative error propagation for xivar.
  - `xiv_numerical(D_ell, a, b, n_points)`: Numerical integration approach to xivar.

- **`s12.py`** — Compute the S12 (integrated squared correlation) statistic.
  - `Tmn(l, l1, l2, a, b)`: Compute and save high-precision Tmn matrix (integral of Legendre polynomial products).
  - `S12(D_ell, M)`: Compute S12 analytically using precomputed Tmn matrix.
  - `S12_vec(D_ell, M)`: Vectorized S12 computation.
  - `S12_err(D_ell, D_ell_err, M)`: Error propagation for S12.
  - `S12_err2(D_ell, D_ell_err, M)`: Alternative error propagation for S12.
  - `s12_numerical(D_ell, a, b, n_points)`: Numerical integration approach to S12.

### Simulation & Monte Carlo

- **`simulation.py`** — Generate Monte Carlo simulations and synthetic power spectra.
  - `MC_calculations(data, data_loader, intervals)`: Process one synthetic realization to compute D_ell, correlation, and derived statistics.
  - `MC_results(data_loader, intervals, n)`: Generate n Monte Carlo realizations from experimental data with error bars, compute their statistics, and save results.

### Visualization & Statistics

- **`plots.py`** — Create publication-quality plots comparing theory, simulation, and experiment.
  - `CorrelationPlots` class: Main plotting interface.
    - `__init__(data_loader)`: Initialize with experimental data.
    - `plot_corr_with_xivar(...)`: Plot correlation function with xivar intervals and comparison bands.
    - `plot_corr_with_S12(...)`: Plot correlation function squared with S12 intervals and comparison bands.
    - `create_histogram_grid(...)`: Create multi-panel histograms comparing theory/simulation distributions to experimental values.
    - `plot_power_and_correlation(...)`: Side-by-side plot of power spectrum and correlation function with confidence bands.
  - `MapPlots` class: Plotting helpers for HEALPix maps.
    - `plot180(...)`: Scatter plot of map vs. opposite-hemisphere map with regression line.
    - `map_contours(...)`: 2D histogram contours for map correlation analysis.
  - Helper functions: `_save_or_show_plot(...)`, `_apply_style(...)`, `_plot_statistic_interval(...)`, `_add_dist_overlay(...)`.

- **`unified_stats.py`** — Unified percentile and p-value computation for comparing statistics to distributions.
  - Functions for computing significance levels and quantiles in a consistent manner across the pipeline.

- **`generate_result_tables.py`** — Convert scalar statistics to LaTeX tables.
  - `generate_all_tables(...)`: Top-level function that takes run directory and produces formatted LaTeX tables and optionally a master PDF.

- **`getdist_stats.py`** — Integration with GetDist for MCMC chain analysis.
  - Helpers to build GetDist MCSamples objects from run outputs.
  - Functions to generate publication-ready LaTeX tables from statistic distributions.

### Utilities

- **`maps.py`** — HEALPix map and pixel utilities.
  - `map_rot_refl(map_data)`: Compute opposite-hemisphere map (180° rotation + reflection).
  - `estimate_coef(x, y)`: Simple linear regression coefficient estimation (used for map correlation analysis).

- **`tools.py`** — Low-level mathematical and utility functions.
  - `@timeit` decorator: Measure and report execution time.
  - `legendre(lmax, x)`: Compute Legendre polynomial P_lmax(x) using three-term recurrence.
  - `A_r(r)`: Compute high-precision coefficient A_r = (2r-1)!! / r! for use in Tmn computation.

- **`__init__.py`** — Package initializer; imports key classes and functions for convenient access.

- **`FLOWCHARTS.md`** — Detailed flowcharts documenting the execution flow of each module's major functions (see [FLOWCHARTS.md](FLOWCHARTS.md) for comprehensive diagrams).

## Usage Notes

### Basic Import Pattern
```python
from functions import data, cosmo, plots, s12, xiv, simulation
```

### Key Classes & Functions

1. **Data_loader** (data.py):
   ```python
   DL = Data_loader(lmax=30)
   exp_values = DL.experimental_values(intervals)  # Returns C180, S12, xivar
   ```

2. **CorrelationPlots** (plots.py):
   ```python
   plotter = CorrelationPlots(DL)
   plotter.plot_corr_with_xivar(corr_theory, est_df, intervals, name, save_path)
   ```

3. **MCMC/Theory Processing** (cosmo.py):
   ```python
   chain_results(DL, intervals, roots, name, n=1000, best_fit=True)
   ```

4. **Simulation** (simulation.py):
   ```python
   MC_results(DL, intervals, n=1000)
   ```

### File Structure & Outputs

- **Input Data**: Planck power spectrum (D_ell) loaded from `maps/` directory.
- **Precomputed Matrices**: Tmn matrices stored in `files/matrix/` with naming convention `Tmn__{upper}__{lower}.npy`.
- **Pickle Cache**: MCMC chain results cached in `files/pickle/` for fast reuse.
- **Run Results**: Statistics and arrays saved in `runs/run_{lmax}_{n_samples}_{mode}_{hash}/`:
  - `theory_bestfit/` — Results from best-fit mode.
  - `theory_mcmc/` — Results from MCMC mode.
  - `simulation/` — Results from Monte Carlo simulations.
  - Each contains: scalar statistics files (s12.npy, xiv.npy, etc.) and arrays (D_ell.npy, Corr.npy).

## Configuration & Entry Points

- See the [root README.md](../README.md) for how to run the full pipeline via `main.py`.
- See [FLOWCHARTS.md](FLOWCHARTS.md) for detailed control flow diagrams.

## Contributing

- Add unit-tested helpers to `tools.py` and keep the public API stable.
- When adding new statistics, include example calls in the main notebooks.
- Update this README and `FLOWCHARTS.md` to document new modules or major functions.
- Ensure error propagation functions are tested against numerical alternatives.
