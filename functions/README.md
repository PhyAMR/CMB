# CMB Analysis Functions

This repository contains a collection of Python scripts designed for the analysis of Cosmic Microwave Background (CMB) data. The functions cover a range of tasks from theoretical calculations and data processing to statistical analysis and visualization.

## Directory Structure

```
functions/
├── __init__.py
├── correlation_function.py
├── cosmology.py
├── data.py
├── maps.py
├── plots.py
├── s12.py
├── simulation.py
├── tools.py
└── xiv.py
```

---

## Performance

For detailed performance metrics of the scripts, see the [Performance Analysis](../Performance.md).

---

## Detailed File Descriptions

### `correlation_function.py`
This module provides the core functions to compute the two-point angular correlation function $C(\theta)$ from the CMB angular power spectrum $D_\ell$. 

-   `correlation_func(D_ell, xvals)`: Calculates $C(\theta)$ using the formula:
    $C(\theta) = \sum_{\ell=2}^{\ell_{max}} \frac{2\ell + 1}{2\ell(\ell + 1)} D_\ell P_\ell(\cos\theta)$, where $P_\ell$ are the Legendre polynomials.
-   `correlation_func_err(error, xvals)` & `correlation_func_err2(error, xvals)`: Propagates the errors from the power spectrum ($\Delta D_\ell$) to the correlation function, assuming uncorrelated errors.

### `cosmology.py`
Interfaces with the `camb` library to compute theoretical CMB power spectra and processes MCMC chains from `getdist`.

-   `compute_cl_cor_pl(parss, lmax, xvals)`: Takes a set of cosmological parameters, passes them to CAMB, and returns the theoretical power spectrum ($D_\ell$) and correlation function for a Planck-like cosmology.
-   `chain_calculations(parss, ...)`: A wrapper function that computes a full set of theoretical observables ($D_\ell$, $C(\theta)$, $S_a^b$, $\bar{\langle\xi\rangle}_a^b$) for a single point in a parameter chain.
-   `chain_results(intervals, ...)`: Processes an entire MCMC chain file. For each set of cosmological parameters in the chain, it computes the theoretical observables and aggregates the results, saving them to a file.

### `maps.py`
Includes functions for the manipulation and analysis of CMB maps in the HEALPix format.

-   `map_rot_refl(map_data)`: Performs a rotation and reflection of a HEALPix map to obtain the "opposite view" of the sky. This is crucial for calculating the $C(180^\circ)$ statistic, which compares a point on the sky with the one directly opposite it.
-   `estimate_coef(x, y)`: A utility function to estimate the coefficients of a simple linear regression, used to analyze the correlation between a map and its opposite.

### `s12.py`
Implements the calculation of the $S_a^b$ statistic, defined as the integral of the squared correlation function over a specified angular range.

-   `Tmn(l, l1, l2, a, b)`: Analytically calculates the elements of the $T_{mn}$ matrix, where $T_{mn} = \int_a^b P_n(x)P_m(x) dx$. This pre-computation is essential for efficiently calculating the $S_a^b$ statistic.
-   `S12(D_ell, M)`: Calculates the $S_a^b$ statistic using the pre-computed $T_{mn}$ matrix ($M$) and the power spectrum $D_\ell$.
-   `S12_err(D_ell, D_ell_err, M)` & `S12_err2(...)`: Propagates the errors from $D_\ell$ to the final $S_a^b$ value.

### `simulation.py`
Contains functions to run Monte Carlo (MC) simulations to understand the impact of observational uncertainties.

-   `MC_calculations(data, ...)`: Similar to `chain_calculations` in `cosmology.py`, but operates on a single Monte Carlo realization of the power spectrum.
-   `MC_results(intervals, data, ...)`: Generates a specified number of mock power spectra by sampling from a Gaussian distribution defined by the experimental $D_\ell$ values and their errors. It then computes the distribution of the derived statistics ($S_a^b$, $\bar{\langle\xi\rangle}_a^b$) for this ensemble of mock universes.

### `tools.py`
A collection of helper and utility functions used throughout the project.

-   `legendre(lmax, x)`: Computes the Legendre polynomial $P_\ell(x)$ using a stable three-term recurrence relation. This is a fundamental mathematical tool for the entire analysis.
-   `A_r(r)`: Computes the coefficient $A_r = \frac{(2r-1)!!}{r!}$, which appears in the analytical integration of Legendre polynomials required for the $T_{mn}$ matrix.

### `xiv.py`
Provides functions to calculate the $\bar{\langle\xi\rangle}_a^b$ statistic, defined as the integrated average of the correlation function over a specified angular range.

-   `xivar(D_ell, a, b)`: Calculates $\bar{\langle\xi\rangle}_a^b$ analytically using the formula:
    $\bar{\langle\xi\rangle}_a^b = \frac{1}{b-a} \sum_\ell \frac{D_\ell}{2\ell(\ell+1)} \left[ P_{\ell+1}(x) - P_{\ell-1}(x) \right]_a^b$.
-   `xivar_err(D_ell_err, a, b)` & `xivar_err2(...)`: Propagates the errors from $D_\ell$ to the final $\bar{\langle\xi\rangle}_a^b$ value.