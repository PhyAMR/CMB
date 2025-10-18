# CMB Analysis Functions

This repository contains a collection of Python scripts designed for the analysis of Cosmic Microwave Background (CMB) data. The functions cover a range of tasks from theoretical calculations and data processing to statistical analysis and visualization.

## Folder Structure

The `functions/` directory contains the following modules:

-   `__init__.py`: Makes the `functions` directory a Python package, allowing for modular imports.

-   `correlation_function.py`: Provides functions to compute the two-point angular correlation function `C(theta)` from the CMB angular power spectrum `D_ell`. It also includes methods for propagating errors.

-   `cosmology.py`: Interfaces with the `camb` library to compute theoretical CMB power spectra and correlation functions for different cosmological models. It includes tools for processing MCMC chains from cosmological parameter estimation software like `getdist`.

-   `data.py`: Contains the `Data_loader` class, a utility for loading, parsing, and processing experimental CMB power spectrum data files (e.g., from the Planck satellite). It provides easy access to the power spectrum, errors, and derived quantities.

-   `maps.py`: Includes functions for the manipulation and analysis of CMB maps in the HEALPix format. Key functionalities include rotating and reflecting maps to study statistical isotropy.

-   `plots.py`: A suite of visualization tools for creating plots commonly used in CMB analysis. This includes plotting power spectra, correlation functions, histograms of statistical estimators, and map comparisons.

-   `s12.py`: Implements the calculation of the `S12` statistic, which is defined as the integral of the squared correlation function over a specified angular range. It provides both analytical (matrix-based) and numerical integration methods.

-   `simulation.py`: Contains functions to run Monte Carlo (MC) simulations. It can generate mock power spectra based on experimental data and uncertainties, and then compute the distribution of various cosmological observables and statistics.

-   `tools.py`: A collection of helper and utility functions used throughout the project. This includes mathematical functions like Legendre polynomials and performance-monitoring tools like timing decorators.

-   `xiv.py`: Provides functions to calculate the `xivar` statistic, defined as the integrated average of the correlation function over a specified angular range. It includes analytical, numerical, and error propagation methods.
