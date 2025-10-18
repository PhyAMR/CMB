# Analysis of Cosmic Microwave Background (CMB) Angular Correlations

This repository contains a comprehensive analysis of the two-point angular correlation function of the Cosmic Microwave Background (CMB) temperature anisotropies. The project investigates potential deviations from the standard cosmological model, the Lambda-CDM (ΛCDM) model, by examining statistical estimators calculated from the Planck 2018 legacy data.

## Scientific Context & Motivation

The standard ΛCDM model has been remarkably successful in describing a wide range of cosmological observations. However, several anomalies have been observed in the CMB data at large angular scales, which appear to be in tension with the predictions of the ΛCDM model. One of the most prominent of these is the lack of correlation in the temperature anisotropies at angles greater than 60 degrees.

This anomaly, along with others like the alignment of low multipoles and hemispherical power asymmetry, challenges the fundamental assumption of statistical isotropy and homogeneity. While these could be statistical flukes, their persistence across multiple experiments (WMAP and Planck) suggests they might be hints of new physics beyond the standard model.

This project focuses on quantifying the statistical significance of the lack of large-scale correlation through the two-point angular correlation function, C(θ). By developing specific statistical estimators and comparing them against theoretical predictions from ΛCDM, we can rigorously test the consistency of the model with the observed data.

## Methodology

The analysis pipeline is implemented in the Jupyter notebooks and Python scripts in this repository. The core workflow is as follows:

1.  **Data Loading**: The analysis begins with the unbinned CMB temperature power spectrum (D_ell) from the Planck 2018 legacy data.
2.  **Correlation Function Calculation**: The two-point angular correlation function, C(θ), is computed from the experimental power spectrum.
3.  **Statistical Estimators**: Two key statistics are used to quantify features in the correlation function over specific angular ranges:
    *   **`xivar`**: The integrated average of the correlation function, which measures the mean correlation over an interval.
    *   **`S12`**: The integral of the squared correlation function, which is sensitive to the overall power within an angular range.
4.  **Theoretical Predictions**: The `camb` library is used to generate theoretical power spectra for various cosmological parameter sets. These parameters are sourced from MCMC chains from the Planck collaboration, processed using `getdist`.
5.  **Comparison**: The experimental values of the `xivar` and `S12` statistics are compared against the distributions predicted by the ΛCDM model from the MCMC chains.
6.  **Simulation**: Monte Carlo simulations are performed to generate mock power spectra based on the experimental data and its uncertainties. This helps in assessing the statistical significance of the observed values and understanding the impact of cosmic variance.

## Repository Structure

-   **`CleanPlanck.ipynb`**: A detailed Jupyter notebook that walks through the entire analysis, from data loading and function definitions to plotting and interpretation. This serves as the primary record of the research workflow.
-   **`compact_planck.ipynb`**: A refactored and more streamlined version of the analysis, designed for clarity and reproducibility. It imports the core logic from the `functions/` modules.
-   **`maps/`**: Contains the CMB data files, including the Planck power spectrum data.
-   **`files/`**: A directory for storing intermediate and final results, such as:
    -   `matrix/`: Pre-calculated `Tmn` matrices used for the analytical computation of the `S12` statistic.
    -   `pickel/`: Pickled Python objects, including results from MCMC chain processing and Monte Carlo simulations.
-   **`functions/`**: A Python package containing the modularized functions for the analysis:
    -   `data.py`: A `Data_loader` class to load and process Planck power spectrum data.
    -   `correlation_function.py`: Functions to compute the two-point correlation function and its error.
    -   `xiv.py`: Functions for calculating the `xivar` statistic.
    -   `s12.py`: Functions for calculating the `S12` statistic.
    -   `cosmology.py`: Interfaces with `camb` and `getdist` to handle theoretical models and MCMC chains.
    -   `simulation.py`: Code for running Monte Carlo simulations.
    -   `plots.py`: A suite of functions for visualizing the results, including power spectra, correlation functions, and histogram comparisons.
    -   `maps.py`: Utilities for handling HEALPix maps.
    -   `tools.py`: General-purpose helper functions.

## Usage

To run the analysis, you will need a Python environment with the following major dependencies installed:

-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scipy`
-   `astropy`
-   `healpy`
-   `camb`
-   `getdist`

The primary entry points for the analysis are the Jupyter notebooks:

1.  **`CleanPlanck.ipynb`**: For a detailed, step-by-step execution of the analysis.
2.  **`compact_planck.ipynb`**: For a quicker, high-level run of the finalized workflow.

Before running, ensure that the required data files are present in the `maps/` directory. Some large data files may not be included in the repository and might need to be downloaded from the Planck Legacy Archive.