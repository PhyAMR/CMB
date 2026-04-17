# Analysis of Cosmic Microwave Background (CMB) Angular Correlations

This repository contains a comprehensive analysis of the two-point angular correlation function of the Cosmic Microwave Background (CMB) temperature anisotropies. The project investigates potential deviations from the standard cosmological model, the Lambda-CDM (ΛCDM) model, by examining statistical estimators calculated from the Planck 2018 legacy data.

## Scientific Context & Motivation

The standard ΛCDM model has been remarkably successful in describing a wide range of cosmological observations. However, several anomalies have been observed in the CMB data at large angular scales, which appear to be in tension with the predictions of the ΛCDM model. One of the most prominent of these is the lack of correlation in the temperature anisotropies at angles greater than 60 degrees.

This anomaly, along with others like the alignment of low multipoles and hemispherical power asymmetry, challenges the fundamental assumption of statistical isotropy and homogeneity. While these could be statistical flukes, their persistence across multiple experiments (WMAP and Planck) suggests they might be hints of new physics beyond the standard model.

This project focuses on quantifying the statistical significance of the lack of large-scale correlation through the two-point angular correlation function, C(θ). By developing specific statistical estimators and comparing them against theoretical predictions from ΛCDM, we can rigorously test the consistency of the model with the observed data.

## Scientific Approach

The analysis pipeline implements the following workflow:

1. **Data Loading**: Load unbinned CMB temperature power spectrum (D_ell) from Planck 2018 legacy data.
2. **Correlation Function Calculation**: Compute C(θ) from D_ell using Legendre polynomial expansion.
3. **Statistical Estimators**: Quantify features in C(θ) over specific angular ranges using:
   - **`xivar`**: Integrated average of the correlation function (measures mean correlation).
   - **`S12`**: Integral of squared correlation function (sensitive to power in angular range).
   - **`C180`**: Correlation at θ = 180° (North-South correlation).
4. **Theoretical Predictions**: Generate theoretical power spectra using CAMB with cosmological parameters from Planck MCMC chains (processed with GetDist).
5. **Comparison & Significance**: Compare experimental values to distributions from theory and simulations.
6. **Monte Carlo Simulation**: Generate synthetic power spectra with realistic noise to assess statistical significance and cosmic variance.

## Getting Started

### Prerequisites

You will need Python 3.7+ with the following major dependencies:

```
numpy, pandas, matplotlib, scipy, astropy, healpy, camb, getdist, pyyaml
```

For table generation, optionally install: `pdflatex` (for PDF output).

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd CMB
   ```

2. Install dependencies (using pip or conda):
   ```bash
   pip install numpy pandas matplotlib scipy astropy healpy camb getdist pyyaml
   ```

3. Download the required Planck data files (see below).

### Download Data

The analysis requires Planck 2018 legacy MCMC chains and power spectrum data. Use the provided script:

```bash
bash download_data.sh
```

This script will:
- Create a `Planck_Data/` directory.
- Download the Planck CosmoParams full grid (compressed file ~1.3 GB).
- Extract MCMC chain files and parameter grids.

**Important Notes:**
- The download includes MCMC chains in GetDist format used for theoretical predictions.
- If you need HEALPix maps for additional analysis, download them manually from [Planck Legacy Archive](https://pla.esac.esa.int/pla/#home) and place them in the `maps/` directory.
- The script can be re-run safely; it will skip files that already exist (use `-c` flag for resume).

## How the Code Works (main.py)

The `main.py` script orchestrates the entire analysis pipeline:

### Configuration System

- **YAML Configuration**: All parameters are specified in `config.yml` (created automatically if missing).
- **Configuration Hash**: The script computes a hash of the config (excluding `roots` parameter) to determine run directories.
- **Smart Reuse**: If only the `roots` parameter changes, the existing run directory is reused; otherwise a new directory is created.

### Run Modes

The analysis supports four modes via the `analysis.mode` configuration:

1. **`bestfit`**: Runs the Planck-style synthesis pipeline with best-fit cosmological parameters.
   - Processes a single best-fit spectrum through the pipeline `n_samples` times.
   - Generates ensemble of recovered spectra and computes scalar statistics.
   - Saves results to `runs/run_.../theory_bestfit/`.

2. **`mcmc`**: Processes MCMC chains to compute statistics across the parameter posterior.
   - Loads GetDist chains from `roots` directories.
   - For each chain sample: generates spectrum with CAMB, processes through pipeline.
   - Computes statistical distributions of derived quantities (S12, xivar, C180).
   - Saves results to `runs/run_.../theory_mcmc/`.

3. **`simulation`**: Generates Monte Carlo realizations from experimental uncertainties.
   - Creates synthetic power spectra by sampling from experimental error distributions.
   - Processes each realization through the pipeline.
   - Builds distribution of statistics to assess impact of cosmic variance.
   - Saves results to `runs/run_.../simulation/`.

4. **`all`**: Runs all three modes sequentially.
   - Produces comprehensive comparison across theoretical, MCMC, and simulated distributions.
   - Creates separate output directories for each mode.

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Read config.yml (or create default)                         │
├─────────────────────────────────────────────────────────────┤
│ Compute config hash → determine run directory               │
├─────────────────────────────────────────────────────────────┤
│ Run analysis (mode: bestfit, mcmc, simulation, or all)      │
│  ├─ Load Data_loader (experimental data)                    │
│  ├─ Load/Generate theoretical spectra (CAMB)               │
│  └─ Process through Planck-style pipeline for statistics    │
├─────────────────────────────────────────────────────────────┤
│ [Optional] Run plotting (CorrelationPlots)                  │
│  └─ Create comparison plots with uncertainty bands          │
├─────────────────────────────────────────────────────────────┤
│ [Optional] Generate LaTeX tables (generate_result_tables)   │
│  └─ Produce publication-ready tables and master PDF         │
└─────────────────────────────────────────────────────────────┘
```

### Key Configuration Options

```yaml
analysis:
  mode: 'bestfit'          # 'bestfit', 'mcmc', 'simulation', 'all'
  lmax: 30                 # Maximum multipole (determines resolution)
  n_samples: 1000          # Number of realizations/MCMC samples to process

intervals: [...]           # Angular intervals (as cosines) for statistics
roots: [...]               # Paths to Planck MCMC chain directories

pipeline:
  nside_in: 2048          # HEALPix resolution for synthesis
  fwhm_in_arcmin: 40.0    # Beam FWHM for convolution
  noise_Cl: null          # Optional noise power spectrum
  mask: null              # Optional sky mask

plotting:
  enabled: true           # Generate plots after computation
  format: 'pdf'           # Output format (pdf, png, or tex)
  
statistics:
  enabled: true           # Generate LaTeX tables
```

### Command-Line Usage

```bash
# Run with default configuration
python main.py

# Create/update configuration interactively
python main.py --create-config

# Run analysis for specific directory
python main.py --run-dir runs/run_30_1000_bestfit_abc123

# Generate plots only (skip analysis)
python main.py --plot-only

# Generate tables only
python main.py --stats-only

# Skip plotting after analysis
python main.py --no-plot
```

### Output Structure

After running, results are organized as:

```
runs/
└── run_{lmax}_{n_samples}_{mode}_{hash}/
    ├── config.yml                    # Configuration used
    ├── theory_bestfit/               # Best-fit mode outputs (if applicable)
    │   ├── D_ell.npy                 # Power spectra array
    │   ├── Corr.npy                  # Correlation functions array
    │   ├── s12.npy, xiv.npy, etc.    # Scalar statistics
    │   └── ...
    ├── theory_mcmc/                  # MCMC mode outputs (if applicable)
    │   ├── D_ell.npy, Corr.npy, ...
    │   └── ...
    ├── simulation/                   # Simulation mode outputs (if applicable)
    │   ├── D_ell.npy, Corr.npy, ...
    │   └── ...
    ├── images_bestfit/               # Plots for best-fit mode
    ├── images_mcmc/                  # Plots for MCMC mode
    ├── images_simulation/            # Plots for simulation mode
    └── tables/                        # Generated LaTeX tables and PDF
```

## Repository Structure

- **`main.py`** — Primary entry point. Orchestrates configuration, analysis dispatch, plotting, and table generation.
- **`config.yml`** — Main configuration file (created automatically if missing).
- **`download_data.sh`** — Script to download Planck 2018 MCMC chains and data.

- **`functions/`** — Modularized Python package with analysis kernels:
  - See [functions/README.md](functions/README.md) for detailed module descriptions.
  - See [functions/FLOWCHARTS.md](functions/FLOWCHARTS.md) for execution flow diagrams.

- **`Jupyter Notebooks`** — Interactive analysis workflows:
  - **`CleanPlanck.ipynb`**: Detailed, step-by-step analysis tutorial. Best for learning the methodology.
  - **`compact_planck.ipynb`**: Streamlined high-level analysis using `functions/` modules.

- **`maps/`** — Directory for CMB data files (Planck power spectra, optional HEALPix maps).

- **`files/`** — Intermediate and cached data:
  - `matrix/` — Pre-calculated Tmn matrices for S12 computation.
  - `pickle/` — Cached MCMC chain results for fast reuse.

## Key Functions & Classes (Quick Reference)

### Data Loading & Management
- `Data_loader` (functions/data.py): Load experimental power spectrum, compute correlation function and statistics.

### Analysis Pipelines
- `chain_results()` (functions/cosmo.py): Process MCMC chains and best-fit spectrum.
- `MC_results()` (functions/simulation.py): Run Monte Carlo simulations.

### Correlation & Statistics
- `correlation_func()` (functions/correlation_function.py): Compute C(θ) from D_ell.
- `xivar()` / `xivar2()` (functions/xiv.py): Compute xivar statistic.
- `S12()` (functions/s12.py): Compute S12 statistic.
- `Tmn()` (functions/s12.py): Compute Tmn integral matrices.

### Visualization
- `CorrelationPlots` (functions/plots.py): Main plotting class for comparison figures.
- `MapPlots` (functions/maps.py): HEALPix map visualization utilities.

### Table Generation
- `generate_all_tables()` (functions/generate_result_tables.py): Produce LaTeX tables and PDF.

For detailed descriptions of all functions and modules, see [functions/README.md](functions/README.md).

## Usage Examples

### Example 1: Run Best-Fit Analysis

```bash
# Edit config.yml to set mode: 'bestfit', n_samples: 1000
python main.py
```

Results will be in `runs/run_30_1000_bestfit_<hash>/` with plots in `images_bestfit/`.

### Example 2: Process MCMC Chains

```bash
# Ensure Planck_Data/ is populated with MCMC chains
# Edit config.yml: mode: 'mcmc', roots: [...]
python main.py
```

Results show distribution of statistics across MCMC posterior.

### Example 3: Monte Carlo Simulations

```bash
# Edit config.yml: mode: 'simulation'
python main.py
```

Assesses impact of cosmic variance and experimental uncertainties.

### Example 4: Run Interactive Notebook

```bash
jupyter notebook CleanPlanck.ipynb
```

Or use the streamlined version:
```bash
jupyter notebook compact_planck.ipynb
```

## Troubleshooting

- **Missing data files**: Run `bash download_data.sh` to download Planck data.
- **Config not found**: The script creates `config.yml` automatically on first run.
- **Slow computation**: Reduce `n_samples` or `lmax` in config.yml; increase `num_workers` for parallelization.
- **Memory issues**: Reduce `nside_in` (HEALPix resolution) in pipeline config.

## References

- Planck Legacy Archive: https://pla.esac.esa.int/pla/#home
- CAMB (Code for Anisotropic Multifluid Boltzmann Advanced) documentation: https://camb.info/
- GetDist: https://getdist.readthedocs.io/
- HEALPix: https://healpix.sourceforge.io/
