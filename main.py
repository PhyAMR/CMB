#!/usr/bin/env python3
"""
Main script for CMB analysis with YAML configuration system.

Features:
- YAML configuration file for all parameters
- Version system based on config file changes
- Smart folder management (only creates new folders when needed)
- Automatic plotting after computation
- Support for both best-fit and MCMC modes
- LaTeX table generation from statistics
"""

import argparse
import numpy as np
import os
import sys
import logging
import yaml
import hashlib
#import shutil
#from datetime import datetime
#from pathlib import Path

# Import analysis modules
from functions.data import Data_loader
from functions.cosmo import chain_results
from functions.simulation import MC_results
from functions.plots import CorrelationPlots, load_run_data

# Configure logging
logging.basicConfig(
    filename='analysis.log',
    filemode = 'w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'analysis': {
        'mode': 'bestfit',  # 'bestfit', 'mcmc', 'simulation', 'all'
        'lmax': 30,
        'n_samples': 1000,
    },
    'intervals': [
        [-0.9999999999999999, 0.5],
        [-0.9999999999999999, -0.9010561423012785],
        [-0.9010561423012785, -0.03888637661577246],
        [-0.03888637661577246, 0.8654808226792662],
        [0.8654808226792662, 0.9999999999999999],
    ],
    'roots': [
        'Planck_Data/base_omegak/CamSpecHM_TT_lowl_lowE/base_omegak_CamSpecHM_TT_lowl_lowE',
        'Planck_Data/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE',
        'Planck_Data/base/CamSpecHM_TT_lowl_lowE/base_CamSpecHM_TT_lowl_lowE',
        'Planck_Data/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE'
    ],
    'pipeline': {
        'nside_in': 2048,
        'fwhm_in_arcmin': 40.0,
        'noise_Cl': None,
        'mask': None,
        'num_workers': None
    },
    'statistics': {
        'enabled': True          # Generate LaTeX tables after computation
    },
    'plotting': {
        'enabled': True,
        'format': 'pdf',
        'colors': {
            'experimental': 'red',
            'simulation': 'blue',
            'theory': 'black'
        }
    },
    'output': {
        'base_dir': 'runs',
        'save_on_param_change': True
    }
}


# ============================================================================
# Configuration Management
# ============================================================================

def create_default_config(config_path):
    """Create a default configuration file."""
    with open(config_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Created default configuration file: {config_path}")


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Creating default configuration file...")
        create_default_config(config_path)
        return DEFAULT_CONFIG
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Merge with defaults for missing keys
    def merge_dicts(default, custom):
        result = default.copy()
        for key, value in custom.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    return merge_dicts(DEFAULT_CONFIG, config)


def compute_config_hash(config, exclude_keys=None):
    """
    Compute hash of configuration for version control.
    
    Args:
        config (dict): Configuration dictionary
        exclude_keys (list): Keys to exclude from hash (e.g., ['roots'])
    
    Returns:
        str: SHA256 hash of config
    """
    if exclude_keys is None:
        exclude_keys = []
    
    # Create a copy without excluded keys
    config_copy = config.copy()
    
    # Remove excluded keys at all levels
    def remove_keys(d, keys):
        if not isinstance(d, dict):
            return d
        result = {}
        for k, v in d.items():
            if k not in keys:
                if isinstance(v, dict):
                    result[k] = remove_keys(v, keys)
                else:
                    result[k] = v
        return result
    
    filtered_config = remove_keys(config_copy, exclude_keys)
    
    # Convert to sorted string for consistent hashing
    config_str = yaml.dump(filtered_config, default_flow_style=False, sort_keys=True)
    
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def find_or_create_run_dir(config):
    """
    Find existing run directory or create new one based on config hash.
    
    Special case: If only 'roots' changed, reuse existing directory.
    Otherwise, create new directory.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (run_dir, is_new, roots_only_changed)
    """
    base_dir = config['output']['base_dir']
    os.makedirs(base_dir, exist_ok=True)
    
    # Compute hashes
    full_hash = compute_config_hash(config)
    hash_no_roots = compute_config_hash(config, exclude_keys=['roots'])
    
    # Generate run directory name
    lmax = config['analysis']['lmax']
    n_samples = config['analysis']['n_samples']
    mode = config['analysis']['mode']
    
    run_name = f"run_{lmax}_{n_samples}_{mode}_{hash_no_roots}"
    run_dir = os.path.join(base_dir, run_name)
    
    # Check if directory exists
    if os.path.exists(run_dir):
        # Load saved config
        saved_config_path = os.path.join(run_dir, 'config.yml')
        
        if os.path.exists(saved_config_path):
            with open(saved_config_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            # Check if only roots changed
            saved_hash_no_roots = compute_config_hash(saved_config, exclude_keys=['roots'])
            current_hash_no_roots = compute_config_hash(config, exclude_keys=['roots'])
            
            if saved_hash_no_roots == current_hash_no_roots:
                # Only roots changed or nothing changed
                roots_changed = (saved_config.get('roots') != config.get('roots'))
                
                logger.info(f"Existing run directory found: {run_dir}")
                if roots_changed:
                    logger.info("Only 'roots' parameter changed - reusing directory")
                
                return run_dir, False, roots_changed
        
        # Config changed significantly - need new directory
        logger.info("Configuration changed - creating new directory")
    
    # Create new directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(run_dir, 'config.yml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created new run directory: {run_dir}")
    logger.info(f"Configuration saved to: {config_save_path}")
    
    return run_dir, True, False


# ============================================================================
# Main Analysis Functions
# ============================================================================

def run_analysis(config, run_dir):
    """
    Run the requested analysis mode.
    
    Args:
        config (dict): Configuration dictionary
        run_dir (str): Run directory path
    
    Returns:
        str: Run directory path
    """
    # Extract config sections
    analysis_cfg = config['analysis']
    pipeline_cfg = config['pipeline']
    
    # Convert intervals from lists to tuples
    intervals = [tuple(i) for i in config['intervals']]
    roots = config['roots']
    
    # Initialize Data_loader
    logger.info(f"Initializing Data_loader with lmax={analysis_cfg['lmax']}")
    DL = Data_loader(lmax=analysis_cfg['lmax'])
    
    # Prepare pipeline parameters
    pipeline_params = {
        'nside_in': pipeline_cfg['nside_in'],
        'fwhm_in_arcmin': pipeline_cfg['fwhm_in_arcmin'],
        'noise_Cl': pipeline_cfg.get('noise_Cl'),
        'mask': pipeline_cfg.get('mask')
    }
    
    # Get the mode
    mode = config['analysis']['mode']
    
    if mode in ['bestfit', 'mcmc']:
        # Single mode
        is_bestfit = (mode == 'bestfit')
        
        chain_results(
            data_loader=DL,
            intervals=intervals,
            roots=roots,
            name=f'analysis_{mode}',
            output_dir=run_dir,
            n=analysis_cfg['n_samples'],
            best_fit=is_bestfit,
            pipeline_params=pipeline_params,
            mode_suffix=mode
        )
    
    elif mode == 'all':
        # Run BOTH modes
        logger.info("Running BOTH best-fit and MCMC modes...")
        
        # 1. Best-fit mode
        logger.info("Step 1/3: Running best-fit analysis...")
        chain_results(
            data_loader=DL,
            intervals=intervals,
            roots=roots,
            name='analysis_bestfit',
            output_dir=run_dir,
            n=analysis_cfg['n_samples'],
            best_fit=True,
            pipeline_params=pipeline_params,
            mode_suffix='bestfit'
        )
        
        # 2. MCMC mode
        logger.info("Step 2/3: Running MCMC analysis...")
        chain_results(
            data_loader=DL,
            intervals=intervals,
            roots=roots,
            name='analysis_mcmc',
            output_dir=run_dir,
            n=analysis_cfg['n_samples'],
            best_fit=False,
            pipeline_params=pipeline_params,
            mode_suffix='mcmc'
        )
        
        # 3. Simulation
        logger.info("Step 3/3: Running MC_results (simulation)...")
        MC_results(
            data_loader=DL,
            intervals=intervals,
            output_dir=run_dir,
            n=analysis_cfg['n_samples']
        )
        logger.info("✓ Simulation complete")
    
    elif mode == 'simulation':
        logger.info("Running MC_results (simulation)...")
        MC_results(
            data_loader=DL,
            intervals=intervals,
            output_dir=run_dir,
            n=analysis_cfg['n_samples']
        )
        logger.info("✓ Simulation complete")
    
    return run_dir


def run_plotting(config, run_dir):
    """
    Generate all plots for a run directory.
    
    Args:
        config (dict): Configuration dictionary
        run_dir (str): Run directory path
    """
    logger.info(f"Generating plots for: {run_dir}")
    
    # Extract config sections
    analysis_cfg = config['analysis']
    plotting_cfg = config['plotting']
    mode = analysis_cfg['mode']
    
    # Convert intervals
    intervals = [tuple(i) for i in config['intervals']]
    
    # Initialize Data_loader
    DL = Data_loader(lmax=analysis_cfg['lmax'])
    
    # Get experimental values
    exp_values = DL.experimental_values(intervals)
    logger.info(f"Computed {len(exp_values)} experimental statistics")
    
    # Load run data
    if not os.path.exists(run_dir):
        logger.error(f"Run directory not found: {run_dir}")
        return
    
    logger.info(f"Loading data from {run_dir}...")
    run_data = load_run_data(run_dir, exp_values,)
    
    # Log what was loaded
    logger.info(f"  - Experimental: {len(run_data['experimental']['scalars'])} statistics")
    
    if run_data['simulation'].get('scalars_df') is not None:
        n_sim = len(run_data['simulation']['scalars_df'])
        logger.info(f"  - Simulation: {n_sim} realizations")
    
    n_models = len(run_data['theory'])
    logger.info(f"  - Theory: {n_models} models")
    for model_name in run_data['theory'].keys():
        if run_data['theory'][model_name]['scalars_df'] is not None:
            n_samples = len(run_data['theory'][model_name]['scalars_df'])
            logger.info(f"      {model_name}: {n_samples} samples")
    
    # Initialize plotter
    colors = plotting_cfg.get('colors', DEFAULT_CONFIG['plotting']['colors'])
    
    if mode == 'all':
        logger.info("Plotting BOTH best-fit and MCMC results...")
        modes = ['bestfit', 'mcmc']
    else:
        logger.info(f"Plotting {mode} results...")
        modes = [mode]
    
    for mode in modes:
        logger.info(f"  - Mode: {mode}")
        CP = CorrelationPlots(DL, output_dir=run_dir, colors=colors, mode=mode)
        
        # Get scalar column names
        scalar_cols = list(exp_values.keys())
        
        # Generate plots for each theory model
        if not run_data['theory']:
            logger.warning("No theory data found. Skipping plots.")
            return
        
        plot_format = plotting_cfg.get('format', 'pdf')
        
        for model_name, model_data in run_data['theory'].items():
            logger.info(f"\nProcessing model: {model_name}")
            
            # Extract data
            scalars_df = model_data['scalars_df']
            D_ell_samples = model_data['D_ell']
            Corr_samples = model_data['Corr']
            
            if scalars_df is None or D_ell_samples is None:
                logger.warning(f"  Skipping {model_name} - incomplete data")
                continue
            
            # Compute mean spectra
            mean_Cl = D_ell_samples.mean(axis=0)
            std_Cl = D_ell_samples.std(axis=0)
            mean_Cor = Corr_samples.mean(axis=0)
            std_Cor = Corr_samples.std(axis=0)
            
            # Safe model name for filenames
            safe_name = model_name.replace('/', '_').replace(' ', '_')
            
            # Plot 1: Individual histograms
            logger.info("  - Generating individual histograms...")
            CP.create_individual_histograms(
                df=scalars_df,
                labels=scalar_cols,
                comparison_data=run_data,
                bins=100,
                base_name=safe_name,
                file_format=plot_format
            )
            
            # Plot 2: Forest plots (separate by statistic type)
            logger.info("  - Generating forest plots...")
            CP.create_forest_plots_by_type(
                df=scalars_df,
                labels=scalar_cols,
                comparison_data=run_data,
                base_name=safe_name,
                file_format=plot_format
            )
            
            # Plot 3: Cumulative convergence
            logger.info("  - Generating cumulative plots...")
            CP.create_cumulative_grid(
                df=scalars_df,
                labels=scalar_cols,
                title=f"Convergence - {model_name}",
                comparison_data=run_data,
                save_path=f"{safe_name}_cumulative.{plot_format}"
            )
            
            # Plot 4: Power spectrum and correlation
            logger.info("  - Generating power spectrum and correlation...")
            CP.plot_power_and_correlation(
                mean_Cl=mean_Cl,
                std_Cl=std_Cl,
                mean_Cor=mean_Cor,
                std_Cor=std_Cor,
                root=model_name,
                save_path=f"{safe_name}_power_corr.{plot_format}"
            )
            
            # Plot 5/6: Combined xi(theta) / xi^2(theta) with xivar/S12
            # bin overlays (replaces the old separate plot_corr_with_xivar
            # / plot_corr_with_S12 calls).
            logger.info("  - Generating xi(theta) / xi^2(theta) plot...")
            xivar_cols = [c for c in scalar_cols if c.startswith('xiv_')]
            s12_cols = [c for c in scalar_cols if c.startswith('s12_')]
            if xivar_cols and s12_cols:
                xivar_df = scalars_df[xivar_cols]
                s12_df = scalars_df[s12_cols]
                CP.plot_xi_and_xi2(
                    corr_samples=Corr_samples,
                    xivar_df=xivar_df,
                    s12_df=s12_df,
                    intervals=intervals,
                    name=model_name,
                    comparison_data=run_data,
                    save_path=f"{safe_name}_xi_xi2.{plot_format}",
                    theory_weights=model_data.get('weights')
                )
            
            # Plot 7: Diagnostics panels (separate by type)
            logger.info("  - Generating diagnostics...")
            CP.create_diagnostics_by_type(
                df=scalars_df,
                labels=scalar_cols,
                comparison_data=run_data,
                base_name=safe_name,
                file_format=plot_format
            )
        
        logger.info(f"  ✓ All plots generated for {model_name}")
    
    logger.info(f"\n✓ All plots saved to: {run_dir}/images/")


def run_table_generation(config, run_dir):
    """
    Generate LaTeX tables from existing statistics files in theory_mcmc/.
    
    This requires that scalar statistics have already been computed:
    - theory_mcmc/{model_name}/S_statistics.txt
    - theory_mcmc/{model_name}/xiv_statistic.txt
    - theory_mcmc/{model_name}/C180_statistics.txt
    - theory_mcmc/{model_name}/columns.txt
    
    This function will:
    1. Create the chain files from scalar statistics:
       - {model_name}_chain.txt (GetDist format)
       - {model_name}_chain.paramnames
       - {model_name}_chain.ranges
    2. Generate LaTeX tables:
       - tables/all_results.tex
       - tables/mcmc/*.tex
       - tables/statistics_summary.txt
    
    Args:
        config (dict): Configuration dictionary
        run_dir (str): Run directory path
    
    Returns:
        str: Run directory path
        
    Raises:
        FileNotFoundError: If theory_mcmc directory or required files not found
    """
    from functions.data import Data_loader
    from functions.generate_result_tables import generate_all_tables
    import pandas as pd
    
    logger.info(f"Generating LaTeX tables for run: {run_dir}")
    
    # Check if theory_mcmc directory exists and has models
    theory_mcmc_dir = os.path.join(run_dir, 'theory_mcmc')
    
    if not os.path.exists(theory_mcmc_dir):
        raise FileNotFoundError(
            f"theory_mcmc directory not found in {run_dir}\n"
            f"Please run the full analysis first to compute statistics."
        )
    
    models = [d for d in os.listdir(theory_mcmc_dir) if os.path.isdir(os.path.join(theory_mcmc_dir, d))]
    
    if not models:
        raise FileNotFoundError(
            f"No model directories found in {theory_mcmc_dir}\n"
            f"Please run the full analysis first to compute statistics."
        )
    
    # Verify that required scalar statistics files exist and create chain files
    logger.info(f"Found {len(models)} model(s) in theory_mcmc/")
    
    for model in models:
        model_dir = os.path.join(theory_mcmc_dir, model)
        
        # Check for columns.txt first (defines what scalars we have)
        columns_file = os.path.join(model_dir, 'columns.txt')
        if not os.path.exists(columns_file):
            raise FileNotFoundError(
                f"Missing columns.txt in {model_dir}\n"
                f"Please run the full analysis first to compute statistics."
            )
        
        # Read column names
        with open(columns_file, 'r') as f:
            scalar_cols = [line.strip() for line in f if line.strip()]
        
        # Check for scalar statistics files
        required_files = []
        
        s_cols = [c for c in scalar_cols if c.startswith('s12_')]
        if s_cols:
            required_files.append('S_statistics.txt')
        
        xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
        if xiv_cols:
            required_files.append('xiv_statistic.txt')
        
        if 'C180' in scalar_cols:
            required_files.append('C180_statistics.txt')
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required statistics files in {model_dir}:\n"
                f"  {', '.join(missing_files)}\n"
                f"Please run the full analysis first to compute statistics."
            )
        
        logger.info(f"  ✓ {model}: all statistics files present")
        
        # ================================================================
        # Create chain files from scalar statistics
        # ================================================================
        logger.info(f"  Creating chain files for {model}...")
        
        data = {}
        
        # Load S12 statistics
        s_path = os.path.join(model_dir, 'S_statistics.txt')
        if os.path.exists(s_path):
            S_mat = np.loadtxt(s_path)
            if S_mat.ndim == 1:
                S_mat = S_mat[:, None]
            s_cols = [c for c in scalar_cols if c.startswith('s12_')]
            for i, col in enumerate(s_cols):
                if i < S_mat.shape[1]:
                    data[col] = S_mat[:, i]
        
        # Load xivar statistics
        xiv_path = os.path.join(model_dir, 'xiv_statistic.txt')
        if os.path.exists(xiv_path):
            xiv_mat = np.loadtxt(xiv_path)
            if xiv_mat.ndim == 1:
                xiv_mat = xiv_mat[:, None]
            xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
            for i, col in enumerate(xiv_cols):
                if i < xiv_mat.shape[1]:
                    data[col] = xiv_mat[:, i]
        
        # Load C180 statistic
        c180_path = os.path.join(model_dir, 'C180_statistics.txt')
        if os.path.exists(c180_path):
            c180_vec = np.loadtxt(c180_path)
            data['C180'] = np.atleast_1d(c180_vec)
        
        if not data:
            logger.error(f"No statistics data found for {model}")
            continue
        
        df_results = pd.DataFrame(data)
        n_samples = len(df_results)
        
        # Create equal weights and dummy loglikes for chain file
        weights = np.ones(n_samples)
        loglikes = np.zeros(n_samples)
        
        # Create paramnames
        derived_cols = list(df_results.columns)
        
        paramnames_file = os.path.join(model_dir, f'{model}_chain.paramnames')
        with open(paramnames_file, 'w') as pf:
            for col in derived_cols:
                label = col.replace('_', '\\_')
                pf.write(f"{col}    ${label}$\n")
        
        logger.info(f"    Created: {model}_chain.paramnames")
        
        # Create chain file (GetDist format: weight, -loglike, params...)
        chain_file = os.path.join(model_dir, f'{model}_chain.txt')
        
        getdist_data = np.column_stack(
            [weights, loglikes, df_results[derived_cols].values]
        )
        
        np.savetxt(chain_file, getdist_data, fmt="%.10e")
        logger.info(f"    Created: {model}_chain.txt")
        
        # Create ranges file
        ranges_file = os.path.join(model_dir, f"{model}_chain.ranges")
        
        # Create a simple ranges file with min/max of each parameter
        with open(ranges_file, 'w') as rf:
            for col in derived_cols:
                data_col = df_results[col].values
                min_val = np.min(data_col)
                max_val = np.max(data_col)
                rf.write(f"{col} {min_val} {max_val}\n")
        
        logger.info(f"    Created: {model}_chain.ranges")
    
    # ================================================================
    # Now generate LaTeX tables
    # ================================================================
    logger.info("Generating LaTeX tables from chain files...")
    
    # Get intervals and experimental values
    intervals = [tuple(i) for i in config['intervals']]
    roots = config['roots']
    
    # Initialize Data_loader
    lmax = config['analysis']['lmax']
    DL = Data_loader(lmax=lmax)
    experimental_values = DL.experimental_values(intervals)
    
    logger.info(f"Computed {len(experimental_values)} experimental statistics")
    
    # Optional: Load derived parameters if available
    derived_params = None
    derived_params_file = os.path.join(run_dir, 'derived_parameters.csv')
    if os.path.exists(derived_params_file):
        derived_params = pd.read_csv(derived_params_file)
        logger.info(f"Loaded {len(derived_params.columns)} derived parameters")
    
    # Generate tables
    try:
        generate_all_tables(
            run_dir,
            roots,
            experimental_values,
            derived_params=derived_params
        )
        logger.info("✓ Tables generated successfully")
        return run_dir
    
    except Exception as e:
        logger.exception(f"Error generating tables: {e}")
        raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='CMB Analysis Pipeline with YAML configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis (creates new run directory)
  python main2.py --config my_config.yml
  
  # Create default config file
  python main2.py --create-config
  
  # Add plots to existing run directory (no computation)
  python main2.py --config my_config.yml --plot-only
  
  # Generate LaTeX tables from existing statistics (creates chain files)
  # Requires theory_mcmc/{model}/S_statistics.txt, xiv_statistic.txt, C180_statistics.txt
  python main2.py --config my_config.yml --stats-only
  
  # Generate tables for specific run directory
  python main2.py --config my_config.yml --stats-only --run-dir runs/run_30_1000_all_abc123
  
  # Skip plotting after computation
  python main2.py --config my_config.yml --no-plot
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to YAML configuration file (default: config.yml)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Only generate plots in existing run directory (skip computation)'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Generate LaTeX tables and chain files from existing statistics'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting after computation'
    )
    
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Specific run directory to use with --plot-only or --stats-only'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle create-config mode
    if args.create_config:
        create_default_config(args.config)
        logger.info(f"Edit {args.config} to customize your analysis")
        return 0
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Print configuration summary
    logger.info("=" * 70)
    logger.info("CMB ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode: {config['analysis']['mode']}")
    logger.info(f"lmax: {config['analysis']['lmax']}")
    logger.info(f"n_samples: {config['analysis']['n_samples']}")
    logger.info(f"nside: {config['pipeline']['nside_in']}")
    logger.info(f"FWHM: {config['pipeline']['fwhm_in_arcmin']} arcmin")
    logger.info(f"Number of intervals: {len(config['intervals'])}")
    logger.info(f"Number of roots: {len(config['roots'])}")
    
    stats_cfg = config.get('statistics', {'enabled': True})
    logger.info(f"Statistics tables enabled: {stats_cfg['enabled']}")
    logger.info("=" * 70)
    
    try:
        # Determine run directory and what to do
        if args.stats_only:
            # --stats-only: use existing or specified directory
            # Create chain files and generate tables
            if args.run_dir:
                run_dir = args.run_dir
            else:
                # Find latest run directory
                base_dir = config['output']['base_dir']
                if not os.path.exists(base_dir):
                    logger.error(f"Base directory does not exist: {base_dir}")
                    logger.error("Cannot use --stats-only without existing run or --run-dir")
                    return 1
                
                runs = sorted([
                    d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))
                ])
                
                if not runs:
                    logger.error(f"No run directories found in {base_dir}")
                    return 1
                
                run_dir = os.path.join(base_dir, runs[-1])
            
            if not os.path.exists(run_dir):
                logger.error(f"Run directory not found: {run_dir}")
                return 1
            
            logger.info(f"Using existing run directory: {run_dir}")
            
            # Generate LaTeX tables and chain files
            if stats_cfg['enabled']:
                try:
                    logger.info("Creating chain files and generating LaTeX tables...")
                    run_table_generation(config, run_dir)
                except FileNotFoundError as e:
                    logger.error(str(e))
                    return 1
        
        elif args.plot_only:
            # --plot-only: use existing directory, only generate plots
            if args.run_dir:
                run_dir = args.run_dir
            else:
                base_dir = config['output']['base_dir']
                if not os.path.exists(base_dir):
                    logger.error(f"Base directory does not exist: {base_dir}")
                    return 1
                
                runs = sorted([
                    d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d))
                ])
                
                if not runs:
                    logger.error(f"No run directories found in {base_dir}")
                    return 1
                
                run_dir = os.path.join(base_dir, runs[-1])
            
            if not os.path.exists(run_dir):
                logger.error(f"Run directory not found: {run_dir}")
                return 1
            
            logger.info(f"Using existing run directory: {run_dir}")
        
        else:
            # Normal mode: create new directory and run full analysis
            run_dir, is_new, roots_only_changed = find_or_create_run_dir(config)
            
            if is_new or config['output'].get('save_on_param_change', True):
                run_analysis(config, run_dir)
            else:
                logger.info("Run directory exists and config unchanged. Skipping computation.")
                logger.info("Use --plot-only to regenerate plots, or --stats-only to generate tables.")
        
        # Generate plots (unless --no-plot)
        if config['plotting']['enabled'] and not args.no_plot:
            run_plotting(config, run_dir)
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✓ Run directory: {run_dir}")
        logger.info(f"✓ Configuration: {run_dir}/config.yml")
        if config['plotting']['enabled'] and not args.no_plot:
            logger.info(f"✓ Plots saved to: {run_dir}/images_{config['analysis']['mode']}/")
        if args.stats_only:
            logger.info(f"✓ Chain files created in: {run_dir}/theory_mcmc/")
            logger.info(f"✓ Tables saved to: {run_dir}/tables/")
        logger.info("=" * 70)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130
    
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())