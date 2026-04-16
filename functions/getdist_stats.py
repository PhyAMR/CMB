import numpy as np
import pandas as pd
import os
import logging
from getdist import mcsamples, loadMCSamples
from getdist.types import ResultTable, NoLineTableFormatter
from scipy import stats

"""
GetDist-based statistics calculator for MCMC posteriors.

This module provides functions to compute percentiles and p-values
using GetDist library, which properly accounts for MCMC weights and
burn-in when computing posterior statistics.

Key features:
- Uses percentiles exclusively (never means or std)
- Supports adding derived parameters to MCMC chains (properly filtered)
- Generates LaTeX tables using GetDist's ResultTable
"""

"""
GetDist-based statistics calculator for MCMC posteriors.

This module provides functions to compute percentiles and p-values
using GetDist library, which properly accounts for MCMC weights and
burn-in when computing posterior statistics.

Key features:
- Uses percentiles exclusively (never means or std)
- Supports adding derived parameters to MCMC chains (properly filtered)
- Generates LaTeX tables using GetDist's ResultTable
"""



logger = logging.getLogger(__name__)


def load_statistics_as_mcsamples(run_dir, model_name, mode='mcmc', root_dir=None):
    """
    Load statistics from run directory as GetDist MCSamples object.
    
    This allows proper computation of posteriors with weights and
    generation of LaTeX tables.
    
    Parameters
    ----------
    run_dir : str
        Base run directory
    model_name : str
        Model name (subdirectory name)
    mode : str
        'mcmc' or 'bestfit'
    
    Returns
    -------
    MCSamples or None
        GetDist samples object with statistics as derived parameters
    """
    model_dir = os.path.join(run_dir, f'theory_{mode}', model_name)
    
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return None
    
    # Check if chain file exists (saved by cosmo_improved)
    #chain_file = os.path.join(model_dir, f'{model_name}_chain.txt')
    paramnames_file = os.path.join(model_dir, f'{model_name}.paramnames')
    
    if os.path.exists(root_dir) and os.path.exists(paramnames_file):
        try:
            # Load as MCSamples from chain files
            samples = loadMCSamples(root_dir,
                settings={'ignore_rows': 0}  # No burn-in, already processed
            )
            logger.info(f"Loaded {samples.numrows} samples for {model_name} from chain files")
            return samples
        except Exception as e:
            logger.warning(f"Could not load chain file: {e}")
    
    # Fallback: Load from .npy files and create MCSamples
    try:
        # Load scalar columns
        cols_path = os.path.join(model_dir, 'columns.txt')
        if not os.path.exists(cols_path):
            return None
        
        with open(cols_path, 'r') as f:
            scalar_cols = [line.strip() for line in f if line.strip()]
        
        # Load statistics
        data = {}
        
        # S12 statistics
        s_path = os.path.join(model_dir, 'S_statistics.txt')
        if os.path.exists(s_path):
            S_mat = np.loadtxt(s_path)
            if S_mat.ndim == 1:
                S_mat = S_mat[:, None]
            s_cols = [c for c in scalar_cols if c.startswith('s12_')]
            for i, col in enumerate(s_cols):
                if i < S_mat.shape[1]:
                    data[col] = S_mat[:, i]
        
        # xivar statistics
        xiv_path = os.path.join(model_dir, 'xiv_statistic.txt')
        if os.path.exists(xiv_path):
            xiv_mat = np.loadtxt(xiv_path)
            if xiv_mat.ndim == 1:
                xiv_mat = xiv_mat[:, None]
            xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
            for i, col in enumerate(xiv_cols):
                if i < xiv_mat.shape[1]:
                    data[col] = xiv_mat[:, i]
        
        # C180 statistic
        c180_path = os.path.join(model_dir, 'C180_statistics.txt')
        if os.path.exists(c180_path):
            c180_vec = np.loadtxt(c180_path)
            data['C180'] = np.atleast_1d(c180_vec)
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        n_samples = len(df)
        samples_array = df.values
        
        # Create parameter names with LaTeX labels
        names = list(df.columns)
        labels = [_format_label_for_getdist(n) for n in names]
        
        # Create MCSamples with equal weights
        weights = np.ones(n_samples)
        
        samples = mcsamples.MCSamples(
            samples=samples_array,
            weights=weights,
            names=names,
            labels=labels,
            label=model_name
        )
        
        logger.info(f"Created MCSamples with {n_samples} samples for {model_name}")
        return samples
        
    except Exception as e:
        logger.exception(f"Error creating MCSamples: {e}")
        return None


def _format_label_for_getdist(param_name):
    """
    Format parameter name as LaTeX label for GetDist.
    
    Parameters
    ----------
    param_name : str
        Parameter name (e.g., 'xiv_60_30', 's12_60_30', 'C180')
    
    Returns
    -------
    str
        LaTeX formatted label
    """
    if param_name == 'C180':
        return r'$C_{180}$'
    
    # xivar: xiv_60_30 -> $\xi_{60,30}$
    if param_name.startswith('xiv_'):
        parts = param_name.split('_')
        if len(parts) == 3:
            return rf'$\xi_{{{parts[1]},{parts[2]}}}$'
    
    # S12: s12_60_30 -> $S_{60,30}$
    if param_name.startswith('s12_'):
        parts = param_name.split('_')
        if len(parts) == 3:
            return rf'$S_{{{parts[1]},{parts[2]}}}$'
    
    return param_name


def add_derived_parameters(samples, derived_df, n_final_samples=None):
    """
    Add derived parameters to MCMC samples with proper filtering.
    
    This method properly handles:
    - Filtering samples to match derived parameter data
    - Burning in samples correctly
    - Adding multiple derived parameters
    
    Parameters
    ----------
    samples : MCSamples
        GetDist samples object (loaded from chain files)
    derived_df : pd.DataFrame
        DataFrame with derived parameter values (one column per parameter)
    n_final_samples : int, optional
        Number of final samples to keep. If None, uses all available.
    
    Returns
    -------
    MCSamples
        Same object with derived parameters added
    """
    if samples is None or derived_df is None:
        logger.warning("Cannot add derived parameters: samples or derived_df is None")
        return samples
    
    try:
        total_rows_after_burn_in = samples.numrows
        df_data = derived_df
        
        # Determine number of final samples to use
        if n_final_samples is None:
            N_final_samples = df_data.shape[0]
        else:
            N_final_samples = n_final_samples
        
        # Check if chain is shorter than target
        if total_rows_after_burn_in < N_final_samples:
            N_final_samples = total_rows_after_burn_in
            start_index = 0
            logger.warning(
                f"Chain shorter than target. "
                f"Using {N_final_samples} rows out of {total_rows_after_burn_in}."
            )
        else:
            # Use the last N_final_samples samples
            start_index = total_rows_after_burn_in - N_final_samples
        
        # Filter samples to keep only the relevant rows
        indices_to_keep = np.arange(start_index, total_rows_after_burn_in)
        samples.filter(indices_to_keep)
        
        logger.info(f"Filtered samples: kept rows {start_index} to {total_rows_after_burn_in}")
        
        # Add each derived parameter as a column
        for param_name in df_data.columns:
            try:
                # Get the parameter values for the selected rows
                param_values = df_data[param_name].values[:N_final_samples]
                
                # Format label for LaTeX
                param_label = _format_label_for_getdist(param_name)
                
                # Add as derived parameter
                samples.addDerived(
                    param_values,
                    name=param_name,
                    label=param_label,
                    comment=f'Derived: {param_name}'
                )
                
                logger.info(f"Added derived parameter: {param_name}")
            
            except Exception as e:
                logger.warning(f"Could not add derived parameter {param_name}: {e}")
                continue
        
        return samples
    
    except Exception as e:
        logger.exception(f"Error adding derived parameters: {e}")
        return samples


def compute_percentiles_getdist(samples, param_name, percentiles=(16, 50, 84)):
    """
    Compute percentiles using GetDist (accounts for weights).
    
    Parameters
    ----------
    samples : MCSamples
        GetDist samples object
    param_name : str
        Parameter name
    percentiles : tuple
        Percentiles to compute (default: (16, 50, 84) for 68% credible interval)
    
    Returns
    -------
    dict
        {'p16': value, 'p50': value, 'p84': value}
    """
    if samples is None:
        return None
    
    try:
        # Get parameter values
        param_values = getattr(samples.getParams(), param_name, None)
        weights = samples.weights
        
        # Sort by parameter values
        sorted_idx = np.argsort(param_values)
        sorted_vals = param_values[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        # Compute weighted cumulative sum
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights = cumsum_weights / cumsum_weights[-1]  # Normalize to [0, 1]
        
        # Compute percentiles
        result = {}
        for p in percentiles:
            target = p / 100.0
            # Find index where cumsum first exceeds target
            idx = np.searchsorted(cumsum_weights, target, side='left')
            idx = min(idx, len(sorted_vals) - 1)
            result[f'p{p}'] = sorted_vals[idx]
        
        return result
        
    except Exception as e:
        logger.warning(f"Could not compute percentiles for {param_name}: {e}")
        return None


def compute_all_percentiles(samples, param_names, percentiles=(16, 50, 84)):
    """
    Compute percentiles for all parameters.
    
    Parameters
    ----------
    samples : MCSamples
        GetDist samples object
    param_names : list
        List of parameter names
    percentiles : tuple
        Percentiles to compute
    
    Returns
    -------
    dict
        {param_name: {'p16': val, 'p50': val, 'p84': val}}
    """
    results = {}
    
    for param in param_names:
        percentile_vals = compute_percentiles_getdist(samples, param, percentiles)
        if percentile_vals:
            results[param] = percentile_vals
    
    return results


def compute_pvalue_from_percentiles(observed_value, p16, p50, p84):
    """
    Compute p-value given observed value and posterior percentiles.
    
    Assumes approximately Gaussian posterior and computes p-value
    based on how many sigma away the observed value is from the median.
    
    Parameters
    ----------
    observed_value : float
        Experimental/simulation value
    p16, p50, p84 : float
        16th, 50th, 84th percentiles of posterior
    
    Returns
    -------
    dict
        {
            'pvalue': p,
            'n_sigma': n,
            'interpretation': str,
            'tension_level': str
        }
    """
    # Estimate sigma from percentiles (68% CI width)
    # For Gaussian: p84 - p50 ≈ 1σ, p50 - p16 ≈ 1σ
    sigma_upper = p84 - p50
    sigma_lower = p50 - p16
    sigma = (sigma_upper + sigma_lower) / 2.0
    
    if sigma == 0 or np.isnan(sigma) or np.isinf(sigma):
        return {
            'pvalue': 1.0,
            'n_sigma': 0.0,
            'interpretation': 'No variance',
            'tension_level': 'N/A'
        }
    
    # Compute number of sigma away
    n_sigma = abs(observed_value - p50) / sigma
    
    # Two-tailed p-value
    pvalue = 2 * (1 - stats.norm.cdf(n_sigma))
    pvalue = max(pvalue, 1e-10)  # Floor to avoid zero
    
    # Interpretation levels
    if pvalue > 0.05:
        interpretation = 'Consistent'
        tension_level = '< 2σ'
    elif pvalue > 0.01:
        interpretation = 'Marginally inconsistent'
        tension_level = '2-3σ'
    elif pvalue > 0.001:
        interpretation = 'Inconsistent'
        tension_level = '3-4σ'
    else:
        interpretation = 'Highly inconsistent'
        tension_level = '> 4σ'
    
    return {
        'pvalue': pvalue,
        'n_sigma': n_sigma,
        'interpretation': interpretation,
        'tension_level': tension_level
    }


def compute_all_pvalues(samples, param_names, experimental_values):
    """
    Compute p-values for all parameters comparing MCMC posterior to data.
    
    Uses GetDist's weighted percentile computation for accurate results.
    
    Parameters
    ----------
    samples : MCSamples
        GetDist samples object
    param_names : list
        List of parameter names
    experimental_values : dict
        {param_name: (value, error)}
    
    Returns
    -------
    dict
        {param_name: {'pvalue': p, 'n_sigma': n, 'interpretation': str, ...}}
    """
    # Get percentiles for all parameters
    percentiles_dict = compute_all_percentiles(samples, param_names)
    
    results = {}
    
    for param in param_names:
        if param not in experimental_values or param not in percentiles_dict:
            continue
        
        exp_val, exp_err = experimental_values[param]
        perc = percentiles_dict[param]
        
        if not all(k in perc for k in ['p16', 'p50', 'p84']):
            continue
        
        pval_info = compute_pvalue_from_percentiles(
            exp_val,
            perc['p16'],
            perc['p50'],
            perc['p84']
        )
        
        results[param] = {
            'exp_value': exp_val,
            'exp_error': exp_err,
            'p16': perc['p16'],
            'p50': perc['p50'],
            'p84': perc['p84'],
            **pval_info
        }
    
    return results


def compute_multivariate_tension(samples, param_names, experimental_means, experimental_cov):
    """
    Compute multivariate chi-squared tension statistic.
    
    Takes into account correlations in both MCMC posterior and experimental data.
    
    Parameters
    ----------
    samples : MCSamples
        GetDist samples object
    param_names : list
        List of parameter names
    experimental_means : np.ndarray
        Experimental mean values
    experimental_cov : np.ndarray
        Experimental covariance matrix
    
    Returns
    -------
    dict
        {'chi2': val, 'dof': n, 'pvalue': p, 'n_sigma': n}
    """
    try:
        # Get percentile-based values (more robust than means)
        percentiles_dict = compute_all_percentiles(samples, param_names)
        
        # Use p50 (median) as best estimate
        mcmc_means = np.array([
            percentiles_dict[p]['p50'] for p in param_names
            if p in percentiles_dict
        ])
        
        # Subset of params that we have percentiles for
        available_params = [p for p in param_names if p in percentiles_dict]
        
        if len(available_params) == 0:
            logger.warning("No parameters available for multivariate tension")
            return None
        
        # Get MCMC covariance from samples
        # Use percentile-based method to compute covariance robustly
        param_arrays = [getattr(samples.getParams(), p, None) for p in available_params]
        param_matrix = np.column_stack(param_arrays)
        
        # Compute weighted covariance
        mcmc_cov = np.cov(param_matrix.T, aweights=samples.weights)
        
        # Subset experimental covariance to available parameters
        n_params = len(available_params)
        exp_cov_subset = experimental_cov[:n_params, :n_params]
        
        # Combined covariance
        cov_combined = exp_cov_subset + mcmc_cov
        
        # Difference (subset of experimental means)
        exp_means_subset = experimental_means[:n_params]
        delta = mcmc_means - exp_means_subset
        
        # Chi-squared
        inv_cov = np.linalg.inv(cov_combined)
        chi2 = delta.T @ inv_cov @ delta
        
        dof = len(available_params)
        pvalue = stats.chi2.sf(chi2, dof)
        
        # Equivalent n-sigma
        pvalue_safe = max(pvalue, 1e-16)
        n_sigma = stats.norm.isf(pvalue_safe / 2)
        
        return {
            'chi2': float(chi2),
            'dof': dof,
            'pvalue': float(pvalue),
            'n_sigma': float(n_sigma)
        }
    
    except Exception as e:
        logger.exception(f"Error computing multivariate tension: {e}")
        return None


def generate_statistics_table(samples, param_names, experimental_values, 
                            mode='mcmc', include_bestfit=False, ncol=1):
    """
    Generate LaTeX table for statistics using GetDist.
    
    Creates a professional LaTeX table with percentiles and p-values.
    
    Parameters
    ----------
    samples : MCSamples
        GetDist samples object
    param_names : list
        Parameter names to include in table
    experimental_values : dict
        {param: (value, error)}
    mode : str
        'mcmc' or 'bestfit' (for display only)
    include_bestfit : bool
        Whether to include best-fit values
    ncol : int
        Number of columns in table
    
    Returns
    -------
    str or ResultTable
        LaTeX table code or ResultTable object
    """
    if samples is None:
        return None
    
    try:
        # Compute all statistics
        percentiles_dict = compute_all_percentiles(samples, param_names)
        pvalues_dict = compute_all_pvalues(samples, param_names, experimental_values)
        
        # Create custom table with statistics and p-values
        # Using ResultTable would give basic limits, but we want to include p-values
        # So we build custom LaTeX
        
        rows = []
        rows.append("\\begin{tabular}{lccccc}")
        rows.append("\\toprule")
        rows.append("Statistic & Experimental & MCMC Median & 68\\% CI & Tension & $p$-value \\\\")
        rows.append("\\midrule")
        
        for param in param_names:
            if param not in percentiles_dict or param not in pvalues_dict:
                continue
            
            # Format statistic name
            param_label = _format_label_for_getdist(param)
            
            # Experimental value
            exp_val, exp_err = experimental_values[param]
            exp_str = f"${exp_val:.4f} \\pm {exp_err:.4f}$"
            
            # MCMC percentiles
            perc = percentiles_dict[param]
            p16, p50, p84 = perc['p16'], perc['p50'], perc['p84']
            
            mcmc_str = f"${p50:.4f}$"
            ci_str = f"$[{p16:.4f}, {p84:.4f}]$"
            
            # P-value and tension
            pval_info = pvalues_dict[param]
            pval = pval_info['pvalue']
            n_sigma = pval_info['n_sigma']
            #tension_level = pval_info['tension_level']
            
            if pval < 0.001:
                pval_str = f"${pval:.2e}$"
            else:
                pval_str = f"${pval:.3f}$"
            
            # Add row
            row = f"{param_label} & {exp_str} & {mcmc_str} & {ci_str} & {n_sigma:.2f}$\\sigma$ & {pval_str} \\\\"
            rows.append(row)
        
        rows.append("\\bottomrule")
        rows.append("\\end{tabular}")
        
        return "\n".join(rows)
    
    except Exception as e:
        logger.exception(f"Error generating table: {e}")
        return None


def generate_full_results_table(samples, param_names, titles=None, ncol=1):
    """
    Generate full GetDist ResultTable with marginalized statistics.
    
    This uses GetDist's built-in table generation for professional output.
    
    Parameters
    ----------
    samples : MCSamples or list of MCSamples
        GetDist samples object(s)
    param_names : list
        Parameter names to include
    titles : list or None
        Titles for each result column
    ncol : int
        Number of columns for multi-column layout
    
    Returns
    -------
    ResultTable
        GetDist ResultTable object
    """
    if not isinstance(samples, list):
        samples = [samples]
    
    try:
        # Create ResultTable
        result_table = ResultTable(
            ncol=ncol,
            results=samples,
            titles=titles,
            limit=1,  # 68% confidence limit (first limit)
            paramList=param_names,
            formatter=NoLineTableFormatter()
        )
        
        return result_table
    
    except Exception as e:
        logger.exception(f"Error creating ResultTable: {e}")
        return None


def save_table_to_file(table_obj, output_path, document=True):
    """
    Save LaTeX table to file.
    
    Parameters
    ----------
    table_obj : str or ResultTable
        LaTeX string or ResultTable object
    output_path : str
        Path to save to
    document : bool
        If True and table_obj is ResultTable, create standalone LaTeX document
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if isinstance(table_obj, str):
        # Plain string - write directly
        with open(output_path, 'w') as f:
            f.write(table_obj)
    else:
        # ResultTable object - use its write method
        table_obj.write(output_path, document=document)
    
    logger.info(f"Table saved to: {output_path}")