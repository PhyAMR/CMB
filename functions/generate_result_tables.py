#!/usr/bin/env python3
"""
Generate comprehensive LaTeX tables for cosmological analysis.

Features:
- Uses GetDist for proper percentile computation with MCMC weights
- Properly adds derived parameters with sample filtering
- P-values for each statistic using percentile-based method
- N-sigma tensions
- Global multivariate chi-squared statistic
"""

import os
import argparse
import numpy as np
import yaml
import pandas as pd
import logging
from getdist import mcsamples
from scipy import stats
from scipy.stats import cauchy

from .getdist_stats import (
    #load_statistics_as_mcsamples,
    add_derived_parameters,
    #compute_all_percentiles,
    #compute_all_pvalues,
    compute_multivariate_tension,
    generate_statistics_table,
    #generate_full_results_table,
    #save_table_to_file,
    _format_label_for_getdist
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_chain_with_derived_params(run_dir, model_name, mode='mcmc', derived_df=None):
    """
    Load MCMC chain or create fake samples object for best-fit and add derived parameters.
    
    Parameters
    ----------
    run_dir : str
        Run directory
    model_name : str
        Model name
    mode : str
        'mcmc' or 'bestfit'
    derived_df : pd.DataFrame, optional
        DataFrame with derived parameters
    
    Returns
    -------
    MCSamples or None
        GetDist samples with derived parameters added
    """
    import pandas as pd
    
    # Load the chain from files
    model_dir = os.path.join(run_dir, f'theory_{mode}', model_name)
    chain_root = os.path.join(model_dir, f'{model_name}_chain')
    
    try:
        if mode == 'mcmc':
            # Load MCMC samples
            samples = mcsamples.loadMCSamples(chain_root, settings={'ignore_rows': 0})
            
            if samples is None:
                logger.warning(f"Could not load samples from {chain_root}")
                return None
            
            # Add derived parameters if provided
            if derived_df is not None and isinstance(derived_df, pd.DataFrame):
                try:
                    samples = add_derived_parameters(samples, derived_df)
                    logger.info(f"Added {len(derived_df.columns)} derived parameters to {model_name}")
                except Exception as e:
                    logger.warning(f"Could not add derived parameters: {e}")
        
        elif mode == 'bestfit':
            # Load best-fit statistics and create fake samples object
            logger.info(f"Loading best-fit statistics for {model_name}...")
            
            # Load scalar columns
            cols_path = os.path.join(model_dir, 'columns.txt')
            if not os.path.exists(cols_path):
                logger.warning(f"columns.txt not found in {model_dir}")
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
                logger.warning(f"No statistics data found for {model_name}")
                return None
            
            df = pd.DataFrame(data)
            n_samples = len(df)
            samples_array = df.values
            
            # Create parameter names with LaTeX labels
            names = list(df.columns)
            labels = [_format_label_for_getdist(n) for n in names]
            
            # Create MCSamples with equal weights (no weights needed for best-fit)
            weights = np.ones(n_samples)
            
            samples = mcsamples.MCSamples(
                samples=samples_array,
                weights=weights,
                names=names,
                labels=labels,
                label=model_name
            )
            
            logger.info(f"Created MCSamples object with {n_samples} samples for {model_name} (best-fit mode)")
        
        else:
            logger.error(f"Unknown mode: {mode}")
            return None
        
        return samples
    
    except Exception as e:
        logger.exception(f"Error loading samples for {model_name}: {e}")
        return None
    
    except Exception as e:
        logger.warning(f"Could not load chain {chain_root}: {e}")
        return None


def generate_statistics_table_mcmc(run_dir, model_name, experimental_values, derived_df=None):
    """
    Generate LaTeX table for MCMC mode using GetDist.
    
    Parameters
    ----------
    run_dir : str
        Run directory
    model_name : str
        Model name
    experimental_values : dict
        {param: (value, error)}
    derived_df : pd.DataFrame, optional
        Derived parameters to add
    
    Returns
    -------
    str
        LaTeX table code
    """
    # Load samples with derived parameters
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='mcmc', derived_df=derived_df
    )
    
    if samples is None:
        logger.warning(f"Could not load samples for {model_name}")
        return None
    
    # Get parameter names
    param_names = list(experimental_values.keys())
    
    # Generate custom table with percentiles and p-values
    table_tex = generate_statistics_table(
        samples,
        param_names,
        experimental_values,
        mode='mcmc'
    )
    
    return table_tex


def generate_statistics_table_bestfit(run_dir, model_name, experimental_values, derived_df=None):
    """
    Generate LaTeX table for best-fit mode using GetDist.
    
    Parameters
    ----------
    run_dir : str
        Run directory
    model_name : str
        Model name
    experimental_values : dict
        {param: (value, error)}
    derived_df : pd.DataFrame, optional
        Derived parameters to add
    
    Returns
    -------
    str
        LaTeX table code
    """
    # Load samples with derived parameters
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='bestfit', derived_df=derived_df
    )
    
    if samples is None:
        logger.warning(f"Could not load best-fit samples for {model_name}")
        return None
    
    # Get parameter names
    param_names = list(experimental_values.keys())
    
    # Generate table using percentiles (not means)
    table_tex = generate_statistics_table(
        samples,
        param_names,
        experimental_values,
        mode='bestfit'
    )
    
    return table_tex

def generate_global_statistics_mcmcv1(run_dir, model_name, experimental_values, derived_df=None):
    """
    Compute global multivariate tension statistic using GetDist percentiles.
    
    Parameters
    ----------
    run_dir : str
        Run directory
    model_name : str
        Model name
    experimental_values : dict
        {param: (value, error)}
    derived_df : pd.DataFrame, optional
        Derived parameters to add
    
    Returns
    -------
    dict
        {'chi2': val, 'dof': n, 'pvalue': p, 'n_sigma': n}
    """
    # Load samples
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='mcmc', derived_df=derived_df
    )
    
    if samples is None:
        return None
    
    # Get parameter names
    param_names = list(experimental_values.keys())
    
    # Build experimental values array and covariance
    exp_means = np.array([experimental_values[p][0] for p in param_names])
    exp_errors = np.array([experimental_values[p][1] for p in param_names])
    exp_cov = np.diag(exp_errors**2)
    
    # Compute multivariate tension using GetDist percentiles
    tension = compute_multivariate_tension(
        samples,
        param_names,
        exp_means,
        exp_cov
    )
    
    return tension

def acat(pvals, weights=None):
    """
    Aggregated Cauchy Association Test (ACAT; Liu & Xie, 2020).

    Combines individual p-values using a weighted Cauchy statistic:
        T = sum(w_i * tan((0.5 - p_i) * pi))
        p_combined = 0.5 - arctan(T) / pi

    Parameters
    ----------
    pvals : array-like
        Individual p-values; all must be in the open interval (0, 1).
    weights : array-like, optional
        Non-negative weights for each p-value. Normalised internally.
        Defaults to uniform weights (1/n each).

    Returns
    -------
    float
        Combined p-value in (0, 1).

    Raises
    ------
    ValueError
        If any p-value is not strictly in (0, 1).
    """
    pvals = np.asarray(pvals, dtype=float)
    if np.any((pvals <= 0) | (pvals >= 1)):
        raise ValueError("All p-values must be in (0, 1)")
    n = len(pvals)
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
    t = np.sum(weights * np.tan((0.5 - pvals) * np.pi))
    return float(0.5 - np.arctan(t) / np.pi), t


def generate_global_statistics_mcmc(run_dir, model_name, experimental_values, derived_df=None):
    """
    Compute global tension statistic using the Cauchy Combination Test (CCT).

    Computes a per-parameter marginal p-value from the weighted MCMC empirical
    CDF, then combines them via the CCT (Liu & Xie, 2020):
        T = mean(tan((0.5 - p_i) * pi))
        p_combined = 0.5 - arctan(T) / pi

    Parameters
    ----------
    run_dir : str
        Run directory
    model_name : str
        Model name
    experimental_values : dict
        {param: (value, error)}
    derived_df : pd.DataFrame, optional
        Derived parameters to add

    Returns
    -------
    dict
        {'chi2': None, 'dof': n, 'pvalue': p, 'n_sigma': n,
         'cauchy_T': float, 'individual_pvalues': list,
         'theory_mean': array, 'theory_std': array, 'obs_values': array}
    """
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='mcmc', derived_df=derived_df
    )
    if samples is None:
        return None

    param_names = list(experimental_values.keys())
    obs_values  = np.array([experimental_values[p][0] for p in param_names])

    # ================================================================
    # Extract parameter arrays from the MCMC chain
    # ================================================================
    param_arrays = []
    for p in param_names:
        try:
            param_vals = getattr(samples.getParams(), p, None)
            if param_vals is None:
                logger.warning(f"Parameter {p} not found in samples")
                return None
            param_arrays.append(np.asarray(param_vals, dtype=float))
        except Exception as e:
            logger.warning(f"Could not extract {p}: {e}")
            return None

    weights = None
    if hasattr(samples, 'weights') and samples.weights is not None:
        weights = np.asarray(samples.weights, dtype=float)
        weights = weights / weights.sum()

    # ================================================================
    # Compute one marginal p-value per parameter (two-sided empirical CDF)
    # ================================================================
    individual_pvalues = []
    theory_means = []
    theory_stds  = []

    for arr, obs, pname in zip(param_arrays, obs_values, param_names):
        if weights is not None:
            mu  = float(np.average(arr, weights=weights))
            var = float(np.average((arr - mu) ** 2, weights=weights))
            std = float(np.sqrt(var))
            sorter = np.argsort(arr)
            F = float(np.interp(obs, arr[sorter], np.cumsum(weights[sorter])))
        else:
            mu  = float(arr.mean())
            std = float(arr.std(ddof=0))
            F   = float(np.mean(arr <= obs))
        pval = 2.0 * min(F, 1.0 - F)
        p_i = max(pval, 1e-16)
        individual_pvalues.append(p_i)
        theory_means.append(mu)
        theory_stds.append(std)
        logger.info(f"  {pname}: mu={mu:.4g}, std={std:.4g}, obs={obs:.4g}, p={p_i:.4e}")

    # ================================================================
    # ACAT: Aggregated Cauchy Association Test
    # ================================================================
    p_arr      = np.array(individual_pvalues)
    combined_p, T = acat(p_arr)
    combined_p = max(combined_p, 1e-16)
    # Recover T from the acat formula for logging
    n_sigma    = float(stats.norm.isf(combined_p/2)) #  / 2 for two-sided, but we use one-sided in acat already
    dof        = len(param_names)

    logger.info(f"ACAT T = {T:.4f}")
    logger.info(f"Combined p-value = {combined_p:.4e}")
    logger.info(f"Tension = {n_sigma:.2f}σ")

    return {
        'chi2':               None,
        'dof':                dof,
        'pvalue':             combined_p,
        'n_sigma':            n_sigma,
        'cauchy_T':           T,
        'individual_pvalues': individual_pvalues,
        'theory_mean':        np.array(theory_means),
        'theory_std':         np.array(theory_stds),
        'obs_values':         obs_values,
    }


def generate_global_statistics_bestfit(run_dir, model_name, experimental_values, derived_df=None):
    """
    Compute global tension statistic using the Cauchy Combination Test (CCT).

    Same as the MCMC variant but uses unweighted best-fit realisations.
    Computes per-parameter marginal p-values from the empirical CDF, then
    combines them via the CCT (Liu & Xie, 2020):
        T = mean(tan((0.5 - p_i) * pi))
        p_combined = 0.5 - arctan(T) / pi

    Parameters
    ----------
    run_dir : str
        Run directory
    model_name : str
        Model name
    experimental_values : dict
        {param: (value, error)}
    derived_df : pd.DataFrame, optional
        Derived parameters to add

    Returns
    -------
    dict
        {'chi2': None, 'dof': n, 'pvalue': p, 'n_sigma': n,
         'cauchy_T': float, 'individual_pvalues': list,
         'theory_mean': array, 'theory_std': array, 'obs_values': array}
    """
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='bestfit', derived_df=derived_df
    )
    if samples is None:
        return None

    param_names = list(experimental_values.keys())
    obs_values  = np.array([experimental_values[p][0] for p in param_names])

    # ================================================================
    # Extract parameter arrays from best-fit realisations
    # ================================================================
    param_arrays = []
    for p in param_names:
        try:
            param_vals = getattr(samples.getParams(), p, None)
            if param_vals is None:
                logger.warning(f"Parameter {p} not found in samples")
                return None
            param_arrays.append(np.asarray(param_vals, dtype=float))
        except Exception as e:
            logger.warning(f"Could not extract {p}: {e}")
            return None

    # ================================================================
    # Compute one marginal p-value per parameter (two-sided empirical CDF)
    # Unweighted for best-fit mode
    # ================================================================
    individual_pvalues = []
    theory_means = []
    theory_stds  = []

    for arr, obs, pname in zip(param_arrays, obs_values, param_names):
        mu  = float(arr.mean())
        std = float(arr.std(ddof=0))
        F   = float(np.mean(arr <= obs))

        pval = 2.0 * min(F, 1.0 - F)
        p_i = max(pval, 1e-16)
        individual_pvalues.append(p_i)
        theory_means.append(mu)
        theory_stds.append(std)
        logger.info(f"  {pname}: mu={mu:.4g}, std={std:.4g}, obs={obs:.4g}, p={p_i:.4e}")

    # ================================================================
    # ACAT: Aggregated Cauchy Association Test
    # ================================================================
    p_arr      = np.array(individual_pvalues)
    combined_p, T = acat(p_arr)
    combined_p = max(combined_p, 1e-16)
    # Recover T from the acat formula for logging
    n_sigma    = float(stats.norm.isf(combined_p/2 )) # / 2 for two-sided, but we are treating the p-values as one-sided here
    dof        = len(param_names)

    logger.info(f"ACAT T = {T:.4f}")
    logger.info(f"Combined p-value = {combined_p:.4e}")
    logger.info(f"Tension = {n_sigma:.2f}σ")

    return {
        'chi2':               None,
        'dof':                dof,
        'pvalue':             combined_p,
        'n_sigma':            n_sigma,
        'cauchy_T':           T,
        'individual_pvalues': individual_pvalues,
        'theory_mean':        np.array(theory_means),
        'theory_std':         np.array(theory_stds),
        'obs_values':         obs_values,
    }


def _inject_global_tension_row(table_tex, global_stats):
    """
    Insert a full-width global-tension row into a LaTeX tabular string,
    placed immediately before the first \\toprule (i.e. above the column headers).

    The number of columns is inferred from the first line that contains '&',
    or falls back to a single spanning cell.

    Parameters
    ----------
    table_tex : str
        LaTeX tabular produced by generate_statistics_table.
    global_stats : dict
        Dict returned by generate_global_statistics_mcmc/bestfit with keys
        'cauchy_T', 'dof', 'pvalue', 'n_sigma'.

    Returns
    -------
    str
        Modified LaTeX string with the tension row prepended inside the tabular.
    """
    if global_stats is None:
        return table_tex

    # Build the cell content
    tension_cell = (
        r"\textbf{Global tension (ACAT):} "
        rf"$T = {global_stats['cauchy_T']:.3f}$,\ "
        rf"$p = {global_stats['pvalue']:.2e}$,\ "
        rf"${global_stats['n_sigma']:.2f}\sigma$"
    )

    # Count columns from the first data/header line containing '&'
    ncols = 1
    for line in table_tex.splitlines():
        stripped = line.strip()
        if '&' in stripped and not stripped.startswith('%'):
            ncols = stripped.count('&') + 1
            break

    tension_row = (
        rf"\multicolumn{{{ncols}}}{{l}}{{{tension_cell}}} \\"
        + "\n"
        + r"\midrule"
    )

    # Insert before the first \toprule
    marker = r"\toprule"
    if marker in table_tex:
        table_tex = table_tex.replace(marker, marker + "\n" + tension_row, 1)
    else:
        # Fallback: prepend after \begin{tabular}{...}
        import re
        table_tex = re.sub(
            r"(\\begin\{tabular\}\{[^}]*\})",
            r"\1\n" + tension_row,
            table_tex,
            count=1,
        )

    return table_tex


def generate_all_tables(run_dir, roots, experimental_values, derived_params=None):
    """
    Generate all LaTeX tables for a run.
    
    Creates:
    - run_dir/tables/mcmc/model_name.tex (one per model)
    - run_dir/tables/bestfit/model_name.tex (one per model)
    - run_dir/tables/all_results.tex (master file)
    - run_dir/tables/statistics_summary.txt (summary with p-values)
    
    Parameters
    ----------
    run_dir : str
        Run directory
    roots : list
        List of model roots
    experimental_values : dict
        {param: (value, error)}
    derived_params : pd.DataFrame, optional
        Derived parameters to add to chains
    """
    tables_dir = os.path.join(run_dir, 'tables')
    mcmc_dir = os.path.join(tables_dir, 'mcmc')
    bestfit_dir = os.path.join(tables_dir, 'bestfit')
    
    os.makedirs(mcmc_dir, exist_ok=True)
    os.makedirs(bestfit_dir, exist_ok=True)
    
    master_tex = []
    master_tex.append("\\documentclass[a4paper,10pt]{article}")
    master_tex.append("\\usepackage{booktabs}")
    master_tex.append("\\usepackage{geometry}")
    master_tex.append("\\geometry{margin=1in}")
    master_tex.append("\\begin{document}")
    master_tex.append("\\title{CMB Analysis Results - GetDist Tables}")
    master_tex.append("\\maketitle")

    

    def parse_config(config):
        rows = []
        for section, content in config.items():
            # Skip the roots section as requested
            if section == 'roots':
                continue
                
            if isinstance(content, dict):
                for key, value in content.items():
                    # Handle nested dicts (like 'colors')
                    if isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            rows.append([f"{section}: {key}: {sub_key}", sub_val])
                    else:
                        rows.append([f"{section}: {key}", value])
                        
            elif isinstance(content, list) and section == 'intervals':
                # Format intervals as [a, b] strings
                for i, interval in enumerate(content):
                    # Rounding for readability in the table
                    formatted_range = f"[{interval[0]:.4f}, {interval[1]:.4f}]"
                    rows.append([f"Interval {i+1}", formatted_range])
                    
        return rows

    # Load the file
    with open(os.path.join(run_dir, 'config.yml'), 'r') as f:
        config = yaml.safe_load(f)

    # Flatten and convert to DataFrame
    flat_data = parse_config(config)
    df = pd.DataFrame(list(flat_data), columns=['Configuration Parameter', 'Value'])

    # Generate LaTeX with specific formatting
    # escape=True handles the underscores in your file paths
    latex_table = df.to_latex(
    index=False, 
    caption="CMB Analysis Configuration", 
    label="tab:analysis_config",
    column_format='lp{6cm}', # Left align keys, 6cm width for values
    bold_rows=False
)
    master_tex.append(str(latex_table).replace('_', ' '))
    
    summary_file = os.path.join(tables_dir, 'statistics_summary.txt')
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("STATISTICS SUMMARY WITH P-VALUES (THEORY-BASED)")
    summary_lines.append("=" * 80)
    
    master_tex.append("\\clearpage")
    for root in roots:
        model_name = root.strip().split('/')[-1]
        
        logger.info(f"Generating tables for {model_name}")
        
        # Check which modes exist
        mcmc_dir_check = os.path.join(run_dir, 'theory_mcmc', model_name)
        bestfit_dir_check = os.path.join(run_dir, 'theory_bestfit', model_name)
        
        has_mcmc = os.path.exists(mcmc_dir_check)
        has_bestfit = os.path.exists(bestfit_dir_check)
        
        master_tex.append(f"\\section{{{model_name.replace('_', ' ')}}}")
        summary_lines.append(f"\n{model_name}")
        summary_lines.append("-" * 80)
        
        # MCMC table
        if has_mcmc:
            logger.info("  - Generating MCMC table with GetDist and derived parameters...")
            
            # Generate custom table with percentiles and p-values
            table_tex = generate_statistics_table_mcmc(
                run_dir, model_name, experimental_values, derived_df=derived_params
            )
            
            if table_tex:
                # Compute global statistics
                global_stats = generate_global_statistics_mcmc(
                    run_dir, model_name, experimental_values, derived_df=derived_params
                )

                # Inject global tension row into the table before column headers
                table_tex = _inject_global_tension_row(table_tex, global_stats)

                # Save individual table (with tension row already embedded)
                with open(os.path.join(mcmc_dir, f'{model_name}.tex'), 'w') as f:
                    f.write(table_tex)
                
                # Add to master
                master_tex.append("\\subsection{MCMC Analysis}")
                master_tex.append("\\begin{table}[h]\\centering")
                master_tex.append(f"\\input{{mcmc/{model_name}.tex}}")
                master_tex.append(f"\\caption{{MCMC results for {model_name.replace('_', '\\_')}}}")
                master_tex.append("\\end{table}")

                if global_stats:
                    summary_lines.append("MCMC Global Tension (ACAT):")
                    summary_lines.append(f"  Cauchy T = {global_stats['cauchy_T']:.3f} (dof={global_stats['dof']})")
                    summary_lines.append(f"  p-value = {global_stats['pvalue']:.2e}")
                    summary_lines.append(f"  Tension = {global_stats['n_sigma']:.2f}σ")
        
        # Best-fit table
        if has_bestfit:
            logger.info("  - Generating best-fit table with GetDist...")
            
            table_tex = generate_statistics_table_bestfit(
                run_dir, model_name, experimental_values, derived_df=derived_params
            )
            
            if table_tex:
                global_stats = generate_global_statistics_bestfit(
                    run_dir, model_name, experimental_values, derived_df=derived_params
                )

                # Inject global tension row into the table before column headers
                table_tex = _inject_global_tension_row(table_tex, global_stats)

                with open(os.path.join(bestfit_dir, f'{model_name}.tex'), 'w') as f:
                    f.write(table_tex)
                
                master_tex.append("\\subsection{Best-Fit Analysis}")
                master_tex.append("\\begin{table}[h]\\centering")
                master_tex.append(f"\\input{{bestfit/{model_name}.tex}}")
                master_tex.append(f"\\caption{{Best-fit results for {model_name.replace('_', '\\_')}}}")
                master_tex.append("\\end{table}")

                if global_stats:
                    summary_lines.append("Best-fit Global Tension (ACAT):")
                    summary_lines.append(f"  Cauchy T = {global_stats['cauchy_T']:.3f} (dof={global_stats['dof']})")
                    summary_lines.append(f"  p-value = {global_stats['pvalue']:.2e}")
                    summary_lines.append(f"  Tension = {global_stats['n_sigma']:.2f}σ")
        
        master_tex.append("\\clearpage")
    
    master_tex.append("\\end{document}")
    
    # Save master file
    master_file = os.path.join(tables_dir, 'all_results.tex')
    with open(master_file, 'w') as f:
        f.write('\n'.join(master_tex))
    
    # Save summary
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info(f"✓ All tables saved to {tables_dir}")
    logger.info(f"✓ Master file: {master_file}")
    logger.info(f"✓ Summary file: {summary_file}")
    
    # Try to compile PDF
    try:
        import subprocess
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'all_results.tex'],
            cwd=tables_dir,
            check=True,
            stdout=subprocess.DEVNULL
        )
        logger.info(f"✓ PDF generated: {os.path.join(tables_dir, 'all_results.pdf')}")
    except Exception as e:
        logger.warning("Could not compile PDF (pdflatex not available): " + str(e))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', required=True, help='Config YAML file')
    args = parser.parse_args()
    
    # Load config to get roots and intervals
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get experimental values
    from functions.data import Data_loader
    
    intervals = [tuple(i) for i in config['intervals']]
    DL = Data_loader(lmax=config['analysis']['lmax'])
    experimental_values = DL.experimental_values(intervals)
    
    # Optional: Load derived parameters if available
    derived_params = None
    derived_params_file = os.path.join(args.run_dir, 'derived_parameters.csv')
    if os.path.exists(derived_params_file):
        derived_params = pd.read_csv(derived_params_file)
        logger.info(f"Loaded {len(derived_params.columns)} derived parameters")
    
    # Generate tables
    generate_all_tables(
        args.run_dir,
        config['roots'],
        experimental_values,
        derived_params=derived_params
    )