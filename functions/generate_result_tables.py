#!/usr/bin/env python3
"""
Generate comprehensive LaTeX tables for cosmological analysis.

Features
--------
- Uses GetDist for proper percentile computation with MCMC weights.
- Experimental *errors* are never used; the observed scalar value alone is
  compared to the empirical ensemble distribution.
- Individual p-values rendered with sigma notation via unified_stats.sigma_label.
- Global significance via the Aggregated Cauchy Association Test (ACAT).
"""

import os
import re
import argparse
import numpy as np
import yaml
import pandas as pd
import logging
from getdist import mcsamples
from scipy import stats

from .getdist_stats import (
    add_derived_parameters,
    compute_multivariate_tension,
    generate_statistics_table,
    _format_label_for_getdist,
)
from .unified_stats import pvalue_to_sigma, sigma_label

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Chain loading
# ---------------------------------------------------------------------------

def load_chain_with_derived_params(run_dir, model_name, mode='mcmc',
                                    derived_df=None):
    """
    Load an MCMC chain or build a fake ``MCSamples`` object for best-fit
    realisations, then optionally attach derived parameters.

    Parameters
    ----------
    run_dir : str
    model_name : str
    mode : str
        ``'mcmc'`` or ``'bestfit'``.
    derived_df : pd.DataFrame, optional
        Derived parameters to inject.

    Returns
    -------
    MCSamples or None
    """
    model_dir  = os.path.join(run_dir, f'theory_{mode}', model_name)
    chain_root = os.path.join(model_dir, f'{model_name}_chain')

    try:
        if mode == 'mcmc':
            samples = mcsamples.loadMCSamples(
                chain_root, settings={'ignore_rows': 0}
            )
            if samples is None:
                logger.warning(f"Could not load samples from {chain_root}")
                return None

            if derived_df is not None and isinstance(derived_df, pd.DataFrame):
                try:
                    samples = add_derived_parameters(samples, derived_df)
                    logger.info(
                        f"Added {len(derived_df.columns)} derived parameters "
                        f"to {model_name}"
                    )
                except Exception as exc:
                    logger.warning(f"Could not add derived parameters: {exc}")

        elif mode == 'bestfit':
            logger.info(f"Loading best-fit statistics for {model_name}…")

            cols_path = os.path.join(model_dir, 'columns.txt')
            if not os.path.exists(cols_path):
                logger.warning(f"columns.txt not found in {model_dir}")
                return None

            with open(cols_path) as f:
                scalar_cols = [line.strip() for line in f if line.strip()]

            data = {}

            s_path = os.path.join(model_dir, 'S_statistics.txt')
            if os.path.exists(s_path):
                S_mat = np.loadtxt(s_path)
                if S_mat.ndim == 1:
                    S_mat = S_mat[:, None]
                s_cols = [c for c in scalar_cols if c.startswith('s12_')]
                for i, col in enumerate(s_cols):
                    if i < S_mat.shape[1]:
                        data[col] = S_mat[:, i]

            xiv_path = os.path.join(model_dir, 'xiv_statistic.txt')
            if os.path.exists(xiv_path):
                xiv_mat = np.loadtxt(xiv_path)
                if xiv_mat.ndim == 1:
                    xiv_mat = xiv_mat[:, None]
                xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
                for i, col in enumerate(xiv_cols):
                    if i < xiv_mat.shape[1]:
                        data[col] = xiv_mat[:, i]

            c180_path = os.path.join(model_dir, 'C180_statistics.txt')
            if os.path.exists(c180_path):
                data['C180'] = np.atleast_1d(np.loadtxt(c180_path))

            if not data:
                logger.warning(f"No statistics data found for {model_name}")
                return None

            df            = pd.DataFrame(data)
            n_samples     = len(df)
            names         = list(df.columns)
            labels        = [_format_label_for_getdist(n) for n in names]

            samples = mcsamples.MCSamples(
                samples=df.values,
                weights=np.ones(n_samples),
                names=names,
                labels=labels,
                label=model_name,
            )
            logger.info(
                f"Created MCSamples with {n_samples} samples for "
                f"{model_name} (best-fit mode)"
            )

        else:
            logger.error(f"Unknown mode: {mode}")
            return None

        return samples

    except Exception as exc:
        logger.exception(f"Error loading samples for {model_name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Table generation wrappers
# ---------------------------------------------------------------------------

def generate_statistics_table_mcmc(run_dir, model_name, experimental_values,
                                    derived_df=None):
    """
    Generate a LaTeX tabular string for the MCMC ensemble.

    Parameters
    ----------
    run_dir : str
    model_name : str
    experimental_values : dict
        ``{param: (value, error)}`` — only the value is used; error is ignored.
    derived_df : pd.DataFrame, optional

    Returns
    -------
    str or None
    """
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='mcmc', derived_df=derived_df
    )
    if samples is None:
        logger.warning(f"Could not load samples for {model_name}")
        return None

    return generate_statistics_table(
        samples,
        list(experimental_values.keys()),
        experimental_values,
        mode='mcmc',
    )


def generate_statistics_table_bestfit(run_dir, model_name, experimental_values,
                                       derived_df=None):
    """
    Generate a LaTeX tabular string for the best-fit ensemble.

    Parameters
    ----------
    run_dir : str
    model_name : str
    experimental_values : dict
        ``{param: (value, error)}`` — only the value is used; error is ignored.
    derived_df : pd.DataFrame, optional

    Returns
    -------
    str or None
    """
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='bestfit', derived_df=derived_df
    )
    if samples is None:
        logger.warning(f"Could not load best-fit samples for {model_name}")
        return None

    return generate_statistics_table(
        samples,
        list(experimental_values.keys()),
        experimental_values,
        mode='bestfit',
    )


# ---------------------------------------------------------------------------
# ACAT implementation
# ---------------------------------------------------------------------------

def acat(pvals, weights=None):
    """
    Aggregated Cauchy Association Test (ACAT; Liu & Xie, 2020).

    Combines individual p-values using a weighted Cauchy statistic::

        T = sum(w_i * tan((0.5 - p_i) * pi))
        p_combined = 0.5 - arctan(T) / pi

    Parameters
    ----------
    pvals : array-like
        Individual p-values; all must be in the open interval (0, 1).
    weights : array-like, optional
        Non-negative weights.  Normalised internally.  Defaults to 1/n each.

    Returns
    -------
    (float, float)
        ``(p_combined, T)``

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
        weights = weights / weights.sum()
    T = float(np.sum(weights * np.tan((0.5 - pvals) * np.pi)))
    return float(0.5 - np.arctan(T) / np.pi), T


# ---------------------------------------------------------------------------
# Global statistics (MCMC and best-fit)
# ---------------------------------------------------------------------------

def _compute_global_statistics(samples, experimental_values, weighted: bool):
    """
    Shared implementation for MCMC and best-fit global tension.

    Computes one empirical p-value per parameter (one-sided CDF), then
    combines them with ACAT.  Experimental *errors* are not used.

    Parameters
    ----------
    samples : MCSamples
    experimental_values : dict
        ``{param: (value, error)}`` — only value consumed.
    weighted : bool
        If True, use sample weights for the CDF (MCMC mode).
        If False, treat all samples equally (best-fit mode).

    Returns
    -------
    dict or None
    """
    param_names = list(experimental_values.keys())
    obs_values  = np.array([experimental_values[p][0] for p in param_names])

    param_arrays = []
    for p in param_names:
        vals = getattr(samples.getParams(), p, None)
        if vals is None:
            logger.warning(f"Parameter {p} not found in samples")
            return None
        param_arrays.append(np.asarray(vals, dtype=float))

    weights = None
    if weighted and hasattr(samples, 'weights') and samples.weights is not None:
        weights = np.asarray(samples.weights, dtype=float)
        weights = weights / weights.sum()

    individual_pvalues = []
    theory_medians     = []
    theory_ci_lo       = []
    theory_ci_hi       = []

    for arr, obs, pname in zip(param_arrays, obs_values, param_names):
        # Percentile-based summary (consistent with the rest of the codebase)
        if weights is not None:
            sorter     = np.argsort(arr)
            cumw       = np.cumsum(weights[sorter])
            p16        = float(np.interp(0.16, cumw, arr[sorter]))
            p50        = float(np.interp(0.50, cumw, arr[sorter]))
            p84        = float(np.interp(0.84, cumw, arr[sorter]))
            F          = float(np.interp(obs,  arr[sorter], cumw))
        else:
            p16 = float(np.percentile(arr, 16))
            p50 = float(np.percentile(arr, 50))
            p84 = float(np.percentile(arr, 84))
            F   = float(np.mean(arr <= obs))

        floor = 1e-16
        p_i   = max(F, floor)
        individual_pvalues.append(p_i)
        theory_medians.append(p50)
        theory_ci_lo.append(p16)
        theory_ci_hi.append(p84)

        logger.info(
            f"  {pname}: median={p50:.4g}, 68%CI=[{p16:.4g},{p84:.4g}], "
            f"obs={obs:.4g}, p={p_i:.4e} ({pvalue_to_sigma(p_i):.2f}σ)"
        )

    p_arr      = np.array(individual_pvalues)
    combined_p, T = acat(p_arr)
    combined_p = max(combined_p, 1e-16)
    n_sigma    = float(stats.norm.isf(combined_p))

    logger.info(f"ACAT T = {T:.4f}")
    logger.info(f"Combined p = {combined_p:.4e}  ({n_sigma:.2f}σ)")

    return {
        'chi2':               None,
        'dof':                len(param_names),
        'pvalue':             combined_p,
        'n_sigma':            n_sigma,
        'cauchy_T':           T,
        'individual_pvalues': individual_pvalues,
        'theory_median':      np.array(theory_medians),
        'theory_ci_lo':       np.array(theory_ci_lo),
        'theory_ci_hi':       np.array(theory_ci_hi),
        'obs_values':         obs_values,
    }


def generate_global_statistics_mcmc(run_dir, model_name, experimental_values,
                                     derived_df=None):
    """
    Compute global ACAT tension for the MCMC ensemble.

    Parameters
    ----------
    run_dir : str
    model_name : str
    experimental_values : dict
        ``{param: (value, error)}`` — only value consumed.
    derived_df : pd.DataFrame, optional

    Returns
    -------
    dict or None
    """
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='mcmc', derived_df=derived_df
    )
    if samples is None:
        return None
    return _compute_global_statistics(samples, experimental_values,
                                      weighted=True)


def generate_global_statistics_bestfit(run_dir, model_name, experimental_values,
                                        derived_df=None):
    """
    Compute global ACAT tension for the best-fit ensemble.

    Parameters
    ----------
    run_dir : str
    model_name : str
    experimental_values : dict
        ``{param: (value, error)}`` — only value consumed.
    derived_df : pd.DataFrame, optional

    Returns
    -------
    dict or None
    """
    samples = load_chain_with_derived_params(
        run_dir, model_name, mode='bestfit', derived_df=derived_df
    )
    if samples is None:
        return None
    return _compute_global_statistics(samples, experimental_values,
                                      weighted=False)


# ---------------------------------------------------------------------------
# LaTeX injection
# ---------------------------------------------------------------------------

def _inject_global_tension_row(table_tex, global_stats):
    """
    Insert a full-width global-tension row immediately after ``\\toprule``.

    The row uses ``sigma_label`` for consistent formatting with the
    individual p-value cells.

    Parameters
    ----------
    table_tex : str
        LaTeX tabular produced by ``generate_statistics_table``.
    global_stats : dict
        Returned by ``generate_global_statistics_mcmc/bestfit``.

    Returns
    -------
    str
    """
    if global_stats is None:
        return table_tex

    pval   = global_stats['pvalue']
    T      = global_stats['cauchy_T']
    # sigma_label() already returns $...$, so embed directly without
    # wrapping in another pair of $ delimiters.
    pv_str = sigma_label(pval)

    tension_cell = (
        r"\textbf{Global tension (ACAT):} "
        rf"$T = {T:.3f}$,\ "
        rf"p = {pv_str}"
    )

    # Infer column count from the first line containing '&'
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

    marker = r"\toprule"
    if marker in table_tex:
        return table_tex.replace(marker, marker + "\n" + tension_row, 1)

    # Fallback: insert after \begin{tabular}{...}
    return re.sub(
        r"(\\begin\{tabular\}\{[^}]*\})",
        r"\1\n" + tension_row,
        table_tex,
        count=1,
    )


# ---------------------------------------------------------------------------
# Master table generator
# ---------------------------------------------------------------------------

def generate_all_tables(run_dir, roots, experimental_values,
                        derived_params=None):
    """
    Generate all LaTeX tables for a run.

    Creates
    -------
    ``run_dir/tables/mcmc/<model>.tex``
    ``run_dir/tables/bestfit/<model>.tex``
    ``run_dir/tables/all_results.tex``
    ``run_dir/tables/statistics_summary.txt``

    Parameters
    ----------
    run_dir : str
    roots : list of str
    experimental_values : dict
        ``{param: (value, error)}`` — only value consumed.
    derived_params : pd.DataFrame, optional
    """
    tables_dir  = os.path.join(run_dir, 'tables')
    mcmc_dir    = os.path.join(tables_dir, 'mcmc')
    bestfit_dir = os.path.join(tables_dir, 'bestfit')

    os.makedirs(mcmc_dir,    exist_ok=True)
    os.makedirs(bestfit_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Master LaTeX document header
    # ------------------------------------------------------------------
    master_tex = [
        r"\documentclass[a4paper,10pt]{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        r"\begin{document}",
        r"\title{CMB Analysis Results - GetDist Tables}",
        r"\maketitle",
    ]

    # ------------------------------------------------------------------
    # Configuration table
    # ------------------------------------------------------------------
    def parse_config(config):
        rows = []
        for section, content in config.items():
            if section == 'roots':
                continue
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            rows.append(
                                [f"{section}: {key}: {sub_key}", sub_val]
                            )
                    else:
                        rows.append([f"{section}: {key}", value])
            elif isinstance(content, list) and section == 'intervals':
                for i, interval in enumerate(content):
                    rows.append(
                        [f"Interval {i+1}",
                         f"[{interval[0]:.4f}, {interval[1]:.4f}]"]
                    )
        return rows

    with open(os.path.join(run_dir, 'config.yml')) as f:
        config = yaml.safe_load(f)

    df_config = pd.DataFrame(
        parse_config(config),
        columns=['Configuration Parameter', 'Value'],
    )
    latex_config = df_config.to_latex(
        index=False,
        caption="CMB Analysis Configuration",
        label="tab:analysis_config",
        column_format='lp{6cm}',
        bold_rows=False,
    )
    master_tex.append(str(latex_config).replace('_', ' '))

    # ------------------------------------------------------------------
    # Per-model tables
    # ------------------------------------------------------------------
    summary_lines = [
        "=" * 80,
        "STATISTICS SUMMARY WITH P-VALUES (THEORY-BASED)",
        "=" * 80,
    ]

    master_tex.append(r"\clearpage")

    for root in roots:
        model_name       = root.strip().split('/')[-1]
        mcmc_dir_check   = os.path.join(run_dir, 'theory_mcmc',    model_name)
        bestfit_dir_check = os.path.join(run_dir, 'theory_bestfit', model_name)
        has_mcmc         = os.path.exists(mcmc_dir_check)
        has_bestfit      = os.path.exists(bestfit_dir_check)

        logger.info(f"Generating tables for {model_name}")
        master_tex.append(
            rf"\section{{{model_name.replace('_', ' ')}}}"
        )
        summary_lines.append(f"\n{model_name}")
        summary_lines.append("-" * 80)

        # MCMC
        if has_mcmc:
            logger.info("  Generating MCMC table…")
            table_tex = generate_statistics_table_mcmc(
                run_dir, model_name, experimental_values,
                derived_df=derived_params,
            )
            if table_tex:
                global_stats = generate_global_statistics_mcmc(
                    run_dir, model_name, experimental_values,
                    derived_df=derived_params,
                )
                table_tex = _inject_global_tension_row(table_tex, global_stats)

                with open(os.path.join(mcmc_dir, f'{model_name}.tex'), 'w') as f:
                    f.write(table_tex)

                master_tex += [
                    r"\subsection{MCMC Analysis}",
                    r"\begin{table}[h]\centering",
                    rf"\input{{mcmc/{model_name}.tex}}",
                    rf"\caption{{MCMC results for {model_name.replace('_', chr(92)+'_')}}}",
                    r"\end{table}",
                ]

                if global_stats:
                    p   = global_stats['pvalue']
                    ns  = global_stats['n_sigma']
                    T   = global_stats['cauchy_T']
                    summary_lines += [
                        "MCMC Global Tension (ACAT):",
                        f"  Cauchy T = {T:.3f}  (dof={global_stats['dof']})",
                        f"  p = {p:.2e}  ({ns:.2f}σ)",
                    ]

        # Best-fit
        if has_bestfit:
            logger.info("  Generating best-fit table…")
            table_tex = generate_statistics_table_bestfit(
                run_dir, model_name, experimental_values,
                derived_df=derived_params,
            )
            if table_tex:
                global_stats = generate_global_statistics_bestfit(
                    run_dir, model_name, experimental_values,
                    derived_df=derived_params,
                )
                table_tex = _inject_global_tension_row(table_tex, global_stats)

                with open(os.path.join(bestfit_dir, f'{model_name}.tex'), 'w') as f:
                    f.write(table_tex)

                master_tex += [
                    r"\subsection{Best-Fit Analysis}",
                    r"\begin{table}[h]\centering",
                    rf"\input{{bestfit/{model_name}.tex}}",
                    rf"\caption{{Best-fit results for {model_name.replace('_', chr(92)+'_')}}}",
                    r"\end{table}",
                ]

                if global_stats:
                    p   = global_stats['pvalue']
                    ns  = global_stats['n_sigma']
                    T   = global_stats['cauchy_T']
                    summary_lines += [
                        "Best-fit Global Tension (ACAT):",
                        f"  Cauchy T = {T:.3f}  (dof={global_stats['dof']})",
                        f"  p = {p:.2e}  ({ns:.2f}σ)",
                    ]

        master_tex.append(r"\clearpage")

    master_tex.append(r"\end{document}")

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    master_file  = os.path.join(tables_dir, 'all_results.tex')
    summary_file = os.path.join(tables_dir, 'statistics_summary.txt')

    with open(master_file, 'w') as f:
        f.write('\n'.join(master_tex))

    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))

    logger.info(f"✓ All tables saved to {tables_dir}")
    logger.info(f"✓ Master file:  {master_file}")
    logger.info(f"✓ Summary file: {summary_file}")

    # ------------------------------------------------------------------
    # Attempt PDF compilation
    # ------------------------------------------------------------------
    try:
        import subprocess
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', 'all_results.tex'],
            cwd=tables_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        logger.info(
            f"✓ PDF generated: {os.path.join(tables_dir, 'all_results.pdf')}"
        )
    except Exception as exc:
        logger.warning(f"Could not compile PDF (pdflatex not available): {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config',  required=True, help='Config YAML file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from functions.data import Data_loader

    intervals          = [tuple(i) for i in config['intervals']]
    DL                 = Data_loader(lmax=config['analysis']['lmax'])
    experimental_values = DL.experimental_values(intervals)

    derived_params      = None
    derived_params_file = os.path.join(args.run_dir, 'derived_parameters.csv')
    if os.path.exists(derived_params_file):
        derived_params = pd.read_csv(derived_params_file)
        logger.info(
            f"Loaded {len(derived_params.columns)} derived parameters"
        )

    generate_all_tables(
        args.run_dir,
        config['roots'],
        experimental_values,
        derived_params=derived_params,
    )