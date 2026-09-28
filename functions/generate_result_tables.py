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
    compute_all_percentiles,
    compute_all_pvalues,
    _format_label_for_getdist,
)
from .unified_stats import pvalue_to_sigma, sigma_label, is_degenerate_stat

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


def _format_ensemble_value_cell(perc_dict, param):
    """
    Format one ensemble's median with its 68% CI attached as
    subscript/superscript (``median_{p16}^{p84}``, matching the
    paper's :math:`\\langle\\xi\\rangle_{\\theta_a}^{\\theta_b}`
    interval-bound notation).

    Parameters
    ----------
    perc_dict : dict
        ``{param: {'p16':…, 'p50':…, 'p84':…}}`` for one ensemble.
    param : str

    Returns
    -------
    str
        LaTeX math string, or a placeholder dash if the ensemble has no
        data for this statistic (missing model).
    """
    if param not in perc_dict:
        return "---"

    perc = perc_dict[param]
    p16, p50, p84 = perc["p16"], perc["p50"], perc["p84"]
    return rf"${p50:.4f}_{{{p16:.4f}}}^{{{p84:.4f}}}$"


def _format_ensemble_pvalue_cell(pval_dict, param):
    """
    Format one ensemble's empirical p-value with sigma notation.

    Parameters
    ----------
    pval_dict : dict
        ``{param: {'pvalue':…, 'floored':…}}`` for one ensemble.
    param : str

    Returns
    -------
    str
        LaTeX math string, or a placeholder dash if there is no
        meaningful p-value (missing model, or an analytically
        degenerate statistic such as ``xiv_180_0``).
    """
    if param not in pval_dict:
        return "---"

    return sigma_label(
        pval_dict[param]["pvalue"],
        bound=pval_dict[param].get("floored", False),
    )


def generate_combined_statistics_table(run_dir, model_name, experimental_values,
                                        derived_df=None):
    """
    Generate a single LaTeX tabular string per model, with the MCMC and
    best-fit ensembles side by side as separate columns instead of two
    separate per-ensemble tables.

    Columns: Statistic | Experimental | MCMC{Value, $p$-value} |
    Best-fit{Value, $p$-value} — each ensemble spans two sub-columns
    (grouped under its own header cell) so the p-value/sigma sits to
    the right of the numerical value instead of stacked below it. The
    ensemble's global ACAT Cauchy statistic is shown right in the group
    header, next to the ensemble's name. Each "Value" cell is the
    median with its 68% credible interval attached as
    subscript/superscript (``median_{p16}^{p84}``). A row above the
    header reports the combined global p-value for each ensemble.

    Parameters
    ----------
    run_dir : str
    model_name : str
    experimental_values : dict
        ``{param: (value, error)}`` — only the value is used; error is
        ignored.
    derived_df : pd.DataFrame, optional

    Returns
    -------
    tuple(str or None, dict or None, dict or None)
        ``(table_tex, mcmc_global, bestfit_global)``. ``table_tex`` is
        None if neither ensemble could be loaded for this model.
        ``mcmc_global``/``bestfit_global`` are the dicts returned by
        ``_compute_global_statistics`` (or None if that ensemble is
        unavailable) — callers can reuse them (e.g. for a summary text
        file) without reloading the chain.
    """
    mcmc_samples = load_chain_with_derived_params(
        run_dir, model_name, mode='mcmc', derived_df=derived_df
    )
    bestfit_samples = load_chain_with_derived_params(
        run_dir, model_name, mode='bestfit', derived_df=derived_df
    )

    if mcmc_samples is None and bestfit_samples is None:
        logger.warning(
            f"Could not load MCMC or best-fit samples for {model_name}"
        )
        return None, None, None

    param_names = list(experimental_values.keys())

    mcmc_perc = (compute_all_percentiles(mcmc_samples, param_names)
                 if mcmc_samples is not None else {})
    mcmc_pval = (compute_all_pvalues(mcmc_samples, param_names, experimental_values)
                 if mcmc_samples is not None else {})
    bestfit_perc = (compute_all_percentiles(bestfit_samples, param_names)
                    if bestfit_samples is not None else {})
    bestfit_pval = (compute_all_pvalues(bestfit_samples, param_names, experimental_values)
                    if bestfit_samples is not None else {})

    mcmc_global = (
        _compute_global_statistics(mcmc_samples, experimental_values, weighted=True)
        if mcmc_samples is not None else None
    )
    bestfit_global = (
        _compute_global_statistics(bestfit_samples, experimental_values, weighted=False)
        if bestfit_samples is not None else None
    )

    try:
        # Ensemble names carry their own global Cauchy statistic T right
        # in the group header, spanning the ensemble's Value/p-value
        # sub-columns.
        mcmc_header = "MCMC"
        if mcmc_global is not None:
            mcmc_header += rf" ($T={mcmc_global['cauchy_T']:.3f}$)"
        bestfit_header = "Best-fit"
        if bestfit_global is not None:
            bestfit_header += rf" ($T={bestfit_global['cauchy_T']:.3f}$)"

        rows = []
        rows.append(r"{\renewcommand{\arraystretch}{2.0}")
        rows.append(r"\begin{tabular}{lccccc}")
        rows.append(r"\toprule")

        tension_parts = []
        if mcmc_global is not None:
            tension_parts.append(rf"MCMC: p = {sigma_label(mcmc_global['pvalue'])}")
        if bestfit_global is not None:
            tension_parts.append(rf"Best-fit: p = {sigma_label(bestfit_global['pvalue'])}")
        if tension_parts:
            tension_cell = (r"\textbf{Global tension (ACAT).} "
                             + r"\quad ".join(tension_parts))
            rows.append(rf"\multicolumn{{6}}{{l}}{{{tension_cell}}} \\")
            rows.append(r"\midrule")

        rows.append(
            rf" & & \multicolumn{{2}}{{c}}{{{mcmc_header}}} & "
            rf"\multicolumn{{2}}{{c}}{{{bestfit_header}}} \\"
        )
        rows.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
        rows.append(
            r"Statistic & Experimental & Value & $p$-value & Value & $p$-value \\"
        )
        rows.append(r"\midrule")

        for param in param_names:
            if param not in mcmc_perc and param not in bestfit_perc:
                continue
            if param not in experimental_values:
                continue

            param_label = _format_label_for_getdist(param)

            # Observed value only — no error
            exp_val = experimental_values[param][0]
            exp_str = rf"${exp_val:.4f}$"

            mcmc_val    = _format_ensemble_value_cell(mcmc_perc, param)
            mcmc_pv     = _format_ensemble_pvalue_cell(mcmc_pval, param)
            bestfit_val = _format_ensemble_value_cell(bestfit_perc, param)
            bestfit_pv  = _format_ensemble_pvalue_cell(bestfit_pval, param)

            rows.append(
                rf"{param_label} & {exp_str} & {mcmc_val} & {mcmc_pv} & "
                rf"{bestfit_val} & {bestfit_pv} \\"
            )

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}}")
        return "\n".join(rows), mcmc_global, bestfit_global

    except Exception as exc:
        logger.exception(f"Error generating combined table for {model_name}: {exc}")
        return None, mcmc_global, bestfit_global


def _inject_combined_global_tension_row(table_tex, mcmc_global, bestfit_global,
                                         ncols=4):
    """
    Insert a full-width row (immediately after ``\\toprule``) reporting
    the global ACAT tension for both ensembles.

    Parameters
    ----------
    table_tex : str
        LaTeX tabular produced by ``generate_combined_statistics_table``.
    mcmc_global : dict or None
        Returned by ``generate_global_statistics_mcmc``.
    bestfit_global : dict or None
        Returned by ``generate_global_statistics_bestfit``.
    ncols : int
        Number of columns in the table (for the ``\\multicolumn`` span).

    Returns
    -------
    str
    """
    parts = []
    if mcmc_global is not None:
        pv = sigma_label(mcmc_global['pvalue'])
        parts.append(rf"MCMC: $T = {mcmc_global['cauchy_T']:.3f}$, p = {pv}")
    if bestfit_global is not None:
        pv = sigma_label(bestfit_global['pvalue'])
        parts.append(rf"Best-fit: $T = {bestfit_global['cauchy_T']:.3f}$, p = {pv}")

    if not parts:
        return table_tex

    tension_cell = r"\textbf{Global tension (ACAT).} " + r"\quad ".join(parts)
    tension_row = (
        rf"\multicolumn{{{ncols}}}{{l}}{{{tension_cell}}} \\"
        + "\n"
        + r"\midrule"
    )

    marker = r"\toprule"
    if marker in table_tex:
        return table_tex.replace(marker, marker + "\n" + tension_row, 1)

    return re.sub(
        r"(\\begin\{tabular\}\{[^}]*\})",
        r"\1\n" + tension_row,
        table_tex,
        count=1,
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
    # Identically-zero statistics (e.g. xi_{180,0}) carry no information and
    # would inject an arbitrary p-value into the Cauchy combination.
    param_names = [p for p in experimental_values if not is_degenerate_stat(p)]
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

        # Resolution limit of the ensemble: p cannot be resolved below 1/n
        # (n = effective number of samples). Same convention as
        # unified_stats.compute_pvalue_unified. Clip both tails so ACAT's
        # tan() stays finite.
        if weights is not None:
            n_eff = 1.0 / float(np.sum(weights ** 2))
        else:
            n_eff = float(len(arr))
        floor = 1.0 / n_eff
        p_i   = min(max(F, floor), 1.0 - floor)
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

    One combined table is produced per model (not per ensemble): each
    table has columns Statistic | Experimental | MCMC | Best-fit, with
    the MCMC and best-fit median/68% CI/p-value packed into their own
    column instead of being split across two separate tables.

    Creates
    -------
    ``run_dir/tables/<model>.tex``
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
    tables_dir = os.path.join(run_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

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
        model_name        = root.strip().split('/')[-1]
        mcmc_dir_check    = os.path.join(run_dir, 'theory_mcmc',    model_name)
        bestfit_dir_check = os.path.join(run_dir, 'theory_bestfit', model_name)
        has_mcmc          = os.path.exists(mcmc_dir_check)
        has_bestfit       = os.path.exists(bestfit_dir_check)

        logger.info(f"Generating combined table for {model_name}")
        master_tex.append(
            rf"\section{{{model_name.replace('_', ' ')}}}"
        )
        summary_lines.append(f"\n{model_name}")
        summary_lines.append("-" * 80)

        if not (has_mcmc or has_bestfit):
            logger.warning(f"  No MCMC or best-fit data found for {model_name}")
            master_tex.append(r"\clearpage")
            continue

        table_tex, mcmc_global, bestfit_global = generate_combined_statistics_table(
            run_dir, model_name, experimental_values,
            derived_df=derived_params,
        )

        if table_tex:
            with open(os.path.join(tables_dir, f'{model_name}.tex'), 'w') as f:
                f.write(table_tex)

            master_tex += [
                r"\begin{table}[h]\centering",
                rf"\input{{{model_name}.tex}}",
                rf"\caption{{Statistical summary for {model_name.replace('_', chr(92)+'_')}: "
                r"experimental value, MCMC, and best-fit ensembles.}}",
                r"\end{table}",
            ]

            if mcmc_global:
                p, ns, T = (mcmc_global['pvalue'], mcmc_global['n_sigma'],
                            mcmc_global['cauchy_T'])
                summary_lines += [
                    "MCMC Global Tension (ACAT):",
                    f"  Cauchy T = {T:.3f}  (dof={mcmc_global['dof']})",
                    f"  p = {p:.2e}  ({ns:.2f}σ)",
                ]
            if bestfit_global:
                p, ns, T = (bestfit_global['pvalue'], bestfit_global['n_sigma'],
                            bestfit_global['cauchy_T'])
                summary_lines += [
                    "Best-fit Global Tension (ACAT):",
                    f"  Cauchy T = {T:.3f}  (dof={bestfit_global['dof']})",
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