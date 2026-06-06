"""
GetDist-based statistics calculator for MCMC posteriors.

Key principles
--------------
- Uses percentiles exclusively (never means or std for characterising the
  theoretical distribution).
- Experimental *errors* are never used.  The observed scalar value alone is
  compared to the empirical distribution.
- Individual p-values in tables are rendered with sigma notation via
  unified_stats.sigma_label().
- The global matplotlib style is inherited from unified_stats (imported below).
"""

import numpy as np
import pandas as pd
import os
import logging
from getdist import mcsamples, loadMCSamples
from getdist.types import ResultTable, NoLineTableFormatter
from scipy import stats

from .unified_stats import (
    compute_percentiles_unified,
    compute_pvalue_unified,
    sigma_label,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

def _format_label_for_getdist(param_name: str) -> str:
    """
    Format a parameter name as a LaTeX label for GetDist / table display.

    Examples
    --------
    ``'xiv_180_60'``  →  ``r'$\\xi_{180,60}$'``
    ``'s12_180_60'``  →  ``r'$S_{180,60}$'``
    ``'C180'``        →  ``r'$C_{180}$'``
    """
    if param_name == "C180":
        return r"$C_{180}$"

    if param_name.startswith("xiv_"):
        parts = param_name.split("_")
        if len(parts) == 3:
            return rf"$\xi_{{{parts[1]},{parts[2]}}}$"

    if param_name.startswith("s12_"):
        parts = param_name.split("_")
        if len(parts) == 3:
            return rf"$S_{{{parts[1]},{parts[2]}}}$"

    return param_name


# ---------------------------------------------------------------------------
# Load statistics as MCSamples
# ---------------------------------------------------------------------------

def load_statistics_as_mcsamples(run_dir, model_name, mode="mcmc",
                                  root_dir=None):
    """
    Load statistics from a run directory as a GetDist ``MCSamples`` object.

    Parameters
    ----------
    run_dir : str
    model_name : str
    mode : str
        ``'mcmc'`` or ``'bestfit'``.
    root_dir : str, optional
        If provided, attempt to load a chain file directly from this path.

    Returns
    -------
    MCSamples or None
    """
    model_dir = os.path.join(run_dir, f"theory_{mode}", model_name)

    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return None

    paramnames_file = os.path.join(model_dir, f"{model_name}.paramnames")

    if root_dir and os.path.exists(root_dir) and os.path.exists(paramnames_file):
        try:
            samples = loadMCSamples(root_dir, settings={"ignore_rows": 0})
            logger.info(
                f"Loaded {samples.numrows} samples for {model_name} from "
                "chain files"
            )
            return samples
        except Exception as exc:
            logger.warning(f"Could not load chain file: {exc}")

    # Fallback: build MCSamples from .txt statistics files
    try:
        cols_path = os.path.join(model_dir, "columns.txt")
        if not os.path.exists(cols_path):
            return None

        with open(cols_path) as f:
            scalar_cols = [line.strip() for line in f if line.strip()]

        data = {}

        s_path = os.path.join(model_dir, "S_statistics.txt")
        if os.path.exists(s_path):
            S_mat = np.loadtxt(s_path)
            if S_mat.ndim == 1:
                S_mat = S_mat[:, None]
            s_cols = [c for c in scalar_cols if c.startswith("s12_")]
            for i, col in enumerate(s_cols):
                if i < S_mat.shape[1]:
                    data[col] = S_mat[:, i]

        xiv_path = os.path.join(model_dir, "xiv_statistic.txt")
        if os.path.exists(xiv_path):
            xiv_mat = np.loadtxt(xiv_path)
            if xiv_mat.ndim == 1:
                xiv_mat = xiv_mat[:, None]
            xiv_cols = [c for c in scalar_cols if c.startswith("xiv_")]
            for i, col in enumerate(xiv_cols):
                if i < xiv_mat.shape[1]:
                    data[col] = xiv_mat[:, i]

        c180_path = os.path.join(model_dir, "C180_statistics.txt")
        if os.path.exists(c180_path):
            data["C180"] = np.atleast_1d(np.loadtxt(c180_path))

        if not data:
            return None

        df            = pd.DataFrame(data)
        samples_array = df.values
        names         = list(df.columns)
        labels        = [_format_label_for_getdist(n) for n in names]

        samples = mcsamples.MCSamples(
            samples=samples_array,
            weights=np.ones(len(df)),
            names=names,
            labels=labels,
            label=model_name,
        )
        logger.info(
            f"Created MCSamples with {len(df)} samples for {model_name}"
        )
        return samples

    except Exception as exc:
        logger.exception(f"Error creating MCSamples: {exc}")
        return None


# ---------------------------------------------------------------------------
# Derived-parameter injection
# ---------------------------------------------------------------------------

def add_derived_parameters(samples, derived_df, n_final_samples=None):
    """
    Add derived parameters to an MCMC ``MCSamples`` object.

    Parameters
    ----------
    samples : MCSamples
    derived_df : pd.DataFrame
        One column per derived parameter.
    n_final_samples : int, optional
        Number of tail samples to keep.  Defaults to ``len(derived_df)``.

    Returns
    -------
    MCSamples
    """
    if samples is None or derived_df is None:
        logger.warning(
            "Cannot add derived parameters: samples or derived_df is None"
        )
        return samples

    try:
        total = samples.numrows
        N     = min(
            derived_df.shape[0] if n_final_samples is None else n_final_samples,
            total,
        )
        start = total - N
        samples.filter(np.arange(start, total))
        logger.info(f"Filtered samples: kept rows {start}–{total}")

        for param_name in derived_df.columns:
            try:
                samples.addDerived(
                    derived_df[param_name].values[:N],
                    name=param_name,
                    label=_format_label_for_getdist(param_name),
                    comment=f"Derived: {param_name}",
                )
                logger.info(f"Added derived parameter: {param_name}")
            except Exception as exc:
                logger.warning(
                    f"Could not add derived parameter {param_name}: {exc}"
                )

        return samples

    except Exception as exc:
        logger.exception(f"Error adding derived parameters: {exc}")
        return samples


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------

def compute_percentiles_getdist(samples, param_name,
                                 percentiles=(16, 50, 84)):
    """
    Compute weight-aware percentiles for a single parameter.

    Parameters
    ----------
    samples : MCSamples
    param_name : str
    percentiles : tuple

    Returns
    -------
    dict or None
        ``{'p16': val, 'p50': val, 'p84': val}``
    """
    if samples is None:
        return None
    try:
        param_values = getattr(samples.getParams(), param_name, None)
        if param_values is None:
            return None
        weights      = np.asarray(samples.weights, dtype=float)
        sorted_idx   = np.argsort(param_values)
        sorted_vals  = param_values[sorted_idx]
        sorted_w     = weights[sorted_idx]
        cumsum_norm  = np.cumsum(sorted_w) / sorted_w.sum()

        result = {}
        for p in percentiles:
            idx = min(
                np.searchsorted(cumsum_norm, p / 100.0, side="left"),
                len(sorted_vals) - 1,
            )
            result[f"p{p}"] = sorted_vals[idx]
        return result

    except Exception as exc:
        logger.warning(
            f"Could not compute percentiles for {param_name}: {exc}"
        )
        return None


def compute_all_percentiles(samples, param_names, percentiles=(16, 50, 84)):
    """
    Compute percentiles for a list of parameters.

    Returns
    -------
    dict
        ``{param_name: {'p16': …, 'p50': …, 'p84': …}}``
    """
    return {
        param: vals
        for param in param_names
        if (vals := compute_percentiles_getdist(samples, param, percentiles))
    }


# ---------------------------------------------------------------------------
# p-value helpers  (no experimental errors)
# ---------------------------------------------------------------------------

def compute_pvalue_from_percentiles(observed_value, p16, p50, p84, values):
    """
    Compute a one-sided empirical p-value from the ensemble.

    Experimental errors are not used.

    Parameters
    ----------
    observed_value : float
    p16, p50, p84 : float
        16th, 50th, 84th percentiles of the ensemble.
    values : array-like
        Full ensemble array (required for empirical CDF).

    Returns
    -------
    dict
        ``{'pvalue': p, 'n_sigma': n, 'interpretation': str,
           'tension_level': str}``
    """
    percentiles_dict = {"p16": p16, "p50": p50, "p84": p84}
    return compute_pvalue_unified(observed_value, percentiles_dict, values)


def compute_all_pvalues(samples, param_names, experimental_values):
    """
    Compute p-values for all parameters.

    Only the observed *value* from ``experimental_values`` is used;
    the associated error is ignored.

    Parameters
    ----------
    samples : MCSamples
    param_names : list
    experimental_values : dict
        ``{param: (value, error)}``  — only ``value`` is consumed.

    Returns
    -------
    dict
        ``{param: {'pvalue': p, 'n_sigma': n, …}}``
    """
    percentiles_dict = compute_all_percentiles(samples, param_names)
    results = {}

    for param in param_names:
        if param not in experimental_values or param not in percentiles_dict:
            continue

        # Unpack but discard the error
        exp_val = experimental_values[param][0]
        perc    = percentiles_dict[param]

        if not all(k in perc for k in ("p16", "p50", "p84")):
            continue

        values = getattr(samples.getParams(), param, None)
        results[param] = compute_pvalue_from_percentiles(
            exp_val,
            perc["p16"],
            perc["p50"],
            perc["p84"],
            values,
        )

    return results


# ---------------------------------------------------------------------------
# Multivariate tension
# ---------------------------------------------------------------------------

def compute_multivariate_tension(samples, param_names,
                                  experimental_means, experimental_cov):
    """
    Compute a multivariate chi-squared tension statistic.

    Parameters
    ----------
    samples : MCSamples
    param_names : list
    experimental_means : np.ndarray
        Observed values (no errors required here — covariance is optional
        and can be set to zero).
    experimental_cov : np.ndarray
        Experimental covariance matrix.  Pass ``np.zeros((n,n))`` if unknown.

    Returns
    -------
    dict or None
        ``{'chi2', 'dof', 'pvalue', 'n_sigma'}``
    """
    try:
        percentiles_dict = compute_all_percentiles(samples, param_names)
        available = [p for p in param_names if p in percentiles_dict]

        if not available:
            logger.warning("No parameters available for multivariate tension")
            return None

        mcmc_means  = np.array([percentiles_dict[p]["p50"] for p in available])
        param_arrs  = [getattr(samples.getParams(), p) for p in available]
        param_mat   = np.column_stack(param_arrs)
        mcmc_cov    = np.cov(param_mat.T, aweights=samples.weights)

        n = len(available)
        exp_cov_sub = experimental_cov[:n, :n]
        cov_total   = exp_cov_sub + mcmc_cov
        delta       = mcmc_means - experimental_means[:n]
        chi2        = float(delta @ np.linalg.inv(cov_total) @ delta)
        pvalue      = float(stats.chi2.sf(chi2, n))
        n_sigma     = float(stats.norm.isf(max(pvalue, 1e-16) / 2))

        return {"chi2": chi2, "dof": n, "pvalue": pvalue, "n_sigma": n_sigma}

    except Exception as exc:
        logger.exception(f"Error computing multivariate tension: {exc}")
        return None


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_statistics_table(samples, param_names, experimental_values,
                               mode="mcmc", include_bestfit=False, ncol=1):
    """
    Generate a LaTeX ``tabular`` string for the given statistics.

    Changes from the previous version
    ----------------------------------
    - The experimental column shows only the observed *value* (no ± error).
    - The p-value column uses sigma notation: ``p  (nσ)``.

    Parameters
    ----------
    samples : MCSamples
    param_names : list
    experimental_values : dict
        ``{param: (value, error)}`` — only the value is displayed/used.
    mode : str
        ``'mcmc'`` or ``'bestfit'`` (cosmetic only).
    include_bestfit : bool
        Unused; kept for API compatibility.
    ncol : int
        Unused; kept for API compatibility.

    Returns
    -------
    str or None
        LaTeX tabular string.
    """
    if samples is None:
        return None

    try:
        percentiles_dict = compute_all_percentiles(samples, param_names)
        pvalues_dict     = compute_all_pvalues(samples, param_names,
                                               experimental_values)

        rows = []
        rows.append(r"\begin{tabular}{lcccc}")
        rows.append(r"\toprule")
        rows.append(
            r"Statistic & Experimental & Median & 68\% CI & $p$-value \\"
        )
        rows.append(r"\midrule")

        for param in param_names:
            if param not in percentiles_dict or param not in pvalues_dict:
                continue

            param_label = _format_label_for_getdist(param)

            # Observed value only — no error
            exp_val = experimental_values[param][0]
            exp_str = rf"${exp_val:.4f}$"

            # Theoretical percentiles
            perc   = percentiles_dict[param]
            p16, p50, p84 = perc["p16"], perc["p50"], perc["p84"]
            med_str = rf"${p50:.4f}$"
            ci_str  = rf"$[{p16:.4f},\ {p84:.4f}]$"

            # p-value with sigma notation
            pval   = pvalues_dict[param]["pvalue"]
            pv_str = sigma_label(pval)

            rows.append(
                rf"{param_label} & {exp_str} & {med_str} & {ci_str} & {pv_str} \\"
            )

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        return "\n".join(rows)

    except Exception as exc:
        logger.exception(f"Error generating table: {exc}")
        return None


# ---------------------------------------------------------------------------
# Full GetDist ResultTable (unchanged functionality, kept for compatibility)
# ---------------------------------------------------------------------------

def generate_full_results_table(samples, param_names, titles=None, ncol=1):
    """
    Generate a GetDist ``ResultTable`` with marginalised statistics.

    Returns
    -------
    ResultTable or None
    """
    if not isinstance(samples, list):
        samples = [samples]
    try:
        return ResultTable(
            ncol=ncol,
            results=samples,
            titles=titles,
            limit=1,
            paramList=param_names,
            formatter=NoLineTableFormatter(),
        )
    except Exception as exc:
        logger.exception(f"Error creating ResultTable: {exc}")
        return None


def save_table_to_file(table_obj, output_path, document=True):
    """
    Save a LaTeX table (string or ResultTable) to *output_path*.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if isinstance(table_obj, str):
        with open(output_path, "w") as f:
            f.write(table_obj)
    else:
        table_obj.write(output_path, document=document)
    logger.info(f"Table saved to: {output_path}")