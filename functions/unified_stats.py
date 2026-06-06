"""
Unified statistics module for percentile and p-value computation.

Provides consistent percentile and p-value calculation across all modules,
using GetDist methods when available, falling back to NumPy.

Design principles
-----------------
- Experimental *errors* are never used in any computation. The observed value
  is compared directly to the empirical distribution of the theoretical ensemble.
- p-values are one-sided empirical CDFs (fraction of realisations <= observed).
- Sigma notation is produced by the Gaussian probit: n_sigma = Phi^{-1}(1 - p).

This module has no matplotlib dependency. All plotting style lives in
``plot_style.py``.
"""

import numpy as np
import logging
from scipy.stats import norm as _norm

logger = logging.getLogger(__name__)

try:
    from getdist import mcsamples as _gd_mcsamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False


# ---------------------------------------------------------------------------
# Sigma / p-value helpers
# ---------------------------------------------------------------------------

def pvalue_to_sigma(p: float) -> float:
    """
    Convert a one-sided p-value to an equivalent number of Gaussian sigma.

        n_sigma = Phi^{-1}(1 - p)

    Parameters
    ----------
    p : float
        p-value in (0, 1).

    Returns
    -------
    float
    """
    p = float(np.clip(p, 1e-16, 1.0 - 1e-16))
    return float(_norm.isf(p))


def sigma_label(p: float) -> str:
    """
    Return a compact LaTeX string such as ``$0.073\\ (1.47\\sigma)$``.

    Intended for use in LaTeX table cells.

    Parameters
    ----------
    p : float

    Returns
    -------
    str
    """
    n = pvalue_to_sigma(p)
    return rf"${p:.3f}\ ({n:.2f}\sigma)$"


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------

def compute_percentiles_weighted(data, weights=None, percentiles=(16, 50, 84)):
    """
    Compute weighted percentiles (GetDist-style when weights are available).

    Falls back to unweighted NumPy percentiles if *weights* is None or all
    weights are equal.

    Parameters
    ----------
    data : array-like
    weights : array-like, optional
    percentiles : tuple
        Default ``(16, 50, 84)`` for a 68 % credible interval.

    Returns
    -------
    dict
        ``{'p16': val, 'p50': val, 'p84': val}``
    """
    data = np.asarray(data, dtype=float)

    if weights is None or len(np.unique(weights)) == 1:
        return {f"p{int(p)}": np.percentile(data, p) for p in percentiles}

    weights        = np.asarray(weights, dtype=float)
    sorted_idx     = np.argsort(data)
    sorted_data    = data[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum_norm    = np.cumsum(sorted_weights) / sorted_weights.sum()

    result = {}
    for p in percentiles:
        idx = min(
            np.searchsorted(cumsum_norm, p / 100.0, side="left"),
            len(sorted_data) - 1,
        )
        result[f"p{int(p)}"] = sorted_data[idx]
    return result


def compute_percentiles_from_mcsamples(samples, param_name,
                                        percentiles=(16, 50, 84)):
    """
    Compute percentiles using a GetDist ``MCSamples`` object.

    Properly accounts for MCMC sample weights.

    Parameters
    ----------
    samples : getdist.mcsamples.MCSamples
    param_name : str
    percentiles : tuple

    Returns
    -------
    dict or None
    """
    if not HAS_GETDIST or samples is None:
        return None
    try:
        param_values = getattr(samples.getParams(), param_name, None)
        if param_values is None:
            return None
        return compute_percentiles_weighted(
            param_values, samples.weights, percentiles
        )
    except Exception as exc:
        logger.debug(
            f"Could not use GetDist percentiles for {param_name}: {exc}"
        )
        return None


def compute_percentiles_unified(data, samples=None, param_name=None,
                                 percentiles=(16, 50, 84)):
    """
    Unified percentile computation.

    Priority: GetDist (weighted) → weighted NumPy → plain NumPy.

    Parameters
    ----------
    data : array-like
    samples : MCSamples, optional
    param_name : str, optional
    percentiles : tuple

    Returns
    -------
    dict
    """
    if samples is not None and param_name is not None:
        result = compute_percentiles_from_mcsamples(
            samples, param_name, percentiles
        )
        if result is not None:
            return result
    return compute_percentiles_weighted(data, weights=None,
                                        percentiles=percentiles)


# ---------------------------------------------------------------------------
# p-value computation  (no experimental errors involved)
# ---------------------------------------------------------------------------

def compute_pvalue_unified(observed_value, percentiles_dict, data=None):
    """
    Compute a one-sided empirical p-value.

    The p-value is the fraction of theoretical realisations less than or
    equal to the observed value::

        p = F(observed) = mean(data <= observed)

    Experimental *errors* are deliberately not used.

    Parameters
    ----------
    observed_value : float
    percentiles_dict : dict
        ``{'p16': val, 'p50': val, 'p84': val}`` — used only to guard
        against degenerate (zero-variance) distributions.
    data : array-like, optional
        Full ensemble.  Required for a meaningful p-value; returns p = 1
        with a warning if absent.

    Returns
    -------
    dict
        ``{'pvalue': float, 'n_sigma': float,
           'interpretation': str, 'tension_level': str}``
    """
    try:
        p16 = percentiles_dict["p16"]
        p84 = percentiles_dict["p84"]
    except KeyError:
        logger.warning("Missing percentile keys in percentiles_dict.")
        return _tension_dict(1.0)

    if data is None:
        logger.warning(
            "compute_pvalue_unified: no data array provided; returning p = 1."
        )
        return _tension_dict(1.0)

    data = np.asarray(data, dtype=float)

    if np.isclose(p84, p16):
        logger.warning("Zero-variance distribution; returning p = 1.")
        return _tension_dict(1.0)

    n      = len(data)
    floor  = 1.0 / n
    F      = float(np.mean(data <= observed_value))
    pvalue = max(F, floor)
    return _tension_dict(pvalue)


def _tension_dict(pvalue: float) -> dict:
    """Build the standard return dict from a p-value."""
    pvalue  = float(np.clip(pvalue, 1e-16, 1.0))
    n_sigma = pvalue_to_sigma(pvalue)

    if pvalue > 0.05:
        interpretation, tension_level = "Consistent",              "< 2σ"
    elif pvalue > 0.01:
        interpretation, tension_level = "Marginally inconsistent", "2–3σ"
    elif pvalue > 0.001:
        interpretation, tension_level = "Inconsistent",            "3–4σ"
    else:
        interpretation, tension_level = "Highly inconsistent",     "> 4σ"

    return {
        "pvalue":         pvalue,
        "n_sigma":        n_sigma,
        "interpretation": interpretation,
        "tension_level":  tension_level,
    }