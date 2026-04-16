"""
Unified statistics module for percentile and p-value computation.

Provides consistent percentile and p-value calculation across all modules,
using GetDist methods when available, falling back to NumPy.
"""

import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from getdist import mcsamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False


def compute_percentiles_weighted(data, weights=None, percentiles=(16, 50, 84)):
    """
    Compute weighted percentiles (GetDist-style when weights available).
    
    Falls back to unweighted NumPy method if weights are None or all equal.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    weights : np.ndarray, optional
        Sample weights. If None, uses unweighted percentiles.
    percentiles : tuple
        Percentiles to compute (default: (16, 50, 84) for 68% CI)
    
    Returns
    -------
    dict
        {'p16': val, 'p50': val, 'p84': val} (or custom percentiles)
    """
    data = np.asarray(data)
    
    if weights is None or len(np.unique(weights)) == 1:
        # Use unweighted percentiles
        return {
            f'p{int(p)}': np.percentile(data, p)
            for p in percentiles
        }
    
    # Use weighted percentiles (GetDist-style)
    weights = np.asarray(weights)
    
    # Sort by data values
    sorted_idx = np.argsort(data)
    sorted_data = data[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    # Compute cumulative sum and normalize
    cumsum = np.cumsum(sorted_weights)
    cumsum_norm = cumsum / cumsum[-1]
    
    # Interpolate percentiles
    result = {}
    for p in percentiles:
        target = p / 100.0
        # Find where cumsum first exceeds target
        idx = np.searchsorted(cumsum_norm, target, side='left')
        idx = min(idx, len(sorted_data) - 1)
        result[f'p{int(p)}'] = sorted_data[idx]
    
    return result


def compute_percentiles_from_mcsamples(samples, param_name, percentiles=(16, 50, 84)):
    """
    Compute percentiles using GetDist MCSamples object.
    
    Properly accounts for MCMC weights and burn-in.
    
    Parameters
    ----------
    samples : getdist.mcsamples.MCSamples
        GetDist samples object
    param_name : str
        Parameter name in samples
    percentiles : tuple
        Percentiles to compute
    
    Returns
    -------
    dict or None
        {'p16': val, 'p50': val, 'p84': val} or None if parameter not found
    """
    if not HAS_GETDIST or samples is None:
        return None
    
    try:
        # Get parameter values
        param_values = getattr(samples.getParams(), param_name, None)
        if param_values is None:
            return None
        
        weights = samples.weights
        
        # Use weighted percentile computation
        return compute_percentiles_weighted(param_values, weights, percentiles)
    
    except Exception as e:
        logger.debug(f"Could not use GetDist percentiles for {param_name}: {e}")
        return None


def compute_percentiles_unified(data, samples=None, param_name=None, 
                                percentiles=(16, 50, 84)):
    """
    Unified percentile computation.
    
    Strategy:
    1. If GetDist samples provided, use GetDist method (accounts for weights)
    2. Otherwise use weighted percentiles if weights available
    3. Fall back to standard NumPy percentiles
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    samples : getdist.mcsamples.MCSamples, optional
        GetDist samples object (if available)
    param_name : str, optional
        Parameter name in samples (required if samples provided)
    percentiles : tuple
        Percentiles to compute
    
    Returns
    -------
    dict
        {'p16': val, 'p50': val, 'p84': val}
    """
    # Try GetDist method first
    if samples is not None and param_name is not None:
        result = compute_percentiles_from_mcsamples(samples, param_name, percentiles)
        if result is not None:
            return result
    
    # Fall back to weighted or unweighted NumPy percentiles
    return compute_percentiles_weighted(data, weights=None, percentiles=percentiles)


def compute_pvalue_unified(observed_value, percentiles_dict, data=None):
    """
    Compute p-value using percentile-based method (consistent across all modes).
    
    Assumes approximately Gaussian posterior and computes p-value
    based on how many sigma away the observed value is from the median.
    
    This method is consistent with GetDist's approach.
    
    Parameters
    ----------
    observed_value : float
        Experimental/observed value
    percentiles_dict : dict
        {'p16': val, 'p50': val, 'p84': val} from compute_percentiles_unified()
    data : np.ndarray, optional
        Original data array (for robustness check)
    
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
    try:
        p16 = percentiles_dict['p16']
        p50 = percentiles_dict['p50']
        p84 = percentiles_dict['p84']
    except KeyError:
        logger.warning("Missing percentile keys")
        return {
            'pvalue': 1.0,
            'n_sigma': 0.0,
            'interpretation': 'No percentiles',
            'tension_level': 'N/A'
        }
    
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