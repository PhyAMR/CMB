"""
This module provides functions to calculate the two-point correlation function
and its associated error from the angular power spectrum of the Cosmic Microwave Background (CMB).
"""

import numpy as np 
from .tools import legendre, timeit

# Vectorize the Legendre polynomial function for efficient computation over arrays.
P = np.vectorize(legendre)

@timeit
def correlation_func(D_ell, xvals):
    """
    Calculates the two-point correlation function C(theta) from the angular power spectrum D_ell.

    The correlation function is computed as a sum over multipoles l:
    C(theta) = sum_{l=2}^{lmax} (2l + 1) / (4 * pi) * C_l * P_l(cos(theta))
    where D_ell = l * (l + 1) / (2 * pi) * C_l.
    This simplifies to:
    C(theta) = sum_{l=2}^{lmax} (2l + 1) / (2 * l * (l + 1)) * D_ell * P_l(cos(theta))

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell, starting from l=2.
        xvals (np.ndarray): An array of cos(theta) values at which to compute the correlation function.

    Returns:
        np.ndarray: The two-point correlation function C(theta) for each value in xvals.
    """
    # Multipole moments, starting from l=2.
    l = np.arange(len(D_ell)) + 2
    
    # Pre-factor for the summation.
    fac2 = (2 * l + 1) / (2 * l * (l + 1)) * D_ell
    
    # Compute the correlation function by summing over l.
    # P(l[:, None], xvals) broadcasts the Legendre polynomial calculation over all l and xvals.
    cor = np.sum(fac2[:, None] * P(l[:, None], xvals), axis=0)
    
    return cor

def correlation_func_err(error, xvals):
    """
    Calculates the error in the correlation function by propagating the errors from D_ell,
    assuming they are uncorrelated and summing them in quadrature.

    Args:
        error (np.ndarray): The error in each D_ell component.
        xvals (np.ndarray): An array of cos(theta) values.

    Returns:
        np.ndarray: The propagated error in the correlation function.
    """
    cor_err = 0

    # Calculate the pre-factor for the error propagation.
    fac_err = [(2 * (l + 2) + 1) / (2 * (l + 2) * ((l + 2) + 1)) * c for l, c in enumerate(error)]
    
    # Sum the squared errors in quadrature.
    for l, f in enumerate(fac_err):
        cor_err += (f * P(l + 2, xvals))**2
        
    return cor_err**0.5

def correlation_func_err2(error, xvals):
    """
    Calculates the error in the correlation function by propagating the errors from D_ell,
    assuming they are uncorrelated and summing them linearly before squaring.
    This method might be less statistically rigorous than summing in quadrature.

    Args:
        error (np.ndarray): The error in each D_ell component.
        xvals (np.ndarray): An array of cos(theta) values.

    Returns:
        np.ndarray: The propagated error in the correlation function.
    """
    cor_err = 0

    # Calculate the pre-factor for the error propagation.
    fac_err = [(2 * (l + 2) + 1) / (2 * (l + 2) * ((l + 2) + 1)) * c for l, c in enumerate(error)]
    
    # Sum the errors linearly.
    for l, f in enumerate(fac_err):
        cor_err += (f * P(l + 2, xvals))
        
    return (cor_err**2)**0.5
