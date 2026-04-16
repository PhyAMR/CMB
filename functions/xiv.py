"""This module provides functions for calculating the xivar statistic, which is the integrated
average of the two-point correlation function over a specified angular range. It includes
analytical, numerical, and error propagation methods.
"""

import numpy as np
from scipy.integrate import simpson
from math import factorial
from .tools import legendre, timeit
from decimal import getcontext, Decimal
from .correlation_function import correlation_func
import logging
logger = logging.getLogger(__name__)

# Vectorize the Legendre polynomial function for efficient computation.
P = np.vectorize(legendre)

@timeit
def xivar(D_ell, a, b):
    """
    Calculates the xivar statistic analytically from the angular power spectrum D_ell.
    Formula: xivar = [1 / (b - a)] * sum_{l=2}^{lmax} [D_ell / (2*l*(l+1))] * integral_{a}^{b} P_l(x) dx
    The integral of P_l(x) is evaluated using the relation: integral(P_l(x)dx) = [P_{l+1}(x) - P_{l-1}(x)] / (2l+1).

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell, starting from l=2.
        a (float): The lower bound of the integral in cos(theta).
        b (float): The upper bound of the integral in cos(theta).

    Returns:
        float: The value of the xivar statistic.
    """
    s = 0
    for i, d in enumerate(D_ell):
        l = i + 2
        fac = d / (2 * l * (l + 1))
        # Analytical integral of the Legendre polynomial P_l(x) from a to b.
        term = Decimal(legendre(l + 1, b) - legendre(l - 1, b)) - \
               Decimal(legendre(l + 1, a) - legendre(l - 1, a))
        s += fac * float(term)
    return s / (b - a)

@timeit
def xivar2(D_ell, a, b):
    """
    A vectorized version of the xivar calculation for improved performance.

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell.
        a (float): The lower bound of the integral.
        b (float): The upper bound of the integral.

    Returns:
        float: The value of the xivar statistic.
    """
    l = np.arange(len(D_ell)) + 2
    fac = D_ell / (2 * l * (l + 1))
    term = (P(l + 1, b) - P(l - 1, b) - P(l + 1, a) + P(l - 1, a))
    s = np.sum(fac * term)
    return s / (b - a)



def xivar_err(D_ell_err, a, b):
    """
    Calculates the error in the xivar statistic by propagating the errors from D_ell,
    summing the contributions in quadrature.

    Args:
        D_ell_err (np.ndarray): The errors in the D_ell values.
        a (float): The lower bound of the integral.
        b (float): The upper bound of the integral.

    Returns:
        float: The propagated error in the xivar statistic.
    """
    s = 0
    for i, d in enumerate(D_ell_err):
        l = i + 2
        fac = d / (2 * l * (l + 1))
        integral = Decimal(legendre(l + 1, b) - legendre(l - 1, b)) - \
                   Decimal(legendre(l + 1, a) - legendre(l - 1, a))
        s += (fac * float(integral) / (b - a))**2
    return s**0.5

def xivar_err2(D_ell_err, a, b):
    """
    An alternative method for calculating the error in xivar, where contributions are
    summed linearly before being squared.

    Args:
        D_ell_err (np.ndarray): The errors in the D_ell values.
        a (float): The lower bound of the integral.
        b (float): The upper bound of the integral.

    Returns:
        float: The propagated error in the xivar statistic.
    """
    s = 0
    for i, d in enumerate(D_ell_err):
        l = i + 2
        fac = d / (2 * l * (l + 1))
        integral = Decimal(legendre(l + 1, b) - legendre(l - 1, b)) - \
                   Decimal(legendre(l + 1, a) - legendre(l - 1, a))
        s += (fac * float(integral) / (b - a))
    return (s**2)**0.5

def xivar_numerical(D_ell, a, b, n_points=2000):
    """
    Computes the xivar statistic numerically from the power spectrum.
    This function first computes the correlation function and then integrates it.
    Formula: xivar = [1 / (b - a)] * integral_{a}^{b} C(theta) d(cos(theta))

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell.
        a (float): The lower bound of the integral in cos(theta).
        b (float): The upper bound of the integral in cos(theta).
        n_points (int): The number of points for the numerical integration grid.

    Returns:
        float: The numerically computed xivar statistic.
    """
    x = np.linspace(a, b, n_points)
    cor = correlation_func(D_ell, x)
    integral = simpson(cor, x)
    return integral / (b - a)

if __name__ == '__main__':
    import time
    import sys
    import os
    # This allows running the script directly to test it
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from functions.data import Data_loader

    logger.info("--- xivar Performance and Accuracy Test ---")
    # Load data using the loader
    data_loader = Data_loader()
    D_ell = data_loader.D_ell

    # Define the integration interval in cos(theta)
    # Corresponds to theta from 60 to 90 degrees
    a = 0.0  # cos(90 deg)
    b = 0.5  # cos(60 deg)

    # --- Test Analytical xivar ---
    start_time_an = time.time()
    xivar_analytical = xivar(D_ell, a, b)
    end_time_an = time.time()
    analytical_duration = end_time_an - start_time_an

    # --- Test Numerical xivar ---
    start_time_num = time.time()
    xivar_numerical_val = xiv_numerical(D_ell, a, b, n_points=5000)
    end_time_num = time.time()
    numerical_duration = end_time_num - start_time_num

    # --- Report Results ---
    logger.info(f"Interval [cos(theta)]: [{a}, {b}] (Theta: [60, 90] degrees)")
    logger.info("\nAnalytical Calculation:")
    logger.info(f"  Result: {xivar_analytical}")
    logger.info(f"  Execution Time: {analytical_duration:.6f} seconds")

    logger.info("\nNumerical Calculation:")
    logger.info(f"  Result: {xivar_numerical_val}")
    logger.info(f"  Execution Time: {numerical_duration:.6f} seconds")

    logger.info("\nComparison:")
    difference = abs(xivar_analytical - xivar_numerical_val)
    relative_difference = difference / abs(xivar_analytical) if xivar_analytical != 0 else 0
    logger.info(f"  Absolute Difference: {difference}")
    logger.info(f"  Relative Difference: {relative_difference:.4%}")
    logger.info("--- End of Test ---")
