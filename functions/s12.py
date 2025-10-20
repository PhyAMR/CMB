"""
This module provides functions for calculating the S12 statistic, which is defined as the integral
of the squared two-point correlation function over a specific angular range. It includes methods
for both analytical (via the Tmn matrix) and numerical computation.
"""

import numpy as np
from decimal import Decimal, getcontext
from math import factorial
from .tools import legendre, A_r
from scipy.integrate import simpson
from .correlation_function import correlation_func

def Tmn(l, l1, l2, a=-1, b=1/2):
    """
    Calculates and saves the Tmn matrix used in the analytical computation of the S12 statistic.
    The Tmn matrix represents the integral of the product of three Legendre polynomials.

    Args:
        l (int): The maximum multipole to compute the matrix for (size of the matrix).
        l1 (int): The lower angular bound in degrees for the filename.
        l2 (int): The upper angular bound in degrees for the filename.
        a (float): The lower bound of the integral in cos(theta).
        b (float): The upper bound of the integral in cos(theta).
    """
    # Set high precision for Decimal calculations to handle large numbers.
    getcontext().prec = 1000
    
    matrix = np.zeros((l, l), dtype=float)
    for i in range(l):
        n = i + 2
        for j in range(l):
            m = j + 2
            integral_val = Decimal(0)
            for r in range(min(m, n) + 1):
                # This formula arises from the analytical integration of P_n(x) * P_m(x).
                term = A_r(r) * A_r(m - r) * A_r(n - r) / A_r(m + n - r) / (2 * m + 2 * n - 2 * r + 1)
                term *= Decimal(legendre(m + n - 2 * r + 1, b) - legendre(m + n - 2 * r - 1, b)) - \
                        Decimal(legendre(m + n - 2 * r + 1, a) - legendre(m + n - 2 * r - 1, a))
                integral_val += term
            matrix[i, j] = np.float64(integral_val)
            
    np.save(f"Tmn_{l1}_{l2}.npy", matrix)

def S12(D_ell, M):
    """
    Calculates the S12 statistic using the pre-computed Tmn matrix.

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell, starting from l=2.
        M (np.ndarray): The pre-computed Tmn matrix.

    Returns:
        float: The value of the S12 statistic.
    """
    getcontext().prec = 1000
    s = Decimal(0)
    for i, xn in enumerate(D_ell):
        n = i + 2
        fac1 = (((2 * n + 1) * xn) / (2 * n * (n + 1)))
        for j, xm in enumerate(D_ell):
            m = j + 2
            fac2 = (((2 * m + 1) * xm) / (2 * m * (m + 1)))
            integral = M[i, j]
            s += Decimal(fac1) * Decimal(fac2) * Decimal(integral)
    return float(s)
    
def S12_vec(D_ell, M):
    """
    Calculates the S12 statistic using a vectorized approach for efficiency.

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell.
        M (np.ndarray): The pre-computed Tmn matrix.

    Returns:
        float: The value of the S12 statistic.
    """
    getcontext().prec = 1000
    D_ell = np.array(D_ell, dtype=float)
    M = np.array(M, dtype=float)
    n = np.arange(2, len(D_ell) + 2)
    f = ((2 * n + 1) * D_ell) / (2 * n * (n + 1))
    return float(f @ M @ f)

def S12_err(D_ell, D_ell_err, M):
    """
    Calculates the error in the S12 statistic by propagating the errors from D_ell.

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell.
        D_ell_err (np.ndarray): The errors in the D_ell values.
        M (np.ndarray): The pre-computed Tmn matrix.

    Returns:
        float: The propagated error in the S12 statistic.
    """
    getcontext().prec = 1000
    s = Decimal(0)
    for i, xn_err in enumerate(D_ell_err):
        n = i + 2
        fac1 = Decimal((2 * n + 1)) / Decimal(2 * n * (n + 1))
        for j, xm in enumerate(D_ell):
            m = j + 2
            fac2 = Decimal((2 * m + 1)) / Decimal(2 * m * (m + 1))
            Amn = fac1 * fac2
            integral = M[i, j]
            # Error propagation assuming uncorrelated errors.
            s += (Amn**2) * (Decimal(integral)**2) * (Decimal(D_ell[i]**2 * xm**2) + Decimal(D_ell[j]**2 * xn_err**2))
    return float(s)**0.5

def S12_err2(D_ell, D_ell_err, M):
    """
    An alternative method for calculating the error in the S12 statistic.

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell.
        D_ell_err (np.ndarray): The errors in the D_ell values.
        M (np.ndarray): The pre-computed Tmn matrix.

    Returns:
        float: The propagated error in the S12 statistic.
    """
    getcontext().prec = 1000
    s1 = Decimal(0)
    s2 = Decimal(0)
    for i, xn_err in enumerate(D_ell_err):
        n = i + 2
        fac1 = Decimal((2 * n + 1)) / Decimal(2 * n * (n + 1))
        for j, xm in enumerate(D_ell):
            m = j + 2
            fac2 = Decimal((2 * m + 1)) / Decimal(2 * m * (m + 1))
            Amn = fac1 * fac2
            integral = M[i, j]
            s1 += Amn * Decimal(integral) * Decimal(D_ell[i] * xm)
            s2 += Amn * Decimal(integral) * Decimal(D_ell[j] * xn_err)
    return float(s1**2 + s2**2)**0.5

def s12_numerical(D_ell, a, b, n_points=2000):
    """
    Computes the S12 statistic numerically by integrating the squared correlation function.
    Formula: S12 = integral from a to b of [C(theta)]^2 d(cos(theta))

    Args:
        D_ell (np.ndarray): The angular power spectrum, D_ell.
        a (float): The lower bound of the integral in cos(theta).
        b (float): The upper bound of the integral in cos(theta).
        n_points (int): The number of points to use for the numerical integration.

    Returns:
        float: The numerically computed S12 statistic.
    """
    x = np.linspace(a, b, n_points)
    cor = correlation_func(D_ell, x)
    cor_sq = cor**2
    integral = simpson(cor_sq, x)
    return integral

if __name__ == '__main__':
    import time
    import sys
    import os
    # This allows running the script directly to test it
    # It adds the project root to the python path
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from functions.data import Data_loader

    print("--- S12 Performance and Accuracy Test ---")
    # Load data using the loader
    data_loader = Data_loader()
    D_ell = data_loader.D_ell

    # Define the integration interval in cos(theta)
    # Corresponds to theta from 60 to 90 degrees
    a = -1  # cos(60 deg)
    b = 0.5  # cos(90 deg)

    # Load the pre-computed Tmn matrix for the analytical calculation
    # The path is relative to the project root
    matrix_path = "files/matrix/Tmn__180__60.npy"
    try:
        M = np.load(matrix_path)
    except FileNotFoundError:
        print(f"Error: Matrix file not found at {matrix_path}")
        print("Please ensure you are running this script from the project's root directory.")
        sys.exit(1)

    # --- Test Analytical S12 ---
    start_time_an = time.time()
    s12_analytical = S12(D_ell, M)
    end_time_an = time.time()
    analytical_duration = end_time_an - start_time_an

    # --- Test Numerical S12 ---
    start_time_num = time.time()
    s12_numerical_val = s12_numerical(D_ell, a, b, n_points=5000)
    end_time_num = time.time()
    numerical_duration = end_time_num - start_time_num

    # --- Report Results ---
    print(f"Interval [cos(theta)]: [{a}, {b}] (Theta: [60, 90] degrees)")
    print("\nAnalytical Calculation:")
    print(f"  Result: {s12_analytical}")
    print(f"  Execution Time: {analytical_duration:.6f} seconds")

    print("\nNumerical Calculation:")
    print(f"  Result: {s12_numerical_val}")
    print(f"  Execution Time: {numerical_duration:.6f} seconds")

    print("\nComparison:")
    difference = abs(s12_analytical - s12_numerical_val)
    relative_difference = difference / abs(s12_analytical) if s12_analytical != 0 else 0
    print(f"  Absolute Difference: {difference}")
    print(f"  Relative Difference: {relative_difference:.4%}")
    print("--- End of Test ---")
