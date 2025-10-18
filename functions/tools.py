"""
This module provides a collection of utility functions used across the CMB analysis codebase,
including mathematical helpers and performance measurement tools.
"""

# Math libraries
import numpy as np
from math import factorial
from decimal import getcontext, Decimal
from scipy.integrate import simpson

# System libraries
import time
import functools

def timeit(func):
    """
    A decorator that measures and prints the execution time of a function.

    Args:
        func (callable): The function to be timed.

    Returns:
        callable: The wrapped function with timing capabilities.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.2f} seconds")
        return result
    return wrapper

def legendre(lmax, x):
    """
    Computes the Legendre polynomial P_l(x) up to a given degree lmax.
    This implementation uses the stable three-term recurrence relation.

    Args:
        lmax (int): The degree of the Legendre polynomial to compute.
        x (float): The point at which to evaluate the polynomial.

    Returns:
        float: The value of P_{lmax}(x).
    """
    if lmax == 0:
        return 1.0
    elif lmax == 1:
        return x
    elif lmax == -1:
        # Define P_{-1}(x) = 0 for convenience in some formulas.
        return 0.0
    else:
        # Initialize the first two polynomials for the recurrence.
        p0 = 1.0  # P_0(x)
        p1 = x    # P_1(x)
        # Iterate from l=2 up to lmax to compute the desired polynomial.
        for l in range(2, lmax + 1):
            p_next = ((2 * l - 1) / l) * x * p1 - ((l - 1) / l) * p0
            p0, p1 = p1, p_next
        return p1

# Vectorize the legendre function to allow it to operate efficiently on numpy arrays.
P = np.vectorize(legendre)

def A_r(r):
    """
    Computes the coefficient A_r, which appears in the analytical integration of Legendre polynomials.
    A_r = (2r-1)!! / r! = (2r)! / (2^r * (r!)^2)
    This implementation uses Decimal for high precision to avoid overflow with large r.

    Args:
        r (int): The index of the coefficient.

    Returns:
        Decimal: The high-precision value of A_r.
    """
    # Numerator is the double factorial (2r-1)!!
    numerator = Decimal(1)
    for i in range(1, r + 1):
        numerator *= Decimal(2 * i - 1)
    
    # Denominator is r!
    denominator = factorial(r)
    
    return numerator / Decimal(denominator)
