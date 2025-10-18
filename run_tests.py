import numpy as np
import sys
import os

# This is needed so we can import from the 'functions' directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

# Import the timeit decorator
from test import timeit

# Import the functions we want to test
from correlation_function import correlation_func, correlation_func_err, correlation_func_err2
from s12 import Tmn, S12, S12_vec, S12_err, S12_err2
from xiv import xivar, xivar2, xivar_err, xivar_err2
from tools import legendre
from maps import map_rot_refl, estimate_coef

# You can add functions that you want to test here
# Example:
# from my_new_function_file import my_new_function
# timed_my_new_function = timeit(my_new_function)


def run_all_tests():
    """
    This function runs a series of tests on the functions in the 'functions' directory.
    You can add more tests here, or modify the existing ones by editing the code below.
    """
    print("--- Starting function tests ---")

    # --- Setup dummy data ---
    # You can adjust these values to test with different data
    D_ell_size = 200
    D_ell = np.random.rand(D_ell_size)
    D_ell_err = np.random.rand(D_ell_size) * 0.1
    xvals = np.linspace(-0.99, 0.99, 180)
    a, b = -0.5, 0.5

    # --- Tests for correlation_function.py ---
    print("\n--- Testing correlation_function.py ---")
    timeit(correlation_func)(D_ell, xvals)
    timeit(correlation_func_err)(D_ell_err, xvals)
    timeit(correlation_func_err2)(D_ell_err, xvals)

    # --- Tests for xiv.py ---
    print("\n--- Testing xiv.py ---")
    timeit(xivar)(D_ell, a, b)
    timeit(xivar2)(D_ell, a, b)
    timeit(xivar_err)(D_ell_err, a, b)
    timeit(xivar_err2)(D_ell_err, a, b)

    # --- Tests for s12.py ---
    print("\n--- Testing s12.py ---")
    # Tmn is slow, so we use a small size for the test
    tmn_size = 10
    print(f"Running Tmn with size {tmn_size}x{tmn_size}...")
    timeit(Tmn)(l=tmn_size, l1=90, l2=60)
    
    try:
        M = np.load('files/matrix/Tmn_90_60.npy')
        
        # We need D_ell to have the same size as M for S12 functions
        D_ell_s12 = np.random.rand(tmn_size)
        D_ell_err_s12 = np.random.rand(tmn_size) * 0.1

        print("\nRunning S12 functions...")
        timeit(S12)(D_ell_s12, M)
        timeit(S12_vec)(D_ell_s12, M)
        timeit(S12_err)(D_ell_s12, D_ell_err_s12, M)
        timeit(S12_err2)(D_ell_s12, D_ell_err_s12, M)
    except FileNotFoundError:
        print("Skipping S12 tests because Tmn matrix file (files/matrix/Tmn_90_60.npy) was not found.")
    except Exception as e:
        print(f"An error occurred during S12 tests: {e}")


    # --- Tests for tools.py ---
    print("\n--- Testing tools.py ---")
    timeit(legendre)(5, 0.5)

    # --- Tests for maps.py ---
    # These functions require healpy, which might not be installed.
    print("\n--- Testing maps.py ---")
    try:
        import healpy as hp
        # Create a dummy map
        nside = 16
        dummy_map = np.arange(hp.nside2npix(nside))
        timeit(map_rot_refl)(dummy_map)
        timeit(estimate_coef)(dummy_map, dummy_map[::-1])
    except ImportError:
        print("Skipping maps.py tests because healpy is not installed.")
        print("You can install it with: pip install healpy")
    except Exception as e:
        print(f"An error occurred during maps.py tests: {e}")


    print("\n--- All tests finished ---")
    print("\nTo test your own functions, you can:")
    print("1. Import your function at the top of this file.")
    print("2. Add a call to it inside the run_all_tests() function, wrapped with timeit().")
    print("   Example: timeit(my_function)(arg1, arg2)")


if __name__ == "__main__":
    run_all_tests()
