import unittest
import time
import numpy as np
import pandas as pd
from .simulation import MC_calculations, MC_results
from .data import Data_loader

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run.")
        return result
    return wrapper

class TestMCSimulation(unittest.TestCase):

    def setUp(self):
        """Set up the test environment by loading data."""
        self.data_loader = Data_loader()
        self.lmax = self.data_loader.lmax
        self.xvals = self.data_loader.xvals
        self.intervals = [(-0.9999999999999999, 0.5), (0.866, 0.9999999999999999),]
        # Prepare data for MC_calculations
        self.mc_calc_data = pd.Series([self.data_loader.D_ell])

    @timing_decorator
    def test_mc_calculations_performance(self):
        """Benchmark the MC_calculations function."""
        # Time each part of the function
        start_time = time.time()
        TTCl = self.mc_calc_data.iloc[0][:self.lmax]
        print(f"TTCl extraction took {time.time() - start_time:.4f} seconds.")

        start_time = time.time()
        from .correlation_function import correlation_func
        correlation_func(TTCl, self.xvals)
        print(f"correlation_func took {time.time() - start_time:.4f} seconds.")

        for a, b in self.intervals:
            start_time = time.time()
            # This path needs to be adjusted if your test running directory is different
            matrix_path = f"files/matrix/Tmn__{round(np.arccos(a) * 180 / np.pi)}__{round(np.arccos(b) * 180 / np.pi)}.npy"
            try:
                M = np.load(matrix_path)
                print(f"Matrix loading for interval ({a}, {b}) took {time.time() - start_time:.4f} seconds.")

                start_time = time.time()
                from .s12 import S12
                S12(TTCl, M)
                print(f"S12 calculation for interval ({a}, {b}) took {time.time() - start_time:.4f} seconds.")
            except FileNotFoundError:
                print(f"Matrix file not found at {matrix_path}, skipping S12 timing.")


            start_time = time.time()
            from .xiv import xivar
            xivar(TTCl, a, b)
            print(f"xivar calculation for interval ({a}, {b}) took {time.time() - start_time:.4f} seconds.")

    @timing_decorator
    def test_mc_results_performance(self):
        """Benchmark the MC_results function."""
        # Prepare data for MC_results
        data_for_mc_results = pd.DataFrame({
            'D_ell': self.data_loader.D_ell,
            'Error': self.data_loader.error
        })
        n = 100

        # Time the main parts of the function
        start_time = time.time()
        data_for_mc_results['dist_per_cl'] = data_for_mc_results.apply(
            lambda row: np.random.normal(loc=row['D_ell'], scale=row['Error'], size=n), axis=1
        )
        print(f"dist_per_cl generation took {time.time() - start_time:.4f} seconds.")

        start_time = time.time()
        distributions = data_for_mc_results['dist_per_cl'].to_list()
        trasp = list(map(list, zip(*distributions)))
        pd.DataFrame({'valores': [np.array(row) for row in trasp]})
        print(f"DataFrame creation took {time.time() - start_time:.4f} seconds.")

        start_time = time.time()
        # Note: This will write a file 'Simulation_100.pkl'
        MC_results(self.intervals, data_for_mc_results, self.xvals, n)
        print(f"MC_results function call took {time.time() - start_time:.4f} seconds.")

if __name__ == '__main__':
    unittest.main()
