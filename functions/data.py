"""
This module provides the Data_loader class, designed to load and process experimental CMB power spectrum data.
It facilitates the calculation of the two-point correlation function and other statistics (xivar, S12)
derived from the loaded data.
"""
import pandas as pd
import numpy as np
from .correlation_function import correlation_func, correlation_func_err2
from .xiv import xivar2, xivar_err2
from .s12 import S12, S12_err2
import time

def time_execution(func):
    """
    A decorator that measures and prints the execution time of a method.
    It is only active if the 'time_it' attribute of the class instance is True.
    """
    def wrapper(self, *args, **kwargs):
        if not self.time_it:
            return func(self, *args, **kwargs)
        
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        
        print(f"Execution time of {func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class Data_loader:
    """
    A class to load, process, and analyze CMB power spectrum data.
    
    Attributes:
        path (str): Path to the data file.
        lmax (int): Maximum multipole to consider.
        time_it (bool): Flag to enable timing of methods.
        df (pd.DataFrame): DataFrame holding the loaded and filtered data.
        ell (np.ndarray): Array of multipoles.
        D_ell (np.ndarray): Array of power spectrum values (D_ell).
        error (np.ndarray): Array of symmetrized errors for D_ell.
        xvals (np.ndarray): Array of cos(theta) values for correlation function calculations.
        theta (np.ndarray): Array of theta values in degrees.
    """
    def __init__(self, path='maps/COM_PowerSpect_CMB-TT-full_R3.01.txt', lmax=200, n_xvals=1800, time_it=False):
        """
        Initializes the Data_loader with data from the specified path.

        Args:
            path (str): The file path to the power spectrum data.
            lmax (int): The maximum multipole (l) to use.
            n_xvals (int): The number of points for the cos(theta) grid.
            time_it (bool): If True, enables timing for decorated methods.
        """
        self.path = path
        self.lmax = lmax
        self.time_it = time_it
        
        try:
            # Load the data, assuming a comma-separated format with comments.
            df_full = pd.read_csv(self.path, sep=',', comment='#')
            self.df = df_full.iloc[:, :4]
            self.df.columns = ['ell', 'D_ell', '-dD_ell', '+dD_ell']
        except Exception as e:
            print(f"Error loading data file {self.path}: {e}")
            return

        # Filter the DataFrame by lmax and convert columns to numpy arrays.
        self.df = self.df[self.df['ell'].astype(float).to_numpy() <= self.lmax].reset_index(drop=True)
        self.ell = self.df['ell'].to_numpy()
        self.D_ell = self.df['D_ell'].to_numpy()
        
        self.dD_ell_neg = self.df['-dD_ell'].to_numpy()
        self.dD_ell_pos = self.df['+dD_ell'].to_numpy()
        # Symmetrize the errors by taking the average.
        self.error = (self.dD_ell_neg + self.dD_ell_pos) / 2

        # Define the grid for the correlation function.
        self.xvals = np.linspace(0.9999999999999999, -0.9999999999999999, n_xvals)  # cos(theta)
        self.theta = np.arccos(self.xvals) * 180 / np.pi  # theta in degrees

        # Cached properties to avoid redundant calculations.
        self._correlation = None
        self._correlation_err = None

    @time_execution
    def get_correlation_function(self, force_recalc=False):
        """
        Calculates and returns the 2-point correlation function C(theta) and its error.
        The result is cached to improve performance on subsequent calls.

        Args:
            force_recalc (bool): If True, forces recalculation even if a cached result exists.
        
        Returns:
            tuple: A tuple containing the correlation function (np.ndarray) and its error (np.ndarray).
        """
        if self._correlation is None or force_recalc:
            self._correlation = correlation_func(self.D_ell, self.xvals)
            self._correlation_err = correlation_func_err2(self.error, self.xvals)
        return self._correlation, self._correlation_err

    @time_execution
    def get_xivar(self, a, b):
        """
        Calculates the xivar statistic and its error for a given interval [a, b] in cos(theta).
        
        Args:
            a (float): The lower bound of the interval (cos(theta_upper)).
            b (float): The upper bound of the interval (cos(theta_lower)).
            
        Returns:
            tuple: A tuple containing the xivar value (float) and its error (float).
        """
        val = xivar2(self.D_ell, a, b)
        err = xivar_err2(self.error, a, b)
        return val, err

    def _load_matrix(self, a, b):
        """
        Loads a pre-calculated Tmn matrix required for the S12 statistic.
        The matrix filename is constructed based on the angular interval.
        This function is order-agnostic with respect to a and b.

        Args:
            a (float): One bound of the interval in cos(theta).
            b (float): The other bound of the interval in cos(theta).
        
        Returns:
            np.ndarray: The loaded Tmn matrix.
        
        Raises:
            FileNotFoundError: If the matrix file does not exist.
        """
        # Convert cos(theta) values to degrees to construct the filename.
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        
        # The file naming convention is Tmn__{larger_angle}__{smaller_angle}.npy
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        
        matrix_path = f"files/matrix/Tmn__{theta_upper}__{theta_lower}.npy"
        
        try:
            return np.load(matrix_path)
        except FileNotFoundError:
            print(f"Matrix file not found: {matrix_path}")
            raise
    @time_execution
    def get_s12(self, a, b):
            """
            Calculates the S12 statistic and its error for a given interval [a, b] in cos(theta).
            
            Args:
                a (float): The lower bound of the interval (cos(theta_upper)).
                b (float): The upper bound of the interval (cos(theta_lower)).
                
            Returns:
                tuple: A tuple containing the S12 value (float) and its error (float).
            """
            M = self._load_matrix(a, b)
            val = S12(self.D_ell, M)
            err = S12_err2(self.D_ell, self.error, M)
            return val, err
    
    @time_execution
    def experimental_values(self, intervals):
            """
            Calculates experimental values for C(180), S12, and xivar for a given set of angular intervals.
    
            Args:
                intervals (list of tuples): A list of tuples, where each tuple (a, b) defines an
                                            interval in cos(theta) between -1 and 1.
    
            Returns:
                dict: A flattened dictionary with correctly ordered keys for C180, s12, and xivar values.
            """
            exp_values = {}
    
            # Calculate C(180)
            corr, corr_err = self.get_correlation_function()
            exp_values['C180'] = (corr[-1], corr_err[-1])
    
            s12_values = {}
            xiv_values = {}

            for a, b in intervals:
                theta_1 = round(np.arccos(a) * 180 / np.pi)
                theta_2 = round(np.arccos(b) * 180 / np.pi)
                theta_upper = max(theta_1, theta_2)
                theta_lower = min(theta_1, theta_2)
                s12_key = f's12_{theta_upper}_{theta_lower}'
                xiv_key = f'xiv_{theta_upper}_{theta_lower}'

                # Calculate S12 for the interval.
                try:
                    s12_val, s12_err = self.get_s12(a, b)
                    s12_values[s12_key] = (s12_val, s12_err)
                except FileNotFoundError:
                    s12_values[s12_key] = (np.nan, np.nan)
    
                # Calculate xivar for the interval.
                xivar_val, xivar_err = self.get_xivar(a, b)
                xiv_values[xiv_key] = (xivar_val, xivar_err)

            exp_values.update(s12_values)
            exp_values.update(xiv_values)

            return exp_values
    

    
