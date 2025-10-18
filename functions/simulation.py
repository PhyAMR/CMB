"""
This module provides functions for running Monte Carlo (MC) simulations to estimate the distribution
of cosmological observables. It is designed to generate mock data based on experimental uncertainties
and then compute statistics for each realization.
"""

import numpy as np
import pandas as pd
import pickle
from .correlation_function import correlation_func
from .s12 import S12
from .xiv import xivar

def MC_calculations(data, lmax, xvals, intervals):
    """
    Performs a set of calculations for a single Monte Carlo realization of the power spectrum.

    Args:
        data (pd.Series): A series containing a single realization of D_ell values.
        lmax (int): The maximum multipole (l) to use.
        xvals (np.ndarray): An array of cos(theta) values for the correlation function.
        intervals (list): A list of tuples, each defining an interval [a, b] for S12 and xivar.

    Returns:
        tuple: A tuple containing the calculated TTCl, TTcor, C180, and S12/xivar statistics.
    """
    # Extract the power spectrum for the current realization.
    TTCl = data.iloc[0][:lmax]
    
    # Calculate the corresponding correlation function.
    TTcor = correlation_func(TTCl, xvals)
    
    # Get the correlation at 180 degrees.
    C180 = TTcor[-1]
    
    th_values = []
    for a, b in intervals:
        # Load the pre-calculated Tmn matrix for the S12 statistic.
        M = np.load(f"files/matrix/Tmn__{round(np.arccos(a) * 180 / np.pi)}__{round(np.arccos(b) * 180 / np.pi)}.npy")
        
        # Calculate S12 and xivar for the interval.
        s12 = S12(TTCl, M)
        th_values.append(s12)
        xiv = xivar(TTCl, a, b)
        th_values.append(xiv)

    return (TTCl, TTcor, C180, *th_values)

def MC_results(intervals, data, xvals, n=1000):
    """
    Generates and processes Monte Carlo simulation results based on experimental data and errors.

    Args:
        intervals (list): A list of tuples defining intervals for S12 and xivar.
        data (pd.DataFrame): A DataFrame with 'D_ell' and 'Error' columns from experimental data.
        xvals (np.ndarray): An array of cos(theta) values.
        n (int): The number of Monte Carlo simulations to generate.
    """
    # Generate n random realizations of the power spectrum for each multipole,
    # assuming a normal distribution centered at the measured D_ell with the given error.
    data['dist_per_cl'] = data.apply(lambda row: np.random.normal(loc=row['D_ell'], scale=row['Error'], size=n), axis=1)
    distributions = data['dist_per_cl'].to_list()

    # Transpose the list of distributions to get a list of n power spectrum realizations.
    trasp = list(map(list, zip(*distributions)))
    df_arrays = pd.DataFrame({'valores': [np.array(row) for row in trasp]})
    
    # Define column names for the results DataFrame.
    chain_cols = ['D_ell', 'Cor']
    for a, b in intervals:
        chain_cols += [f's12_{round(np.arccos(a) * 180 / np.pi)}_{round(np.arccos(b) * 180 / np.pi)}',
                       f'xiv_{round(np.arccos(a) * 180 / np.pi)}_{round(np.arccos(b) * 180 / np.pi)}']
    chain_cols += ['C180']
    
    data_dict = {}

    try:
        # Apply the calculation function to each MC realization.
        chain_result = df_arrays.apply(MC_calculations, axis=1, args=(200, xvals, intervals))
        df_chain = pd.DataFrame(chain_result.tolist(), columns=chain_cols, index=df_arrays.index)
        df = pd.concat([df_chain, df_arrays], axis=1)

        # Calculate statistics (mean and std) across all realizations.
        stacked_Cl = np.vstack(df['D_ell'].values)
        stacked_Cor = np.vstack(df['Cor'].values)
        
        mean_Cl = stacked_Cl.mean(axis=0)
        std_Cl = stacked_Cl.std(axis=0, ddof=0)
        mean_Cor = stacked_Cor.mean(axis=0)
        std_Cor = stacked_Cor.std(axis=0, ddof=0)

        # Store the results in a dictionary.
        data_dict['Simulation'] = (df[chain_cols], mean_Cl, std_Cl, mean_Cor, std_Cor)

    except Exception as e:
        print(f'Error during MC processing: {e}')

    finally:
        # Save the results dictionary to a pickle file.
        with open(f'Simulation_{n}.pkl', 'wb') as f:
            pickle.dump(data_dict, f)