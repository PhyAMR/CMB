"""
This module provides functions for cosmological calculations, primarily interfacing with the CAMB library
to compute theoretical power spectra (Cl) and correlation functions. It also includes utilities for
processing Markov Chain Monte Carlo (MCMC) chains from cosmological parameter estimation.
"""

import numpy as np
import pandas as pd
import pickle
import camb
import camb.correlations
from getdist import loadMCSamples
from .s12 import S12
from .xiv import xivar

def compute_cl_cor_pl(parss, lmax, xvals):
    """
    Computes the CMB power spectrum (Cl) and correlation function for a Planck-like cosmology.

    Args:
        parss (dict): A dictionary of cosmological parameters.
        lmax (int): The maximum multipole (l) to compute.
        xvals (np.ndarray): An array of cos(theta) values for the correlation function.

    Returns:
        tuple: A tuple containing:
            - TTCl (np.ndarray): The temperature power spectrum (D_ell) from l=2 to lmax.
            - TTcor (np.ndarray): The temperature correlation function.
            - TTcor[-1] (float): The correlation at 180 degrees (cos(theta)=-1).
    """
    # Set cosmological parameters using CAMB's set_params function.
    pars = camb.set_params(ombh2=parss['omegabh2'], omch2=parss['omegach2'], H0=parss['H0'], omk=parss['omegak'],
                           YHe=parss['yheused'], nnu=parss['nnu'], nrun=parss['nrun'], Alens=parss['Alens'],
                           ns=parss['ns'], As=np.exp(parss['logA']) * 1e-10, w=-1, wa=parss['wa'],
                           mnu=parss['mnu'], tau=parss['tau'])
    
    # Get the cosmological results from CAMB.
    resu = camb.get_results(pars)
    
    # Get the CMB power spectra.
    cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    totCL = cls['total']
    
    # Extract the TT power spectrum (D_ell), starting from l=2.
    TTCl = totCL[2:, 0]
    
    # Compute the correlation function from the power spectrum.
    totCorr = camb.correlations.cl2corr(totCL, xvals, lmax)
    TTcor = totCorr[:, 0]
    
    return TTCl, TTcor, TTcor[-1]

def compute_cl_cor_dv(parss, lmax, xvals):
    """
    Computes the CMB power spectrum (Cl) and correlation function for a dark energy model.

    Args:
        parss (dict): A dictionary of cosmological parameters.
        lmax (int): The maximum multipole (l) to compute.
        xvals (np.ndarray): An array of cos(theta) values for the correlation function.

    Returns:
        tuple: A tuple containing:
            - TTCl (np.ndarray): The temperature power spectrum (D_ell) from l=2 to lmax.
            - TTcor (np.ndarray): The temperature correlation function.
            - TTcor[-1] (float): The correlation at 180 degrees (cos(theta)=-1).
    """
    # Set cosmological parameters for a dark energy model.
    pars = camb.set_params(ombh2=parss['ombh2'], omch2=parss['omch2'], H0=parss['H0'], omk=0,
                           YHe=parss['YHe'], ns=parss['ns'], As=np.exp(parss['logA']) * 1e-10,
                           w=parss['w'], mnu=parss['mnu'], tau=parss['tau'])
    
    # Get the cosmological results from CAMB.
    resu = camb.get_results(pars)
    
    # Get the CMB power spectra.
    cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    totCL = cls['total']
    
    # Extract the TT power spectrum (D_ell), starting from l=2.
    TTCl = totCL[2:, 0]
    
    # Compute the correlation function from the power spectrum.
    totCorr = camb.correlations.cl2corr(totCL, xvals, lmax)
    TTcor = totCorr[:, 0]
    
    return TTCl, TTcor, TTcor[-1]

def expand_dict_values(dict1, dict2):
    """
    Expands the values of dict2 to match the array length of values in dict1 and merges them.
    This is useful for combining MCMC chain parameters with fixed cosmological parameters.

    Args:
        dict1 (dict): A dictionary where values are arrays of the same length.
        dict2 (dict): A dictionary where values are scalars.

    Returns:
        dict: A merged dictionary where values from dict2 are expanded to arrays.
    """
    # Get the target length from the arrays in the first dictionary.
    first_key = next(iter(dict1))
    target_length = len(dict1[first_key])

    # Expand the scalar values in dict2 into arrays of the target length.
    expanded_dict2 = {key: [value] * target_length for key, value in dict2.items()}
    
    # Merge the two dictionaries.
    z = dict1 | expanded_dict2
    return z

def chain_calculations(parss, lmax, xvals, intervals, c):
    """
    Performs a series of calculations for a given set of cosmological parameters from an MCMC chain.
    This includes computing Cl, correlation function, S12, and xivar statistics.

    Args:
        parss (pd.Series): A row from a DataFrame containing cosmological parameters.
        lmax (int): The maximum multipole (l).
        xvals (np.ndarray): An array of cos(theta) values.
        intervals (list): A list of tuples, each defining an interval [a, b] for S12 and xivar.
        c (int): An index to determine which cosmology computation to use (Planck-like or dark energy).

    Returns:
        tuple: A tuple containing TTCl, TTcor, C180, and the calculated S12 and xivar values.
    """
    # Choose the appropriate cosmology computation based on the index c.
    if c <= 4:
        TTCl, TTcor, C180 = compute_cl_cor_pl(parss, lmax, xvals)
    else:
        TTCl, TTcor, C180 = compute_cl_cor_dv(parss, lmax, xvals)
    
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

def chain_results(intervals, xvals, roots, name, n=1000):
    """
    Processes MCMC chains to compute cosmological observables and statistics for each sample.

    Args:
        intervals (list): A list of tuples defining intervals for S12 and xivar.
        xvals (np.ndarray): An array of cos(theta) values.
        roots (list): A list of file paths to the MCMC chain files (from getdist).
        name (str): The base name for the output pickle file.
        n (int): The number of samples to process from the tail of each chain.
    """
    # Define column names for the results DataFrame.
    chain_cols = ['D_ell', 'Cor']
    for a, b in intervals:
        chain_cols += [f's12_{round(np.arccos(a) * 180 / np.pi)}_{round(np.arccos(b) * 180 / np.pi)}',
                       f'xiv_{round(np.arccos(a) * 180 / np.pi)}_{round(np.arccos(b) * 180 / np.pi)}']
    chain_cols += ['C180']
    
    data_dict = {}

    for i, root in enumerate(roots):
        try:
            # Load the MCMC samples.
            samples = loadMCSamples(file_root=root)
            print(f'Processing: {root}')
            print('_' * 40)

            # Get the parameters and fixed values from the chain.
            params = samples.getParams()
            fixed = samples.ranges.fixedValueDict()

            # Create a dictionary of the chain data.
            chain_data = {p.name: getattr(params, p.name, np.nan) for p in samples.paramNames.names}
            
            # Combine chain parameters with fixed parameters.
            data = expand_dict_values(chain_data, fixed)
            df = pd.DataFrame.from_dict(data).tail(n)

            # Apply the chain_calculations function to each sample.
            chain_result = df.apply(chain_calculations, axis=1, args=(200, xvals, intervals, i))
            df_chain = pd.DataFrame(chain_result.tolist(), columns=chain_cols, index=df.index)
            df = pd.concat([df, df_chain], axis=1)

            # Stack the results for Cl and correlation function to compute mean and std.
            stacked_Cl = np.vstack(df['D_ell'].values)
            stacked_Cor = np.vstack(df['Cor'].values)

            mean_Cl = stacked_Cl.mean(axis=0)
            std_Cl = stacked_Cl.std(axis=0, ddof=0)
            mean_Cor = stacked_Cor.mean(axis=0)
            std_Cor = stacked_Cor.std(axis=0, ddof=0)

            # Store the results in the dictionary.
            short_name = root.strip().split('/')[-1]
            data_dict[short_name] = (df[chain_cols], mean_Cl, std_Cl, mean_Cor, std_Cor)

        except Exception as e:
            print(f'Error in {root}: {e}')
            continue

        finally:
            # Save the results dictionary to a pickle file.
            with open(f'{name}.pkl', 'wb') as f:
                pickle.dump(data_dict, f)