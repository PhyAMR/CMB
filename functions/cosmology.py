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
    try:
        # Set cosmological parameters using CAMB's set_params function.
        print("compute_cl_cor_pl parameters:")
        for key, value in parss.items():
            print(f"  {key}: {value} (type: {type(value)})")

        pars = camb.set_params(ombh2=parss['omegabh2'], omch2=parss['omegach2'], H0=parss['H0'], omk=parss['omegak'],
                               YHe=parss['yheused'], nnu=parss['nnu'], nrun=parss['nrun'], Alens=parss['Alens'],
                               ns=parss['ns'], As=np.exp(parss['logA']) * 1e-10, w=float(-1), wa=parss['wa'],
                               mnu=parss['mnu'], tau=parss['tau'])
    except TypeError as e:
        print(f"TypeError in compute_cl_cor_pl: {e}")
        for key, value in parss.items():
            print(f"  Parameter {key}: {value} (type: {type(value)})")
        raise

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
    try:
        # Set cosmological parameters for a dark energy model.
        print("compute_cl_cor_dv parameters:")
        for key, value in parss.items():
            print(f"  {key}: {value} (type: {type(value)})")

        pars = camb.set_params(ombh2=parss['ombh2'], omch2=parss['omch2'], H0=parss['H0'], omk=0,
                               YHe=parss['YHe'], ns=parss['ns'], As=np.exp(parss['logA']) * 1e-10,
                               w=parss['w'], mnu=parss['mnu'], tau=parss['tau'])
    except TypeError as e:
        print(f"TypeError in compute_cl_cor_dv: {e}")
        for key, value in parss.items():
            print(f"  Parameter {key}: {value} (type: {type(value)})")
        raise

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

def chain_calculations(parss, data_loader, intervals, c):
    """
    Performs a series of calculations for a given set of cosmological parameters from an MCMC chain.
    This includes computing Cl, correlation function, S12, and xivar statistics.

    Args:
        parss (pd.Series): A row from a DataFrame containing cosmological parameters.
        data_loader (Data_loader): An instance of the Data_loader class.
        intervals (list): A list of tuples, each defining an interval [a, b] for S12 and xivar.
        c (int): An index to determine which cosmology computation to use (Planck-like or dark energy).

    Returns:
        pd.Series: A series containing TTCl, TTcor, C180, and named S12 and xivar values.
    """
    lmax = data_loader.lmax
    xvals = data_loader.xvals

    # Choose the appropriate cosmology computation based on the index c.
    if c <= 4:
        TTCl, TTcor, C180 = compute_cl_cor_pl(parss, lmax, xvals)
    else:
        TTCl, TTcor, C180 = compute_cl_cor_dv(parss, lmax, xvals)
    
    results = {
        'D_ell': TTCl,
        'Cor': TTcor,
        'C180': C180,
    }
    
    for a, b in intervals:
        # Make loading order-agnostic
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        matrix_path = f"files/matrix/Tmn__{theta_upper}__{theta_lower}.npy"
        M = np.load(matrix_path)
        
        s12_key = f's12_{theta_upper}_{theta_lower}'
        xiv_key = f'xiv_{theta_upper}_{theta_lower}'

        # Calculate S12 and xivar for the interval.
        s12 = S12(TTCl, M)
        results[s12_key] = s12
        xiv = xivar(TTCl, a, b)
        results[xiv_key] = xiv

    return pd.Series(results)

def chain_results(data_loader, intervals, roots, name, n=1000):
    """
    Processes MCMC chains to compute cosmological observables and statistics for each sample.

    Args:
        data_loader (Data_loader): An instance of the Data_loader class.
        intervals (list): A list of tuples defining intervals for S12 and xivar.
        roots (list): A list of file paths to the MCMC chain files (from getdist).
        name (str): The base name for the output pickle file.
        n (int): The number of samples to process from the tail of each chain.
    """
    xvals = data_loader.xvals
    lmax = data_loader.lmax

    # Define column names for the results DataFrame.
    chain_cols = ['D_ell', 'Cor', 'C180']
    for a, b in intervals:
        # Make column naming order-agnostic
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        chain_cols += [f's12_{theta_upper}_{theta_lower}',
                       f'xiv_{theta_upper}_{theta_lower}']
    
    data_dict = {}

    for i, root in enumerate(roots):
        try:
            # Load the MCMC samples.
            samples = loadMCSamples(file_root=root)
            print(f'Processing: {root}')
            print('_' * 70)

            # Get the parameters and fixed values from the chain.
            params = samples.getParams()
            fixed = samples.ranges.fixedValueDict()
            # Create a dictionary of the chain data.
            chain_data = {p.name: getattr(params, p.name, np.nan) for p in samples.paramNames.names}
            # Combine chain parameters with fixed parameters.
            data = expand_dict_values(chain_data, fixed)
            df = pd.DataFrame.from_dict(data).tail(n)
            # Apply the chain_calculations function to each sample.
            df_chain = df.apply(chain_calculations, axis=1, args=(data_loader, intervals, i))
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
