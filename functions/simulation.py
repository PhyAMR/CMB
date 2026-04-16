"""
This module provides functions for running Monte Carlo (MC) simulations to estimate the distribution
of cosmological observables. It is designed to generate mock data based on experimental uncertainties
and then compute statistics for each realization.
"""

import numpy as np
import pandas as pd
import pickle
import os
from .correlation_function import correlation_func
from .s12 import S12
from .xiv import xivar
from scipy.stats import chi2
import healpy as hp
from multiprocessing import Pool, cpu_count
import logging
logger = logging.getLogger(__name__)
def MC_calculations(data, data_loader, intervals):
    """
    Performs a set of calculations for a single Monte Carlo realization of the power spectrum.

    Args:
        data (pd.Series): A series containing a single realization of D_ell values.
        data_loader (Data_loader): An instance of the Data_loader class.
        intervals (list): A list of tuples, each defining an interval [a, b] for S12 and xivar.

    Returns:
        tuple: A tuple containing the calculated TTCl, TTcor, C180, and S12/xivar statistics.
    """
    lmax = data_loader.lmax
    xvals = data_loader.xvals
    
    # Extract the power spectrum for the current realization.
    TTCl = data.iloc[0][:lmax]
    
    # Calculate the corresponding correlation function.
    TTcor = correlation_func(TTCl, xvals)
    
    # Get the correlation at 180 degrees.
    C180 = TTcor[-1]
    
    th_values = []
    for a, b in intervals:
        # Load the pre-calculated Tmn matrix for the S12 statistic.
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        matrix_path = f"files/matrix/Tmn__{theta_upper}__{theta_lower}.npy"
        try:
            M = np.load(matrix_path)
        except FileNotFoundError:
            try:
                from .s12 import Tmn
                logger.info(f"Tmn matrix not found for interval ({theta_upper},{theta_lower}). Computing and saving to {matrix_path}...")
                Tmn(len(TTCl), theta_upper, theta_lower, a, b)
                M = np.load(matrix_path)
            except Exception as e:
                logger.exception(f"Failed to compute Tmn for ({theta_upper},{theta_lower}): {e}")
                raise
        
        # Calculate S12 and xivar for the interval.
        s12 = S12(TTCl, M)
        th_values.append(s12)
        xiv = xivar(TTCl, a, b)
        th_values.append(xiv)

    return (TTCl, TTcor, C180, *th_values)

def MC_results(data_loader, intervals, output_dir=None, n=1000):
    """
    Generates and processes Monte Carlo simulation results based on a Data_loader object.

    Args:
        data_loader (Data_loader): An instance of the Data_loader class containing experimental data.
        intervals (list): A list of tuples defining intervals for S12 and xivar.
        n (int): The number of Monte Carlo simulations to generate.
    """
    # Use the DataFrame and error from the Data_loader object.

    # Ensure output_dir defaults to run_{lmax}_{n} when not provided
    if output_dir is None:
        output_dir = f"run_{data_loader.lmax}_{n}"

    data = data_loader.df.copy()
    data['Error'] = data_loader.error

    # Generate n random realizations of the power spectrum for each multipole.
    # Use truncated normal distribution to ensure all D_ell values remain positive.
    def generate_Dl_realizations(row, n):
        """
        Generate n realizations for a given row. If the provided D_ell is an array (full spectrum)
        use the parallel planck-style pipeline, otherwise fall back to the original per-ell sampling.
        """
        D_ell = row['D_ell']


        # === original per-ell Monte Carlo sampling (fallback) ===
        ell = int(row['ell'])
        D_ell_scalar = float(D_ell)

        F = ell * (ell + 1) / (2 * np.pi)
        C_ell = D_ell_scalar / F      # convert back to C_ell

        results = []

        for _ in range(n):
            # Storage for a_lm from m=-ell..+ell
            a_lm = np.zeros(2*ell+1, dtype=np.complex128)

            # m = 0 → purely real Gaussian
            a_lm[ell] = np.random.normal(0, np.sqrt(C_ell))

            # m > 0
            for m in range(1, ell+1):
                re = np.random.normal(0, np.sqrt(C_ell/2))
                im = np.random.normal(0, np.sqrt(C_ell/2))
                a_lm[ell + m] = re + 1j*im

                # m < 0 via symmetry
                a_lm[ell - m] = (-1)**m * np.conj(a_lm[ell + m])

            # Compute C_ell estimator
            Cl_hat = np.sum(np.abs(a_lm)**2) / (2*ell + 1)

            # Convert to D_ell
            Dl_hat = F * Cl_hat

            results.append(Dl_hat)


        # Return the per-ell Dl realizations (one value per simulation)
        return results
    

    data['dist_per_cl'] = data.apply(lambda row: generate_Dl_realizations(row, n), axis=1)

    distributions = data['dist_per_cl'].to_list()

    # Transpose the list of distributions to get a list of n power spectrum realizations.
    trasp = list(map(list, zip(*distributions)))
    df_arrays_mid = pd.DataFrame({'valores': [np.array(row) for row in trasp]})
    logger.debug(f"df_arrays_mid['valores'] sample: {df_arrays_mid['valores']}")
    df_arrays = planck_style_pipeline_parallel(
                ell=data_loader.ell,
                D_ell_theory=df_arrays_mid['valores'],
                nside_in=2048,
                fwhm_in_arcmin=40.0,
                noise_Cl=4.5e-5,
                mask=None,
                estimate_Cl=True,
                num_workers=None,
            )
    # Define column names for the results DataFrame.
    chain_cols = ['D_ell', 'Cor', 'C180']
    for a, b in intervals:
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        chain_cols += [f's12_{theta_upper}_{theta_lower}',
                       f'xiv_{theta_upper}_{theta_lower}']
    
    data_dict = {}

    try:
        # Apply the calculation function to each MC realization.
        chain_result = df_arrays.apply(MC_calculations, axis=1, args=(data_loader, intervals))
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
        logger.exception(f'Error during MC processing: {e}')

    finally:
        # Save results to output directory if provided
        sim_out = output_dir if output_dir else '.'
        sim_dir = os.path.join(sim_out, 'simulation')
        os.makedirs(sim_dir, exist_ok=True)
        # Save D_ell and Corr matrices
        try:
            np.save(os.path.join(sim_dir, 'D_ell.npy'), stacked_Cl)
            np.save(os.path.join(sim_dir, 'Corr.npy'), stacked_Cor)
        except NameError:
            # If stacking failed earlier, skip saving arrays
            pass
        # Save scalar statistics columns order
        scalar_cols = [c for c in chain_cols if c not in ['D_ell','Cor']]
        with open(os.path.join(sim_dir, 'columns.txt'), 'w') as cf:
            cf.write('\n'.join(scalar_cols))
        # Build S, xiv, C180 matrices
        s_cols = [c for c in scalar_cols if c.startswith('s12_')]
        xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
        try:
            if s_cols:
                S_mat = np.vstack([df[c].values for c in s_cols]).T
                np.savetxt(os.path.join(sim_dir, 'S_statistics.txt'), S_mat)
            if xiv_cols:
                xiv_mat = np.vstack([df[c].values for c in xiv_cols]).T
                np.savetxt(os.path.join(sim_dir, 'xiv_statistic.txt'), xiv_mat)
            if 'C180' in scalar_cols:
                C180_vec = df['C180'].values
                np.savetxt(os.path.join(sim_dir, 'C180_statistics.txt'), C180_vec)
        except Exception:
            pass
        # Also save a lightweight pickle for backward compatibility
        try:
            with open(os.path.join(sim_dir, f'Simulation_{n}.pkl'), 'wb') as f:
                pickle.dump(data_dict, f)
        except Exception:
            pass


# --- Multiprocessing helpers for Planck-style pipelines ---
def _single_realization_worker(args):
    """Worker function for a single simulation realization."""
    (cl_map_row, ell, nside_in, fwhm_in, noise_Cl, mask, estimate_Cl) = args

    ell_max = int(np.max(ell))
    full_cl = np.zeros(ell_max + 1)
    full_cl[ell] = cl_map_row
    if noise_Cl is not None:
        full_cl[ell] += noise_Cl

    # 1. Generate map (heavy SHT)
    try:
        m = hp.synfast(full_cl, nside=nside_in, lmax=ell_max, pixwin=True,
                       fwhm=np.radians(fwhm_in/60))
    except TypeError:
        m = hp.synfast(full_cl, nside=nside_in, lmax=ell_max, pixwin=True,
                       fwhm=np.radians(fwhm_in/60))

    # 2. Masking
    if mask is not None:
        m = hp.ma(m)
        m.mask = mask < 0.9
        m = m.filled(0)

    # 3. Estimation
    if estimate_Cl:
        cl_est = hp.anafast(m, lmax=ell_max)
        return cl_est[ell]
    return m


def planck_style_pipeline_parallel(
    ell, D_ell_theory, nside_in, fwhm_in_arcmin, noise_Cl,
    mask=None, estimate_Cl=True, num_workers=None
):
    """Run a Planck-like simulation pipeline in parallel.

    Args:
        ell (array-like): Multipole indices to simulate (e.g., np.arange(2, lmax+1)).
        D_ell_theory (ndarray): Shape (n_sims, len(ell)) of theoretical D_ell values.
        nside_in (int): healpy nside for synfast.
        fwhm_in_arcmin (float): Beam FWHM in arcminutes.
        noise_Cl (array-like or None): Optional noise power spectrum to add on top of theory (C_ell units).
        mask (array-like or None): Optional mask to apply to synthesized maps.
        estimate_Cl (bool): If True, return estimated Cls (anafast) per realization; otherwise return maps.
        num_workers (int or None): Number of worker processes.

    Returns:
        ndarray: Simulated D_ell outputs of shape (n_sims, len(ell)).
    """
    ell = np.asarray(ell, dtype=int)
    # Accept pandas Series or list/tuple of arrays and ensure a proper 2D ndarray
    if isinstance(D_ell_theory, (pd.Series, list, tuple)):
        D_ell_theory = np.asarray(list(D_ell_theory), dtype=float)
    else:
        D_ell_theory = np.asarray(D_ell_theory, dtype=float)
    n_sims = D_ell_theory.shape[0]
    logger.debug(f"Number of simulations to run: {n_sims}")
    # Convert D_ell to C_ell for all rows at once
    F = ell * (ell + 1) / (2 * np.pi)
    C_ell = D_ell_theory / F

    # Beam and pixel window
    sigma = np.radians(fwhm_in_arcmin / 60) / np.sqrt(8 * np.log(2))
    b_ell = np.exp(-0.5 * ell * (ell + 1) * sigma**2)
    p_ell = hp.pixwin(nside_in, lmax=int(np.max(ell)))[ell]

    Cl_to_simulate = C_ell * (b_ell**2) * (p_ell**2)

    if num_workers is None:
        num_workers = max(1, int(cpu_count() - 4))

    tasks = [
        (Cl_to_simulate[i], ell, nside_in, fwhm_in_arcmin, noise_Cl, mask, estimate_Cl)
        for i in range(n_sims)
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.map(_single_realization_worker, tasks)

    Cl_matrix = np.array(results)

    # Convert back to D_ell
    D_ell_out = Cl_matrix * F
    # Return as a DataFrame where each row contains the D_ell array for a realization.
    df_out = pd.DataFrame({'D_ell_arr': [D_ell_out[i] for i in range(n_sims)]})
    return df_out
