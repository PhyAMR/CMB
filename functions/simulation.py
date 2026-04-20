"""
This module provides functions for running Monte Carlo (MC) simulations to estimate the distribution
of cosmological observables. It uses the experimental D_ell spectrum and runs it through the
Planck-style pipeline n times to generate mock realizations.
"""

import numpy as np
import pandas as pd
import pickle
import os
from .correlation_function import correlation_func
from .s12 import S12, Tmn
from .xiv import xivar
import healpy as hp
from multiprocessing import Pool, cpu_count
import logging

logger = logging.getLogger(__name__)


def MC_calculations(D_ell, data_loader, intervals):
    """
    Performs a set of calculations for a single Monte Carlo realization of the power spectrum.

    Args:
        D_ell (np.ndarray): A single realization of D_ell values.
        data_loader (Data_loader): An instance of the Data_loader class.
        intervals (list): A list of tuples, each defining an interval [a, b] for S12 and xivar.

    Returns:
        dict: A dictionary containing the calculated D_ell, correlation, C180, and S12/xivar statistics.
    """
    lmax = data_loader.lmax
    xvals = data_loader.xvals
    
    # Calculate the corresponding correlation function.
    TTcor = correlation_func(D_ell, xvals)
    
    # Get the correlation at 180 degrees.
    C180 = TTcor[-1]
    
    results = {
        'D_ell': D_ell,
        'Cor': TTcor,
        'C180': C180
    }
    
    for a, b in intervals:
        # Load the pre-calculated Tmn matrix for the S12 statistic.
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        matrix_path = f"files/matrix/Tmn__{theta_upper}__{theta_lower}.npy"
        try:
            M = np.load(matrix_path)
            
            if M.shape[0] != len(D_ell):
                logger.info(
                    f"Tmn size mismatch ({M.shape[0]} vs {len(D_ell)}), "
                    f"recomputing for ({theta_upper}, {theta_lower})"
                )
                Tmn(len(D_ell), theta_upper, theta_lower, a, b)
                M = np.load(matrix_path)
        
        except FileNotFoundError:
            logger.info(
                f"Tmn not found, computing for ({theta_upper}, {theta_lower})"
            )
            Tmn(len(D_ell), theta_upper, theta_lower, a, b)
            M = np.load(matrix_path)
        
        # Calculate S12 and xivar for the interval.
        s12_key = f's12_{theta_upper}_{theta_lower}'
        xiv_key = f'xiv_{theta_upper}_{theta_lower}'
        
        results[s12_key] = S12(D_ell, M)
        results[xiv_key] = xivar(D_ell, a, b)
    
    return results


# ============================================================================
# Planck-Style Simulation Pipeline (same as cosmo.py)
# ============================================================================

def _single_realization_worker(args):
    """Worker function for a single simulation realization."""
    (cl_map_row, ell, nside_in, fwhm_in, noise_Cl, mask, estimate_Cl, seed) = args
    
    np.random.seed(seed)
    
    ell_max = int(np.max(ell))
    full_cl = np.zeros(ell_max + 1)
    full_cl[ell] = cl_map_row
    
    if noise_Cl is not None:
        full_cl[ell] += noise_Cl
    
    try:
        m = hp.synfast(
            full_cl,
            nside=nside_in,
            lmax=ell_max,
            pixwin=True,
            fwhm=np.radians(fwhm_in / 60)
        )
    except TypeError:
        m = hp.synfast(
            full_cl,
            nside=nside_in,
            lmax=ell_max,
            pixwin=True,
            fwhm=np.radians(fwhm_in / 60)
        )
    
    if mask is not None:
        # Calculate sky fraction (percentage of pixels not masked)
        mask_map = hp.read_map(mask)
        f_sky = np.mean(mask_map >= 0.9)
        
        m = hp.ma(m)
        m.mask = mask_map < 0.9
        m = m.filled(0)
    else:
        f_sky = 1.0
    
    if estimate_Cl:
        cl_est = hp.anafast(m, lmax=ell_max)
        # Correct for the loss of power due to the mask
        return cl_est[ell] / f_sky
    
    return m


def planck_style_pipeline_parallel(
    ell,
    D_ell_theory,
    nside_in=2048,
    fwhm_in_arcmin=40.0,
    noise_Cl=None,
    mask=None,
    estimate_Cl=True,
    num_workers=None
):
    """Run Planck-like simulation pipeline in parallel.
    
    Args:
        ell (array-like): Multipole indices to simulate (e.g., np.arange(2, lmax+1)).
        D_ell_theory (ndarray): Shape (n_sims, len(ell)) of theoretical D_ell values.
        nside_in (int): healpy nside for synfast.
        fwhm_in_arcmin (float): Beam FWHM in arcminutes.
        noise_Cl (array-like or None): Optional noise power spectrum to add on top of theory.
        mask (array-like or None): Optional mask to apply to synthesized maps.
        estimate_Cl (bool): If True, return estimated Cls (anafast) per realization.
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
    
    if D_ell_theory.ndim == 1:
        D_ell_theory = D_ell_theory.reshape(1, -1)
    
    n_sims = D_ell_theory.shape[0]
    logger.info(f"Running Planck pipeline on {n_sims} spectra")
    
    # Convert D_ell to C_ell for all rows at once
    F = ell * (ell + 1) / (2 * np.pi)
    C_ell = D_ell_theory / F
    
    # Beam and pixel window
    sigma = np.radians(fwhm_in_arcmin / 60) / np.sqrt(8 * np.log(2))
    b_ell = np.exp(-0.5 * ell * (ell + 1) * sigma**2)
    p_ell = hp.pixwin(nside_in, lmax=int(np.max(ell)))[ell]
    
    Cl_to_simulate = C_ell * (b_ell**2) * (p_ell**2)
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 8)
    
    logger.debug(f"Using {num_workers} parallel workers")
    
    tasks = []
    for i in range(n_sims):
        unique_seed = np.random.randint(0, 2**31 - 1)
        tasks.append((
            Cl_to_simulate[i],
            ell,
            nside_in,
            fwhm_in_arcmin,
            noise_Cl,
            mask,
            estimate_Cl,
            unique_seed
        ))
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(_single_realization_worker, tasks)
    
    Cl_matrix = np.array(results)
    D_ell_out = Cl_matrix * F
    
    return D_ell_out


def MC_results(data_loader, intervals, output_dir=None, n=1000):
    """
    Generates and processes Monte Carlo simulation results based on experimental D_ell.
    
    Takes the experimental D_ell spectrum from data_loader and runs it through
    the Planck-style pipeline n times to generate mock realizations.

    Args:
        data_loader (Data_loader): An instance of the Data_loader class containing experimental data.
        intervals (list): A list of tuples defining intervals for S12 and xivar.
        output_dir (str or None): Output directory. If None, uses default.
        n (int): The number of times to run the pipeline on the experimental D_ell.
    """
    # Ensure output_dir defaults appropriately when not provided
    if output_dir is None:
        output_dir = f"run_{data_loader.lmax}_{n}"
    
    logger.info(f"Running Monte Carlo simulations with {n} realizations")
    logger.info(f"Using experimental D_ell from data_loader")
    
    # ================================================================
    # Get experimental D_ell and prepare for pipeline
    # ================================================================
    D_ell_experimental = data_loader.D_ell
    ell = data_loader.ell
    lmax = data_loader.lmax
    
    logger.info(f"Experimental D_ell shape: {D_ell_experimental.shape}")
    logger.info(f"Running pipeline {n} times on this spectrum")
    
    # Create n copies of the experimental spectrum
    D_ell_theory_stack = np.tile(D_ell_experimental, (n, 1))
    
    # ================================================================
    # Step 1: Run Planck-style pipeline on all realizations
    # ================================================================
    logger.info("Step 1: Running Planck-style pipeline on all realizations...")
    
    D_ell_recovered_all = planck_style_pipeline_parallel(
        ell=ell,
        D_ell_theory=D_ell_theory_stack,
        nside_in=2048,
        fwhm_in_arcmin=40.0,
        noise_Cl=None,
        mask=None,
        estimate_Cl=True,
        num_workers=None,
    )
    
    logger.info(f"Pipeline complete. Recovered D_ell shape: {D_ell_recovered_all.shape}")
    
    # ================================================================
    # Step 2: Compute statistics for each realization
    # ================================================================
    logger.info("Step 2: Computing statistics for each realization...")
    
    # Define column names for the results DataFrame
    chain_cols = ['D_ell', 'Cor', 'C180']
    for a, b in intervals:
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        chain_cols += [f's12_{theta_upper}_{theta_lower}',
                       f'xiv_{theta_upper}_{theta_lower}']
    
    results_list = []
    for i, D_ell_recovered in enumerate(D_ell_recovered_all):
        if (i + 1) % max(1, n // 10) == 0:
            logger.info(f"  Processed {i + 1}/{n} realizations")
        
        stats = MC_calculations(D_ell_recovered, data_loader, intervals)
        results_list.append(stats)
    
    df = pd.DataFrame(results_list)
    
    logger.info("Statistics computation complete")
    
    # ================================================================
    # Step 3: Calculate aggregate statistics and save results
    # ================================================================
    try:
        stacked_Cl = np.vstack(df['D_ell'].values)
        stacked_Cor = np.vstack(df['Cor'].values)
        
        mean_Cl = stacked_Cl.mean(axis=0)
        std_Cl = stacked_Cl.std(axis=0, ddof=0)
        mean_Cor = stacked_Cor.mean(axis=0)
        std_Cor = stacked_Cor.std(axis=0, ddof=0)
        
        logger.info(f"Mean D_ell: {mean_Cl[:5]} ...")
        logger.info(f"Std D_ell: {std_Cl[:5]} ...")
        
    except Exception as e:
        logger.exception(f'Error during statistics calculation: {e}')
        stacked_Cl = None
        stacked_Cor = None
    
    finally:
        # ================================================================
        # Step 4: Save results to output directory
        # ================================================================
        logger.info(f"Saving results to {output_dir}...")
        
        sim_dir = os.path.join(output_dir, 'simulation')
        os.makedirs(sim_dir, exist_ok=True)
        
        # Save D_ell and Corr matrices
        if stacked_Cl is not None:
            np.save(os.path.join(sim_dir, 'D_ell.npy'), stacked_Cl)
            np.save(os.path.join(sim_dir, 'Corr.npy'), stacked_Cor)
            logger.info(f"Saved D_ell and Corr arrays to {sim_dir}")
        
        # Save scalar statistics columns order
        scalar_cols = [c for c in chain_cols if c not in ['D_ell', 'Cor']]
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
            
            logger.info(f"Saved scalar statistics to {sim_dir}")
        except Exception as e:
            logger.warning(f"Could not save some statistics files: {e}")
        
        # Save a lightweight pickle for backward compatibility
        try:
            with open(os.path.join(sim_dir, f'Simulation_{n}.pkl'), 'wb') as f:
                pickle.dump({'Simulation': (df[chain_cols],)}, f)
            logger.info(f"Saved pickle file: Simulation_{n}.pkl")
        except Exception as e:
            logger.warning(f"Could not save pickle file: {e}")
        
        logger.info(f"✓ Monte Carlo simulation complete. Results saved to: {sim_dir}")