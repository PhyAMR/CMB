"""
Improved cosmological calculations - saves best_fit and MCMC separately.

Key improvements:
- Saves theory_bestfit/ and theory_mcmc/ directories separately
- Allows 'all' mode to save both
- Proper directory structure for getdist analysis
"""

import numpy as np
import pandas as pd
import os
import camb
import camb.correlations
import healpy as hp
from multiprocessing import Pool, cpu_count
from getdist import loadMCSamples
import logging

from .correlation_function import correlation_func
from .s12 import S12, Tmn
from .xiv import xivar

logger = logging.getLogger(__name__)


# ============================================================================
# Planck-Style Simulation Pipeline (same as before)
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
        mask = hp.read_map(mask)
        f_sky = np.mean(mask >= 0.9) 
        
        m = hp.ma(m)
        m.mask = mask < 0.9
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
    """Run Planck-like simulation pipeline in parallel."""
    ell = np.asarray(ell, dtype=int)
    
    if isinstance(D_ell_theory, (pd.Series, list, tuple)):
        D_ell_theory = np.asarray(list(D_ell_theory), dtype=float)
    else:
        D_ell_theory = np.asarray(D_ell_theory, dtype=float)
    
    if D_ell_theory.ndim == 1:
        D_ell_theory = D_ell_theory.reshape(1, -1)
    
    n_sims = D_ell_theory.shape[0]
    logger.info(f"Running Planck pipeline on {n_sims} spectra")
    
    F = ell * (ell + 1) / (2 * np.pi)
    C_ell = D_ell_theory / F
    
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


# ============================================================================
# CAMB Interface Functions
# ============================================================================

def compute_cl_from_params_planck(parss, lmax):
    """Compute theoretical D_ell from cosmological parameters (Planck-like)."""
    try:
        pars = camb.set_params(
            ombh2=parss['omegabh2'],
            omch2=parss['omegach2'],
            H0=parss['H0'],
            omk=parss['omegak'],
            YHe=parss['yheused'],
            nnu=parss['nnu'],
            nrun=parss['nrun'],
            Alens=parss['Alens'],
            ns=parss['ns'],
            As=np.exp(parss['logA']) * 1e-10,
            w=float(-1),
            wa=parss['wa'],
            mnu=parss['mnu'],
            tau=parss['tau']
        )
    except TypeError as e:
        logger.exception(f"TypeError in compute_cl_from_params_planck: {e}")
        for key, value in parss.items():
            logger.debug(f"  {key}: {value} (type: {type(value)})")
        raise
    
    resu = camb.get_results(pars)
    cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    totCL = cls['total']
    TTCl_theory = totCL[2:, 0]
    
    return TTCl_theory


def compute_cl_from_params_dark_energy(parss, lmax):
    """Compute theoretical D_ell from cosmological parameters (dark energy model)."""
    try:
        pars = camb.set_params(
            ombh2=parss['ombh2'],
            omch2=parss['omch2'],
            H0=parss['H0'],
            omk=0,
            YHe=parss['YHe'],
            ns=parss['ns'],
            As=np.exp(parss['logA']) * 1e-10,
            w=parss['w'],
            mnu=parss['mnu'],
            tau=parss['tau']
        )
    except TypeError as e:
        logger.exception(f"TypeError in compute_cl_from_params_dark_energy: {e}")
        for key, value in parss.items():
            logger.debug(f"  {key}: {value} (type: {type(value)})")
        raise
    
    resu = camb.get_results(pars)
    cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    totCL = cls['total']
    TTCl_theory = totCL[2:, 0]
    
    return TTCl_theory


def expand_dict_values(dict1, dict2):
    """Expand scalar values in dict2 to match array lengths in dict1."""
    first_key = next(iter(dict1))
    target_length = len(dict1[first_key])
    
    expanded_dict2 = {
        key: [value] * target_length
        for key, value in dict2.items()
    }
    
    return dict1 | expanded_dict2


# ============================================================================
# Statistics Computation Functions
# ============================================================================

def compute_correlation_and_statistics(D_ell, xvals, intervals):
    """Compute correlation function and statistics from D_ell."""
    Cor = correlation_func(D_ell, xvals)
    C180 = Cor[-1]
    
    results = {
        'Cor': Cor,
        'C180': C180
    }
    
    for a, b in intervals:
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
        
        s12_key = f's12_{theta_upper}_{theta_lower}'
        xiv_key = f'xiv_{theta_upper}_{theta_lower}'
        
        results[s12_key] = S12(D_ell, M)
        results[xiv_key] = xivar(D_ell, a, b)
    
    return results


def process_single_spectrum(D_ell, data_loader, intervals):
    """Process a single recovered D_ell: compute correlation and statistics."""
    stats = compute_correlation_and_statistics(
        D_ell,
        data_loader.xvals,
        intervals
    )
    
    stats['D_ell'] = D_ell
    
    return stats


# ============================================================================
# MCMC Chain Processing
# ============================================================================

def process_mcmc_sample(parss, data_loader, intervals, model_index,
                        pipeline_params):
    """Process a single MCMC sample: compute Cl, run pipeline once, compute stats."""
    lmax = data_loader.lmax
    xvals = data_loader.xvals
    ell = np.arange(2, lmax + 1, dtype=int)
    
    if model_index <= 3:
        D_ell_theory = compute_cl_from_params_planck(parss, lmax)
    else:
        D_ell_theory = compute_cl_from_params_dark_energy(parss, lmax)
    
    D_ell_recovered = planck_style_pipeline_parallel(
        ell=ell,
        D_ell_theory=D_ell_theory.reshape(1, -1),
        nside_in=pipeline_params['nside_in'],
        fwhm_in_arcmin=pipeline_params['fwhm_in_arcmin'],
        noise_Cl=pipeline_params.get('noise_Cl'),
        mask=pipeline_params.get('mask'),
        estimate_Cl=True,
        num_workers=1
    )[0]
    
    stats = compute_correlation_and_statistics(
        D_ell_recovered,
        xvals,
        intervals
    )
    
    stats['D_ell'] = D_ell_recovered
    
    return pd.Series(stats)


# ============================================================================
# Main Processing Function - IMPROVED
# ============================================================================

def chain_results(
    data_loader,
    intervals,
    roots,
    name,
    output_dir=None,
    n=1000,
    best_fit=True,
    pipeline_params=None,
    mode_suffix=None  # NEW: allows explicit control over directory name
):
    """
    Process MCMC chains or best-fit models with Planck-style pipeline.
    
    NEW: Saves to theory_bestfit/ or theory_mcmc/ directories to allow
    'all' mode to save both without overwriting.
    
    Parameters
    ----------
    data_loader : Data_loader
        Instance with experimental data and configuration
    intervals : list of tuples
        Angular intervals (a, b) for statistics
    roots : list of str
        Paths to MCMC chain files (without extension)
    name : str
        Base name for output files (currently unused)
    output_dir : str or None
        Output directory. If None, uses f"run_{lmax}_{n}"
    n : int, default=1000
        Number of realizations (best-fit) or samples to process (MCMC)
    best_fit : bool, default=True
        If True, use best-fit mode; if False, use MCMC mode
    pipeline_params : dict or None
        Pipeline parameters
    mode_suffix : str or None
        Explicit suffix for theory directory (e.g., 'bestfit', 'mcmc')
        If None, auto-determined from best_fit flag
    
    Returns
    -------
    None
        Results are saved to:
        output_dir/
            theory_bestfit/ or theory_mcmc/
                model_name/
                    D_ell.npy, Corr.npy, columns.txt, etc.
    """
    lmax = data_loader.lmax
    #xvals = data_loader.xvals
    #ell = np.arange(2, lmax + 1, dtype=int)
    
    if output_dir is None:
        output_dir = f"run_{lmax}_{n}"
    
    if pipeline_params is None:
        pipeline_params = {
            'nside_in': 2048,
            'fwhm_in_arcmin': 40.0,
            'noise_Cl': None,
            'mask': None
        }
    
    # Determine theory directory suffix
    if mode_suffix is None:
        mode_suffix = 'bestfit' if best_fit else 'mcmc'
    
    # Define column names for results
    chain_cols = ['D_ell', 'Cor', 'C180']
    for a, b in intervals:
        theta_1 = round(np.arccos(a) * 180 / np.pi)
        theta_2 = round(np.arccos(b) * 180 / np.pi)
        theta_upper = max(theta_1, theta_2)
        theta_lower = min(theta_1, theta_2)
        chain_cols += [
            f's12_{theta_upper}_{theta_lower}',
            f'xiv_{theta_upper}_{theta_lower}'
        ]
    
    # Process each root
    for i, root in enumerate(roots):
        short_name = root.strip().split('/')[-1]
        logger.info(f"Processing: {short_name} (mode: {mode_suffix})")
        
        try:
            if best_fit:
                # ============================================================
                # BEST-FIT MODE
                # ============================================================
                logger.info(f"Best-fit mode: Running pipeline {n} times")
                
                try:
                    powers = pd.read_csv(
                        root + '.minimum.theory_cl',
                        sep=r'\s+',
                        #comment='#'
                    )
                except FileNotFoundError:
                    logger.error(
                        f"Best-fit file not found: {root}.minimum.theory_cl"
                    )
                    continue
                
                mask = (powers['#'] >= 2) & (powers['#'] <= lmax)
                powers = powers[mask].sort_values('#')
                ell_theory = powers['#'].values.astype(int)
                D_ell_theory = powers['L'].values
                
                expected_ell = data_loader.ell
                if not np.array_equal(ell_theory, expected_ell):
                    raise ValueError(
                        f"Best-fit file does not contain ell=2..{lmax}. "
                        f"Found: {ell_theory}"
                    )
                
                logger.info(f"Loaded theoretical spectrum: {len(D_ell_theory)} multipoles")
                
                D_ell_theory_stack = np.tile(D_ell_theory, (n, 1))
                
                logger.info(f"Running pipeline on {n} realizations...")
                
                D_ell_recovered_all = planck_style_pipeline_parallel(
                    ell=expected_ell,
                    D_ell_theory=D_ell_theory_stack,
                    nside_in=pipeline_params['nside_in'],
                    fwhm_in_arcmin=pipeline_params['fwhm_in_arcmin'],
                    noise_Cl=pipeline_params.get('noise_Cl'),
                    mask=pipeline_params.get('mask'),
                    estimate_Cl=True,
                    num_workers=None
                )
                
                logger.info("Pipeline complete. Computing statistics...")
                
                results_list = []
                for j in range(n):
                    stats = process_single_spectrum(
                        D_ell_recovered_all[j],
                        data_loader,
                        intervals
                    )
                    results_list.append(stats)
                
                df = pd.DataFrame(results_list)
            
            else:
                # ============================================================
                # MCMC MODE
                # ============================================================
                logger.info(f"MCMC mode: Processing {n} samples")
                
                try:
                    samples = loadMCSamples(file_root=root)
                except Exception as e:
                    logger.error(f"Failed to load MCMC samples: {e}")
                    continue
                
                params = samples.getParams()
                fixed = samples.ranges.fixedValueDict()
                chain_data = {
                    p.name: getattr(params, p.name, np.nan)
                    for p in samples.paramNames.names
                }
                
                data = expand_dict_values(chain_data, fixed)
                df_params = pd.DataFrame.from_dict(data).tail(n)
                weights = samples.weights[-len(df_params):]
                loglikes = samples.loglikes[-len(df_params):]
                logger.info(f"Processing {len(df_params)} MCMC samples...")
                
                df_results = df_params.apply(
                    process_mcmc_sample,
                    axis=1,
                    args=(data_loader, intervals, i, pipeline_params)
                )
                
                df = pd.concat([df_params, df_results], axis=1)
            
            # ================================================================
            # Save Results - IMPROVED DIRECTORY STRUCTURE
            # ================================================================
            
            stacked_Cl = np.vstack(df['D_ell'].values)
            stacked_Cor = np.vstack(df['Cor'].values)
            
            mean_Cl = stacked_Cl.mean(axis=0)
            std_Cl = stacked_Cl.std(axis=0, ddof=0)
            #mean_Cor = stacked_Cor.mean(axis=0)
            #std_Cor = stacked_Cor.std(axis=0, ddof=0)
            
            logger.info(f"Mean D_ell: {mean_Cl[:5]} ...")
            logger.info(f"Std D_ell: {std_Cl[:5]} ...")
            
            # NEW: Save to theory_{mode_suffix}/model_name/
            model_dir = os.path.join(output_dir, f'theory_{mode_suffix}', short_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save arrays
            np.save(os.path.join(model_dir, 'D_ell.npy'), stacked_Cl)
            np.save(os.path.join(model_dir, 'Corr.npy'), stacked_Cor)
            
            logger.info(f"Saved D_ell and Corr to {model_dir}")
            
            # Save scalar statistics
            scalar_cols = [c for c in chain_cols if c not in ['D_ell', 'Cor']]
            
            with open(os.path.join(model_dir, 'columns.txt'), 'w') as cf:
                cf.write('\n'.join(scalar_cols))
            
            # Save S12 statistics
            s_cols = [c for c in scalar_cols if c.startswith('s12_')]
            if s_cols:
                S_mat = np.vstack([df[c].values for c in s_cols]).T
                np.savetxt(os.path.join(model_dir, 'S_statistics.txt'), S_mat)
            
            # Save xivar statistics
            xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
            if xiv_cols:
                xiv_mat = np.vstack([df[c].values for c in xiv_cols]).T
                np.savetxt(os.path.join(model_dir, 'xiv_statistic.txt'), xiv_mat)
            
            # Save C180
            if 'C180' in scalar_cols:
                C180_vec = df['C180'].values
                np.savetxt(os.path.join(model_dir, 'C180_statistics.txt'), C180_vec)
            
            # NEW: Save chain file for getdist (for MCMC mode)
            if not best_fit and isinstance(df, pd.DataFrame):
                # Save samples in getdist format
                # This allows getdist to compute proper posteriors with weights
                chain_file = os.path.join(model_dir, f'{short_name}_chain.txt')
                
                # Select only the derived statistics columns for the chain file
                derived_cols = [
            c for c in df.columns if c not in ["D_ell", "Cor"]
        ]

                # IMPORTANT: GetDist .txt files REQUIRE [weight, -loglike, params...]
                # We create a new array with weights and loglikes at the front
                getdist_data = np.column_stack(
                    [weights, loglikes, df[derived_cols].values]
                )
                
                np.savetxt(chain_file, getdist_data, fmt="%.10e")
                # Create .paramnames file for getdist
                paramnames_file = os.path.join(model_dir, f'{short_name}_chain.paramnames')
                with open(paramnames_file, 'w') as pf:
                    for col in derived_cols:
                        # Format: param_name    param_label
                        label = col.replace('_', '\\_')
                        pf.write(f"{col}    ${label}$\n")

                ranges_file = os.path.join(model_dir, f"{short_name}.ranges")

                # If the original samples had ranges, we propagate them
                if samples.ranges:
                    # This saves the min/max for all parameters in the new short_name.ranges
                    samples.ranges.saveToFile(ranges_file)
                    logger.info(f"Propagated ranges to {ranges_file}")
                else:
                    logger.warning("No ranges found in original samples; skipping ranges file.")
                
                logger.info(f"Saved chain file for getdist: {chain_file}")
            
            logger.info(f"✓ Completed processing for {short_name}")
        
        except Exception as e:
            logger.exception(f"Error processing {root}: {e}")
            continue
    
    logger.info(f"All processing complete. Results saved to: {output_dir}/theory_{mode_suffix}/")