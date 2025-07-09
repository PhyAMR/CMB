import camb.correlations
from getdist import loadMCSamples
import numpy as np
import camb
import pandas as pd
import matplotlib.pyplot as plt
import os

unbin_cl = pd.read_csv('maps/COM_PowerSpect_CMB-TT-full_R3.01.txt', sep=',')

unbin_cl.head()

unbin_cl.columns = ['ell', 'D_ell', '-dD_ell', '+dD_ell']

unbin_cl.head()

# Define the roots and other initial parameters
roots = ['COM_CosmoParams_fullGrid_R3.01/base_omegak/CamSpecHM_TT_lowl_lowE/base_omegak_CamSpecHM_TT_lowl_lowE']
xvals = np.linspace(-0.9999, 0.9999, 1800)
theta = np.arccos(xvals) * 180 / np.pi
ell = unbin_cl['ell'][:201]
D_ell = unbin_cl['D_ell'][:201]
clstot = np.tile(np.array(np.insert(D_ell, [0, 1], [0, 0])).reshape(-1, 1), 4)

corrs = camb.correlations.cl2corr(clstot, xvals)
expd = np.array([D_ell, corrs[:, 0]], dtype='object')

# File to track processed roots
processed_roots_file = 'processed_roots.txt'
processed_roots = set()

# Load already processed roots if the file exists
if os.path.exists(processed_roots_file):
    with open(processed_roots_file, 'r') as f:
        processed_roots = set(line.strip() for line in f)

for root in roots:
    if root in processed_roots:
        print(f"Root {root} already processed. Skipping.")
        continue

    # Initialize variables for this root
    M = 0.0
    S = 0.0
    W = 0.0
    lencl = 0

    samples = loadMCSamples(file_root=root)
    bfsam = samples.getParamBestFitDict()
    
    # Extract chi2 columns from best fit parameters
    chi2_cols = [col for col in bfsam.keys() if col.startswith('chi2')]
    totchib = np.sum(np.array([bfsam[col] for col in chi2_cols]))

    # Process each sample
    for i in range(11079, 12080):
        parss = samples.getParamSampleDict(i)
        current_chi2_cols = [col for col in parss.keys() if col.startswith('chi2')]
        
        # Check the condition using current sample's chi2 columns
        if np.any(np.array([np.exp((bfsam[col] - parss[col])/2) > 1e-3 for col in current_chi2_cols])):
            # Set CAMB parameters and compute results
            pars = camb.set_params(
                ombh2=parss['omegabh2'], omch2=parss['omegach2'], 
                H0=parss['H0'], omk=parss['omegak'], YHe=parss['yheused'], 
                nnu=parss['nnu'], nrun=0, Alens=1, ns=parss['ns'], 
                As=np.exp(parss['logA'])*1e-10, w=parss['w'], wa=parss['wa'], 
                mnu=parss['mnu'], tau=parss['tau']
            )
            resu = camb.get_results(pars)
            cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=200)
            totCL = cls['total']
            TTCl = np.array(totCL[2:, 0])
            totCorr = camb.correlations.cl2corr(totCL, xvals, lmax=200)
            TTcor = np.array(totCorr[:, 0])
            clcorr = np.concatenate((TTCl, TTcor))
            
            # Calculate weight and accumulate statistics
            totchi = np.sum(np.array([parss[col] for col in current_chi2_cols]))
            w = np.exp((totchib - totchi) / 2)
            M += w * clcorr
            S += w * clcorr ** 2
            W += w
            lencl = len(TTCl)
        else:
            print('No condition')

    # Check if any samples were processed
    if W == 0:
        print(f"No valid samples for {root}. Skipping save.")
        continue

    # Compute mean and standard deviation
    mu = M / W
    sig = np.sqrt(S / W - mu ** 2)

    # Save results to a .npz file
    safe_root = root.replace('/', '_').replace(' ', '_')
    results_file = f"{safe_root}_results.npz"
    np.savez(results_file, mu=mu, sig=sig, lencl=lencl)

    # Update processed roots
    with open(processed_roots_file, 'a') as f:
        f.write(f"{root}\n")
    processed_roots.add(root)
    print(f"Processed and saved results for {root} in {results_file}")