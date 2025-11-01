"""
Diagnostic script to trace through MC_results and chain_calculations step by step
to identify where values might be going wrong.
"""

import numpy as np
import pandas as pd
import pickle
import sys
from functions.data import Data_loader
from functions.correlation_function import correlation_func
from functions.s12 import S12
from functions.xiv import xivar
import camb
import camb.correlations

print("="*100)
print("DIAGNOSTIC SCRIPT: Tracing MC_results and chain_calculations")
print("="*100)

# Configuration
intervals = [(-1, 0.5)]  # Just one interval for simplicity
n_simulations = 3  # Small number for debugging

print("\n" + "="*100)
print("PART 1: MC_RESULTS DIAGNOSTIC")
print("="*100)

# Initialize data loader
data_loader = Data_loader(lmax=200, n_xvals=1800)
print("\n[1] Data Loader initialized")
print(f"    lmax = {data_loader.lmax}")
print(f"    xvals shape = {data_loader.xvals.shape}")
print(f"    xvals range = [{data_loader.xvals.min():.6f}, {data_loader.xvals.max():.6f}]")

# Get the experimental data
data = data_loader.df.copy()
data['Error'] = data_loader.error

print("\n[2] Experimental data loaded")
print(f"    DataFrame shape: {data.shape}")
print(f"    Columns: {list(data.columns)}")
print(f"    First few rows:")
print(data.head())
print(f"\n    D_ell statistics:")
print(f"      Min: {data['D_ell'].min():.2f}")
print(f"      Max: {data['D_ell'].max():.2f}")
print(f"      Mean: {data['D_ell'].mean():.2f}")

# Generate random realizations
print("\n[3] Generating random realizations...")
data['dist_per_cl'] = data.apply(lambda row: np.random.normal(loc=row['D_ell'], scale=row['Error'], size=n_simulations), axis=1)
distributions = data['dist_per_cl'].to_list()

print(f"    Number of multipoles: {len(distributions)}")
print(f"    Samples per multipole: {len(distributions[0])}")
print(f"    First multipole (l=2) samples: {distributions[0]}")

# Transpose to get realizations
print("\n[4] Transposing distributions...")
trasp = list(map(list, zip(*distributions)))
print(f"    Number of realizations: {len(trasp)}")
print(f"    Length of each realization: {len(trasp[0])}")
print(f"    First realization (first 5 values): {trasp[0][:5]}")

# Create DataFrame with arrays
df_arrays = pd.DataFrame({'valores': [np.array(row) for row in trasp]})
print("\n[5] Created df_arrays DataFrame")
print(f"    Shape: {df_arrays.shape}")
print(f"    Column: {df_arrays.columns.tolist()}")
print(f"    First row type: {type(df_arrays['valores'].iloc[0])}")
print(f"    First row shape: {df_arrays['valores'].iloc[0].shape}")
print(f"    First row first 5 values: {df_arrays['valores'].iloc[0][:5]}")

# Now manually process ONE realization to see what happens
print("\n[6] Processing first realization manually...")
first_realization = df_arrays.iloc[0]
print(f"    Type of first_realization: {type(first_realization)}")
print(f"    first_realization.index: {first_realization.index.tolist()}")

# This simulates what MC_calculations receives
print("\n[7] Extracting TTCl (simulating MC_calculations line 30)...")
print(f"    first_realization.iloc[0] type: {type(first_realization.iloc[0])}")
print(f"    first_realization.iloc[0] shape: {first_realization.iloc[0].shape}")
print(f"    first_realization['valores'] type: {type(first_realization['valores'])}")
print(f"    first_realization['valores'] shape: {first_realization['valores'].shape}")

# The FIXED version
TTCl = first_realization.iloc[0][:data_loader.lmax]
print(f"\n    Using iloc[0]:")
print(f"      TTCl shape: {TTCl.shape}")
print(f"      TTCl first 5 values: {TTCl[:5]}")
print(f"      TTCl min: {TTCl.min():.2f}, max: {TTCl.max():.2f}")

# Calculate correlation function
print("\n[8] Calculating correlation function...")
TTcor = correlation_func(TTCl, data_loader.xvals)
print(f"    TTcor shape: {TTcor.shape}")
print(f"    TTcor first 5 values: {TTcor[:5]}")
print(f"    TTcor last 5 values: {TTcor[-5:]}")
print(f"    C180 (TTcor[-1]): {TTcor[-1]:.6e}")

# Calculate S12 and xivar for the interval
print("\n[9] Calculating S12 and xivar...")
for a, b in intervals:
    theta_1 = round(np.arccos(a) * 180 / np.pi)
    theta_2 = round(np.arccos(b) * 180 / np.pi)
    theta_upper = max(theta_1, theta_2)
    theta_lower = min(theta_1, theta_2)
    print(f"    Interval: cos(theta) ∈ [{a}, {b}] → theta ∈ [{theta_lower}°, {theta_upper}°]")
    
    matrix_path = f"files/matrix/Tmn__{theta_upper}__{theta_lower}.npy"
    try:
        M = np.load(matrix_path)
        print(f"    Matrix M loaded: shape {M.shape}")
        print(f"    Matrix M statistics: min={M.min():.6e}, max={M.max():.6e}, mean={M.mean():.6e}")
        
        s12 = S12(TTCl, M)
        print(f"    S12 value: {s12:.6e}")
        
        xiv = xivar(TTCl, a, b)
        print(f"    XIV value: {xiv:.6e}")
    except FileNotFoundError:
        print(f"    Matrix file not found: {matrix_path}")

print("\n" + "="*100)
print("PART 2: CHAIN_CALCULATIONS DIAGNOSTIC")
print("="*100)

# Load chain data to get one set of parameters
print("\n[10] Loading chain data...")
chain_file = 'chain_planck_10.pkl'
try:
    with open(chain_file, 'rb') as f:
        chain_data = pickle.load(f)
    
    chain_name = list(chain_data.keys())[0]
    print(f"    Chain loaded: {chain_name}")
    
    df_chain_full = chain_data[chain_name][0]
    print(f"    Chain DataFrame shape: {df_chain_full.shape}")
    print(f"    Chain columns: {list(df_chain_full.columns)}")
    
    # Get first row of parameters
    first_params = df_chain_full.iloc[0]
    print(f"\n[11] First parameter set:")
    param_cols = [col for col in first_params.index if not col.startswith(('D_ell', 'Cor', 'C180', 's12', 'xiv'))]
    for col in param_cols[:10]:  # Show first 10 parameters
        print(f"    {col}: {first_params[col]}")
    
    # Simulate compute_cl_cor_pl
    print("\n[12] Computing power spectrum with CAMB...")
    parss = first_params
    lmax = data_loader.lmax
    xvals = data_loader.xvals
    
    # Check if we have the required parameters
    required_params = ['omegabh2', 'omegach2', 'H0', 'omegak', 'yheused', 'nnu', 'nrun', 'Alens', 'ns', 'logA', 'wa', 'mnu', 'tau']
    print(f"    Checking for required parameters:")
    for param in required_params:
        if param in parss.index:
            print(f"      {param}: {parss[param]}")
        else:
            print(f"      {param}: NOT FOUND")
    
    # Compute with CAMB
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
            w=-1, 
            wa=parss['wa'],
            mnu=parss['mnu'], 
            tau=parss['tau']
        )
        
        resu = camb.get_results(pars)
        cls = resu.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
        totCL = cls['total']
        
        print(f"\n[13] CAMB results:")
        print(f"    totCL shape: {totCL.shape}")
        print(f"    totCL[0:5, 0] (l=0 to 4): {totCL[0:5, 0]}")
        
        TTCl_chain = totCL[2:, 0]
        print(f"\n    TTCl_chain (from l=2) shape: {TTCl_chain.shape}")
        print(f"    TTCl_chain first 5 values: {TTCl_chain[:5]}")
        print(f"    TTCl_chain statistics:")
        print(f"      Min: {TTCl_chain.min():.2f}")
        print(f"      Max: {TTCl_chain.max():.2f}")
        print(f"      Mean: {TTCl_chain.mean():.2f}")
        
        # Compute correlation function from CAMB
        totCorr = camb.correlations.cl2corr(totCL, xvals, lmax)
        TTcor_chain = totCorr[:, 0]
        
        print(f"\n[14] CAMB correlation function:")
        print(f"    TTcor_chain shape: {TTcor_chain.shape}")
        print(f"    TTcor_chain first 5 values: {TTcor_chain[:5]}")
        print(f"    TTcor_chain last 5 values: {TTcor_chain[-5:]}")
        print(f"    C180 (TTcor_chain[-1]): {TTcor_chain[-1]:.6e}")
        
        # Calculate S12 and xivar for chain
        print("\n[15] Calculating S12 and xivar for chain...")
        for a, b in intervals:
            theta_1 = round(np.arccos(a) * 180 / np.pi)
            theta_2 = round(np.arccos(b) * 180 / np.pi)
            theta_upper = max(theta_1, theta_2)
            theta_lower = min(theta_1, theta_2)
            
            matrix_path = f"files/matrix/Tmn__{theta_upper}__{theta_lower}.npy"
            try:
                M = np.load(matrix_path)
                
                s12_chain = S12(TTCl_chain, M)
                print(f"    S12 (chain) value: {s12_chain:.6e}")
                
                xiv_chain = xivar(TTCl_chain, a, b)
                print(f"    XIV (chain) value: {xiv_chain:.6e}")
            except FileNotFoundError:
                print(f"    Matrix file not found: {matrix_path}")
        
    except Exception as e:
        print(f"    Error computing with CAMB: {e}")
        import traceback
        traceback.print_exc()
    
except FileNotFoundError:
    print(f"    Chain file not found: {chain_file}")
except Exception as e:
    print(f"    Error loading chain: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*100)
print("PART 3: COMPARISON WITH EXPERIMENTAL VALUES")
print("="*100)

print("\n[16] Getting experimental values...")
exp_values = data_loader.experimental_values(intervals)

for key, (val, err) in exp_values.items():
    print(f"    {key}: {val:.6e} ± {err:.6e}")

print("\n" + "="*100)
print("PART 4: UNIT CHECK")
print("="*100)

print("\n[17] Checking units consistency...")
print(f"    Experimental D_ell range: [{data_loader.D_ell.min():.2f}, {data_loader.D_ell.max():.2f}]")
print(f"    MC D_ell range: [{TTCl.min():.2f}, {TTCl.max():.2f}]")
if 'TTCl_chain' in locals():
    print(f"    Chain D_ell range: [{TTCl_chain.min():.2f}, {TTCl_chain.max():.2f}]")
    print(f"\n    Ratio (Chain/Experimental) D_ell mean: {TTCl_chain.mean() / data_loader.D_ell.mean():.6f}")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)
