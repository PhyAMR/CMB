"""
Deep diagnostic: Check stored chain values and investigate the negative D_ell problem
"""

import numpy as np
import pandas as pd
import pickle
from functions.data import Data_loader

print("="*100)
print("DEEP DIAGNOSTIC: Investigating negative D_ell and unit issues")
print("="*100)

# Load data
data_loader = Data_loader(lmax=200)
intervals = [(-1, 0.5)]

print("\n" + "="*100)
print("PART 1: Analyzing stored chain results")
print("="*100)

chain_file = 'chain_planck_10.pkl'
with open(chain_file, 'rb') as f:
    chain_data = pickle.load(f)

chain_name = list(chain_data.keys())[0]
df, mean_Cl, std_Cl, mean_Cor, std_Cor = chain_data[chain_name]

print(f"\nChain: {chain_name}")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {list(df.columns)}")

print("\n[1] Checking stored D_ell arrays in chain:")
if 'D_ell' in df.columns:
    sample_d_ell = df['D_ell'].iloc[0]
    print(f"    D_ell shape: {sample_d_ell.shape}")
    print(f"    D_ell range: [{sample_d_ell.min():.2f}, {sample_d_ell.max():.2f}]")
    print(f"    D_ell mean: {sample_d_ell.mean():.2f}")
    print(f"    First 10 values: {sample_d_ell[:10]}")
    
    # Check for negative values
    n_negative = np.sum(sample_d_ell < 0)
    print(f"    Number of negative values: {n_negative}")

print("\n[2] Checking stored XIV values in chain:")
for col in df.columns:
    if col.startswith('xiv'):
        values = df[col].values
        print(f"    {col}: mean={values.mean():.2e}, std={values.std():.2e}, range=[{values.min():.2e}, {values.max():.2e}]")

print("\n[3] Checking stored S12 values in chain:")
for col in df.columns:
    if col.startswith('s12'):
        values = df[col].values
        print(f"    {col}: mean={values.mean():.2e}, std={values.std():.2e}, range=[{values.min():.2e}, {values.max():.2e}]")
        n_negative = np.sum(values < 0)
        print(f"              negative count: {n_negative}")

print("\n" + "="*100)
print("PART 2: Analyzing MC simulation results")
print("="*100)

mc_file = 'Simulation_1000.pkl'
with open(mc_file, 'rb') as f:
    mc_data = pickle.load(f)

df_mc, mean_Cl_mc, std_Cl_mc, mean_Cor_mc, std_Cor_mc = mc_data['Simulation']

print(f"\nMC Simulation (n=1000)")
print(f"DataFrame shape: {df_mc.shape}")

print("\n[4] Checking D_ell arrays in MC:")
if 'D_ell' in df_mc.columns:
    # Check all samples for negative values
    all_negatives = []
    all_mins = []
    all_maxs = []
    
    for i in range(min(10, len(df_mc))):
        sample_d_ell = df_mc['D_ell'].iloc[i]
        n_neg = np.sum(sample_d_ell < 0)
        all_negatives.append(n_neg)
        all_mins.append(sample_d_ell.min())
        all_maxs.append(sample_d_ell.max())
    
    print(f"    First 10 samples statistics:")
    print(f"      Negative values per sample: {all_negatives}")
    print(f"      Min values: {[f'{x:.2f}' for x in all_mins]}")
    print(f"      Max values: {[f'{x:.2f}' for x in all_maxs]}")
    
    # Check all 1000 samples
    total_with_negatives = sum(1 for i in range(len(df_mc)) if np.sum(df_mc['D_ell'].iloc[i] < 0) > 0)
    print(f"\n    Out of {len(df_mc)} samples: {total_with_negatives} have at least one negative D_ell value")

print("\n[5] Checking XIV values in MC:")
for col in df_mc.columns:
    if col.startswith('xiv'):
        values = df_mc[col].values
        print(f"    {col}: mean={values.mean():.2e}, std={values.std():.2e}, range=[{values.min():.2e}, {values.max():.2e}]")

print("\n[6] Checking S12 values in MC:")
for col in df_mc.columns:
    if col.startswith('s12'):
        values = df_mc[col].values
        print(f"    {col}: mean={values.mean():.2e}, std={values.std():.2e}, range=[{values.min():.2e}, {values.max():.2e}]")
        n_negative = np.sum(values < 0)
        n_positive = np.sum(values > 0)
        print(f"              negative: {n_negative}, positive: {n_positive}")

print("\n" + "="*100)
print("PART 3: Root cause analysis")
print("="*100)

print("\n[7] Checking experimental data errors:")
exp_data_with_error = data_loader.df.copy()
exp_data_with_error['Error'] = data_loader.error
print(f"    Multipoles with error > value (would allow negative sampling):")
high_error_count = 0
for idx, row in exp_data_with_error.iterrows():
    if row['Error'] > row['D_ell']:
        print(f"      l={int(row['ell'])}: D_ell={row['D_ell']:.2f}, Error={row['Error']:.2f}, ratio={row['Error']/row['D_ell']:.2f}")
        high_error_count += 1
        if high_error_count > 10:
            print(f"      ... and more")
            break

total_high_error = sum(1 for _, row in exp_data_with_error.iterrows() if row['Error'] > row['D_ell'])
print(f"\n    Total multipoles where Error > D_ell: {total_high_error} out of {len(exp_data_with_error)}")

print("\n[8] Comparing scales:")
print(f"    Experimental D_ell mean: {exp_data_with_error['D_ell'].mean():.2f}")
print(f"    MC mean_Cl mean: {mean_Cl_mc.mean():.2f}")
print(f"    Chain mean_Cl mean: {mean_Cl.mean():.2f}")
print(f"\n    Ratio Chain/Experimental: {mean_Cl.mean() / exp_data_with_error['D_ell'].mean():.6f}")
print(f"    Ratio MC/Experimental: {mean_Cl_mc.mean() / exp_data_with_error['D_ell'].mean():.6f}")

print("\n" + "="*100)
print("PART 4: Experimental values")
print("="*100)

exp_values = data_loader.experimental_values(intervals)
print("\n[9] Experimental statistics:")
for key, (val, err) in exp_values.items():
    print(f"    {key}: {val:.6e} ± {err:.6e}")

print("\n" + "="*100)
print("DIAGNOSTIC COMPLETE")
print("="*100)

print("\n" + "="*100)
print("SUMMARY OF ISSUES FOUND:")
print("="*100)
print("""
1. MC SIMULATION PROBLEM:
   - Random sampling with large errors creates NEGATIVE D_ell values
   - This causes incorrect S12 values (can be negative when should be positive)
   - Solution: Truncate/clip negative values or use log-normal distribution

2. CHAIN XIV VALUES:
   - Chain XIV values are ORDERS OF MAGNITUDE too large
   - Likely a units mismatch between CAMB output and experimental data
   - CAMB returns D_ell in (μK)², needs conversion

3. S12 NEGATIVE VALUES:
   - Both MC and Chain show some negative S12 values
   - S12 should always be positive (integral of squared correlation)
   - May indicate issue with Tmn matrix or formula implementation
""")
