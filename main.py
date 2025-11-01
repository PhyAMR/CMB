# Math
import numpy as np
from math import factorial
from decimal import getcontext, Decimal
from scipy.integrate import simpson

import pandas as pd
# Cosmology
from astropy.io import fits
import healpy as hp
import camb
import camb.correlations
from getdist import loadMCSamples

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# import scienceplots 
#plt.style.use(['science','ieee'])
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'text.usetex': False  # Set True if LaTeX installed
})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from matplotlib.patches import Rectangle

# System
import pickle
import os

# Functions
from functions.tools import legendre, P, timeit, A_r
from functions.correlation_function import correlation_func, correlation_func_err, correlation_func_err2
from functions.xiv import xivar, xivar2, xivar_num, xivar_err, xivar_err2
from functions.s12 import Tmn, S12, S12_vec, S12_err, S12_err2
from functions.maps import map_rot_refl, estimate_coef
from functions.cosmology import compute_cl_cor_pl, compute_cl_cor_dv, expand_dict_values, chain_calculations, chain_results
from functions.plots import MapPlots, CorrelationPlots
from functions.simulation import MC_calculations, MC_results

from functions.data import Data_loader

def get_user_choice():
    """Presents a menu to the user and returns their choice."""
    while True:
        print("\nPlease select an option:")
        print("1. Run chain_results")
        print("2. Run MC_results")
        print("3. Run both")
        print("4. None (only plot existing results)")
        choice = input("Enter your choice (1-4): ")
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def swap_prefix(col):
    if col.startswith("xiv_"):
        return col.replace("xiv_", "s12_", 1)
    elif col.startswith("s12_"):
        return col.replace("s12_", "xiv_", 1)
    return col  # unchanged if it doesn't start with either prefix



def main():
    # Load all the data and variables
    DL = Data_loader(lmax=200)

    intervals = [(-0.9999999999999999, 0.5), (0.866, 0.9999999999999999), (0.5, 0.866), (0, 0.5), (-0.5, 0), (-0.866, -0.5), (-0.9999999999999999, -0.866)]

    roots_planck = ['Planck_Data/base_omegak/CamSpecHM_TT_lowl_lowE/base_omegak_CamSpecHM_TT_lowl_lowE',
            'Planck_Data/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE',
             'Planck_Data/base/CamSpecHM_TT_lowl_lowE/base_CamSpecHM_TT_lowl_lowE',
             'Planck_Data/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE']

    # Statistics
    exp_values = DL.experimental_values(intervals)

    # Get the desired column order from the experimental values.
    ordered_cols = list(exp_values.keys())

    print("Experimental Values:", exp_values)
    n = 1000#int(input("Select number of samples/realizations:"))
    choice = '4' #get_user_choice()

    if choice == '1':

        print("Running chain_results...")
        chain_results(DL, intervals, roots_planck, f'chain_planck_{n}', n)
        print("chain_results finished.")
    elif choice == '2':
        print("Running MC_results...")
        MC_results(DL, intervals, n=n)
        print("MC_results finished.")
    elif choice == '3':
        print("Running both chain_results and MC_results...")
        chain_results(DL, intervals, roots_planck, f'chain_planck_{n}', n)
        MC_results(DL, intervals, n=n)
        print("Both processes finished.")
    elif choice == '4':
        print("Skipping calculations. Will attempt to plot existing results.")

    # Load the simulation results from the pickle file.
    sim_data_dict = None
    if os.path.exists(f'Simulation_{n}.pkl'):
        print(f"Loading simulation results from Simulation_{n}.pkl...")
        with open(f'Simulation_{n}.pkl', 'rb') as f:
            sim_data_dict = pickle.load(f)
    else:
        print(f"Simulation_{n}.pkl not found. Skipping plots that require it.")

    # Plotting
    output_dir = "images/script_outputs"
    # Set to 'png', 'tex', or None to display plots
    save_format = 'png' 

    if save_format and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load chain results
    data_dict = None
    if os.path.exists(f'chain_planck_{n}.pkl'):
        print(f"Loading chain results from chain_planck_{n}.pkl...")
        with open(f'chain_planck_{n}.pkl', 'rb') as f1:
            data_dict = pickle.load(f1)
    else:
        print(f"chain_planck_{n}.pkl not found. Skipping plots that require it.")

    if not data_dict and not sim_data_dict:
        print("No data found to plot. Exiting.")
        return

    # Instantiate plotter with Data_loader object
    CP = CorrelationPlots(DL)

    # Prepare simulation data
    if sim_data_dict:
        df_simu, cl_simu, _, _, _ = sim_data_dict['Simulation']
        # Prepare simulation data for histogram plotting by removing non-scalar columns
        df_simu_hist = df_simu.drop(columns=['D_ell', 'Cor'], errors='ignore')
        # Reorder the columns to match the experimental data
        df_simu_hist = df_simu_hist[ordered_cols]
        #print(df_simu_hist.head())

    # Loop through chain results and plot
    if data_dict:
        for root_name, (df, mean_Cl, std_Cl, mean_Cor, std_Cor) in data_dict.items():
            # Apply renaming
            print("Before swapping the columns")
            print(df.head())
            df_fixed = df.copy() #rename(columns=swap_prefix) #df_fixed is a work around because chain results is swapping xiv and s12 values
            print("After swapping the columns")
            print(df_fixed.head())
            est_df = df_fixed.copy()

            xiv_df = df_fixed.filter(regex=r'xiv_')
            s12_df = df_fixed.filter(regex=r's12_')
            est_df.drop(columns=['D_ell', 'Cor'], inplace=True)
            
            # Reorder the columns to match the experimental data
            est_df = est_df[ordered_cols]

            
            # Sanitize root_name for use in filenames
            safe_root_name = root_name.replace('/', '_')

            # Define save paths
            hist_exp_th_path = f"{output_dir}/{safe_root_name}_hist_exp_vs_th.{save_format}" if save_format else None
            hist_exp_sim_path = f"{output_dir}/{safe_root_name}_hist_exp_vs_sim.{save_format}" if save_format else None
            hist_th_sim_path = f"{output_dir}/{safe_root_name}_hist_th_vs_sim.{save_format}" if save_format else None
            xivar_path = f"{output_dir}/{safe_root_name}_xivar.{save_format}" if save_format else None
            s12_path = f"{output_dir}/{safe_root_name}_s12.{save_format}" if save_format else None
            power_corr_path = f"{output_dir}/{safe_root_name}_power_corr.{save_format}" if save_format else None
            
            CP.create_histogram_grid(est_df, ordered_cols, f"Histograms for {root_name} Exp vs Th", comparison_data=exp_values, figsize=(20, 20), bins=100, save_path=hist_exp_th_path)
            
            if sim_data_dict:
                CP.create_histogram_grid(df_simu_hist, ordered_cols, f"Histograms for {root_name} Exp vs Sim", comparison_data=exp_values, figsize=(20, 20), bins=100, save_path=hist_exp_sim_path)    
                CP.create_histogram_grid(df_simu_hist, ordered_cols, f"Histograms for {root_name} Th vs Sim", comparison_data=est_df, figsize=(20, 20), bins=100, save_path=hist_th_sim_path)

            CP.plot_corr_with_xivar(mean_Cor, xiv_df, intervals, root_name, save_path=xivar_path)
            CP.plot_corr_with_S12(mean_Cor, s12_df, intervals, root_name, save_path=s12_path)
            CP.plot_power_and_correlation(mean_Cl, std_Cl,
                                          mean_Cor, std_Cor,
                                          root=root_name, figsize=(18, 7), save_path=power_corr_path)

if __name__ == "__main__":
    main()
