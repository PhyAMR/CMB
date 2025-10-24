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

def main():
    # Load all the data and variables
    DL = Data_loader(lmax=201)

    intervals = [(-0.9999999999999999, 0.5), (0.866, 0.9999999999999999), (0.5, 0.866), (0, 0.5), (-0.5, 0), (-0.866, -0.5), (-0.9999999999999999, -0.866)]

    roots_planck = ['COM_CosmoParams_fullGrid_R3.01/base_omegak/CamSpecHM_TT_lowl_lowE/base_omegak_CamSpecHM_TT_lowl_lowE',
            'COM_CosmoParams_fullGrid_R3.01/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE',
             'COM_CosmoParams_fullGrid_R3.01/base/CamSpecHM_TT_lowl_lowE/base_CamSpecHM_TT_lowl_lowE',
             'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE']

    # Statistics
    raw_exp_values = DL.experimental_values(intervals)
    # Flatten the dictionary to match the format expected by the plotting code
    exp_values = {}
    exp_values['C180'] = raw_exp_values['C(180)']

    for interval_label, (val, err) in raw_exp_values['s12'].items():
        # interval_label is like '30-60'. We need to convert to 's12_60_30'
        lower, upper = interval_label.split('-')
        exp_values[f's12_{upper}_{lower}'] = (val, err)

    for interval_label, (val, err) in raw_exp_values['xivar'].items():
        lower, upper = interval_label.split('-')
        exp_values[f'xiv_{upper}_{lower}'] = (val, err)

    print("Experimental Values:", exp_values)

    # Montecarlo Simulation
    # The MC_results function expects a DataFrame with an 'Error' column.
    # The Data_loader class stores the error as an attribute, so we add it to the DataFrame.
    sim_df = DL.df.copy()
    sim_df['Error'] = DL.error

    # Run the Monte Carlo simulation with 100 realizations.
    # MC_results(intervals, sim_df, DL.xvals, n=100)

    # Load the simulation results from the pickle file.
    with open('files/pickel/Simulation_1000.pkl', 'rb') as f:
        sim_data_dict = pickle.load(f)

    # Plotting
    output_dir = "images/script_outputs"
    # Set to 'png', 'tex', or None to display plots
    save_format = 'png' 

    if save_format and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load chain results
    with open('files/pickel/chain_results_planck.pkl', 'rb') as f1:
        data_dict = pickle.load(f1)

    # Instantiate plotter with Data_loader object
    CP = CorrelationPlots(DL)

    # Prepare simulation data
    df_simu, cl_simu, _, _, _ = sim_data_dict['Simulation']
    # Prepare simulation data for histogram plotting by removing non-scalar columns
    df_simu_hist = df_simu.drop(columns=['D_ell', 'Cor'], errors='ignore')

    # Loop through chain results and plot
    for root_name, (df, mean_Cl, std_Cl, mean_Cor, std_Cor) in data_dict.items():
        xiv_df = df.filter(regex=r'xiv_')
        s12_df = df.filter(regex=r's12_')
        est_df = df.copy()
        est_df.drop(columns=['D_ell', 'Cor'], inplace=True)
        
        # Sanitize root_name for use in filenames
        safe_root_name = root_name.replace('/', '_')

        # Define save paths
        hist_exp_th_path = f"{output_dir}/{safe_root_name}_hist_exp_vs_th.{save_format}" if save_format else None
        hist_exp_sim_path = f"{output_dir}/{safe_root_name}_hist_exp_vs_sim.{save_format}" if save_format else None
        hist_th_sim_path = f"{output_dir}/{safe_root_name}_hist_th_vs_sim.{save_format}" if save_format else None
        xivar_path = f"{output_dir}/{safe_root_name}_xivar.{save_format}" if save_format else None
        s12_path = f"{output_dir}/{safe_root_name}_s12.{save_format}" if save_format else None
        power_corr_path = f"{output_dir}/{safe_root_name}_power_corr.{save_format}" if save_format else None
        
        CP.create_histogram_grid(est_df, est_df.columns, f"Histograms for {root_name} Exp vs Th", comparison_data=exp_values, figsize=(15, 15), bins=100, save_path=hist_exp_th_path)
        CP.create_histogram_grid(df_simu_hist, df_simu_hist.columns, f"Histograms for {root_name} Exp vs Sim", comparison_data=exp_values, figsize=(15, 15), bins=100, save_path=hist_exp_sim_path)    
        CP.create_histogram_grid(df_simu_hist, df_simu_hist.columns, f"Histograms for {root_name} Th vs Sim", comparison_data=est_df, figsize=(15, 15), bins=100, save_path=hist_th_sim_path)

        CP.plot_corr_with_xivar(mean_Cor, xiv_df, intervals, root_name, save_path=xivar_path)
        CP.plot_corr_with_S12(mean_Cor, s12_df, intervals, root_name, save_path=s12_path)
        CP.plot_power_and_correlation(mean_Cl, std_Cl,
                                      mean_Cor, std_Cor,
                                      root=root_name, figsize=(18, 7), save_path=power_corr_path)

if __name__ == "__main__":
    main()
