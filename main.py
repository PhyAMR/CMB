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
from functions.plots import plot_corr_with_xivar, plot_corr_with_S12, create_histogram_grid, plot_power_and_correlation, create_histogram_grid2, plot180, map_contours
from functions.simulation import MC_calculations, MC_results

from functions.data import Data_loader

def main():
    DL = Data_loader()

    # Some of the variables are not going to be imported since they can be accessed easily with DL.variable (i.e. data_loader.lmax)
    corr, corr_err = DL.get_correlation_function(force_recalc=True)

    intervals = [(-0.9999999999999999, 0.5), (0.866, 0.9999999999999999), (0.5, 0.866), (0, 0.5), (-0.5, 0), (-0.866, -0.5), (-0.9999999999999999, -0.866)]

    roots_planck = ['COM_CosmoParams_fullGrid_R3.01/base_omegak/CamSpecHM_TT_lowl_lowE/base_omegak_CamSpecHM_TT_lowl_lowE',
            'COM_CosmoParams_fullGrid_R3.01/base_omegak/plikHM_TTTEEE_lowl_lowE/base_omegak_plikHM_TTTEEE_lowl_lowE',
             'COM_CosmoParams_fullGrid_R3.01/base/CamSpecHM_TT_lowl_lowE/base_CamSpecHM_TT_lowl_lowE',
             'COM_CosmoParams_fullGrid_R3.01/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE']

    exp_values = {}
    for i,j in intervals:
        xiv, xiv_err = DL.get_xivar(i,j)
        s12, s12_err = DL.get_s12(j,i)
        exp_values[f"xiv_{round(np.arccos(i)*180/np.pi)}_{round(np.arccos(j)*180/np.pi)}"] = (xiv, xiv_err)
        exp_values[f"s12_{round(np.arccos(i)*180/np.pi)}_{round(np.arccos(j)*180/np.pi)}"] = (s12, s12_err)
    exp_values["C180"] =  (corr[-1], corr_err[-1])
    print(exp_values)

if __name__ == "__main__":
    main()