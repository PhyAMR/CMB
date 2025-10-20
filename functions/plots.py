"""This module provides a collection of functions for creating various plots related to CMB analysis,
including power spectra, correlation functions, statistical distributions, and map comparisons.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import Rectangle
import healpy as hp

from .correlation_function import correlation_func, correlation_func_err2
from .xiv import xivar, xivar_err2
from .s12 import S12, S12_err2
from .maps import estimate_coef

def plot180(map_data, opposite_map, map_name, filename, lower=False):
    """
    Creates a scatter plot of a map against its rotated/reflected version and fits a linear regression.
    This is used to visualize the correlation at 180 degrees.

    Args:
        map_data (np.ndarray): The original HEALPix map.
        opposite_map (np.ndarray): The transformed (opposite) map.
        map_name (str): The name of the map for the plot label.
        filename (str): The base name for the output image file.
        lower (bool): If True, degrades the resolution of the map by a factor of 2.
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'text.usetex': False
    })
    
    if lower:
        # Downgrade the map resolution if requested.
        low_nside = hp.get_nside(map_data) // 2
        map_data = hp.ud_grade(map_data, low_nside)

    # Calculate the product of the maps to estimate C(180).
    mult = map_data * opposite_map
    
    # Estimate the linear regression coefficients.
    a, b = estimate_coef(x=map_data, y=opposite_map)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(map_data, opposite_map, marker='o', c='blue', 
               label=rf'Impainting {map_name} $C(180)={np.mean(mult):.3f}$', s=2, alpha=0.05)
    ax.plot(map_data, b * map_data + a, color='red', label=rf'$y={b:.3f}x + {a:.3f}$')
    ax.set_xlabel(r'T [$K_{CMB}$]')
    ax.set_ylabel(r'Inverted map T [$K_{CMB}$]')
    ax.grid(True, which='both', ls='--', alpha=0.6)
    ax.legend(frameon=True, loc='best')

    plt.tight_layout()
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def map_contours(map_data, opposite_map, filename):
    """
    Creates a 2D histogram (density plot) of a map against its opposite, with contours.

    Args:
        map_data (np.ndarray): The original HEALPix map.
        opposite_map (np.ndarray): The transformed (opposite) map.
        filename (str): The base name for the output image file.
    """
    fig = plt.figure(figsize=(8, 8))

    h, xx, yy, _ = plt.hist2d(map_data, opposite_map, bins=np.linspace(-200, 200, 100))
    x = (xx[1:] + xx[:-1]) / 2
    y = (yy[1:] + yy[:-1]) / 2
    
    plt.colorbar()
    plt.grid(True, which='both', ls='--', alpha=0.6, color='red')
    plt.axis('equal')
    plt.contour(x, y, h, levels=[4000, 6000, 10000], colors='k')
    plt.contour(x, -y, h, levels=[4000, 6000, 10000], colors='w')
    
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_corr_with_xivar(xvals, unbin_cl, corr_th, est_df, intervals, name):
    """
    Plots the correlation function and overlays the xivar statistic for specified intervals.

    Args:
        xvals (np.ndarray): Array of cos(theta) values.
        unbin_cl (dict): Dictionary containing experimental 'D_ell' and 'Error'.
        corr_th (np.ndarray): Theoretical correlation function.
        est_df (pd.DataFrame): DataFrame with theoretical estimates for xivar.
        intervals (list): List of tuples defining the angular intervals.
        name (str): Name for the plot title.
    """
    corr = correlation_func(unbin_cl['D_ell'][:200], xvals)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    ax.scatter(np.arccos(xvals) * 180 / np.pi, corr, s=2, marker='o', c='r', label='Correlation function data')
    ax.scatter(np.arccos(xvals) * 180 / np.pi, corr_th, s=2, marker='o', c='b', label='Correlation function model')

    for (a, b), i in zip(intervals, est_df.columns):
        mean_sq = xivar(unbin_cl['D_ell'][:200], a, b)
        mean_sq_err = xivar_err2(unbin_cl['Error'][:200], a, b)
        
        ax.fill_between([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                        [mean_sq - mean_sq_err], [mean_sq + mean_sq_err], alpha=0.4, color='r')
        ax.plot([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                [mean_sq, mean_sq], ls='-.', alpha=0.8, color='r',
                label=rf'$\xi_{{{round(np.arccos(a)*180/np.pi)}}}^{{{round(np.arccos(b)*180/np.pi)}}}$: {mean_sq:.2f} ± {mean_sq_err:.2f}')
    
        ax.fill_between([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                        [np.mean(est_df[i]) - np.std(est_df[i])], [np.mean(est_df[i]) + np.std(est_df[i])], alpha=0.4, color='b')
        ax.plot([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                [np.mean(est_df[i]), np.mean(est_df[i])], ls='-.', alpha=0.8, color='b',
                label=f'{i}: {np.mean(est_df[i]):.2f} ± {np.std(est_df[i]):.2f}')

    plt.ylim(-400, 400)
    plt.xlabel(r"$\theta$ [º]")
    plt.ylabel(r"$C(\theta)$")
    plt.title(r"$\xi_a^b$ analysis for " + str(name))
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_corr_with_S12(xvals, unbin_cl, cor_th, est_df, intervals, name):
    """
    Plots the squared correlation function and overlays the S12 statistic for specified intervals.

    Args:
        xvals (np.ndarray): Array of cos(theta) values.
        unbin_cl (dict): Dictionary containing experimental 'D_ell' and 'Error'.
        cor_th (np.ndarray): Theoretical correlation function.
        est_df (pd.DataFrame): DataFrame with theoretical estimates for S12.
        intervals (list): List of tuples defining the angular intervals.
        name (str): Name for the plot title.
    """
    corr = correlation_func(unbin_cl['D_ell'][:200], xvals)
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    ax.scatter(np.arccos(xvals) * 180 / np.pi, corr**2, s=2, marker='o', c='r', label='Correlation function data')
    ax.scatter(np.arccos(xvals) * 180 / np.pi, cor_th**2, s=2, marker='o', c='b', label='Correlation function model')

    for (a, b), i in zip(intervals, est_df.columns):
        M = np.load(f"Tmn__$\\{round(np.arccos(a) * 180 / np.pi)}__{round(np.arccos(b) * 180 / np.pi)}.npy")
        mean_sq = S12(unbin_cl['D_ell'][:200], M)
        mean_sq_err = S12_err2(unbin_cl['D_ell'][:200], unbin_cl['Error'][:200], M)
        
        ax.fill_between([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                        [mean_sq - mean_sq_err], [mean_sq + mean_sq_err], alpha=0.4, color='r')
        ax.plot([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                [mean_sq, mean_sq], ls='-.', alpha=0.8, color='r',
                label=rf'$S_{{{round(np.arccos(a)*180/np.pi)}}}^{{{round(np.arccos(b)*180/np.pi)}}}$: {mean_sq:.2f} ± {mean_sq_err:.2f}')
        
        ax.fill_between([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                        [np.mean(est_df[i]) - np.std(est_df[i])], [np.mean(est_df[i]) + np.std(est_df[i])], alpha=0.4, color='b')
        ax.plot([np.arccos(a) * 180 / np.pi, np.arccos(b) * 180 / np.pi],
                [np.mean(est_df[i]), np.mean(est_df[i])], ls='-.', alpha=0.4, color='b',
                label=f'{i}: {np.mean(est_df[i]):.2f} ± {np.std(est_df[i]):.2f}')
    
    plt.yscale('symlog', linthresh=10)
    plt.xlabel(r"$\theta$ [º]")
    plt.ylabel(r"$C(\theta)^2$")
    plt.title(r"$S_a^b$ analysis for " + str(name))
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_histogram_grid(df, labels, exp_values, title, figsize=(15, 15), bins='auto'):
    """
    Creates a grid of histograms comparing theoretical distributions with experimental values.

    Args:
        df (pd.DataFrame): DataFrame with theoretical data columns.
        labels (list): List of titles for each subplot.
        exp_values (dict): Dictionary with experimental values and errors, keys matching df.columns.
        title (str): The main title for the entire figure.
        figsize (tuple): The size of the figure.
        bins: The bin specification for the histograms.
    """
    n_cols = len(df.columns)
    if len(labels) != n_cols or set(df.columns) != set(exp_values.keys()):
        raise ValueError("Mismatch in number of columns, labels, or experimental values keys.")
    
    n_rows = int(np.ceil(n_cols / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, (col, label) in enumerate(zip(df.columns, labels)):
        ax = axes[i]
        data = df[col].dropna()
        if len(data) == 0:
            continue

        exp_value, exp_err = exp_values[col]
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        sigmas = np.sqrt((exp_value - mean_val)**2 / (exp_err**2 + std_val**2))

        n, _, _ = ax.hist(data, bins=bins, edgecolor='b', alpha=0.7)

        rect = Rectangle((exp_value - exp_err, 0), 2 * exp_err, np.max(n),
                         color='r', alpha=0.3, label=f'± {exp_err:.2f}')
        ax.add_patch(rect)
        ax.axvline(exp_value, color='r', linestyle='--', label=f'{exp_value:.2f}')
        
        rect2 = Rectangle((mean_val - std_val, 0), 2 * std_val, np.max(n),
                          color='b', alpha=0.3, label=f'± {std_val:.2f}')
        ax.add_patch(rect2)
        ax.axvline(mean_val, color='b', linestyle='--', label=f'{mean_val:.2f}')

        ax.set_title(f"{label}\nΔ = {sigmas:.2f} σ", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_power_and_correlation(unbin_cl, mean_Cl, std_Cl,
                                xvals, mean_Cor, std_Cor,
                                root="Title", figsize=(18, 7)):
    """
    Plots the power spectrum and correlation function side-by-side with uncertainty bands.

    Args:
        unbin_cl (dict): Dictionary with experimental power spectrum data.
        mean_Cl (np.ndarray): Mean theoretical power spectrum.
        std_Cl (np.ndarray): Standard deviation of theoretical power spectrum.
        xvals (np.ndarray): Array of cos(theta) values.
        mean_Cor (np.ndarray): Mean theoretical correlation function.
        std_Cor (np.ndarray): Standard deviation of theoretical correlation function.
        root (str): Base title for the plots.
        figsize (tuple): Figure size.
    """
    theta = np.arccos(xvals) * 180 / np.pi
    corr = correlation_func(unbin_cl['D_ell'][:200], xvals)
    corr_err = correlation_func_err2(unbin_cl['Error'][:200], xvals)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Power Spectrum Plot
    ax1.set_title(f"{root} - Power Spectrum")
    ax1.errorbar(unbin_cl['ell'][:200], unbin_cl['D_ell'][:200],
                 yerr=(unbin_cl['-dD_ell'][:200], unbin_cl['+dD_ell'][:200]),
                 fmt='o', color='red', ecolor=colors.to_rgba('blue', 0.2),
                 elinewidth=1, capsize=2, markersize=2, label="Power spectrum data")
    ax1.plot(unbin_cl['ell'][:200], mean_Cl, label="Power spectrum mean", color='k')
    ax1.fill_between(unbin_cl['ell'][:200], mean_Cl - 5 * std_Cl, mean_Cl + 5 * std_Cl,
                     color='k', alpha=0.2, label=r'$5\sigma$')
    ax1.fill_between(unbin_cl['ell'][:200], mean_Cl - std_Cl, mean_Cl + std_Cl,
                     color='k', alpha=0.4, label=r'$1\sigma$')
    ax1.legend()
    ax1.grid(True)

    # Correlation Function Plot
    ax2.set_title(f"{root} - Correlation Function")
    ax2.errorbar(theta, corr, yerr=abs(corr_err), fmt='o', color='red',
                 ecolor=colors.to_rgba('blue', 0.2), elinewidth=1, capsize=2,
                 markersize=2, label='Correlation function data')
    ax2.plot(theta, mean_Cor, label="Correlation function mean", color='k')
    ax2.fill_between(theta, mean_Cor - 5 * std_Cor, mean_Cor + 5 * std_Cor,
                     color='k', alpha=0.2, label=r'$5\sigma$')
    ax2.fill_between(theta, mean_Cor - std_Cor, mean_Cor + std_Cor,
                     color='k', alpha=0.4, label=r'$1\sigma$')
    ax2.axhline(0, color='k')
    ax2.legend()
    ax2.grid(True)

    # Inset Plot for Correlation Function
    axins = inset_axes(ax2, width="30%", height="30%", loc='center', borderpad=1)
    axins.plot(theta, mean_Cor, color='k')
    axins.errorbar(theta, corr, yerr=corr_err, fmt='o', color='red',
                   ecolor=colors.to_rgba('blue', 0.2), elinewidth=1, capsize=2, markersize=2)
    axins.fill_between(theta, mean_Cor - 5 * std_Cor, mean_Cor + 5 * std_Cor, color='k', alpha=0.2)
    axins.fill_between(theta, mean_Cor - std_Cor, mean_Cor + std_Cor, color='k', alpha=0.4)
    axins.set_ylim(-600, 500)
    axins.set_xlim(20, 180)
    axins.grid(True)
    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="k", lw=1)

    plt.tight_layout()
    plt.show()

def create_histogram_grid2(df, labels, exp_df, title, figsize=(15, 15), bins='auto'):
    """
    Creates a grid of histograms comparing two different data distributions (e.g., model vs. simulation).

    Args:
        df (pd.DataFrame): DataFrame for the first set of distributions.
        labels (list): List of titles for each subplot.
        exp_df (pd.DataFrame): DataFrame for the second set of distributions to compare against.
        title (str): The main title for the figure.
        figsize (tuple): The size of the figure.
        bins: The bin specification for the histograms.
    """
    n_cols = len(df.columns)
    
    if len(labels) != n_cols or list(exp_df.columns) != list(df.columns):
        raise ValueError("Mismatch in number of columns, labels, or experimental DataFrame format.")
    
    n_rows = int(np.ceil(n_cols / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, (col, label) in enumerate(zip(df.columns, labels)):
        ax = axes[i]
        data = df[col].dropna()
        data_exp = exp_df[col].dropna()
        if len(data) == 0 or len(data_exp) == 0:
            continue

        mean_val = np.mean(data)
        std_val = np.std(data)
        exp_value = np.mean(data_exp)
        exp_err = np.std(data_exp)
        sigmas = np.sqrt((exp_value - mean_val)**2 / (exp_err**2 + std_val**2))

        n, bins_hist, _ = ax.hist(data, bins=bins, edgecolor='b', alpha=0.7)
        n2, _, _ = ax.hist(data_exp, bins=bins_hist, edgecolor='r', alpha=0.7)

        rect = Rectangle((exp_value - exp_err, 0), 2 * exp_err, np.max(n2),
                         color='r', alpha=0.3, label=f'± {exp_err:.2f}')
        ax.add_patch(rect)
        ax.axvline(exp_value, color='r', linestyle='--', label=f'{exp_value:.2f}')

        rect2 = Rectangle((mean_val - std_val, 0), 2 * std_val, np.max(n),
                          color='b', alpha=0.3, label=f'± {std_val:.2f}')
        ax.add_patch(rect2)
        ax.axvline(mean_val, color='b', linestyle='--', label=f'{mean_val:.2f}')

        ax.set_title(f"{label}\nΔ = {sigmas:.2f} σ", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    for j in range(n_cols, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()