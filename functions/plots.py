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
import os
import pandas as pd

from .correlation_function import correlation_func, correlation_func_err2
from .xiv import xivar, xivar_err2
from .s12 import S12, S12_err2
from .maps import estimate_coef

try:
    import tikzplotlib
except ImportError:
    matplotlib2tikz = None

def _save_or_show_plot(save_path):
    """
    Handles saving the current plot to a file or showing it.
    The format is determined by the file extension in save_path.
    Supported formats: .png, .tex.
    """
    if not save_path:
        plt.show()
        plt.close()
        return

    file_ext = os.path.splitext(save_path)[1].lower()
    
    if file_ext == '.png':
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    elif file_ext == '.tex':
        if matplotlib2tikz is None:
            print("Warning: matplotlib2tikz is not installed. Cannot save as .tex. Please install it using 'pip install matplotlib2tikz'.")
        else:
            try:
                matplotlib2tikz.save(save_path)
            except Exception as e:
                print(f"Error saving with matplotlib2tikz: {e}")
    else:
        print(f"Warning: Unsupported file format '{file_ext}'. Plot not saved. Supported formats: .png, .tex")
    
    plt.close()

class MapPlots:
    """
    A class for plotting CMB map data.
    """
    def plot180(self, map_data, opposite_map, map_name, save_path=None, lower=False):
        """
        Creates a scatter plot of a map against its rotated/reflected version and fits a linear regression.
        This is used to visualize the correlation at 180 degrees.

        Args:
            map_data (np.ndarray): The original HEALPix map.
            opposite_map (np.ndarray): The transformed (opposite) map.
            map_name (str): The name of the map for the plot label.
            save_path (str, optional): The full path to save the plot to (e.g., 'plot.png' or 'plot.tex').
                                       If None, the plot is shown. Defaults to None.
            lower (bool): If True, degrades the resolution of the map by a factor of 2.
        """
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'text.usetex': False
        })
        
        if lower:
            low_nside = hp.get_nside(map_data) // 2
            map_data = hp.ud_grade(map_data, low_nside)

        mult = map_data * opposite_map
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
        _save_or_show_plot(save_path)

    def map_contours(self, map_data, opposite_map, save_path=None):
        """
        Creates a 2D histogram (density plot) of a map against its opposite, with contours.

        Args:
            map_data (np.ndarray): The original HEALPix map.
            opposite_map (np.ndarray): The transformed (opposite) map.
            save_path (str, optional): The full path to save the plot to (e.g., 'plot.png' or 'plot.tex').
                                       If None, the plot is shown. Defaults to None.
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
        
        _save_or_show_plot(save_path)

class CorrelationPlots:
    """
    A class for plotting correlation functions and related statistics, designed to work with a Data_loader object.
    """
    def __init__(self, data_loader):
        """
        Initializes the CorrelationPlots class with a Data_loader instance.

        Args:
            data_loader (Data_loader): An instance of the Data_loader class containing the experimental data.
        """
        self.DL = data_loader
        self._apply_style()

    @staticmethod
    def _apply_style():
        """Sets up the matplotlib style for the plots."""
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'text.usetex': False
        })

    @staticmethod
    def _plot_statistic_interval(ax, interval, data_mean, data_err, model_mean, model_std, data_color, model_color, data_symbol, model_label):
        """Helper to plot a single statistic interval."""
        a, b = interval
        theta_a_deg = np.arccos(a) * 180 / np.pi
        theta_b_deg = np.arccos(b) * 180 / np.pi

        ax.fill_between([theta_a_deg, theta_b_deg],
                        [data_mean - data_err], [data_mean + data_err], alpha=0.4, color=data_color)
        ax.plot([theta_a_deg, theta_b_deg],
                [data_mean, data_mean], ls='-.', alpha=0.8, color=data_color,
                label=rf'${data_symbol}_{{{round(theta_a_deg)}}}^{{{round(theta_b_deg)}}}$: {data_mean:.2f} ± {data_err:.2f}')
        
        ax.fill_between([theta_a_deg, theta_b_deg],
                        [model_mean - model_std], [model_mean + model_std], alpha=0.4, color=model_color)
        ax.plot([theta_a_deg, theta_b_deg],
                [model_mean, model_mean], ls='-.', alpha=0.8, color=model_color,
                label=f'{model_label}: {model_mean:.2f} ± {model_std:.2f}')

    @staticmethod
    def _add_dist_overlay(ax, mean, std, max_n, color, label):
        """Adds a rectangle for std and a line for mean to a histogram plot."""
        rect = Rectangle((mean - std, 0), 2 * std, max_n,
                         color=color, alpha=0.3, label=f'{label} std: {std:.2f}')
        ax.add_patch(rect)
        ax.axvline(mean, color=color, linestyle='--', label=f'{label} mean: {mean:.2f}')

    def plot_corr_with_xivar(self, corr_th, est_df, intervals, name, save_path=None):
        """
        Plots the correlation function and overlays the xivar statistic for specified intervals.

        Args:
            corr_th (np.ndarray): Theoretical correlation function.
            est_df (pd.DataFrame): DataFrame with theoretical estimates for xivar.
            intervals (list): List of tuples defining the angular intervals.
            name (str): Name for the plot title.
            save_path (str, optional): The full path to save the plot to. Defaults to None.
        """
        lmax = self.DL.lmax
        xvals = self.DL.xvals
        corr = self.DL._correlation 
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.scatter(np.arccos(xvals) * 180 / np.pi, corr, s=2, marker='o', c='r', label='Correlation function data')
        ax.scatter(np.arccos(xvals) * 180 / np.pi, corr_th, s=2, marker='o', c='b', label='Correlation function model')

        for (a, b), i in zip(intervals, est_df.columns):
            mean_sq, mean_sq_err = self.DL.get_xivar(a, b)
            
            self._plot_statistic_interval(
                ax, (a, b), mean_sq, mean_sq_err, 
                np.mean(est_df[i]), np.std(est_df[i]),
                'r', 'b', r'\xi', i
            )
        
        ax.set_ylim(-400, 400)
        ax.set_xlabel(r"$\theta$ [º]")
        ax.set_ylabel(r"$C(\theta)$")
        ax.set_title(r"$\xi_a^b$ analysis for " + str(name))
        ax.legend(fontsize=7)
        ax.grid(True)
        plt.tight_layout()
        
        _save_or_show_plot(save_path)

    def plot_corr_with_S12(self, cor_th, est_df, intervals, name, matrix_dir='.', save_path=None):
        """
        Plots the squared correlation function and overlays the S12 statistic for specified intervals.

        Args:
            cor_th (np.ndarray): Theoretical correlation function.
            est_df (pd.DataFrame): DataFrame with theoretical estimates for S12.
            intervals (list): List of tuples defining the angular intervals.
            name (str): Name for the plot title.
            matrix_dir (str): Directory where the Tmn matrices are stored.
            save_path (str, optional): The full path to save the plot to. Defaults to None.
        """
        lmax = self.DL.lmax
        xvals = self.DL.xvals
        corr = self.DL._correlation
        
        fig, ax = plt.subplots(figsize=(15, 8))

        ax.scatter(np.arccos(xvals) * 180 / np.pi, corr**2, s=2, marker='o', c='r', label='Correlation function data')
        ax.scatter(np.arccos(xvals) * 180 / np.pi, cor_th**2, s=2, marker='o', c='b', label='Correlation function model')

        for (a, b), i in zip(intervals, est_df.columns):
            try:
                mean_sq, mean_sq_err = self.DL.get_s12(a, b)
            except FileNotFoundError:
                print(f"Warning: Matrix file for interval ({a}, {b}) not found. Skipping.")
                continue
            
            self._plot_statistic_interval(
                ax, (a, b), mean_sq, mean_sq_err,
                np.mean(est_df[i]), np.std(est_df[i]),
                'r', 'b', 'S', i
            )
        
        ax.set_yscale('symlog', linthresh=10)
        ax.set_xlabel(r"$\theta$ [º]")
        ax.set_ylabel(r"$C(\theta)^2$")
        ax.set_title(r"$S_a^b$ analysis for " + str(name))
        ax.legend(fontsize=7)
        ax.grid(True)
        plt.tight_layout()

        _save_or_show_plot(save_path)

    def create_histogram_grid(self, df, labels, title, comparison_data=None, figsize=(20, 20), bins='auto', ncols=2, save_path=None):
        """
        Creates a grid of histograms, optionally comparing with another dataset.

        Args:
            df (pd.DataFrame): DataFrame with primary data columns.
            labels (list): List of titles for each subplot.
            title (str): The main title for the entire figure.
            comparison_data (pd.DataFrame or dict, optional): Data to compare against.
            figsize (tuple): The size of the figure.
            bins: The bin specification for the histograms.
            ncols (int): Number of columns in the subplot grid.
            save_path (str, optional): The full path to save the plot to. Defaults to None.
        """
        #print(df.columns)
        n_labels = len(labels)
        
        n_rows = int(np.ceil(n_labels / ncols))
        fig, axes = plt.subplots(n_rows, ncols, figsize=figsize)
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16)

        for i, col in enumerate(labels):
            #print(f"{i}: {col}")
            ax = axes[i]
            if col not in df.columns:
                ax.text(0.5, 0.5, f"'{col}' not in data", ha='center', va='center')
                ax.set_title(col, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            data = df[col].dropna()
            if len(data) == 0:
                continue

            mean_val = np.mean(data)
            std_val = np.std(data)
            
            n, bins_hist, _ = ax.hist(data, bins=bins, histtype='step', edgecolor='b', label='Model')
            self._add_dist_overlay(ax, mean_val, std_val, np.max(n), 'b', 'Model')

            sigmas = np.nan
            if comparison_data is not None:
                exp_value, exp_err = None, None
                if isinstance(comparison_data, pd.DataFrame):
                    if col not in comparison_data.columns:
                        continue
                    
                    print(comparison_data.columns)

                    data_exp = comparison_data[col].dropna()
                    if len(data_exp) == 0:
                        continue
                    exp_value = np.mean(data_exp)
                    exp_err = np.std(data_exp)
                    
                    n2, _, _ = ax.hist(data_exp, bins=bins_hist, histtype='step', edgecolor='r', label='Data')
                    self._add_dist_overlay(ax, exp_value, exp_err, np.max(n2), 'r', 'Data')

                elif isinstance(comparison_data, dict):
                    if col not in comparison_data:
                        continue
                    exp_value, exp_err = comparison_data[col]
                    self._add_dist_overlay(ax, exp_value, exp_err, np.max(n), 'r', 'Data')
                
                else:
                    raise TypeError("comparison_data must be a pandas DataFrame or a dictionary.")

                if exp_value is not None and exp_err is not None and (exp_err**2 + std_val**2 > 0):
                    sigmas = np.sqrt((exp_value - mean_val)**2 / (exp_err**2 + std_val**2))

            ax.set_title(f"{col}\nΔ = {sigmas:.2f} σ", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)

        for j in range(n_labels, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        _save_or_show_plot(save_path)

    def plot_power_and_correlation(self, mean_Cl, std_Cl, mean_Cor, std_Cor,
                                    root="Title", figsize=(18, 7), save_path=None):
        """
        Plots the power spectrum and correlation function side-by-side with uncertainty bands.

        Args:
            mean_Cl (np.ndarray): Mean theoretical power spectrum.
            std_Cl (np.ndarray): Standard deviation of theoretical power spectrum.
            mean_Cor (np.ndarray): Mean theoretical correlation function.
            std_Cor (np.ndarray): Standard deviation of theoretical correlation function.
            root (str): Base title for the plots.
            figsize (tuple): Figure size.
            save_path (str, optional): The full path to save the plot to. Defaults to None.
        """
        lmax = self.DL.lmax
        xvals = self.DL.xvals
        theta = np.arccos(xvals) * 180 / np.pi
        corr, corr_err = self.DL.get_correlation_function()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Power Spectrum Plot
        ax1.set_title(f"{root} - Power Spectrum")
        ax1.errorbar(self.DL.ell, self.DL.D_ell,
                     yerr=(self.DL.dD_ell_neg, self.DL.dD_ell_pos),
                     fmt='o', color='red', ecolor=colors.to_rgba('blue', 0.2),
                     elinewidth=1, capsize=2, markersize=2, label="Power spectrum data")
        ax1.plot(self.DL.ell, mean_Cl, label="Power spectrum mean", color='k')
        ax1.fill_between(self.DL.ell, mean_Cl - 5 * std_Cl, mean_Cl + 5 * std_Cl,
                         color='k', alpha=0.2, label=r'$5\sigma$')
        ax1.fill_between(self.DL.ell, mean_Cl - std_Cl, mean_Cl + std_Cl,
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
        
        _save_or_show_plot(save_path)
