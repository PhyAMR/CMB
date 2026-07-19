"""
Unified plotting module for CMB analysis - Article Version with GetDist P-values.

Key improvements:
- Proper LaTeX formatting for all labels
- P-values computed using percentile-based method (consistent across all modes)
- Individual histograms instead of grid
- Forest plots separated by statistic type
- Diagnostic plots separated by statistic type
- Mode-dependent plotting (bestfit, mcmc, or all)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import seaborn as sns
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle
#import healpy as hp
import re
#from scipy import stats

from .unified_stats import compute_percentiles_unified, compute_pvalue_unified
from .plot_style import apply_style, COLORS


logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import tikzplotlib
    matplotlib2tikz = tikzplotlib
except ImportError:
    matplotlib2tikz = None


# ============================================================================
# Label Formatting Utilities
# ============================================================================

def format_label_latex(label):
    """
    Convert raw label to properly formatted LaTeX label.
    
    Examples:
        'C180' -> '$C_{180}$'
        'xiv_60_30' -> '$\\xi_{60,30}$'  
        's12_60_30' -> '$S_{60,30}$'
        's12_180_150' -> '$S_{180,150}$'
    
    Args:
        label (str): Raw label string
    
    Returns:
        str: LaTeX-formatted label
    """
    # Handle C180
    if label == 'C180':
        return r'$C_{180}$'
    
    # Handle xivar: xiv_60_30 -> $\xi_{60,30}$
    xiv_match = re.match(r'xiv_(\d+)_(\d+)', label)
    if xiv_match:
        upper, lower = xiv_match.groups()
        return rf'$\xi_{{{upper},{lower}}}$'
    
    # Handle S12: s12_60_30 -> $S_{60,30}$
    s12_match = re.match(r's12_(\d+)_(\d+)', label)
    if s12_match:
        upper, lower = s12_match.groups()
        return rf'$S_{{{upper},{lower}}}$'
    
    # Fallback: return as-is
    return label


def get_statistic_type(label):
    """
    Determine the statistic type from label.
    
    Args:
        label (str): Label string
    
    Returns:
        str: 'C180', 'xivar', 's12', or 'other'
    """
    if label == 'C180':
        return 'C180'
    elif label.startswith('xiv_'):
        return 'xivar'
    elif label.startswith('s12_'):
        return 's12'
    else:
        return 'other'


def group_labels_by_type(labels):
    """
    Group labels by statistic type.
    
    Args:
        labels (list): List of label strings
    
    Returns:
        dict: {type: [labels]} mapping
    """
    groups = {'C180': [], 'xivar': [], 's12': [], 'other': []}
    
    for label in labels:
        stat_type = get_statistic_type(label)
        groups[stat_type].append(label)
    
    return groups


# ============================================================================
# P-Value Computation (Unified across all modes)
# ============================================================================

def compute_percentiles(data, samples=None, param_name=None, weights=None):
    """
    Unified percentile computation using GetDist when available.

    Falls back to explicit-weight NumPy percentiles, and finally to
    plain NumPy percentiles if neither GetDist samples nor weights are
    supplied.

    Parameters
    ----------
    data : np.ndarray
        Data array
    samples : getdist.mcsamples.MCSamples, optional
        GetDist samples object
    param_name : str, optional
        Parameter name if using GetDist samples
    weights : np.ndarray, optional
        Per-sample weights aligned with ``data`` (e.g. MCMC importance
        weights loaded from a chain file). Ignored if ``samples`` is
        supplied, since GetDist carries its own weights in that case.

    Returns
    -------
    dict
        {'p16': val, 'p50': val, 'p84': val}
    """
    return compute_percentiles_unified(data, samples=samples,
                                        param_name=param_name,
                                        weights=weights)


def _aligned_weights(mask, weights):
    """
    Select the subset of *weights* aligned with a boolean/NaN-drop mask.

    Parameters
    ----------
    mask : pd.Series or np.ndarray of bool
        Mask used to select non-null values from a column (e.g. from
        ``series.notna()``).
    weights : np.ndarray or None
        Full-length weight array (same length as the original,
        un-masked column), or None if no weights are available.

    Returns
    -------
    np.ndarray or None
    """
    if weights is None:
        return None
    mask_arr = np.asarray(mask)
    weights_arr = np.asarray(weights)
    if len(weights_arr) != len(mask_arr):
        # Lengths don't line up (e.g. mismatched ensemble) - safer to
        # skip weighting than to silently misalign samples.
        logger.debug(
            "Skipping weights: length mismatch (%d weights vs %d rows)",
            len(weights_arr), len(mask_arr)
        )
        return None
    return weights_arr[mask_arr]




def compute_pvalue_from_percentiles(observed_value, p16, p50, p84, values):
    """
    Compute p-value from percentiles using unified method.
    
    Consistent with GetDist and all other modules.
    
    Parameters
    ----------
    observed_value : float
        Experimental/simulation value
    p16, p50, p84 : float
        16th, 50th, 84th percentiles
    
    Returns
    -------
    dict
        {'pvalue': p, 'n_sigma': n, 'interpretation': str, 'tension_level': str}
    """
    percentiles_dict = {'p16': p16, 'p50': p50, 'p84': p84}
    return compute_pvalue_unified(observed_value, percentiles_dict, values)


# ============================================================================
# Module-level Helper Functions
# ============================================================================

def _save_or_show_plot(save_path):
    """
    Save plot to file or display it.
    Supported formats: .pdf, .png, .tex
    
    Args:
        save_path (str or None): Path to save file. If None, shows plot.
    """
    if not save_path:
        plt.show()
        plt.close()
        return

    file_ext = os.path.splitext(save_path)[1].lower()
    
    if file_ext in ['.pdf', '.png']:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format=file_ext[1:])
    elif file_ext == '.tex':
        if matplotlib2tikz is None:
            logger.warning("tikzplotlib not installed. Cannot save as .tex")
        else:
            try:
                matplotlib2tikz.save(save_path)
            except Exception as e:
                logger.exception(f"Error saving with tikzplotlib: {e}")
    else:
        logger.warning(f"Unsupported format '{file_ext}'. Supported: .pdf, .png, .tex")
    
    plt.close()


def _build_output_path(output_dir, category, filename, mode):
    """
    Build full output path: output_dir/images_{mode}/category/filename
    
    Args:
        output_dir (str): Base run directory
        category (str): Subdirectory (e.g., 'histograms', 'correlation')
        filename (str): Output filename
        mode : str
            'auto', 'bestfit', 'mcmc', or 'all'
    
    Returns:
        str: Full path with directories created
    """
    if not output_dir:
        return filename
    
    final_dir = os.path.join(output_dir, f'images_{mode}', category)
    os.makedirs(final_dir, exist_ok=True)
    return os.path.join(final_dir, filename)


# ============================================================================
# Data Loading Function
# ============================================================================

def load_scalars(base_dir):
    """
    Load scalar statistics from a directory.
    
    Parameters
    ----------
    base_dir : str
        Directory containing statistics files
    
    Returns
    -------
    dict
        {
            'scalars_df': pd.DataFrame or None,
            'D_ell': np.ndarray or None,
            'Corr': np.ndarray or None,
            'columns': list of column names,
            'weights': np.ndarray or None
                Per-sample weights read from a GetDist-style chain file
                (``*_chain.txt``, first column) when present. This is
                what carries the MCMC likelihood/importance weighting
                through to the plotting percentiles - it is None for
                best-fit/simulation ensembles, which have no such
                weighting (all realisations are equally likely).
        }
    """
    # Read column names
    columns_path = os.path.join(base_dir, 'columns.txt')
    if not os.path.exists(columns_path):
        logger.warning(f"columns.txt not found in {base_dir}")
        return {
            'scalars_df': None,
            'D_ell': None,
            'Corr': None,
            'columns': [],
            'weights': None
        }
    
    with open(columns_path, 'r') as f:
        scalar_cols = [line.strip() for line in f if line.strip()]
    
    data = {}
    
    # Load S12 statistics
    s_path = os.path.join(base_dir, 'S_statistics.txt')
    if os.path.exists(s_path):
        S_mat = np.loadtxt(s_path)
        if S_mat.ndim == 1:
            S_mat = S_mat[:, None]
        s_cols = [c for c in scalar_cols if c.startswith('s12_')]
        for i, col in enumerate(s_cols):
            if i < S_mat.shape[1]:
                data[col] = S_mat[:, i]
    
    # Load xivar statistics
    xiv_path = os.path.join(base_dir, 'xiv_statistic.txt')
    if os.path.exists(xiv_path):
        xiv_mat = np.loadtxt(xiv_path)
        if xiv_mat.ndim == 1:
            xiv_mat = xiv_mat[:, None]
        xiv_cols = [c for c in scalar_cols if c.startswith('xiv_')]
        for i, col in enumerate(xiv_cols):
            if i < xiv_mat.shape[1]:
                data[col] = xiv_mat[:, i]
    
    # Load C180
    c180_path = os.path.join(base_dir, 'C180_statistics.txt')
    if os.path.exists(c180_path):
        c180_vec = np.loadtxt(c180_path)
        data['C180'] = np.atleast_1d(c180_vec)
    
    # Load D_ell and Corr
    D_ell_path = os.path.join(base_dir, 'D_ell.npy')
    Corr_path = os.path.join(base_dir, 'Corr.npy')
    
    D_ell = np.load(D_ell_path) if os.path.exists(D_ell_path) else None
    Corr = np.load(Corr_path) if os.path.exists(Corr_path) else None

    scalars_df = pd.DataFrame(data) if data else None

    # Load MCMC chain weights when available (mcmc mode only). The chain
    # file is written by cosmo.chain_results() as
    # "{short_name}_chain.txt" with columns [weight, -loglike, params...],
    # in the same row order as the S_/xiv_/C180 statistics files above -
    # so the weight column aligns directly with `scalars_df` by row.
    weights = None
    chain_candidates = glob.glob(os.path.join(base_dir, '*_chain.txt'))
    if chain_candidates:
        chain_path = chain_candidates[0]
        try:
            chain_weights = np.loadtxt(chain_path, usecols=0)
            chain_weights = np.atleast_1d(chain_weights)
            if scalars_df is not None and len(chain_weights) == len(scalars_df):
                weights = chain_weights
            else:
                logger.warning(
                    f"Chain weights length ({len(chain_weights)}) does not "
                    f"match scalars ({0 if scalars_df is None else len(scalars_df)}) "
                    f"in {base_dir}; ignoring weights."
                )
        except Exception as exc:
            logger.warning(f"Could not read chain weights from {chain_path}: {exc}")

    return {
        'scalars_df': scalars_df,
        'D_ell': D_ell,
        'Corr': Corr,
        'columns': scalar_cols,
        'weights': weights
    }


def load_run_data(run_dir, experimental_values, mode='auto'):
    """
    Load data from run directory with new structure.
    
    Handles both theory_bestfit/ and theory_mcmc/ directories.
    
    Parameters
    ----------
    run_dir : str
        Run directory path
    experimental_values : dict
        Experimental values
    mode : str
        'auto', 'bestfit', 'mcmc', or 'all'
    
    Returns
    -------
    dict
        Structure with experimental, simulation, and theory data
    """
    result = {
        'experimental': {'scalars': experimental_values},
        'simulation': {},
    }
    
    # Load simulation
    sim_dir = os.path.join(run_dir, 'simulation')
    if os.path.exists(sim_dir):
        result['simulation'] = load_scalars(sim_dir)
        logger.info(f"Loaded simulation data from {sim_dir}")
    else:
        logger.warning(f"Simulation directory not found: {sim_dir}")
    
    # Load theory - UPDATED LOGIC
    if mode == 'all':
        # Load both theory_bestfit and theory_mcmc
        result['theory_bestfit'] = {}
        result['theory_mcmc'] = {}
        
        # Load theory_bestfit/
        bestfit_dir = os.path.join(run_dir, 'theory_bestfit')
        if os.path.exists(bestfit_dir):
            for model_name in sorted(os.listdir(bestfit_dir)):
                model_dir = os.path.join(bestfit_dir, model_name)
                if os.path.isdir(model_dir):
                    result['theory_bestfit'][model_name] = load_scalars(model_dir)
                    logger.info(f"Loaded theory_bestfit/{model_name}")
        else:
            logger.warning(f"theory_bestfit directory not found: {bestfit_dir}")
        
        # Load theory_mcmc/
        mcmc_dir = os.path.join(run_dir, 'theory_mcmc')
        if os.path.exists(mcmc_dir):
            for model_name in sorted(os.listdir(mcmc_dir)):
                model_dir = os.path.join(mcmc_dir, model_name)
                if os.path.isdir(model_dir):
                    result['theory_mcmc'][model_name] = load_scalars(model_dir)
                    logger.info(f"Loaded theory_mcmc/{model_name}")
        else:
            logger.warning(f"theory_mcmc directory not found: {mcmc_dir}")
    
    else:
        # Load single mode
        result['theory'] = {}
        
        # Determine which directories to try
        if mode == 'bestfit':
            theory_dirs = [os.path.join(run_dir, 'theory_bestfit')]
        elif mode == 'mcmc':
            theory_dirs = [os.path.join(run_dir, 'theory_mcmc')]
        elif mode == 'auto':
            # Try in order: theory_bestfit, theory_mcmc, theory (legacy)
            theory_dirs = [
                os.path.join(run_dir, 'theory_bestfit'),
                os.path.join(run_dir, 'theory_mcmc'),
                os.path.join(run_dir, 'theory')
            ]
        else:
            logger.error(f"Unknown mode: {mode}")
            return result
        
        # Load from first available directory
        loaded = False
        for theory_dir in theory_dirs:
            if os.path.exists(theory_dir):
                for model_name in sorted(os.listdir(theory_dir)):
                    model_dir = os.path.join(theory_dir, model_name)
                    if os.path.isdir(model_dir):
                        result['theory'][model_name] = load_scalars(model_dir)
                        logger.info(f"Loaded theory/{model_name} from {theory_dir}")
                
                loaded = True
                break  # Use first available and stop
        
        if not loaded:
            logger.warning(f"No theory directory found for mode={mode}")
    
    return result


# ============================================================================
# CorrelationPlots Class
# ============================================================================

class CorrelationPlots:
    """
    Class for plotting correlation functions, power spectra, and statistics.
    
    Optimized for article-quality output with:
    - Proper LaTeX label formatting
    - Individual plots instead of grids
    - Separated plots by statistic type
    - Unified percentile-based p-value computation
    """
    
    def __init__(self, DL, output_dir=None, colors=None, mode='all', use_sim=True):
        """
        Initialize plotter.
        
        Args:
            DL (Data_loader): Data loader instance
            output_dir (str): Base directory for saving plots
            colors (dict): Color scheme for experimental/simulation/theory
            mode (str): 'auto', 'bestfit', 'mcmc', or 'all'
        """
        self.DL = DL
        self.output_dir = output_dir

        # Base palette from plot_style; caller may override individual keys.
        self.colors = {
            'experimental': COLORS['experimental'],
            'simulation':   COLORS['theory_fill'],
            'theory':       COLORS['theory'],
        }
        if colors:
            self.colors.update(colors)

        self.mode = mode
        self.use_sim = use_sim

        apply_style()

    @staticmethod
    def _apply_style():
        """Delegate to the centralised plot_style module."""
        apply_style()
    
    # Helper Methods
    
    @staticmethod
    def _plot_statistic_interval(ax, interval, data_mean, data_err,
                                  model_mean, model_std,
                                  data_color, model_color,
                                  data_symbol, model_label):
        """Plot interval band for a single statistic.
        
        Experimental errors are not shown; only the observed mean is drawn.
        """
        a, b = interval
        theta_a_deg = np.arccos(a) * 180 / np.pi
        theta_b_deg = np.arccos(b) * 180 / np.pi

        # Observed value — single dashed line, no error band
        ax.plot([theta_a_deg, theta_b_deg],
                [data_mean, data_mean], ls='--', alpha=0.9, color=data_color,
                linewidth=1.8,
                label=rf'${data_symbol}_{{{round(theta_a_deg)}}}^{{{round(theta_b_deg)}}}$: {data_mean:.2f}')

        # Model 68 % band + median
        ax.fill_between([theta_a_deg, theta_b_deg],
                         [model_mean - model_std], [model_mean + model_std],
                         alpha=0.35, color=model_color)
        ax.plot([theta_a_deg, theta_b_deg],
                [model_mean, model_mean], ls='-.', alpha=0.8, color=model_color,
                linewidth=1.8,
                label=f'{model_label}: {model_mean:.2f}')
    
    @staticmethod
    def _add_dist_overlay(ax, mean, std, max_n, color, label):
        """Add rectangle for std and line for mean to histogram."""
        rect = Rectangle((mean - std, 0), 2 * std, max_n,
                          color=color, alpha=0.3,
                          label=f'{label} std: {std:.2f}')
        ax.add_patch(rect)
        ax.axvline(mean, color=color, linestyle='--',
                    label=f'{label} mean: {mean:.2f}')
    
    def _prepare_comparison_data(self, comparison_data):
        """
        Extract experimental, simulation, and theory data from comparison_data.

        Args:
            comparison_data: Can be:
                - None
                - dict from load_run_data()

        Returns:
            tuple: (exp_data, sim_data, theory_data, theory_samples, theory_weights)
                ``theory_weights`` holds per-row MCMC/likelihood weights
                (aligned with ``theory_data``'s rows) when available -
                None for best-fit/simulation ensembles.
        """
        if comparison_data is None:
            return None, None, None, None, None

        # If it's the standardized structure from load_run_data()
        if isinstance(comparison_data, dict):
            exp = comparison_data.get('experimental', {}).get('scalars', None)
            sim = comparison_data.get('simulation', {}).get('scalars_df', None)

            # Theory can have multiple models - take first if exists
            theory_dict = comparison_data.get('theory', {})
            if theory_dict:
                first_model = next(iter(theory_dict.values()))
                theory = first_model.get('scalars_df', None)
                theory_samples = first_model.get('samples', None)
                theory_weights = first_model.get('weights', None)
            else:
                theory = None
                theory_samples = None
                theory_weights = None

            return exp, sim, theory, theory_samples, theory_weights

        # Fallback: treat as single dataset
        if isinstance(comparison_data, pd.DataFrame):
            return None, comparison_data, None, None, None

        return None, None, None, None, None
    
    # ========================================================================
    # Individual Histogram Methods with Unified P-value Computation
    # ========================================================================
    
    def create_individual_histograms(self, df, labels, comparison_data=None,
                                     bins=25, base_name='stat', file_format='pdf'):
        """
        Create individual histogram for each statistic with unified p-values.

        Experimental errors are never shown; only the observed value is marked.
        The p-value in the title uses sigma notation.

        Args:
            df (DataFrame): Data with statistics
            labels (list): List of statistic labels
            comparison_data (dict): Comparison data structure
            bins (int): Number of histogram bins
            base_name (str): Base name for output files
            file_format (str): File format ('pdf' or 'png')
        """
        from .plot_style import COLORS as _C
        exp_data, sim_data, theory_data, _, theory_weights = \
            self._prepare_comparison_data(comparison_data)

        for label in labels:
            if label not in df.columns:
                continue

            mask = df[label].notna()
            values = df[label][mask].values
            if len(values) == 0:
                continue
            weights_matched = _aligned_weights(mask, theory_weights)

            fig, ax = plt.subplots(figsize=(9.5, 6.5))

            # Histogram bars
            ax.hist(values, bins=bins,
                    color=_C['theory_fill'], alpha=0.85,
                    edgecolor='#1A252F', linewidth=0.8, rwidth=0.95,
                    label='Theory Distribution', zorder=2)

            # Percentiles — weighted by the chain's likelihood/importance
            # weights when available (MCMC mode), otherwise an
            # equal-weight (plain) percentile is used.
            perc = compute_percentiles(values, weights=weights_matched)
            p16, p50, p84 = perc['p16'], perc['p50'], perc['p84']

            # 68% CI shown as dotted low/high lines rather than a shaded
            # band.
            ax.axvline(p16, color=_C['ci_band'], linestyle=':', linewidth=1.8,
                       zorder=3, label='68% CI')
            ax.axvline(p84, color=_C['ci_band'], linestyle=':', linewidth=1.8,
                       zorder=3)
            ax.axvline(p50, color=_C['theory'], linestyle='-', linewidth=2.2,
                       zorder=4, label='Median')

            # Observed value — no error, and no p-value is computed or
            # displayed here; the histogram only shows where the
            # observed value falls relative to the distribution.
            if exp_data and isinstance(exp_data, dict) and label in exp_data:
                exp_val = exp_data[label][0]   # discard error
                ax.axvline(exp_val, color=_C['experimental'],
                           linestyle='--', linewidth=2.2, zorder=5,
                           label='Experimental')

            # Simulation median overlay
            if self.use_sim and sim_data is not None and label in sim_data.columns:
                sim_vals = sim_data[label].dropna().values
                if len(sim_vals) > 0:
                    sim_p50 = compute_percentiles(sim_vals)['p50']
                    ax.axvline(sim_p50, color=_C['theory_fill'],
                               linestyle=':', linewidth=2.0, zorder=3,
                               label='Simulation median')

            formatted_label = format_label_latex(label)

            ax.set_xlabel(formatted_label)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution: {formatted_label}')
            ax.legend(loc='best')
            plt.tight_layout()

            safe_label = label.replace('_', '-')
            filename = f"{base_name}_hist_{safe_label}.{file_format}"

            if self.output_dir:
                save_path = _build_output_path(
                    self.output_dir, 'histograms', filename, self.mode
                )
            else:
                save_path = filename

            _save_or_show_plot(save_path)

        logger.info(f"✓ Individual histograms saved for {len(labels)} statistics")
    
    # ========================================================================
    # Forest Plot Methods (Separated by Type)
    # ========================================================================
    
    def create_forest_plots_by_type(self, df, labels, comparison_data=None,
                                     base_name='model', file_format='pdf'):
        """
        Create separate forest plots for each statistic type using percentiles.
        
        Args:
            df (DataFrame): Data with statistics
            labels (list): List of statistic labels
            comparison_data (dict): Comparison data structure
            base_name (str): Base name for output files
            file_format (str): File format
        """
        # Group labels by type
        groups = group_labels_by_type(labels)
        
        # Create forest plot for each type
        for stat_type, type_labels in groups.items():
            if not type_labels:
                continue
            
            if stat_type == 'C180':
                self._create_single_forest_plot(
                    df, type_labels, comparison_data,
                    title=r'$C_{180}$ Statistic',
                    save_name=f'{base_name}_forest_C180.{file_format}'
                )
            elif stat_type == 'xivar':
                self._create_single_forest_plot(
                    df, type_labels, comparison_data,
                    title=r'$\xi$ Statistics',
                    save_name=f'{base_name}_forest_xivar.{file_format}'
                )
            elif stat_type == 's12':
                self._create_single_forest_plot(
                    df, type_labels, comparison_data,
                    title=r'$S$ Statistics',
                    save_name=f'{base_name}_forest_s12.{file_format}'
                )
        
        logger.info("✓ Forest plots saved by type")
    
    def _create_single_forest_plot(self, df, labels, comparison_data,
                                   title, save_name):
        """
        Create a single forest plot using percentiles.
        
        Args:
            df (DataFrame): Data
            labels (list): Labels to plot
            comparison_data (dict): Comparison data
            title (str): Plot title
            save_name (str): Filename to save
        """
        exp_data, sim_data, theory_data, _, theory_weights = \
            self._prepare_comparison_data(comparison_data)

        # Determine figure size based on number of parameters
        n_params = len(labels)
        figsize = (10, max(6, n_params * 0.4))

        fig, ax = plt.subplots(figsize=figsize)

        y = np.arange(len(labels))

        for i, label in enumerate(labels):
            # Plot theory data using percentiles (likelihood-weighted
            # when MCMC chain weights are available)
            if label in df.columns:
                mask = df[label].notna()
                arr = df[label][mask].values
                if len(arr) > 0:
                    w = _aligned_weights(mask, theory_weights)
                    perc = compute_percentiles(arr, weights=w)
                    p16, p50, p84 = perc['p16'], perc['p50'], perc['p84']
                    
                    ax.hlines(i, xmin=p16, xmax=p84,
                            color=self.colors['theory'], linewidth=3,
                            label='Theory 68% CI' if i==0 else '')
                    ax.plot(p50, i, 'o', color=self.colors['theory'],
                           markersize=8, label='Theory median' if i==0 else '')
            
            # Overlay experimental — observed value only, no error bar
            if exp_data and isinstance(exp_data, dict) and label in exp_data:
                val = exp_data[label][0]   # discard error
                ax.scatter(val, i, color=self.colors['experimental'],
                           marker='|', s=200, linewidths=3,
                           label='Experimental' if i == 0 else '')
            
            # Overlay simulation
            if sim_data is not None and label in sim_data.columns:
                sim_vals = sim_data[label].dropna().values
                if len(sim_vals) > 0:
                    sim_perc = compute_percentiles(sim_vals)
                    sp16, sp50, sp84 = sim_perc['p16'], sim_perc['p50'], sim_perc['p84']
                    
                    ax.hlines(i, xmin=sp16, xmax=sp84,
                            color=self.colors['simulation'], linewidth=1.5,
                            alpha=0.7, label='Simulation 68% CI' if i==0 else '')
                    ax.plot(sp50, i, 'D', color=self.colors['simulation'],
                           markersize=5, alpha=0.7,
                           label='Simulation median' if i==0 else '')
        
        # Format y-tick labels with LaTeX
        formatted_labels = [format_label_latex(la) for la in labels]
        
        ax.set_yticks(y)
        ax.set_yticklabels(formatted_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Value', fontsize=14)
        ax.set_ylabel('Statistic', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        if self.output_dir:
            save_path = _build_output_path(self.output_dir, 'forest', save_name, self.mode)
        else:
            save_path = save_name
        
        _save_or_show_plot(save_path)
    
    # ========================================================================
    # Diagnostic Plot Methods (Separated by Type)
    # ========================================================================
    
    def create_diagnostics_by_type(self, df, labels, comparison_data=None,
                                   base_name='model', file_format='pdf'):
        """
        Create separate diagnostic panels for each statistic type.
        
        Args:
            df (DataFrame): Data with statistics
            labels (list): List of statistic labels
            comparison_data (dict): Comparison data structure
            base_name (str): Base name for output files
            file_format (str): File format
        """
        # Group labels by type
        groups = group_labels_by_type(labels)
        
        # Create diagnostics for each type
        for stat_type, type_labels in groups.items():
            if not type_labels:
                continue
            
            if stat_type == 'C180':
                self._create_single_diagnostic(
                    df, type_labels, comparison_data,
                    title=r'$C_{180}$ Diagnostics',
                    save_name=f'{base_name}_diagnostics_C180.{file_format}'
                )
            elif stat_type == 'xivar':
                self._create_single_diagnostic(
                    df, type_labels, comparison_data,
                    title=r'$\xi$ Diagnostics',
                    save_name=f'{base_name}_diagnostics_xivar.{file_format}'
                )
            elif stat_type == 's12':
                self._create_single_diagnostic(
                    df, type_labels, comparison_data,
                    title=r'$S$ Diagnostics',
                    save_name=f'{base_name}_diagnostics_s12.{file_format}'
                )
        
        logger.info("✓ Diagnostic plots saved by type")
    
    def _create_single_diagnostic(self, df, labels, comparison_data,
                                  title, save_name):
        """
        Create a single 3-panel diagnostic figure using percentiles.
        
        Panels:
        1. Density comparison
        2. Forest plot (percentile-based credible intervals)
        3. Cumulative mean convergence
        
        Args:
            df (DataFrame): Data
            labels (list): Labels to plot
            comparison_data (dict): Comparison data
            title (str): Plot title
            save_name (str): Filename to save
        """
        exp_data, sim_data, theory_data, _, theory_weights = \
            self._prepare_comparison_data(comparison_data)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=16)
        
        # Panel 1: Density comparison (first few statistics)
        plot_labels = labels[:min(5, len(labels))]
        
        for label in plot_labels:
            if label in df.columns:
                formatted_label = format_label_latex(label)
                sns.kdeplot(df[label].dropna(), ax=axes[0], fill=True,
                           label=formatted_label, alpha=0.4)
        
        if theory_data is not None:
            for label in plot_labels:
                if label in theory_data.columns:
                    formatted_label = format_label_latex(label)
                    sns.kdeplot(theory_data[label].dropna(), ax=axes[0],
                               fill=True, color='gray', alpha=0.25,
                               label=f'{formatted_label} (alt)', linestyle='--')
        
        axes[0].set_title('Density Comparison')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].legend(fontsize=8, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Forest plot (68% credible intervals using percentiles)
        forest_labels = labels[:min(20, len(labels))]
        y = np.arange(len(forest_labels))
        
        for i, label in enumerate(forest_labels):
            if label in df.columns:
                mask = df[label].notna()
                arr = df[label][mask].values
                if len(arr) > 0:
                    w = _aligned_weights(mask, theory_weights)
                    perc = compute_percentiles(arr, weights=w)
                    p16, p50, p84 = perc['p16'], perc['p50'], perc['p84']

                    axes[1].hlines(i, xmin=p16, xmax=p84,
                                 color=self.colors['theory'], linewidth=2)
                    axes[1].plot(p50, i, 'o', color=self.colors['theory'],
                               markersize=6)
            
            # Overlay experimental — observed value only
            if exp_data and isinstance(exp_data, dict) and label in exp_data:
                val = exp_data[label][0]   # discard error
                axes[1].scatter(val, i, color=self.colors['experimental'],
                                marker='|', s=200, linewidths=3)
        
        formatted_forest_labels = [format_label_latex(la) for la in forest_labels]
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(formatted_forest_labels, fontsize=9)
        axes[1].set_title('68% Credible Intervals')
        axes[1].set_xlabel('Value')
        axes[1].grid(axis='x', linestyle='--', alpha=0.6)
        axes[1].invert_yaxis()
        
        # Panel 3: Cumulative mean convergence (first statistic)
        if len(labels) > 0:
            first_label = next((la for la in labels if la in df.columns), None)
            if first_label:
                run_mask = df[first_label].notna()
                runs = df[first_label][run_mask].values
                if len(runs) > 0:
                    cum_mean = np.cumsum(runs) / np.arange(1, len(runs) + 1)
                    axes[2].plot(cum_mean, color=self.colors['theory'],
                               label='Running Mean', linewidth=2)

                    # Add percentile bands (likelihood-weighted when
                    # chain weights are available)
                    run_weights = _aligned_weights(run_mask, theory_weights)
                    perc = compute_percentiles(runs, weights=run_weights)
                    p16, p50, p84 = perc['p16'], perc['p50'], perc['p84']
                    
                    axes[2].axhspan(p16, p84, color=self.colors['theory'],
                                   alpha=0.2, label='68% range')
                    axes[2].axhline(p50, color=self.colors['theory'],
                                   linestyle='--', label='Median')
                    
                    # Overlay experimental — observed value only
                    if exp_data and isinstance(exp_data, dict) and first_label in exp_data:
                        val = exp_data[first_label][0]   # discard error
                        axes[2].axhline(val, color=self.colors['experimental'],
                                        linestyle='--', linewidth=2,
                                        label='Experimental')
                    
                    formatted_first = format_label_latex(first_label)
                    axes[2].set_title(f'Convergence: {formatted_first}')
                    axes[2].set_xlabel('Sample Number')
                    axes[2].set_ylabel('Cumulative Mean')
                    axes[2].legend(fontsize=9)
                    axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.output_dir:
            save_path = _build_output_path(self.output_dir, 'diagnostics', save_name, self.mode)
        else:
            save_path = save_name
        
        _save_or_show_plot(save_path)
    
    # ========================================================================
    # Cumulative Grid (Unchanged - keep for compatibility)
    # ========================================================================
    
    def create_cumulative_grid(self, df, labels, title,
                                comparison_data=None, figsize=(18, 12),
                                ncols=3, save_path=None):
        """
        Create grid of cumulative mean convergence plots.
        
        Args:
            df (DataFrame): Data
            labels (list): List of labels
            title (str): Plot title
            comparison_data (dict): Comparison data
            figsize (tuple): Figure size
            ncols (int): Number of columns
            save_path (str): Path to save
        """
        exp_data, sim_data, theory_data, _, theory_weights = \
            self._prepare_comparison_data(comparison_data)

        params = [c for c in labels if c in df.columns][:12]
        nrows = int(np.ceil(len(params) / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        fig.suptitle(title, fontsize=16)

        for i, label in enumerate(params):
            run_mask = df[label].notna()
            runs = df[label][run_mask].values
            if len(runs) > 0:
                cum_mean = np.cumsum(runs) / np.arange(1, len(runs) + 1)
                axes[i].plot(cum_mean, color=self.colors['theory'],
                           label='Running Mean', linewidth=2)

                # Add percentile band (likelihood-weighted when chain
                # weights are available)
                run_weights = _aligned_weights(run_mask, theory_weights)
                perc = compute_percentiles(runs, weights=run_weights)
                p16, p50, p84 = perc['p16'], perc['p50'], perc['p84']

                axes[i].axhspan(p16, p84, color=self.colors['theory'], alpha=0.15)
                axes[i].axhline(p50, color=self.colors['theory'], linestyle='--',
                               alpha=0.7, label='Median')

            # Overlay experimental
            if exp_data and isinstance(exp_data, dict) and label in exp_data:
                val = exp_data[label][0]   # discard error
                axes[i].axhline(val, color=self.colors['experimental'],
                                linestyle='--', label='Experimental', linewidth=2)

            # Overlay theory
            if theory_data is not None and label in theory_data.columns:
                theory_mask = theory_data[label].notna()
                theory_vals = theory_data[label][theory_mask]
                if len(theory_vals) > 0:
                    theory_w = _aligned_weights(theory_mask, theory_weights)
                    theory_perc = compute_percentiles(theory_vals, weights=theory_w)
                    tp50 = theory_perc['p50']
                    axes[i].axhline(tp50, color=self.colors['theory'],
                                  linestyle='-', alpha=0.7, label='Theory Median')
            
            formatted_label = format_label_latex(label)
            axes[i].set_title(formatted_label, fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)
        
        for j in range(len(params), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if save_path and self.output_dir:
            save_path = _build_output_path(self.output_dir, 'cumulative',
                                          os.path.basename(save_path), self.mode)
        _save_or_show_plot(save_path)
    
    # ========================================================================
    # Correlation Function Plots
    # ========================================================================

    def plot_xi_and_xi2(self, corr_samples, xivar_df, s12_df, intervals,
                         name, comparison_data=None, save_path=None,
                         theory_weights=None):
        """
        Combined two-panel figure: xi(theta) and xi^2(theta).

        Top panel shows the correlation function ensemble (median +
        68% band across theta), overlaid with the per-interval xivar
        statistic (Planck value and model median/68% CI, drawn as a
        horizontal segment/marker spanning each angular bin).

        Bottom panel shows the squared correlation function (log scale),
        overlaid the same way with the per-interval S statistic.

        Replaces the previous separate ``plot_corr_with_xivar`` /
        ``plot_corr_with_S12`` methods (kept below, commented out, for
        reference/history — not deleted).

        Args:
            corr_samples (np.ndarray): Ensemble of theoretical
                correlation functions, shape (n_samples, len(self.DL.theta)).
            xivar_df (DataFrame): xivar statistics, one column per interval.
            s12_df (DataFrame): S12 statistics, one column per interval.
            intervals (list): Angular intervals (a, b) in cos(theta).
            name (str): Model name (used only for the output filename).
            comparison_data (dict): Comparison data structure (for the
                experimental/Planck values).
            save_path (str): Path to save the figure.
            theory_weights (np.ndarray, optional): Per-sample weights
                (e.g. MCMC likelihood/importance weights) aligned with
                the rows of `corr_samples`, `xivar_df` and `s12_df`.
                When given, every percentile below is likelihood-weighted
                rather than a plain NumPy percentile.
        """
        from .unified_stats import compute_percentiles_weighted

        theta = self.DL.theta
        exp_data, _, _, _, _ = self._prepare_comparison_data(comparison_data)

        corr_samples = np.asarray(corr_samples)

        def _percentiles_over_theta(matrix):
            """Weighted 16/50/84 percentiles at each theta point."""
            n_points = matrix.shape[1]
            p16 = np.empty(n_points)
            p50 = np.empty(n_points)
            p84 = np.empty(n_points)
            for j in range(n_points):
                perc = compute_percentiles_weighted(matrix[:, j], weights=theory_weights)
                p16[j], p50[j], p84[j] = perc['p16'], perc['p50'], perc['p84']
            return p16, p50, p84

        xi_16, xi_50, xi_84 = _percentiles_over_theta(corr_samples)
        xi2_16, xi2_50, xi2_84 = _percentiles_over_theta(corr_samples ** 2)

        # Per-interval bin edges (degrees). All intervals are drawn with a
        # solid line, except the classic [60°, 180°] interval (the one
        # used for S_1/2 in the literature), which is dashed to set it
        # apart from the data-driven partition.
        theta_bins = [tuple(np.arccos([a, b]) * 180 / np.pi) for a, b in intervals]

        def _is_classic_interval(theta_bin, tol=1.0):
            lo, hi = min(theta_bin), max(theta_bin)
            return abs(lo - 60) < tol and abs(hi - 180) < tol

        ls_bins = ['--' if _is_classic_interval(tb) else '-' for tb in theta_bins]

        xi_mean_Planck, xi_mean_16, xi_mean_50, xi_mean_84 = [], [], [], []
        s_Planck, s_16, s_50, s_84 = [], [], [], []

        for (a, b), xiv_col, s_col in zip(intervals, xivar_df.columns, s12_df.columns):
            # xivar: Planck value (no error shown) + weighted model percentiles
            xi_mean_Planck.append(
                exp_data[xiv_col][0] if exp_data and xiv_col in exp_data else np.nan
            )
            mask = xivar_df[xiv_col].notna()
            arr = xivar_df[xiv_col][mask].values
            w = _aligned_weights(mask, theory_weights)
            perc = compute_percentiles(arr, weights=w)
            xi_mean_16.append(perc['p16'])
            xi_mean_50.append(perc['p50'])
            xi_mean_84.append(perc['p84'])

            # S statistic: Planck value (no error shown) + weighted model percentiles
            s_Planck.append(
                exp_data[s_col][0] if exp_data and s_col in exp_data else np.nan
            )
            mask_s = s12_df[s_col].notna()
            arr_s = s12_df[s_col][mask_s].values
            w_s = _aligned_weights(mask_s, theory_weights)
            perc_s = compute_percentiles(arr_s, weights=w_s)
            s_16.append(perc_s['p16'])
            s_50.append(perc_s['p50'])
            s_84.append(perc_s['p84'])

        fig, axes = plt.subplots(figsize=(14, 12), nrows=2, sharex="col",
                                  squeeze=False, gridspec_kw={"hspace": 0})

        ax = axes[0, 0]
        ax.set_ylabel(r"$\xi(\theta)$ [$\mu$K$^2$]")
        ax.plot(theta, xi_50, "k-")
        ax.fill_between(theta, xi_16, xi_84, color="k", alpha=.25)
        ax.grid(c="k", alpha=.1)

        for theta_bin, planck, model_16, model_50, model_84, ls in zip(
                theta_bins, xi_mean_Planck, xi_mean_16, xi_mean_50, xi_mean_84, ls_bins):
            ax.plot(theta_bin, 2 * (planck,), f'r{ls}')
            ax.plot(theta_bin, 2 * (model_50,), f'k{ls}')
            x_mid = (theta_bin[0] + theta_bin[1]) / 2
            ax.errorbar([x_mid], [model_50],
                        yerr=[[model_50 - model_16], [model_84 - model_50]],
                        fmt='none', ecolor='k', elinewidth=1.5,
                        capsize=4, capthick=1.5)

        ax = axes[1, 0]
        ax.set_ylabel(r"$\xi^2(\theta)$ [$\mu$K$^4$]")
        ax.set_yscale("log")
        ax.plot(theta, xi2_50, "k-")
        ax.fill_between(theta, xi2_16, xi2_84, color="k", alpha=.25)
        ax.grid(c="k", alpha=.1)

        for theta_bin, planck, model_16, model_50, model_84, ls in zip(
                theta_bins, s_Planck, s_16, s_50, s_84, ls_bins):
            ax.plot(theta_bin, 2 * (planck,), f'r{ls}')
            ax.plot(theta_bin, 2 * (model_50,), f'k{ls}')
            x_mid = (theta_bin[0] + theta_bin[1]) / 2
            ax.errorbar([x_mid], [model_50],
                        yerr=[[model_50 - model_16], [model_84 - model_50]],
                        fmt='none', ecolor='k', elinewidth=1.5,
                        capsize=4, capthick=1.5)

        ax.set_xlabel(r"$\theta$ [º]")

        if save_path and self.output_dir:
            save_path = _build_output_path(self.output_dir, 'correlation',
                                          os.path.basename(save_path), self.mode)
        _save_or_show_plot(save_path)

    # ------------------------------------------------------------------
    # DEPRECATED — replaced by plot_xi_and_xi2() above.
    # Kept commented out (not deleted) for reference.
    # ------------------------------------------------------------------
    # def plot_corr_with_xivar(self, corr_theory, xivar_df, intervals,
    #                           name, comparison_data=None, save_path=None):
    #     """
    #     Plot correlation function with xivar statistic overlays.
    #
    #     Args:
    #         corr_theory (np.ndarray): Theoretical correlation
    #         xivar_df (DataFrame): xivar statistics
    #         intervals (list): Angular intervals
    #         name (str): Model name
    #         comparison_data (dict): Comparison data
    #         save_path (str): Path to save
    #     """
    #     xvals = self.DL.xvals
    #     corr_exp, _ = self.DL.get_correlation_function()
    #
    #     exp_vals, sim_data, _, _ = self._prepare_comparison_data(comparison_data)
    #
    #     fig, ax = plt.subplots(figsize=(15, 8))
    #
    #     # Plot correlation functions
    #     ax.scatter(np.arccos(xvals) * 180 / np.pi, corr_exp,
    #                s=2, marker='o', c=self.colors['experimental'],
    #                label='Correlation function data')
    #     ax.scatter(np.arccos(xvals) * 180 / np.pi, corr_theory,
    #                s=2, marker='o', c=self.colors['theory'],
    #                label='Correlation function model')
    #
    #     # Overlay xivar intervals
    #     for (a, b), col in zip(intervals, xivar_df.columns):
    #         mean_exp, _ = self.DL.get_xivar(a, b)   # error discarded
    #         arr = xivar_df[col].dropna().values
    #         perc = compute_percentiles(arr)
    #         mean_th = perc['p50']
    #         std_th = (perc['p84'] - perc['p16']) / 2
    #
    #         self._plot_statistic_interval(
    #             ax, (a, b), mean_exp, None, mean_th, std_th,
    #             self.colors['experimental'], self.colors['theory'],
    #             r'\xi', col
    #         )
    #
    #     ax.set_ylim(-400, 400)
    #     ax.set_xlabel(r"$\theta$ [º]", fontsize=14)
    #     ax.set_ylabel(r"$C(\theta)$", fontsize=14)
    #     ax.set_title(rf"$\xi_a^b$ analysis for {name}", fontsize=16)
    #     ax.legend(fontsize=7)
    #     ax.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #
    #     if save_path and self.output_dir:
    #         save_path = _build_output_path(self.output_dir, 'correlation',
    #                                       os.path.basename(save_path), self.mode)
    #     _save_or_show_plot(save_path)
    #
    # def plot_corr_with_S12(self, corr_theory, s12_df, intervals,
    #                         name, comparison_data=None, save_path=None):
    #     """
    #     Plot squared correlation function with S12 statistic overlays.
    #
    #     Args:
    #         corr_theory (np.ndarray): Theoretical correlation
    #         s12_df (DataFrame): S12 statistics
    #         intervals (list): Angular intervals
    #         name (str): Model name
    #         comparison_data (dict): Comparison data
    #         save_path (str): Path to save
    #     """
    #     xvals = self.DL.xvals
    #     corr_exp, _ = self.DL.get_correlation_function()
    #     _, sim_data, _, _ = self._prepare_comparison_data(comparison_data)
    #
    #     fig, ax = plt.subplots(figsize=(15, 8))
    #
    #     # Plot squared correlation functions
    #     ax.scatter(np.arccos(xvals) * 180 / np.pi, corr_exp**2,
    #                s=2, marker='o', c=self.colors['experimental'],
    #                label='Correlation function data')
    #     ax.scatter(np.arccos(xvals) * 180 / np.pi, corr_theory**2,
    #                s=2, marker='o', c=self.colors['theory'],
    #                label='Correlation function model')
    #
    #     # Overlay S12 intervals
    #     for (a, b), col in zip(intervals, s12_df.columns):
    #         try:
    #             mean_exp, _ = self.DL.get_s12(a, b)   # error discarded
    #         except FileNotFoundError:
    #             continue
    #
    #         arr = s12_df[col].dropna().values
    #         perc = compute_percentiles(arr)
    #         mean_th = perc['p50']
    #         std_th = (perc['p84'] - perc['p16']) / 2
    #
    #         self._plot_statistic_interval(
    #             ax, (a, b), mean_exp, None, mean_th, std_th,
    #             self.colors['experimental'], self.colors['theory'],
    #             'S', col
    #         )
    #
    #     ax.set_yscale('symlog', linthresh=10)
    #     ax.set_xlabel(r"$\theta$ [º]", fontsize=14)
    #     ax.set_ylabel(r"$C(\theta)^2$", fontsize=14)
    #     ax.set_title(rf"$S_a^b$ analysis for {name}", fontsize=16)
    #     ax.legend(fontsize=7)
    #     ax.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #
    #     if save_path and self.output_dir:
    #         save_path = _build_output_path(self.output_dir, 'correlation',
    #                                       os.path.basename(save_path), self.mode)
    #     _save_or_show_plot(save_path)

    def plot_power_and_correlation(self, mean_Cl, std_Cl, mean_Cor, std_Cor,
                                     root="Title", save_path=None):
        """
        Side-by-side power spectrum and correlation function with uncertainty bands.
        
        Args:
            mean_Cl (np.ndarray): Mean theoretical power spectrum
            std_Cl (np.ndarray): Std dev of power spectrum
            mean_Cor (np.ndarray): Mean theoretical correlation
            std_Cor (np.ndarray): Std dev of correlation
            root (str): Plot title base
            save_path (str): Path to save
        """
        xvals = self.DL.xvals
        theta = np.arccos(xvals) * 180 / np.pi
        corr_exp, corr_err = self.DL.get_correlation_function()
        
        ell = self.DL.ell
        D_ell_exp = self.DL.D_ell
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        # Left panel: Power spectrum
        ax1.scatter(ell, D_ell_exp, s=5, c=self.colors['experimental'],
                   label='Experimental Data', zorder=3)
        ax1.plot(ell, mean_Cl, color=self.colors['theory'],
                linewidth=2, label='Theory Mean', zorder=2)
        ax1.fill_between(ell, mean_Cl - std_Cl, mean_Cl + std_Cl,
                        color=self.colors['theory'], alpha=0.3,
                        label=r'Theory $\pm 1\sigma$', zorder=1)
        
        ax1.set_xlabel(r'Multipole $\ell$', fontsize=14)
        ax1.set_ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
        ax1.set_title('Power Spectrum', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Correlation function
        ax2.scatter(theta, corr_exp, s=5, c=self.colors['experimental'],
                   label='Experimental Data', zorder=3)
        ax2.plot(theta, mean_Cor, color=self.colors['theory'],
                linewidth=2, label='Theory Mean', zorder=2)
        ax2.fill_between(theta, mean_Cor - std_Cor, mean_Cor + std_Cor,
                        color=self.colors['theory'], alpha=0.3,
                        label=r'Theory $\pm 1\sigma$', zorder=1)
        
        ax2.set_xlabel(r'$\theta$ [degrees]', fontsize=14)
        ax2.set_ylabel(r'$C(\theta)$ [$\mu K^2$]', fontsize=14)
        ax2.set_title('Correlation Function', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Power Spectrum & Correlation: {root}', fontsize=16)
        plt.tight_layout()
        
        if save_path and self.output_dir:
            save_path = _build_output_path(self.output_dir, 'powerspectrum',
                                          os.path.basename(save_path), self.mode)
        _save_or_show_plot(save_path)