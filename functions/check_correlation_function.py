
import numpy as np
import matplotlib.pyplot as plt
from functions.cosmology import compute_cl_cor_pl
from functions.correlation_function import correlation_func

def main():
    """
    This script compares the two-point correlation function computed by
    the user's implementation in `functions.correlation_function.py` and
    the CAMB library's implementation as used in `functions.cosmology.py`.
    """
    # 1. Set up cosmological parameters (Planck 2018 best-fit)
    parss = {
        'omegabh2': 0.02242,
        'omegach2': 0.11933,
        'H0': 67.66,
        'omegak': 0,
        'yheused': 0.245,
        'nnu': 3.046,
        'nrun': 0,
        'Alens': 1.0,
        'ns': 0.9665,
        'logA': 3.047,
        'wa': 0,
        'mnu': 0.06,
        'tau': 0.0561
    }

    # 2. Set up lmax and theta values
    lmax = 2500
    theta_deg_full = np.linspace(0, 180, 361)
    
    # Exclude endpoints to avoid division by zero in CAMB's cl2corr
    theta_deg_calc = np.linspace(0.001, 179.999, 359)
    theta_rad_calc = np.deg2rad(theta_deg_calc)
    xvals_calc = np.cos(theta_rad_calc)

    # 3. Compute D_ell and correlation function using CAMB's function from cosmology.py
    # Note: compute_cl_cor_pl returns D_ell from l=2.
    # D_ell = l(l+1)Cl / 2pi
    ttcl_camb, camb_corr, _ = compute_cl_cor_pl(parss, lmax, xvals_calc)

    # 4. Compute correlation function using the implementation from correlation_function.py
    # This function expects D_ell starting from l=2.
    my_corr = correlation_func(ttcl_camb, xvals_calc)

    # 5. Plot the results
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot both correlation functions
    ax1.plot(theta_deg_calc, camb_corr, label='CAMB correlation function (from cosmology.py)')
    ax1.plot(theta_deg_calc, my_corr, label='Custom correlation function', linestyle='--')
    ax1.set_ylabel('C(theta) [μK^2]')
    ax1.set_title('Comparison of Correlation Function Implementations')
    ax1.legend()
    
    # Plot the difference
    difference = my_corr - camb_corr
    ax2.plot(theta_deg_calc, difference, color='C2')
    ax2.set_xlabel('Theta (degrees)')
    ax2.set_ylabel('Difference (Custom - CAMB) [μK^2]')
    ax2.set_title('Difference Between Implementations')
    
    plt.tight_layout()
    output_filename = 'correlation_function_comparison.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == '__main__':
    main()
