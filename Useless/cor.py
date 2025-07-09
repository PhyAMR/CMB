import healpy as hp
import numpy as np
import treecorr
import matplotlib.pyplot as plt

# Scientific plotting setup
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': False  # Set True if LaTeX installed
})

fig, ax = plt.subplots(figsize=(8, 6))

# Load first map
print('Loading Map 1')
map1_path = 'maps/COM_CMB_IQU-commander_2048_R3.00_full.fits'  # Update path
map1 = hp.read_map(map1_path, field=5) * 1e6  # Field 5, convert to μK
nside = hp.get_nside(map1)
low_nside = nside // 2
map1_low = hp.ud_grade(map1, low_nside)

# Load second map (change path/field as needed)
print('Loading Map 2')
#map2_path = 'maps/COM_CMB_IQU-commander_2048_R3.00_full.fits'  # Update path
#map2 = hp.read_map(map2_path, field=5) * 1e6  # Field 6, convert to μK
#map2_low = hp.ud_grade(map2, low_nside)

# Get coordinates (same for both maps)
theta, phi = hp.pix2ang(low_nside, np.arange(len(map1_low)))
ra = np.degrees(phi)
dec = np.degrees(0.5*np.pi - theta)

# Create catalogs
cat1 = treecorr.Catalog(ra=ra, dec=dec, k=map1_low, ra_units='deg', dec_units='deg')
#cat2 = treecorr.Catalog(ra=ra, dec=dec, k=map2_low, ra_units='deg', dec_units='deg')

# Correlation configuration
config = {
    'bin_size': 0.1,      # Degrees
    'min_sep': 180.0-0.1,       # Degrees
    'max_sep': 180.0,      # Degrees
    'sep_units': 'deg',
    'bin_slop': 0.01
}

# Compute correlations
print('Calculating autocorrelation 1')
kk = treecorr.KKCorrelation(config)
kk.process(cat1)

r = kk.meanr
ax.plot(r, kk.xi, 'b-', label='Correlation of COMMANDER map')
print('Calculating autocorrelation 2')

#kk.process(cat2)
#ax.plot(r, kk.xi, 'r--', label='Correlation of SMICA map')

print('Calculating cross-correlation')

#kk.process(cat1, cat2)
#ax.plot(r, kk.xi, 'g:', label='Cross correlation')



# Plot data with error bars

ax.set_xlabel(r'Angular Separation [$^{\circ}$]')
ax.set_ylabel(r'Correlation $\xi(\theta)$')
#ax.set_xscale('log')
ax.grid(True, which='both', ls='--', alpha=0.6)
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig('correlations.png', dpi=300, bbox_inches='tight')
plt.close()