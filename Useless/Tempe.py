import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific plotting setup
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'text.usetex': False  # Set True if LaTeX installed
})

fig, ax = plt.subplots(figsize=(8, 6))
rot_phi=180
rot_theta=180
r = hp.Rotator(rot=[rot_phi, rot_theta], deg=True)

# Load first map
print('Loading Map 1')
map1_path = 'maps/COM_CMB_IQU-commander_2048_R3.00_full.fits'  # Update path
map1 = hp.read_map(map1_path, field=5) * 1e6  # Field 5, convert to μK
nside = hp.get_nside(map1)
low_nside = nside // 2
map1_low = hp.ud_grade(map1, low_nside)
print(map1_low.shape)
print(map1_low[:20])
# Load second map (change path/field as needed)
print('Loading Map 2')
map2_path = 'maps/COM_CMB_IQU-commander_2048_R3.00_full.fits'  # Update path
map2 = r.rotate_map_pixel(map1_low)#hp.read_map(map2_path, field=5) * 1e6  # Field 6, convert to μK
print(map2[:20])


mult = map1_low*map2
print('Expected T*T_{inv}',np.sum(mult)/(len(mult)))
ax.scatter(map1_low,map2,marker='o',c='blue', label= 'Impainting Commander', s=2, alpha=0.05) #map1_low, cmap='viridis'
ax.set_xlabel(r'T [$K_{CMB}$]')
ax.set_ylabel(r'Inverted map T [$K_{CMB}$]')
#ax.set_xscale('log')
ax.grid(True, which='both', ls='--', alpha=0.6)
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig('Temperaturegal.png', dpi=300, bbox_inches='tight')
plt.close() 