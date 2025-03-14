"""
Script that generates the figure 7 for section 5.1

To run this script excute

python3 Plot_figures_L10_bifurcation.py

from within the Paper_Figures directory.
"""
import numpy as np
import glob, h5py

import sys
import os

sys.path.append(os.path.abspath("../"))

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Plot_Tools import Spectral_To_Gridpoints
from Main import result, Kinetic_Energy
from Matrix_Operators import cheb_radial
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


d = 0.3521  # Fixed gap-width
RES = 25  # number of contour levels to use
markersize = 10


# %%
print('Load above')
dir = '/home/pmannix/Spatial_Localisation/SpectralDoubleDiffusiveConvection/Paper_Data/Figure_L10_TSN_Ras400/'


# %%

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6), layout='constrained')

# A) Plot the bifurcation diagram

filename = dir + 'L10_Plus/L10Plus_1.h5'
obj = result()
with h5py.File(filename, 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])

ax.plot(obj.Ra, obj.KE, 'k:', linewidth=2, label=r'$L^{A+}_{10}$')

filename = dir + 'L10_Minus/L10Minus_1.h5'
obj = result()
with h5py.File(filename, 'r') as f:
    ff = f["Bifurcation"]
    for key in ff.keys():
        setattr(obj, key, ff[key][()])

ax.plot(obj.Ra, obj.KE, 'k-', linewidth=2, label=r'$L^{A-}_{10}$')

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra_T$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.legend(loc=3,fontsize=25)
ax.set_ylim([0, .75e-03])
ax.set_xlim([8000, 8375])

plt.savefig('Bifurcation_L10_Ras400_TSN.png', format='png', dpi=100)
plt.show()