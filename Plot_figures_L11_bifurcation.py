import numpy as np
import glob, h5py
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


d = 0.31325  # Fixed gap-width
RES = 25  # number of contour levels to use
markersize = 10


def Plot_full_bif(folder, ax, line='k-', shift=False):
    """
    Plot out the bifurcation diagram and locate all the states
    corresponding to the fold points
    """ 
    def add_to_fig(obj):
        index = np.where(obj.Ra_dot[:-1]*obj.Ra_dot[1:] < 0)
        _, idx = np.unique(np.round(obj.Ra[index], 3), return_index=True)
        idx = np.sort(idx)
        Ra_f = obj.Ra[index][idx]
        KE_f = obj.KE[index][idx]

        if shift == True:
            ax.plot(obj.Ra - 2*np.ones(len(obj.Ra)), obj.KE, line)
        else:
            ax.plot(obj.Ra, obj.KE, line)
            #ax.plot(Ra_f, KE_f, 'ro', markersize=markersize)

        # Return Saddles
        if len(obj.Y_FOLD) != 0:
            return obj.Y_FOLD[idx], obj.Ra[index][idx]
        else:
            return [], []
        
    X_fold = []
    Ra_fold = []
    N_fm_fold = []
    N_r_fold = []
    for filename in glob.glob(folder + '/*.h5'):
      
        obj = result()
        with h5py.File(filename, 'r') as f:
            ff = f["Bifurcation"]
            for key in ff.keys():
                setattr(obj, key, ff[key][()])
            N_fm = f['Parameters']["N_fm"][()]
            N_r = f['Parameters']["N_r"][()]

            X_f, Ra_f = add_to_fig(obj)
            for X_i, Ra_i in zip(X_f, Ra_f):
                X_fold.append(X_i)
                N_fm_fold.append(N_fm)
                N_r_fold.append(N_r)
                Ra_fold.append(Ra_i)

    return X_fold, N_r_fold, N_fm_fold, Ra_fold

# %%
print('Load above')
dir = '/home/pmannix/SpectralDoubleDiffusiveConvection/Figure_L11/'

# %%
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
# L = 11 two pulse close up Ras=150
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6), layout='constrained')

# A) Plot the bifurcation diagram
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'one_pulse/', ax, line='k-')
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'two_pulse_l12plus/', ax, line='k-.')

ax.set_ylabel(r'$\mathcal{E}$', fontsize=25)
ax.set_xlabel(r'$Ra$', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.set_xlim([2660, 4560])
ax.set_ylim([0, 5])

# C) Add inset to show pitchfork
axins = inset_axes(ax, width="70%", height="60%", loc='upper right', borderpad=2)

X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'one_pulse/', axins, line='k-')
X_folds, Nr_folds, Nfm_folds, Ra_folds = Plot_full_bif(dir + 'two_pulse_l12plus/', axins, line='k-.')

axins.annotate(r'$\ell^1_{11}$', xy=(4405,0.00075), textcoords='data', fontsize=25, rotation =0)
axins.annotate(r'$\ell^2_{11}$', xy=(4405,0.0055), textcoords='data', fontsize=25, rotation =0)

axins.annotate(r'$\ell = 11$', xy=(4515,0.0001), textcoords='data', fontsize=25, rotation =-20)
axins.annotate(r'$\ell = 12^+$', xy=(4540,0.0001), textcoords='data', fontsize=25, rotation =-22.5)

axins.set_ylabel(r'$\mathcal{E}$', fontsize=25)
axins.set_xlabel(r'$Ra$', fontsize=25)
axins.tick_params(axis='both', labelsize=25)
axins.set_xlim([4400, 4560])
axins.set_ylim([0, 0.0075])

plt.savefig('L11_bifurcation_Ras150.png', format='png', dpi=400)
plt.show()

# %%
