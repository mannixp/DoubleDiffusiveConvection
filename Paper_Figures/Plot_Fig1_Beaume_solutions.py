"""
This script generates a bifurcation diagram and spatial solutions for a
planar bifurcation problem. It reads data from specified files, processes
the data, and creates contour plots to visualize the results. The script
uses Matplotlib for plotting and NumPy for numerical operations.

To run this script, ensure you have the required libraries installed:
- numpy
- matplotlib
You can install them using pip:
pip install numpy matplotlib

To execute the script, run the following command in your terminal:
python plot_solutions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm

# Uncomment the following lines to use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def cmap(Z, RES=12):
    """
    Generate a custom colormap and normalization for visualizing data.

    This function creates a colormap based on the 'RdBu_r' colormap, with a 
    specified number of levels. The color corresponding to the value 0 is 
    replaced with white to emphasize the zero level. The function also 
    returns the levels and a normalization object for mapping data values 
    to the colormap.

    Parameters:
    -----------
    Z : numpy.ndarray
        A 2D array of data values for which the colormap will be created. 
        The maximum absolute value in `Z` determines the range of the colormap.
    RES : int, optional
        The number of levels in the colormap. Default is 12. It must be even.
        This parameter is used to create evenly spaced levels for the colormap.
    Returns:
    --------
    levels : numpy.ndarray
        An array of evenly spaced levels for the colormap, symmetric around 0.
    custom_cmap : matplotlib.colors.ListedColormap
        A custom colormap with the color corresponding to 0 replaced by white.
    norm : matplotlib.colors.BoundaryNorm
        A normalization object for mapping data values to the colormap.

    Notes:
    ------
    - The function assumes that the input array `Z` is 2D and contains numeric 
      values.

    Example:
    --------
    >>> import numpy as np
    >>> Z = np.random.randn(10, 10)
    >>> levels, custom_cmap, norm = cmap(Z)
    """

    print(np.asarray([min(Z.flatten()), max(Z.flatten())]))
    Max = np.max(abs(Z))

    assert RES % 2 == 0  # ensure RES is even

    # Define levels (must include 0)
    levels = np.linspace(-Max, Max, RES)

    # Create RdBu_r colormap with 12 colors
    original_cmap = plt.get_cmap('RdBu_r', len(levels) - 1)
    colors = original_cmap(np.arange(original_cmap.N))

    # Find the index of the bin that contains 0
    # Since levels are symmetric, 0 will be in the middle bin
    zero_bin_index = (len(levels) - 1) // 2  # e.g., 6 if 12 bins

    # Replace that color with white
    colors[zero_bin_index] = [1, 1, 1, 1]  # RGBA for white
    custom_cmap = ListedColormap(colors)

    # Create normalization
    norm = BoundaryNorm(levels, custom_cmap.N)

    return levels, custom_cmap, norm


def transform(file):
    """
    Transforms data from a file into structured arrays for plotting.

    This function reads data from a file, extracts specific columns, and reshapes
    the data into 2D arrays suitable for plotting. It also removes an offset from
    the computed values.

    Args:
        file (str): Path to the file containing the data. The file should be in a
                    format readable by `numpy.loadtxt` and contain at least 5 columns.

    Returns:
        tuple: A tuple containing three 2D numpy arrays:
            - X (numpy.ndarray): 2D array of x-coordinates (reshaped from column 1).
            - Z (numpy.ndarray): 2D array of z-coordinates (reshaped from column 2).
            - T (numpy.ndarray): 2D array of transformed values (computed from column 5).
    """
    # Load data from the file
    data = np.loadtxt(file)

    # Extract columns for plotting (column 1, column 2, and computed value for color)
    x = data[:, 0]  # Column 1 - but python is zero-indexed
    z = data[:, 1]  # Column 2
    t = data[:, 4]  # Column 5

    Nz = 17
    Nx = 500
    X = np.zeros((Nz, Nx))
    Z = np.zeros((Nz, Nx))
    T = np.zeros((Nz, Nx))
    for i in range(Nx):
        X[:, i] = x[i*Nz:(i+1)*Nz]
        Z[:, i] = z[i*Nz:(i+1)*Nz]
        T[:, i] = t[i*Nz:(i+1)*Nz]

    T += (Z - 1/2)  # Remove the offset

    # Print file and max/min
    print(file, np.asarray([min(T.flatten()), max(T.flatten())]), '\n')

    return X, Z, T


print('Load above')
dir = '/home/pmannix/Spatial_Localisation/SpectralDoubleDiffusiveConvection/Paper_Data/Beamue2011_Data/'

# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

fig = plt.figure(figsize=(16, 6), layout='constrained')

# Create a GridSpec with custom width ratios
gs = GridSpec(6, 2, figure=fig, width_ratios=[0.4, 1.59])  # Adjust width_ratios as needed

# Create the left column as a single subplot
ax = fig.add_subplot(gs[:, 0])

# Create the right column subplots
axs = np.empty((6, 1), dtype=object)
for i in range(6):
    axs[i, 0] = fig.add_subplot(gs[i, 1], aspect=1)


# A) Bifurcation diagram
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

# Load data from bifdiag1.dat and bifdiag2.dat
data1 = np.loadtxt(dir + 'Planarfigure1_2/bifdiag1.dat')
data2 = np.loadtxt(dir + 'Planarfigure1_2/bifdiag2.dat')

# Extract columns for plotting (column 2 and column 7, zero-indexed)
x1, y1 = data1[:, 1], data1[:, 6]
x2, y2 = data2[:, 1], data2[:, 6]

# Create the plot
ax.plot(x1, y1, 'k-', linewidth=2, label=r'$L^{1-}_{10}$')
ax.plot(x2, y2, 'k:', linewidth=2, label=r'$L^{1+}_{10}$')

# Add labels, legend, and title
ax.set_xlabel(r'$Ra_T$', fontsize=30)
ax.set_ylabel(r'$E$', fontsize=30)

# Show the plot
ax.set_xlim(2660, 2800)
ax.set_ylim(0, 290)
ax.tick_params(axis='both', which='major', labelsize=20)

E = np.array([35-2, 65-4, 90-3,   25-3, 50-3, 75])
Ra= np.array([2700-4, 2700-7, 2700-7,   2710-6, 2700-6, 2700-6])
letters = [r'1',r'2',r'3', r'a',r'b',r'c']
for i, xy in enumerate(zip(Ra, E)):
    ax.annotate(letters[i], xy=xy, textcoords='data', fontsize=15)

# B) Spatial solutions
# ~~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

files = ['Planarfigure2_2/SOL_sn1.plt','Planarfigure2_2/SOL_sn2.plt',
        'Planarfigure2_2/SOL_sn3.plt','Planarfigure2_2/SOL_sna.plt',
        'Planarfigure2_2/SOL_snb.plt','Planarfigure2_2/SOL_snc.plt']

# Create a contour plot
for i, file in enumerate(files):
    X, Z, T = transform(dir + file)
    levels, custom_cmap, norm = cmap(T)
    axs[::-1, 0][i].contour(X, Z, T, levels=levels, cmap=custom_cmap, norm=norm)
    axs[::-1, 0][i].contourf(X, Z, T, levels=levels, cmap=custom_cmap, norm=norm)
    axs[::-1, 0][i].set_xticks([])
    axs[::-1, 0][i].set_yticks([])


letters = [r'1',r'2',r'3', r'a',r'b',r'c']
for i, letter in enumerate(letters):
    axs[::-1, 0][i].annotate(letter, xy=(-0.025, 0.5), xycoords='axes fraction', fontsize=20, color='k')

# ~~~~~~~~~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

# Save the figure
plt.savefig('planar_bifurcation_diagram.png', dpi=100)
plt.show()
