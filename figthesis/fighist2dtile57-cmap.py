import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as _colors
from scipy import interpolate

import figlatex
import hist2d
import colormap

command = '-m 100000 -L 1 -t -l 500 darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'

###########################

def naivelinear(colors=['black', '#f55', 'white'], N=256, return_pos=False):
    rgb0 = np.array([_colors.to_rgb(color) for color in colors])
    t0 = np.linspace(0, 1, len(rgb0))
    t = np.linspace(0, 1, N)
    rgb = interpolate.interp1d(t0, rgb0, axis=0)(t)
    rt = _colors.ListedColormap(rgb)
    if return_pos:
        rt = (rt, t0)
    return rt

cmap1 = naivelinear()
cmap2 = colormap.uniform()

figs = []

for ifile, cmap in enumerate([cmap1, cmap2]):

    figname = f'fighist2dtile57-cmap-{ifile}'
    fig = plt.figure(num=figname, clear=True, figsize=[9, 4])
    
    save = f'figthesis/{figname}.npz'
    if not os.path.exists(save):
        hist = hist2d.Hist2D(command.split())
        print(f'save {save}...')
        hist.save(save, compress=True)
    print(f'load {save}...')
    hist = hist2d.Hist2D.load(save)
    
    hist.hist2d(fig, cmap=cmap)

    figs.append(fig)

for fig in figs:
    fig.show()
