import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import hist2d
import colormap

commands = [
    f'-m 100000 -L 1 darksidehd/LF_TILE21_77K_54V_{VoV}VoV_2.wav'
    for VoV in [65, 69, 73]
]

###########################

figs = []

cmap = colormap.uniform()

for ifile, cmd in enumerate(commands):

    figname = f'fighist2dtile21-{ifile}'
    fig = plt.figure(num=figname, clear=True, figsize=[9, 4])
    
    save = f'figthesis/{figname}.npz'
    if not os.path.exists(save):
        hist = hist2d.Hist2D(cmd.split())
        print(f'save {save}...')
        hist.save(save, compress=True)
    print(f'load {save}...')
    hist = hist2d.Hist2D.load(save)
    
    hist.hist2d(fig, cmap=cmap)

    figs.append(fig)

for fig in figs:
    figlatex.save(fig)

for fig in figs:
    fig.show()
