import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import hist2d
import colormap

commands = [
    '-m 100000 -L 860 -v 860 -l 8900 darksidehd/LF_TILE15_77K_73V_9VoV_1.wav',
    '-m 100000 -L 750 -v 750 -l 8900 darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav',
    '-m 100000 -L 750 -v 750 -l 8900 darksidehd/nuvhd_lf_3x_tile59_77K_64V_6VoV_1.wav',
]

###########################

figs = []

cmap = colormap.uniform()

for ifile, cmd in enumerate(commands):

    figname = f'fighist2dtile155759-{ifile}'
    fig = plt.figure(num=figname, clear=True, figsize=[9, 3.55])
    
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
    fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
    print(figlatex.figlatex(fig))

for fig in figs:
    fig.show()
