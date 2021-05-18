import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import hist2d
import colormap

cmd = '-m 100000 -t -l 500 -L 1 darksidehd/nuvhd_lf_3x_tile52_77K_64V_6VoV_1.wav'

###########################

cmap = colormap.uniform()

figname = 'fighist2dtile52'
fig = plt.figure(num=figname, clear=True, figsize=[9, 4])

save = f'figthesis/{figname}.npz'
if not os.path.exists(save):
    hist = hist2d.Hist2D(cmd.split())
    print(f'save {save}...')
    hist.save(save, compress=True)
print(f'load {save}...')
hist = hist2d.Hist2D.load(save)

hist.hist2d(fig, cmap=cmap)

figlatex.save(fig)

fig.show()
