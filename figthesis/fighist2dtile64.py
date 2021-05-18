import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import hist2d
import colormap

cmd = '-m 100000 -L 15250 -U 15750 darksidehd/merged_000886.root:64'

###########################

figs = []

cmap = colormap.uniform()

figname = 'fighist2dtile64'
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
