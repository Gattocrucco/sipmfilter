from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

flengths = [64, 128]

################

figs = []

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    event = ap21.sim.getexpr('event[argmax(mainpos)]', f'(length=={flengths[0]})&(mainnpe==1)&(event!=22209)')

    row = []
    for i, length in enumerate(flengths):
        fig = plt.figure(num=f'figlptail2-{j}{i}', clear=True, figsize=[4.5, 3.5])
    
        ilen = np.searchsorted(ap21.sim.filtlengths, length)
        ap21.sim.plotevent(ap21.datalist, event, ilen, zoom='main', fig=fig)
        ax, = fig.get_axes()
        textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower left')

        row.append(fig)
    
    figs.append(row)

for row in figs:
    for fig in row:
        fig.tight_layout()
        fig.show()

figlatex.save(figs)
