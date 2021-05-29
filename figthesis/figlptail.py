from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5
flengths = [64, 128]

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

event = ap21.sim.getexpr('event[argmax(mainpos)]', f'(length=={flengths[0]})&(mainnpe==1)&(event!=22209)')
# If you look at event 22209, your eyes will cry blood
# Serious version: it is one of those delayed for real pulses.

figs = []
for i, length in enumerate(flengths):
    fig = plt.figure(num=f'figlptail-{i}', clear=True, figsize=[4.5, 3])
    
    ilen = np.searchsorted(ap21.sim.filtlengths, length)
    ap21.sim.plotevent(ap21.datalist, event, ilen, zoom='main', fig=fig)
    ax, = fig.get_axes()
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower left')

    figs.append(fig)

for fig in figs:
    fig.tight_layout()
    fig.show()

figlatex.save([figs])
