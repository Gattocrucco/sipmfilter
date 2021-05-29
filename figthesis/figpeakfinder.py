from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5
length = 128

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

event0 = ap21.sim.getexpr('event[argmin(ptBpos - ptApos)]', f'(mainnpe > 0) & (ptAnpe > 0) & (ptBnpe > 0) & (length == {length})')
event1 = ap21.sim.getexpr('event[argmin(apApos - mainpos)]', f'(mainnpe > 0) & (apAnpe > 0) & (length == {length})')

figs = []
for i, event in enumerate([event0, event1]):
    fig = plt.figure(num=f'figpeakfinder-{i}', clear=True, figsize=[4.5, 3.5])
    
    if i == 0:
        zoom = 'pretrigger'
        vovloc = 'lower right'
        legloc = 'lower left'
    else:
        zoom = 'main'
        vovloc = 'lower left'
        legloc = 'lower right'
    
    ilen = np.searchsorted(ap21.sim.filtlengths, length)
    ap21.sim.plotevent(ap21.datalist, event, ilen, zoom=zoom, fig=fig)
    ax, = fig.get_axes()
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc=vovloc)
    ax.legend(loc=legloc, fontsize='x-small')

    figs.append(fig)

for fig in figs:
    fig.tight_layout()
    fig.show()

figlatex.save([figs])
