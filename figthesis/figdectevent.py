from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

vov = 9.5
ievt = 59689

################

fig, axs = plt.subplots(1, 2, num='figdectevent', clear=True, figsize=[9, 3], sharey=True, sharex=True)

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

ilen = np.searchsorted(ap21.sim.filtlengths, ap21.params['aplength'])
for ax, apampl in zip(axs, [True, False]):
    ap21.sim.plotevent(ap21.datalist, ievt, ilen, zoom='main', fig=ax, apampl=apampl)

textbox.textbox(axs[0], f'{vov} VoV', fontsize='medium', loc='lower left')

fig.tight_layout()
fig.show()

figlatex.save(fig)
