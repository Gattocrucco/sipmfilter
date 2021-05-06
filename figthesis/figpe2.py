from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

lengths = {
    5.5: 128,
    7.5: 64,
    9.5: 64
}

################

figs = []

for i, (vov, length) in enumerate(lengths.items()):
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    fig = plt.figure(num=f'figpe2-{i}', clear=True, figsize=[8, 4])

    ap21.sim.hist('mainheight', f'~saturated & ~closept & (mainpos >= 0) & (length == {length})', yscale='log', fig=fig, nbins=1000, selection=False)
    ax, = fig.get_axes()
    ax.set_xlabel('Laser peak height')
    textbox.textbox(ax, f'{vov} VoV\nFilter {length} ns', fontsize='medium', loc='center right')
    ilen = np.searchsorted(ap21.sim.filtlengths, length)
    boundaries = ap21.sim.computenpeboundaries(ilen)
    afterpulse_tile21.vlines(ax, boundaries, linestyle=':')
    
    figs.append(fig)

for fig in figs:
    fig.tight_layout()
    fig.show()

figlatex.save(figs)
