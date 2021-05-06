from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

lengths = {
    5.5: 128,
    7.5: 64,
    9.5: 64,
}

################

gen = np.random.default_rng(202105060104)

pages = []

for j, (vov, length) in enumerate(lengths.items()):
    
    figs = []

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    
    allevents = ap21.sim.eventswhere('all(mainpos < 0, axis=0)')
    events = gen.choice(allevents, 12, replace=False)
    
    for i, event in enumerate(events):
        
        fig = plt.figure(num=f'figverymissing-{j}{i}', clear=True, figsize=[3, 2.7])
        figs.append(fig)

        ilen = np.searchsorted(ap21.sim.filtlengths, length)
        assert ap21.sim.filtlengths[ilen] == length
        ap21.sim.plotevent(ap21.datalist, event, ilen, zoom='main', fig=fig)
        
        ax, = fig.get_axes()
        ax.legend().set_visible(False)
        ax.set_xlabel('')
        textbox.textbox(ax, f'Filter {length} ns', fontsize='medium', loc='lower right')
        textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower left')
        
        fig.tight_layout()
    
    figs = np.array(figs).reshape(4, 3)
    pages.append(figs)

for figs in pages:
    figlatex.save(figs)
