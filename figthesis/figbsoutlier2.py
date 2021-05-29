from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

figs = []

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    
    row = []
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    datalist = ap21.datalist

    i0 = ap21.sim.getexpr('argmin(baseline)')
    i1 = ap21.sim.getexpr('argmax(baseline)')

    for i, ievt in enumerate([i0, i1]):
        fig = plt.figure(num=f'figbsoutlier2-{j}{i}', clear=True, figsize=[4.5, 3.5])

        ap21.sim.plotevent(datalist, ievt, 2, zoom='all', fig=fig)
        ax, = fig.get_axes()
        ax.legend(loc='lower left', fontsize='small')
        textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower right')

        fig.tight_layout()
        fig.show()
    
        row.append(fig)
    
    figs.append(row)

figlatex.save(figs)
