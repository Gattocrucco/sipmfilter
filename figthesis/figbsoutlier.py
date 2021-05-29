from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)
datalist = ap21.datalist

i0 = ap21.sim.getexpr('argmin(baseline)')
i1 = ap21.sim.getexpr('argmax(baseline)')

figs = []
for i, ievt in enumerate([i0, i1]):
    fig = plt.figure(num=f'figbsoutlier-{i}', clear=True, figsize=[4.5, 3])

    ap21.sim.plotevent(datalist, ievt, 2, zoom='all', fig=fig)
    ax, = fig.get_axes()
    ax.legend(loc='lower left', fontsize='small')
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower right')

    fig.tight_layout()
    fig.show()
    
    figs.append(fig)

figlatex.save([figs])
