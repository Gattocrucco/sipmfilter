from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5
low = 955
high = 956

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)
datalist = ap21.datalist

events = ap21.sim.eventswhere(f'(baseline >= {low}) & (baseline <= {high})')
assert len(events) >= 2, len(events)

figs = []
for i, ievt in enumerate(events[:2]):
    fig = plt.figure(num=f'figbstail-{i}', clear=True, figsize=[4.5, 3])

    ap21.sim.plotevent(datalist, ievt, 2, zoom='all', fig=fig)
    ax, = fig.get_axes()
    ax.legend(loc='lower right', fontsize='x-small')
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower left')

    fig.tight_layout()
    fig.show()
    
    figs.append(fig)

figlatex.save([figs])
