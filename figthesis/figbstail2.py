from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

bounds = {
    5.5: (955, 956),
    7.5: (955, 956),
    9.5: (955, 956),
}

figs = []

for j, (vov, (low, high)) in enumerate(bounds.items()):
    
    row = []
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    datalist = ap21.datalist
    
    events = ap21.sim.eventswhere(f'(baseline >= {low}) & (baseline <= {high})')
    assert len(events) >= 2, len(events)

    for i, ievt in enumerate(events[:2]):
        fig = plt.figure(num=f'figbstail2-{j}{i}', clear=True, figsize=[4.5, 3.5])

        ap21.sim.plotevent(datalist, ievt, 2, zoom='all', fig=fig)
        ax, = fig.get_axes()
        textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower left')

        fig.tight_layout()
        fig.show()
    
        row.append(fig)
    
    figs.append(row)

figlatex.save(figs)
