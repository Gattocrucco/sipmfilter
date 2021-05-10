from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

# 1 PE
# cuts = {
#     5.5: (0, 60, 35, 60),
#     7.5: (0, 50, 50, 90),
#     9.5: (0, 50, 80, 130),
# }

# 2 PE
cuts = {
    5.5: (20, 140, 70, 110),
    7.5: (20, 100, 100, 160),
    9.5: (20, 90, 160, 250),
}

fig, axs = plt.subplots(1, 3, num='figdect', clear=True, figsize=[9, 3], sharex=True)

for j, ((vov, cut), ax) in enumerate(zip(cuts.items(), axs)):

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    length = ap21.params['aplength']
    npe = ap21.params['npe']
    cond = f'(length == {length}) & (apApos >= 0) & (mainnpebackup == {npe})'
    x = 'apApos - mainposbackup'
    y = 'apAprom'
    inbox = f'({x}>={cut[0]}) & ({x}<={cut[1]}) & ({y}>={cut[2]}) & ({y}<={cut[3]})'
    count = ap21.sim.getexpr(f'count_nonzero({inbox})', cond)
    total = ap21.sim.getexpr(f'count_nonzero({cond})')
    suspect = ap21.sim.getexpr(f'count_nonzero({inbox} & (mainnpe != mainnpebackup))', cond)
    ap21.sim.scatter(x, y, cond, fig=ax, selection=False, markersize=2)
    
    ax.legend(fontsize='small', loc='upper right', title='Filter length (entries)', title_fontsize='small')
    ax.set_xlabel('Delay from laser peak [ns]')
    if ax.is_first_col():
        ax.set_ylabel('Prominence')
    else:
        ax.set_ylabel(None)
    textbox.textbox(ax, f'{vov} VoV\nbox = {count}\n= {count/total*100:.2g} %\nsuspect = {suspect}', fontsize='small', loc='center right', bbox=dict(alpha=0.85))
    
    afterpulse_tile21.hlines(ax, ap21.apboundaries, linestyle=':')
    ax.set_ylim(0, ap21.apboundaries[2] * 1.1)
    afterpulse_tile21.vspan(ax, *cut[:2])
    afterpulse_tile21.hspan(ax, *cut[2:])

ax.set_xlim(-10, 200)

fig.tight_layout()
fig.show()

figlatex.save(fig)
