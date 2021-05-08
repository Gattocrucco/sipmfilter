from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

cond = '(length == 128) & (apApos >= 0) & (mainnpebackup == 1)'
x = 'apApos - mainposbackup'
y = [
    ('apAprom', 'Prominence'),
    ('apAheight', 'Height'),
    ('apAamplh', 'Amplitude'),
    ('apAapamplh', 'Corrected amplitude'),
]

figs = []
for i, (expr, label) in enumerate(y):
    fig = plt.figure(num=f'figapampl-{i}', clear=True, figsize=[4.5, 3])
    ap21.sim.scatter(x, expr, cond, fig=fig, selection=False, markersize=2)
    ax, = fig.get_axes()
    ax.set_xlabel('Delay from laser peak [ns]')
    ax.set_ylabel(label)
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='center right')
    lim = ax.get_ylim()
    afterpulse_tile21.hlines(ax, ap21.apboundaries, linestyle=':')
    ax.set_ylim(lim)
    figs.append(fig)

for fig in figs:
    fig.figlatex_options = dict(saveaspng=True)
    fig.tight_layout()
    fig.show()

figlatex.save(np.reshape(figs, (2, 2)))
