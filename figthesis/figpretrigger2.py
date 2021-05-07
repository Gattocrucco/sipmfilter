from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

figs = []

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    fig = plt.figure(num=f'figpretrigger2-{j}0', clear=True, figsize=[4.5, 4])

    ap21.ptscatter(fig=fig, selection=False)
    ax, = fig.get_axes()
    ax.set_xlabel('Position of first pre-trigger pulse [ns]')
    ax.set_ylabel('Amplitude')
    fig.figlatex_options = dict(saveaspng=True)

    row = [fig]

    fig = plt.figure(num=f'figpretrigger2-{j}1', clear=True, figsize=[4.5, 4])

    ap21.pthist(fig=fig, selection=False)
    ax, = fig.get_axes()
    ax.set_xlabel('Amplitude of first pre-trigger pulse')

    row.append(fig)
    figs.append(row)

for row in figs:
    for fig in row:
        fig.tight_layout()

figlatex.save(figs)
