from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox
import colormap

length = [128, 64, 64]

################

figs = []

for j, (vov, flen) in enumerate(zip(afterpulse_tile21.AfterPulseTile21.defaultparams, length)):
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    fig = plt.figure(num=f'figlaserpos2-{j}0', clear=True, figsize=[4.5, 3.5])

    ap21.sim.hist('mainpos-offset', 'mainnpe==1', fig=fig, selection=False)
    ax, = fig.get_axes()
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower center')
    ax.set_xlabel('Laser peak position [ns]')

    row = [fig]

    fig = plt.figure(num=f'figlaserpos2-{j}1', clear=True, figsize=[4.5, 3.5])

    ap21.sim.hist2d('mainpos-offset', 'mainampl', f'(mainnpe==1)&(length=={flen})', fig=fig, cmap=colormap.uniform(), selection=False)
    ax, _ = fig.get_axes()
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower center')
    textbox.textbox(ax, f'{flen} ns filter', fontsize='medium', loc='upper left')
    ax.set_xlabel('Laser peak position [ns]')
    ax.set_ylabel('Peak height')

    row.append(fig)
    
    figs.append(row)

for row in figs:
    for fig in row:
        fig.tight_layout()
        fig.show()

figlatex.save(figs)
