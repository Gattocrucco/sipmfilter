from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox
import colormap

vov = 5.5

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

fig = plt.figure(num='figlaserpos-0', clear=True, figsize=[4.5, 3])

ap21.sim.hist('mainpos-offset', 'mainnpe==1', fig=fig, selection=False)
ax, = fig.get_axes()
textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower center')
ax.set_xlabel('Laser peak position [ns]')

figs = [fig]

fig = plt.figure(num='figlaserpos-1', clear=True, figsize=[4.5, 3])

ap21.sim.hist2d('mainpos-offset', 'mainampl', '(mainnpe==1)&(length==128)', fig=fig, cmap=colormap.uniform(), selection=False)
ax, _ = fig.get_axes()
textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower center')
ax.set_xlabel('Laser peak position [ns]')
ax.set_ylabel('Peak height')

figs.append(fig)

for fig in figs:
    fig.tight_layout()
    fig.show()

figlatex.save([figs])
