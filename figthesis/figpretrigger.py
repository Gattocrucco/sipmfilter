from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

vov = 5.5

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

fig = plt.figure(num='figpretrigger-0', clear=True, figsize=[4.5, 3])

ap21.ptscatter(fig=fig, selection=False, vovloc='upper left')
ax, = fig.get_axes()
ax.set_xlabel('Position of first pre-trigger pulse [ns]')
ax.set_ylabel('Amplitude')
fig.figlatex_options = dict(saveaspng=True)

figs = [fig]

fig = plt.figure(num='figpretrigger-1', clear=True, figsize=[4.5, 3])

ap21.pthist(fig=fig, selection=False, vovloc='center right')
ax, = fig.get_axes()
ax.set_xlabel('Amplitude of first pre-trigger pulse')

figs.append(fig)

for fig in figs:
    fig.tight_layout()
    fig.show()

figlatex.save([figs])
