from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

fig = plt.figure(num='figapscatter-0', clear=True, figsize=[4.5, 3])

ap21.apscatter(fig=fig, selection=False, markersize=2, vovloc='center right')
ax, = fig.get_axes()
ax.set_xlabel('Delay from laser peak [ns]')
ax.set_ylabel('Corrected amplitude')
fig.figlatex_options = dict(saveaspng=True)

figs = [fig]

fig = plt.figure(num='figapscatter-1', clear=True, figsize=[4.5, 3])

ap21.aphist(fig=fig, selection=False, vovloc='center right')
ax, = fig.get_axes()
ax.set_xlabel('Corrected amplitude')

figs.append(fig)

for fig in figs:
    fig.tight_layout()
    fig.show()

figlatex.save([figs])
