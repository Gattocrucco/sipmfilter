from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

vov = 5.5

################

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

fig = plt.figure(num='figbaseline', clear=True, figsize=[6.4, 3])

ap21.sim.hist('baseline', yscale='log', fig=fig)
ax, = fig.get_axes()
textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='lower center')
ax.set_xlabel('Baseline [ADC digit]')

fig.tight_layout()
fig.show()

figlatex.save(fig)
