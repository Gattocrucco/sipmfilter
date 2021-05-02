import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy
import num2si
import template as _template

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
snr = 5
mftau = 256
locfilter = 2
loctau = 16
wlen = 64
wlmargin = 16

###########################

noise = toy.DataCycleNoise(allow_break=True)
noise.load(f'noises/{prefix}-noise.npz')
template = _template.Template.load(f'templates/{prefix}-template.npz')
sim = toy.Toy(template, [loctau, mftau], [snr], noise)
sim.run(100, seed=202102190959)
wcenter = sim.window_center([locfilter], [0], [0])
sim.run_window([wlen], [wlmargin], wcenter)

fig, axs = plt.subplots(1, 2, num='figwindowevent', clear=True, sharex=True, sharey=True, figsize=[8.94, 4.41])

sim.plot_event       (ievent=0, ifilter=locfilter, isnr=0, itau=0,                     ax=axs[0])
sim.plot_event_window(ievent=0,                    isnr=0, itau=1, iwlen=0, icenter=1, ax=axs[1])

axs[0].legend(fontsize='small', framealpha=0.95)
axs[1].legend(fontsize='small', framealpha=0.95)

fig.tight_layout()
fig.show()

figlatex.save(fig)
