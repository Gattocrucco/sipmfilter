import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy
import num2si
import template as _template

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
tau = 128
snr = 5

###########################

noise = toy.DataCycleNoise(allow_break=True)
noise.load(f'noises/{prefix}-noise.npz')
template = _template.Template.load(f'templates/{prefix}-template.npz')
sim = toy.Toy(template, [tau, 256], [snr], noise)
sim.run(1, seed=202102171737)

fig, ax = plt.subplots(num='figtoyevent', clear=True, figsize=[8.32, 3.59])

sim.plot_event(0, None, 0, 0, ax)

ax.legend(fontsize='small', framealpha=0.95)

fig.tight_layout()
fig.show()

figlatex.save(fig)
