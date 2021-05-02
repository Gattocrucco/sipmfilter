import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy
import num2si
import template as _template

templfile = 'templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz'
noisefile = 'noises/merged_000886-adc_W201_Ch00.npz'
tau = 256
snr = 5

###########################

noise = toy.DataCycleNoise(maxcycles=1, chunk_skip=1000)
noise.load(noisefile)
template = _template.Template.load(templfile)
sim = toy.Toy(template, [tau], [snr], noise)
sim.run(1000, pbar=10, seed=202102172153)

fig, axs = plt.subplots(2, 2, num='figlochist', clear=True, figsize=[7, 4.3])

sim.plot_loc(0, 0, axs=axs, center=True)

fig.tight_layout()
fig.show()

figlatex.save(fig)
