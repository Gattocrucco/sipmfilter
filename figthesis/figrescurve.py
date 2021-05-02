import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy
import num2si
import template as _template

templfile = 'templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz'
noisefile = 'noises/merged_000886-adc_W201_Ch00.npz'
tau = [4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384]
snr = np.linspace(1.8, 6, 60)

###########################

simfile = 'figthesis/figrescurve.npz'

if not os.path.exists(simfile):
    noise = toy.DataCycleNoise(maxcycles=1, chunk_skip=1000)
    noise.load(noisefile)
    template = _template.Template.load(templfile)
    sim = toy.Toy(template, tau, snr, noise)
    sim.run(1000, pbar=10, seed=202102181210)
    print(f'save {simfile}')
    sim.save(simfile)

print(f'load {simfile}')
sim = toy.Toy.load(simfile)
assert np.array_equal(sim.tau, tau)
assert np.array_equal(sim.snr, snr)

fig, axs = plt.subplots(2, 2, num='figrescurve', clear=True, sharex=True, sharey=True, figsize=[7.49, 4.52])

sim.plot_loc_all(sampleunit=False, axs=axs)

for ax in axs.reshape(-1):
    if ax.is_last_row():
        ax.set_xlabel('SNR (before filtering)')
    if ax.is_first_col():
        ax.set_ylabel('Temporal resolution [ns]')

fig.tight_layout()
fig.show()

figlatex.save(fig)
