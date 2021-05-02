import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy
import num2si

noisefile = 'noises/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz'
N = 300
timebase = [1, 8, 16, 32]

###########################

noise = toy.DataCycleNoise(timebase=1)
noise.load(noisefile)

fig, ax = plt.subplots(num='fignoise', clear=True, figsize=[6.4, 3.32])

ax.set_xlabel(f'Time [ns]')

n = noise.generate(1, 3 * N)[0]
plotkw = [
    dict(color='#aaa'),
    dict(color='#000'),
    dict(color='#000', linestyle='--', marker='o', markerfacecolor='#f55'),
    dict(color='#f55', marker='o', markerfacecolor='#fff'),
]
for tb, kw in zip(timebase, plotkw):
    nr = toy.downsample(n, tb)
    x = (tb - 1) / 2 + tb * np.arange(len(nr)) - N
    kwargs = dict(
        label=num2si.num2si(1e9 / tb, format='%.3g') + 'Sa/s',
        marker='.',
    )
    kwargs.update(kw)
    ax.plot(x, nr, **kwargs)

ax.legend(loc='lower left', title='Sampling frequency', framealpha=0.95, fontsize='small', title_fontsize='small')

ax.set_xlim(0, N)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
