import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy
import num2si
import textbox

config = [
    # label, noisefile, time [ns], timebase [ns]
    ('LNGS noise'  , 'noises/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz', 150, [1, 8, 16, 32]),
    ('Proto0 noise', 'noises/merged_000886-adc_W201_Ch00.npz'            , 150, [   8, 16, 32]),
]

plotkw = {
     1: dict(color='#aaa'),
     8: dict(color='#000'),
    16: dict(color='#000', linestyle='--', marker='o', markerfacecolor='#f55'),
    32: dict(color='#f55', marker='o', markerfacecolor='#fff'),
}

###########################

fig, axs = plt.subplots(1, 2, num='fignoise', clear=True, figsize=[9, 3.3], sharex=True, sharey=True)

for ax, (label, noisefile, N, timebase) in zip(axs, config):
    
    basetb = np.gcd.reduce(timebase)
    noise = toy.DataCycleNoise(timebase=basetb)
    noise.load(noisefile)

    ax.set_xlabel(f'Time [ns]')

    n = noise.generate(1, 3 * N)[0]
    for tb in timebase:
        nr = toy.downsample(n, tb // basetb)
        x = (tb - 1) / 2 + tb * np.arange(len(nr)) - N
        kwargs = dict(
            label=num2si.num2si(1e9 / tb, format='%.3g') + 'Sa/s',
            marker='.',
        )
        kwargs.update(plotkw[tb])
        ax.plot(x, nr, **kwargs)
    
    textbox.textbox(ax, label, fontsize='medium', loc='upper left')
    ax.legend(loc='lower left', ncol=2, title='Sampling frequency', framealpha=0.95, fontsize='small', title_fontsize='small')

    ax.set_xlim(0, N)
    ax.set_ylim(-3, 3)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
