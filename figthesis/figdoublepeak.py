import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import readwav
import colormap
import textbox

file = 'darksidehd/LF_TILE21_77K_54V_65VoV_1.wav'
trigger = 8969 - 11
length = 1500
eventbin = 200
cache = 'figthesis/figdoublepeak.npz'

###########################

if not os.path.exists(cache):
    
    data = readwav.readwav(file)
    data = data[:, 0]

    Q = np.mean(data[:, trigger:trigger + length], axis=1)
    
    print(f'save {cache}...')
    np.savez_compressed(cache, Q=Q)

print(f'load {cache}...')
with np.load(cache) as arch:
    Q = arch['Q']

lower, upper = np.quantile(Q, [0.01, 1])
Q = Q[(Q >= lower) & (Q <= upper)]

fig, axs = plt.subplots(1, 2, num='figdoublepeak', clear=True, figsize=[8, 4])

ax = axs[0]

_, bins, _ = ax.hist(Q, bins=200, histtype='step', color='black', zorder=2)
bw = bins[1] - bins[0]

ax.set_xlabel(f'Mean over {length / 1000:.2g} $\\mu$s [ADC digit]')
ax.set_ylabel(f'Count per bin ({bw:.2g} digit)')

_, name = os.path.split(file)
textbox.textbox(ax, name, fontsize='medium', loc='upper left')

ax = axs[1]

xbins = np.arange(0, len(Q) + 1, eventbin)
ybins = np.histogram_bin_edges(Q, 100)
_, _, _, im = ax.hist2d(np.arange(len(Q)), Q, bins=(xbins, ybins), cmin=1, cmap=colormap.uniform(), zorder=2)
bw = ybins[1] - ybins[0]

fig.colorbar(im, label=f'Count per bin ({eventbin} event x {bw:.2g} digit)')

ax.set_xlabel('Event')
ax.set_ylabel(axs[0].get_xlabel())

for ax in axs:
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
