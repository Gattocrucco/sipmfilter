import glob

import numpy as np
from matplotlib import pyplot as plt

import readwav

files = list(sorted(glob.glob('darksidehd/*TILE21*73*.wav')))

start = 8970
length = 1500

values = np.empty(0)
for file in files:
    data = readwav.readwav(file)
    spurious = np.any(data[:, 0, :8000] < 700, axis=-1)
    baseline = np.mean(data[:, 0, :8000], axis=-1)
    x = np.mean(data[:, 0, start:start + length], axis=-1)
    values = np.append(values, (baseline - x)[~spurious])

fig, ax = plt.subplots(num='fingerplot_tile21', clear=True)

ax.hist(values, bins=600, histtype='step')
ax.set_ylabel('Count per bin')
ax.set_xlabel('ADC unit')
ax.set_title(files[0] + ', etc.')

fig.tight_layout()
fig.show()
