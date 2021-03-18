"""
Histogram together the trigger time of all wav files listed on the command
line, ignoring files without trigger channel.
"""

import sys
import os

from matplotlib import pyplot as plt
import numpy as np

import readwav
import firstbelowthreshold

files = sys.argv[1:]

fig, ax = plt.subplots(num='triggerhist', clear=True)

for file in files:
    data = readwav.readwav(file)
    if data.shape[1] != 2:
        continue
    trigger = firstbelowthreshold.firstbelowthreshold(data[:, 1], 600)
    bins = -0.5 + np.arange(np.min(trigger), np.max(trigger) + 2)
    ax.hist(trigger, bins, histtype='step', label=os.path.split(file)[1])

ax.legend(fontsize='small')

ax.minorticks_on()
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.tight_layout()
fig.show()
