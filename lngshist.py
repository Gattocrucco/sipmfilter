"""
Plot the 2D histogram of trigger-aligned events of an LNGS wav. Usage:

    lngshist.py filename [maxevents [length [start]]]

maxevents = number of events read from the file, default 1000.
length = number of samples read per event, default 2000.
start = starting sample relative to the trigger leading edge, default 0.
"""

import sys

import numpy as np
from matplotlib import pyplot as plt, colors
import numba

import readwav
import textbox
import runsliced

filename = sys.argv[1]
maxevents = 1000
length = 2000
start = 0
try:
    maxevents = int(sys.argv[2])
    length = int(sys.argv[3])
    start = int(sys.argv[4])
except IndexError:
    pass

data = readwav.readwav(filename, mmap=True)

nevents = min(len(data), maxevents)

h = np.zeros((length, 1024), int)

@numba.njit(cache=True)
def accumhist(hist, data, start, length):
    for event in data:
        signal = event[0]
        trigger = event[1]
        for itrig, sample in enumerate(trigger):
            if sample < 600:
                break
        for i in range(length):
            isample = itrig + start + i
            if 0 <= isample < len(signal):
                sample = signal[isample]
                hist[i, sample] += 1

runsliced.runsliced(lambda s: accumhist(h, data[s], start, length), nevents, 100)

fig, ax = plt.subplots(num='lngshist', clear=True, figsize=[10.47,  4.8 ])

im = ax.imshow(h.T, origin='lower', cmap='magma', norm=colors.LogNorm(), aspect='auto', extent=(-0.5+start, length-0.5+start, -0.5, h.shape[1]-0.5))
fig.colorbar(im, label='Counts per bin', fraction=0.1)

ax.set_title(filename)
ax.set_xlabel('Samples after trigger leading edge @ 1 GSa/s')
ax.set_ylabel('ADC value')
textbox.textbox(ax, f'first {nevents}/{len(data)} events', fontsize='medium', loc='lower right')

fig.tight_layout()
fig.show()
