"""
Make a fingerplot with an LNGS wav. Usage:

    fingerplot.py [filename [filter_length [maxevents]]]

defaults:
filename = darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav
filter_length = 1500
maxevents = 1000
"""

import sys
import os

import numpy as np
from matplotlib import pyplot as plt

import readwav
import single_filter_analysis
import integrate
import textbox

filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
length = 1500
maxevents = 1000
try:
    filename = sys.argv[1]
    length = int(sys.argv[2])
    maxevents = int(sys.argv[3])
except:
    pass

data = readwav.readwav(filename, mmap=False, maxevents=maxevents)
mask = ~np.any(data[:, 0, :8000] < 700, axis=-1)
if data.shape[1] == 2:
    trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=length, delta_ma=length)
    value = value[:, 0]
else:
    baseline = np.median(data[:, 0, :8000], axis=-1)
    start = 8969 - 21
    value = np.mean(data[:, 0, start:start + length], axis=-1)
corr_value = baseline - value

fig = plt.figure(num='fingerplot', clear=True)
snr, _, _ = single_filter_analysis.single_filter_analysis(corr_value[mask], fig, return_full=True)
print(f'SNR = {snr:.3g}')

ax = fig.get_axes()[0]
ax.set_title(os.path.split(filename)[1])
textbox.textbox(ax, f"""\
first {len(data)} events
ignored {np.count_nonzero(~mask)} events
movavg {length} ns""", fontsize='small', loc='center right')

fig.tight_layout()
fig.show()
