"""
Make a fingerplot with an LNGS wav. Usage:

    fingerplot.py [filename [filter_length [maxevents]]]

defaults:
filename = nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav
filter_length = 1500
maxevents = 1000
"""

import sys

import numpy as np
from matplotlib import pyplot as plt

import readwav
import single_filter_analysis
import integrate

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
length = 1500
maxevents = 1000
try:
    filename = sys.argv[1]
    length = int(sys.argv[2])
    maxevents = int(sys.argv[3])
except:
    pass

data = readwav.readwav(filename, mmap=False, maxevents=maxevents)
mask = ~readwav.spurious_signals(data)
trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=length, delta_ma=length)
corr_value = baseline - value[:, 0]

fig = plt.figure(num='fingerplot', clear=True)
snr, _, _ = single_filter_analysis.single_filter_analysis(corr_value[mask], fig, return_full=True)
print(f'SNR = {snr:.3g}')

fig.tight_layout()
fig.show()
