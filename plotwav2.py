"""
Plot an event from an LNGS wav and the moving average filter output.
"""

import numpy as np
from matplotlib import pyplot as plt
import readwav
import fighelp

filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, maxevents=1, mmap=False)

def moving_average(x, n):
    s = np.concatenate([[0], np.cumsum(x)])
    return (s[n:] - s[:-n]) / n

signal = data[0, 0]
trigger = data[0, 1]
ma = moving_average(signal, 1000)

fig = fighelp.figwithsize([8.21, 5.09], resetfigcount=True)

ax = fig.subplots(1, 1)

ax.plot(signal, '-', color='red', label='signal', linewidth=1)
ax.plot(trigger, '-', color='blue', label='trigger', linewidth=1)
ax.plot(np.arange(999, len(signal)), ma, '-', color='green', label='filter', linewidth=1)
ax.set_ylim(np.min(signal) - 10, np.max(trigger) + 10)
ax.set_xlim(0, len(signal) - 1)

ax.legend(loc='best')
ax.set_title('Laser on SiPM')
ax.set_xlabel(f'Time [ns]')
ax.set_ylabel('ADC reading [10 bit]')
ax.grid()

fighelp.saveaspng(fig)

plt.show()
