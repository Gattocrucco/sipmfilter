import numpy as np
from matplotlib import pyplot as plt
import readwav
import fighelp

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)

signal = data[:, 0, :].reshape(-1)
trigger = data[:, 1, :].reshape(-1)

print('computing global histogram...')

fig = fighelp.figwithsize([11.8, 4.8], resetfigcount=True)

ax = fig.subplots(1, 1)
ax.set_title('Histogram of all data')
ax.set_xlabel('ADC value')
ax.set_ylabel('occurences')

ax.plot(np.bincount(signal, minlength=1024), drawstyle='steps', label='signal')
ax.plot(np.bincount(trigger, minlength=1024), drawstyle='steps', label='trigger')

ax.set_yscale('symlog')
ax.set_ylim(-1, ax.get_ylim()[1])
ax.grid()
ax.legend(loc='best')

fighelp.saveaspng(fig)

fig = fighelp.figwithsize([8.21, 5.09])

ax = fig.subplots(1, 1)

start = 0
s = slice(start, start + 125000)
ax.plot(signal[s], ',', color='red', label='signal')
ax.plot(trigger[s], ',', color='blue', label='trigger')
ax.set_ylim(-1, 2**10 + 1)

ax.legend(loc='best')
ax.set_title('Original signal')
ax.set_xlabel(f'Sample number (starting from {start})')
ax.set_ylabel('ADC value')

fighelp.saveaspng(fig)

plt.show()
