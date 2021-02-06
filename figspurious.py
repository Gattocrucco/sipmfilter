import numpy as np
from matplotlib import pyplot as plt

import readwav
import single_filter_analysis
import integrate
import figlatex

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
events700 = [0, 1, 8]

###########################

data = readwav.readwav(filename, mmap=False)

signal  = data[:, 0]
baseline = signal[:, :8900]
spurious = np.flatnonzero(np.any(baseline < 700, axis=-1))

fig, ax = plt.subplots(num='figspurious', clear=True, figsize=[9.79, 4.8])

for i, j in enumerate(spurious[events700]):
    ax.plot(signal[j], linewidth=1, label=f'event {j}', color=[i / len(events700)] * 3, zorder=10 - i)
ax.axvline(9000, color='#f55', linestyle='--', label='laser trigger')

def plots(i):
    print(spurious[i])
    ax.cla()
    ax.plot(signal[spurious[i]], linewidth=1)
    fig.show()

def plotk(i):
    ax.cla()
    ax.plot(signal[i], linewidth=1)
    fig.show()

ax.legend(loc='best')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('ADC scale [10 bit]')
ax.minorticks_on()
ax.set_xlim(0, signal.shape[1])
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
