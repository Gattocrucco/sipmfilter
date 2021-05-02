import numpy as np
from matplotlib import pyplot as plt

import readwav
import single_filter_analysis
import integrate
import figlatex

events = [0, 1]
filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'

###########################

data = readwav.readwav(filename, maxevents=np.max(events) + 1, mmap=False)
data = data[events]

signal  = data[:, 0].reshape(-1)
trigger = data[:, 1].reshape(-1)

fig, ax = plt.subplots(num='figlngs', clear=True, figsize=[9.75, 4.8])

ax.plot(trigger, color='#f55', linewidth=1, label='laser trigger')
ax.plot(signal, color='black', linewidth=1, label='PDM output')

kw = dict(label='event boundary', color='black', linestyle='--')
for i in range(1, len(events)):
    ax.axvline(data.shape[-1] * i, **kw)
    kw.pop('label', None)

ax.legend(loc='best')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('ADC scale [10 bit]')
ax.minorticks_on()
ax.set_xlim(0, len(signal))
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
