import numpy as np
from matplotlib import pyplot as plt

import readwav
import figlatex
import make_template

filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
length = 2100

###########################

data = readwav.readwav(filename, mmap=False)
ignore = readwav.spurious_signals(data)
template = make_template.make_template(data, ignore, norm=False, length=length)

fig, ax = plt.subplots(num='figtemplate', clear=True, figsize=[6.4, 3.32])

ax.plot(template, color='black')

ax.set_xlabel('Time [ns]')
ax.set_ylabel('ADC scale')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
