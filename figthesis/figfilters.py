import numpy as np
from matplotlib import pyplot as plt

import toy
import figlatex
import readwav
import template as _template

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
length = 1024

###########################

data = readwav.readwav('darksidehd/' + prefix + '.wav', maxevents=1, mmap=False)
signal = data[:, 0]

template = _template.Template.load('templates/' + prefix + '-template.npz')
templ, offset = template.matched_filter_template(length, timebase=1)

filt = toy.Filter(signal, template.baseline)
fsignal = filt.all(templ)[:, 0]

print(f'filter length = {length} ns')

fig, ax = plt.subplots(num='figfilters', clear=True)

ax.plot(fsignal[0], color='#f55', linewidth=1, label=toy.Filter.name(0))
for i in range(3):
    ax.plot(fsignal[i + 1], color='black', linestyle=[':', '--', '-'][i], label=toy.Filter.name(i + 1))

ax.legend(loc='best')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('ADC scale')
ax.set_xlim(8000, 15000)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
