import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

import toy
import figlatex
import readwav
import template as _template

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
ifilter = 1
length = 2048
sample = 1800

###########################

data = readwav.readwav('darksidehd/' + prefix + '.wav', maxevents=1, mmap=False)
event = data[0]
signal, trigger = event

template = _template.Template.load('templates/' + prefix + '-template.npz')
templ, offset = template.matched_filter_template(length, timebase=1)

filt = toy.Filter(signal[None], template.baseline)
fsignal = filt.all(templ)[ifilter, 0]

print(f'filter is {toy.Filter.name(ifilter)}')
print(f'filter length = {length} ns')

fig, ax = plt.subplots(num='figfiltersample', clear=True, figsize=[9.6, 4.8])

ax.plot(signal, color='#f55', label='unfiltered waveform')
ax.plot(trigger, color='black', linestyle='--', label='laser trigger')
ax.plot(fsignal, color='black', linestyle='-', label='filter output')

triglead = np.flatnonzero(trigger < 600)[0]
x = triglead + sample
y = fsignal[x]

ax.axvline(x, color='black', linestyle=':')
ax.plot(x, y, color='black', marker='.', markersize=10, linewidth=4)
yarrow = 650
ax.quiver(triglead, yarrow, sample, 0, angles='xy', scale_units='xy', scale=1, zorder=10)

ax.axvspan(triglead - 8100, triglead - 100, color='#ddd')

ax.legend(loc='best')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('ADC scale')
ax.set_xlim(0, 15000)
ax.set_ylim(600, 1000)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
