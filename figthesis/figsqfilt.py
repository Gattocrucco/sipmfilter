import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import read
import num2si

filespec = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
event = 12

###########################

data, trigger, freq, ndigit = read.read(filespec, firstevent=event, maxevents=1)

nsamp = int(1e-6 * freq)

wf = data[0]
baseline = data[0, :trigger[0]]
wf = wf - np.mean(baseline) + np.std(baseline)
c = np.pad(np.cumsum(wf), (1, 0))
m = (c[nsamp:] - c[:-nsamp]) / nsamp
f = m[:-nsamp] - m[nsamp:]

kw = dict(
    num='figsqfilt',
    clear=True,
    figsize=[9, 3.62],
    gridspec_kw=dict(width_ratios=[2, 1]),
)
fig, axs = plt.subplots(1, 2, **kw)

ax, axf = axs

ax.plot(wf, color='#f55', label='x[t]')
ax.plot(len(wf) - len(m) + np.arange(len(m)), m, color='#000', linestyle=':', label='m = mean(x[t:t + 1μs])')
ax.plot(len(wf) - len(f) + np.arange(len(f)), f, color='#000', linestyle='-', label='y = m[t] $-$ m[t + 1μs]')

ax.legend()

_, name = os.path.split(filespec)
ax.set_title(f'{name}[0]')
ax.set_ylabel('Value [ADC unit, arbitrary offset]')

xy = np.array([
    (0, 0),
    (0, 1/nsamp),
    (nsamp, 1/nsamp),
    (nsamp, -1/nsamp),
    (2 * nsamp, -1/nsamp),
    (2 * nsamp, 0),
])
axf.plot(*xy.T, '-k')

axf.set_title('Filter')

for ax in axs:
    ax.set_xlabel(f'Sample number @ {num2si.num2si(freq)}Sa/s')
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
