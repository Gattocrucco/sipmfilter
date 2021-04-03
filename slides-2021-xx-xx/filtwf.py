import sys

from matplotlib import pyplot as plt
import numpy as np

import read
import num2si

filespec = sys.argv[1]

data, trigger, freq, ndigit = read.read(filespec, maxevents=1)

nsamp = int(1e-6 * freq)

wf = data[0]
baseline = data[0, :trigger[0]]
wf = wf - np.mean(baseline) + np.std(baseline)
c = np.concatenate([[0], np.cumsum(wf)])
m = (c[nsamp:] - c[:-nsamp]) / nsamp
f = m[:-nsamp] - m[nsamp:]

fig, ax = plt.subplots(num='filtwf', clear=True)

ax.plot(wf, color='#f55', label='x[t]')
ax.plot(len(wf) - len(m) + np.arange(len(m)), m, color='#000', linestyle=':', label='m = mean(x[t:t + 1μs])')
ax.plot(len(wf) - len(f) + np.arange(len(f)), f, color='#000', linestyle='-', label='y = m[t] $-$ m[t + 1μs]')

ax.legend()

ax.set_title(f'{filespec}[0]')
ax.set_xlabel(f'Sample number @ {num2si.num2si(freq)}Sa/s')
ax.set_ylabel('Value [ADC unit, arbitrary offset]')

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()
