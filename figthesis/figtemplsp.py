import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

import toy
import figlatex
import figspectra
import template as _template

templfile = 'templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz'
maxfreq = 15

###################

if not os.path.exists(figspectra.npfile):
    figspectra.savenp()

print(f'read {figspectra.npfile}')
arch = np.load(figspectra.npfile)
f1 = arch['f1']
f2 = arch['f2']
s1 = arch['s1']
s2 = arch['s2']

template = _template.Template.load(templfile)
templ, offset = template.matched_filter_template(3000, timebase=1)

def adapt(f, s):
    f /= 1e6
    s *= 1e6
    end = np.searchsorted(f, maxfreq) + 1
    f, s = f[:end], s[:end]
    return f, s / np.max(s)

f, s = signal.periodogram(templ, fs=1e9, window='boxcar')
cdf = np.cumsum(s) / np.sum(s)

f, s = adapt(f, s)
f1, s1 = adapt(f1, s1)
f2, s2 = adapt(f2, s2)

fig, ax = plt.subplots(num='figtemplsp', clear=True, figsize=[6.4, 3.79])
axr = ax.twinx()

line1, = ax.plot( f[1:], s[1:] , color='black')
line2, = ax.plot(f1[1:], s1[1:], color='black', linestyle='--')
line3, = ax.plot(f2[1:], s2[1:], color='black', linestyle=':')
line4, = axr.plot(f, cdf[:len(f)] * 100, color='#f55')

axr.legend([line1, line2, line3, line4], [
    'Signal spectrum',
    'Proto0 noise spectrum',
    'LNGS noise spectrum',
    'Cumulative signal spectrum (right scale)'
], loc='lower right', fontsize='small', framealpha=0.9)
ax.set_xlabel('Frequency [MHz]')
ax.set_ylabel('Power spectral density [MHz$^{-1}$]')
axr.set_ylabel('Spectral power [%]')
ax.set_xlim(ax.get_xlim()[0], maxfreq)
axr.set_ylim(np.array(ax.get_ylim()) * 100)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
