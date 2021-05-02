import os

import numpy as np
from matplotlib import pyplot as plt

import read
import textbox
import num2si
import figlatex
import fdiffrate
import uncertainties
from uncertainties import umath

file = 'figthesis/figfakerate/merged_000886.root_31.npz'

########################

if not os.path.exists(file):
    raise ValueError(f'{file} does not exist, run figfakerate.py')

keys = [
    'thr',
    'thrcounts',
    'thr_theory',
    'thrcounts_theory',
    'effnsamples',
    'nsamples',
    'nevents',
    'freq',
    'sdev',
    'nsamp',
    'veto',
    'vetocount',
    'k2',
    'errsdev',
    'errk2',
]

print(f'load {file}...')
with np.load(file) as arch:
    globals().update({
        key: arch[key] for key in keys
    })

fig, ax = plt.subplots(num='figfakerate1', clear=True, figsize=[7.86, 4.8])    

ratefactor = freq / (nevents * effnsamples)
cond = thrcounts_theory >= np.min(thrcounts[thrcounts > 0])
ax.plot(thr_theory[cond] / sdev, ratefactor * thrcounts_theory[cond], color='#f55', label='Theory')

nz = np.flatnonzero(thrcounts)
start = max(0, nz[0] - 1)
end = min(len(thr), nz[-1] + 2)
s = slice(start, end)
ax.plot(thr[s] / sdev, ratefactor * thrcounts[s], 'k.-', label='Data')

ax.set_title(os.path.split(file)[1].replace('.npz', ''))
ax.set_xlabel('Threshold [$\\sigma$]')
ax.set_ylabel('Rate [cps]')

ax.axhspan(0, ratefactor, color='#ddd')
ax.legend(loc='upper right')

ax.set_yscale('log')
ax.set_ylim(1, 1e6)

axr = ax.twinx()
axr.set_yscale(ax.get_yscale())
axr.set_ylim(np.array(ax.get_ylim()) / ratefactor)
axr.set_ylabel('Count')

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')

k2 = uncertainties.ufloat(k2, errk2)
sdev = uncertainties.ufloat(sdev, errsdev)

info = f"""\
{nevents} events ({nevents * nsamples / freq * 1e3:.0f} ms)
sampling frequency {num2si.num2si(freq)}Sa/s
samples/event {nsamples} ({nsamples / freq * 1e6:.0f} µs)
effective samples/event {effnsamples} ({effnsamples / freq * 1e6:.0f} µs)
moving average {nsamp} Sa ({nsamp / freq * 1e6:.1f} μs)
difference delay {nsamp} Sa ({nsamp / freq * 1e6:.1f} μs)
dead time {nsamp} Sa ({nsamp / freq * 1e6:.1f} μs)
σ = {sdev:S}
k2 = –{-k2:S}
veto if any sample < {veto} (vetoed {vetocount})"""

textbox.textbox(ax, info, fontsize='small', loc='lower center', bbox=dict(alpha=0.8))

fig.tight_layout()
fig.show()

figlatex.save(fig)
