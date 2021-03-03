import sys
import os

from matplotlib import pyplot as plt
import numpy as np

files = list(sorted(sys.argv[1:]))

fig, ax = plt.subplots(num='fdrallplot', clear=True)

ax.set_xlabel('Threshold [Filter output sdev]')
ax.set_ylabel('Rate [cps]')

for ifile, file in enumerate(files):
    with np.load(file) as arch:
        
        thr = arch['thr']
        thrcounts = arch['thrcounts']
        thr_theory = arch['thr_theory']
        thrcounts_theory = arch['thrcounts_theory']
        effnsamples = arch['effnsamples']
        nevents = arch['nevents']
        freq = arch['freq']
        sdev = arch['sdev']
        
        nz = np.flatnonzero(thrcounts)
        start = max(0, nz[0] - 1)
        end = min(len(thr), nz[-1] + 2)
        s = slice(start, end)
        
        ratefactor = freq / (nevents * effnsamples)
        cond = thr_theory >= np.min(thr)
        cond &= thr_theory <= np.max(thr)
        cond &= thrcounts_theory >= np.min(thrcounts[thrcounts > 0])
        
        label = os.path.split(file)[1].replace('.npz', '')
        
        kw = dict(color=f'C{ifile}')
        ax.plot(thr_theory[cond] / sdev, ratefactor * thrcounts_theory[cond], '-', **kw)
        ax.plot(thr[s] / sdev, ratefactor * thrcounts[s], '.--', label=label, **kw)

if len(files) <= 6:
    fontsize = 'medium'
elif len(files) <= 10:
    fontsize = 'small'
else:
    fontsize = 'x-small'
ax.legend(fontsize=fontsize)

ax.set_yscale('log')
ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()
