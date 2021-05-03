import glob
import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import readwav
import firstbelowthreshold

archive = 'figthesis/figtriggerhist.npz'
if not os.path.exists(archive):
    
    files = list(sorted(glob.glob('darksidehd/*.wav')))

    outfiles = []
    hists = []
    for file in files:
        data = readwav.readwav(file)
        if data.shape[1] != 2:
            continue
        trigger = firstbelowthreshold.firstbelowthreshold(data[:, 1], 600)
        hists.append(np.bincount(trigger))
        outfiles.append(file)
    
    length = max(len(h) for h in hists)
    hists = [np.pad(h, (0, length - len(h))) for h in hists]
    
    print(f'save {archive}...')
    np.savez_compressed(archive, files=outfiles, hists=hists)
    
print(f'load {archive}...')
with np.load(archive) as arch:
    files = arch['files']
    hists = arch['hists']

for file in files:
    _, name = os.path.split(file)
    print(name)

h = np.sum(hists, axis=0)
bins = -0.5 + np.arange(len(h) + 1)

nz = np.flatnonzero(h)
start = nz[0]
end = 1 + nz[-1]

def plothist(ax, bins, h, **kw):
    x = np.repeat(bins, 2)
    y = np.pad(np.repeat(h, 2), (1, 1))
    return ax.plot(x, y, **kw)

fig, ax = plt.subplots(num='figtriggerhist', clear=True, figsize=[6.4, 2.38])

plothist(ax, bins[start:end + 1], h[start:end], color='black')

_, up = ax.get_ylim()
ax.set_ylim(0, up)

ax.set_xlabel('Trigger leading edge [ns, Sa]')
ax.set_ylabel('Count per bin')

ax.minorticks_on()
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
