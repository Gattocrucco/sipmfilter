import glob
import os

from matplotlib import pyplot as plt

import template

V = [65, 69, 73]
files = [
    list(sorted(glob.glob(f'darksidehd/LF_TILE21_77K_54V_{v}VoV_*.wav')))
    for v in V
]

fig, ax = plt.subplots(num='templates', clear=True)

for i, (v, F) in enumerate(zip(V, files)):
    plotkw = dict(linewidth=1, color=f'C{i}', label=f'{(v - 54)/2} VoV')
    for file in F:
        _, tfile = os.path.split(file)
        tfile, _ = os.path.splitext(tfile)
        tfile = 'templates/' + tfile + '-template.npz'
        templ = template.Template.load(tfile)
        
        y, = templ.generate(templ.template_length, [0], timebase=1, aligned=True, randampl=False)
        ax.plot(y, **plotkw)
        plotkw.pop('label', None)

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')
ax.legend()
ax.set_xlabel('Sample number @ 1 GSa/s')
ax.set_ylabel('ADC units')
ax.set_xlim(0, 800)

fig.tight_layout()
fig.show()
