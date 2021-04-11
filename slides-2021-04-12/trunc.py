import os

from matplotlib import pyplot as plt
import numpy as np

import template

file = f'darksidehd/LF_TILE21_77K_54V_65VoV_1.wav'

fig, ax = plt.subplots(num='trunc', clear=True)

_, tfile = os.path.split(file)
tfile, _ = os.path.splitext(tfile)
tfile = 'templates/' + tfile + '-template.npz'
templ = template.Template.load(tfile)
    
template_offset = [
    templ.matched_filter_template(length, norm=False, aligned=True, timebase=1)
    for length in [64, 128]
]
for i, (y, offset) in enumerate(reversed(template_offset)):
    kw = dict(linewidth=i + 1, color=f'#060', alpha=(i + 1) / len(template_offset))
    ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')
ax.legend(title='Template length [ns]')
ax.set_xlabel('Sample number @ 1 GSa/s')
ax.set_ylim(ax.get_ylim()[0], 0)

fig.tight_layout()
fig.show()
