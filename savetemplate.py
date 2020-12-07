"""
Make a template with LNGS data, save it, load it back and do some plots.
"""

from matplotlib import pyplot as plt
import numpy as np

import toy
import readwav

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
source = f'{prefix}.wav'
dest = f'{prefix}-template.npz'

data = readwav.readwav(source, mmap=False)
ignore = readwav.spurious_signals(data)
template = toy.Template()
template.make(data, 7 * 512, ~ignore)
print(f'saving template to {dest}...')
template.save(dest)

template = toy.Template()
template.load(dest)

fig = plt.figure('savetemplate', figsize=[6.4 , 6.93])
fig.clf()

axs = fig.subplots(3, 1)

ax = axs[0]

ax.set_title('Full template @ 1 GSa/s')
ax.plot(template.template)
ax.grid()

ax = axs[1]

ax.set_title('Matched filter templates @ 125 MSa/s')
template_offset = [
    template.matched_filter_template(length, norm=False)
    for length in [4, 8, 16, 32, 64]
]
for i, (y, offset) in enumerate(reversed(template_offset)):
    kw = dict(linewidth=i + 1, color='#060', alpha=(i + 1) / len(template_offset))
    ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

ax.legend(title='Template length', loc='best', fontsize='small')
ax.grid()
ax.set_ylim(-90, 0)

ax = axs[2]

ax.set_title('Matched filter templates @ 1 GSa/s')
template_offset = [
    template.matched_filter_template(length, norm=False, timebase=1)
    for length in np.array([2, 4, 8, 16, 32]) * 8
]
for i, (y, offset) in enumerate(reversed(template_offset)):
    kw = dict(linewidth=i + 1, color='#060', alpha=(i + 1) / len(template_offset))
    ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

ax.legend(title='Template length', loc='best', fontsize='small')
ax.grid()
ax.set_ylim(-90, 0)

fig.tight_layout()
fig.show()
