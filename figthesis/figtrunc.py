from matplotlib import pyplot as plt
import numpy as np

import template
import figlatex

files = [
    f'templates/LF_TILE21_77K_54V_{bias}VoV_1-template.npz'
    for bias in [65, 69, 73]
]
lengths = {
    5.5: 128,
    7.5: 64,
    9.5: 64,
}
styles = [
    dict(color='black', linestyle='-'),
    dict(color='black', linestyle='--'),
    dict(color='black', linestyle=':'),
]

################

fig, ax = plt.subplots(num='figtrunc', clear=True, figsize=[6.4, 3.5])

for (vov, length), file, style in zip(lengths.items(), files, styles):
    templobj = template.Template.load(file)
    
    templ, offset = templobj.matched_filter_template(length, norm=False, aligned=True, timebase=1)
    ax.plot(np.arange(len(templ)) + offset, templ, label=f'{vov} VoV ({length} ns)', **style)

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')
ax.legend(title='Overvoltage (length)')
ax.set_xlabel('Sample number after trigger @ 1 GSa/s')
ax.set_ylim(ax.get_ylim()[0], 0)

fig.tight_layout()
fig.show()

figlatex.save(fig)
