import numpy as np
from matplotlib import pyplot as plt

import readwav
import figlatex
import template as _template

file = 'templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz'

config = [
    # aligned, plotkw
    (False    , dict(label='Unaligned', color='#f55' , linestyle='-')),
    ('trigger', dict(label='Trigger'  , color='black', linestyle='-')),
    (True     , dict(label='Filter'   , color='black', linestyle=':')),
]

###########################

templ = _template.Template.load(file)
print(templ.template_rel_std)

fig, axs = plt.subplots(1, 2, num='figtemplate', clear=True, figsize=[9, 3.3], sharey=True)

for aligned, plotkw in config:
    y, = templ.generate(templ.template_length, [0], timebase=1, aligned=aligned, randampl=False)
    for ax in axs:
        ax.plot(y, **plotkw)

axs[0].legend(loc='best', title='Alignment')
axs[1].set_xlim(0, 200)

for ax in axs:
    ax.set_xlabel('Time [ns]')
    if ax.is_first_col():
        ax.set_ylabel('ADC scale')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
