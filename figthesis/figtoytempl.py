import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy

templfile = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz'

###########################

templ = toy.Template.load(templfile)

fig, ax = plt.subplots(num='figtoytempl', clear=True, figsize=[6.4, 3.32])

ax.plot(templ.template, color='black')

ax.set_xlabel('Time [ns]')

ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
