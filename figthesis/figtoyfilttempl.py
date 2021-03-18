import numpy as np
from matplotlib import pyplot as plt

import figlatex
import template as _template

templfile = 'templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz'

###########################

template = _template.Template.load(templfile)

fig, ax = plt.subplots(num='figtoyfilttempl', clear=True, figsize=[6.4, 3.32])

ax.set_xlabel('Sample number @ 125 MSa/s')

template_offset = [
    template.matched_filter_template(length, norm=False)
    for length in [4, 8, 16, 32, 64]
]
for i, (y, offset) in enumerate(reversed(template_offset)):
    kw = dict(linewidth=i + 1, color='#600', alpha=(i + 1) / len(template_offset))
    ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

ax.legend(title='Template length', loc='best', fontsize='medium')

ax.set_ylim(-90, 0)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
