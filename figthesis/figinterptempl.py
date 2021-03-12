import numpy as np
from matplotlib import pyplot as plt

import figlatex
import toy

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'

###########################

templ = toy.Template.load(prefix + '-template.npz')

signal_loc = np.linspace(0, 1, 3 * 8 + 1)
event_length = templ.template_length // 8 + int(np.max(np.ceil(signal_loc))) + 1
simulated_signal = templ.generate(event_length, signal_loc, randampl=False)

fig, ax = plt.subplots(num='figinterptempl', clear=True, figsize=[6.4, 4])

ax.set_xlabel('Sample number @ 125 MSa/s')

for i in range(len(signal_loc)):
    kw = dict(
        label=f'{signal_loc[i]:.2f}',
        color=[i / len(signal_loc)] + 2 * [i / len(signal_loc) * 5/16],
    )
    ax.plot(simulated_signal[i], **kw)

ax.legend(loc='best', title_fontsize='large', title='Signal start [sample]', ncol=4)

ax.set_xlim(7, 13)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
