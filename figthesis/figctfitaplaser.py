from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

vov = 5.5

################

fig, axs = plt.subplots(1, 2, num='figctfitaplaser', clear=True, figsize=[9, 3.5], gridspec_kw=dict(width_ratios=[2, 1]))

ap21 = afterpulse_tile21.AfterPulseTile21(vov)

kw = dict(selection=False, overflow=True, fitlabel=['fit Borel', 'fit Geometric'])

ax = axs[0]
_, fig1, _ = ap21.maindict(fig2=ax, fixzero=False, vovloc='upper right', **kw)
plt.close(fig1)
ax.set_xlabel(f'Laser pulses PE')
ax.legend(loc='lower left', fontsize='small')

ax = axs[1]
_, fig1, _ = ap21.apdict(fig2=ax, vovloc='lower left', **kw)
plt.close(fig1)
ax.set_xlabel(f'Afterpulses PE')
ax.legend(loc='upper right', fontsize='small')

for ax in axs:
    if not ax.is_first_col():
        ax.set_ylabel(None)

fig.tight_layout()
fig.show()

figlatex.save(fig)
