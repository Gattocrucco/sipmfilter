from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

fig, axs = plt.subplots(3, 2, num='figctfitap', clear=True, figsize=[9, 12], sharex='row')

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    for k, overflow in enumerate([True, False]):
        ax = axs[j, k]
        
        kw = dict(
            overflow=overflow,
            selection=False,
            vovloc='lower left',
            fitlabel=['fit Borel', 'fit Geometric'],
        )
        
        _, fig1, _ = ap21.apdict(fig2=ax, **kw)
        plt.close(fig1)
        
        ax.set_xlabel(f'Afterpulses PE')
        ax.legend(loc='upper right', fontsize='small')

fig.tight_layout()

figlatex.save(fig)
