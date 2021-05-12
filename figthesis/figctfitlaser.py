from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

figaxs = [
    plt.subplots(3, 2, num=f'figctfitlaser-{i}', clear=True, figsize=[9, 12], sharex='row')
    for i in range(2)
]

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    for i, fixzero in enumerate([False, True]):
        fig, axs = figaxs[i]
        
        for k, overflow in enumerate([True, False]):
            ax = axs[j, k]
            
            kw = dict(
                fixzero=fixzero,
                overflow=overflow,
                selection=False,
                vovloc='center left',
                boxloc='upper right',
                fitlabel=['fit Borel', 'fit Geometric'],
            )
            legendloc='lower left'
            if vov == 9.5:
                kw.update(boxloc='lower left')
                legendloc='lower right'
            
            _, fig1, _ = ap21.maindict(fig2=ax, **kw)
            plt.close(fig1)
            
            ax.set_xlabel(f'Laser pulses PE')
            ax.legend(loc=legendloc, fontsize='small')

for fig, _ in figaxs:
    fig.tight_layout()
    figlatex.save(fig)
