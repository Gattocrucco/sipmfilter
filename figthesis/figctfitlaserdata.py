from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21

fig, axs = plt.subplots(3, 1, num=f'figctfitlaserdata', clear=True, figsize=[9, 12], sharex='row')

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    ax = axs[j]
    
    kw = dict(
        fixzero=False,
        overflow=True,
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
    
    # extract data from plot
    xy = ax.lines[0].get_xydata()
    xyoverflow = ax.lines[5].get_xydata()
    
    print(f'# VOV = {vov}')
    print('# X\tY')
    for x, y in xy:
        print(f'{x}\t{y}')
    
    print('# overflow')
    x, y = xyoverflow[0]
    print(f'{x}\t{y}')

fig.tight_layout()
