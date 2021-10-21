from matplotlib import pyplot as plt

import figlatex
import afterpulse_tile21
import textbox

fig, axs = plt.subplots(1, 3, num='figctfitrandomdata', clear=True, figsize=[9, 3.5])

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    ax = axs[j]
    (fit1, fit2), fig1, _ = ap21.ptdict(fig2=ax, vovloc='lower left', selection=False, fitlabel=['fit Borel', 'fit Geometric'])
    plt.close(fig1)
    ax.set_xlabel(f'Pre-trigger pulses PE')
    ax.legend(loc='upper right', fontsize='small')
    if not ax.is_first_col():
        ax.set_ylabel(None)

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
