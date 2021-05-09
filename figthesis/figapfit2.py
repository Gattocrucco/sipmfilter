from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

figs = []

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    fig = plt.figure(num=f'figapfit2-{j}0', clear=True, figsize=[6, 4])

    (fit1, fit2), fig1, _ = ap21.apfittau(fig2=fig, vovloc='lower left', selection=False)
    plt.close(fig1)
    ax, = fig.get_axes()
    dcut = ap21.params['dcut']
    ax.set_xlabel(f'Delay from {dcut} ns after laser peak [ns]')

    figs.append(fig)

    fig = plt.figure(num=f'figapfit2-{j}1', clear=True, figsize=[3, 4])
    
    length = ap21.params['aplength']
    cond = f'(length=={length})&(apApos>=0)&(mainnpebackup==1)&(apAapamplh>{ap21.apcut})'
    ap21.sim.hist('apApos - mainposbackup', cond, yscale='log', fig=fig, selection=False)
    ax, = fig.get_axes()
    ax.set_xlabel('Delay from laser peak [ns]')
    ax.set_ylabel(None)
    textbox.textbox(ax, f'{vov} VoV', fontsize='medium', loc='center right')

    figs.append(fig)

for fig in figs:
    fig.tight_layout()

figlatex.save(np.reshape(figs, (3, 2)))
