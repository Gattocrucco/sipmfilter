from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

figs = []

for j, vov in enumerate(afterpulse_tile21.AfterPulseTile21.defaultparams):

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)

    row = [
        plt.figure(num=f'figapscatter2-{j}{i}', clear=True, figsize=[4.5, 4])
        for i in range(2)
    ]
    
    for fig in row:
        ap21.apscatter(fig=fig, selection=False, markersize=2, vovloc='center right')
        ax, = fig.get_axes()
        ax.set_xlabel('Delay from laser peak [ns]')
        ax.set_ylabel('Corrected amplitude')
    
    row[0].figlatex_options = dict(saveaspng=True)
    
    ax, = row[1].get_axes()
    ax.set_xlim(-10, 500)
    ax.set_ylim(-10, ap21.apboundaries[1] * 1.1)

    figs += row

    # amplitude histogram
    # fig = plt.figure(num=f'figapscatter2-{j}1', clear=True, figsize=[4.5, 4])
    #
    # ap21.aphist(fig=fig, selection=False, vovloc='center right')
    # ax, = fig.get_axes()
    # ax.set_xlabel('Corrected amplitude')
    #
    # figs.append(fig)

figlatex.save(np.reshape(figs, (3, 2)))
