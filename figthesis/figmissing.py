from matplotlib import pyplot as plt
import numpy as np

import figlatex
import afterpulse_tile21
import textbox

styles = {
    5.5: dict(color='black', linestyle='-'),
    7.5: dict(color='black', linestyle='--'),
    9.5: dict(color='black', linestyle=':'),
}

fig, axs = plt.subplots(1, 2, num='figmissing', clear=True, figsize=[8, 3], sharex=True, sharey=True)

for vov, style in styles.items():

    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    
    length = ap21.sim.filtlengths
    total = ap21.sim.getexpr('len(event)')
    missing = ap21.sim.getexpr('count_nonzero(mainpos < 0, axis=-1)')
    cum = ap21.sim.getexpr('count_nonzero(logical_and.accumulate(mainpos < 0, axis=0), axis=-1)')
    
    axs[0].plot(length, missing / total * 100, label=f'{vov} VoV', **style)
    axs[1].plot(length, cum / total * 100, **style)
    
    print(f'{vov} VoV: {cum[-1]} hard misses {cum[-1] / total * 100:.2g} %')

for ax in axs:
    if ax.is_first_col():
        ax.legend()
        ax.set_ylabel('Events missing laser peak [%]')
        ax.set_xscale('log')
        _, up = ax.get_ylim()
        ax.set_ylim(0, up)
    ax.set_xlabel('Filter length [ns]')
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle=':')
    
fig.tight_layout()
fig.show()

figlatex.save(fig)
