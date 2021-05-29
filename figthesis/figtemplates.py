import os
import glob

import numpy as np
from matplotlib import pyplot as plt

import readwav
import figlatex
import template
import afterpulse_tile21

fig, ax = plt.subplots(num='figtemplates', clear=True, figsize=[6.7, 3.2])

styles = {
    5.5: dict(color='#f55'),
    7.5: dict(color='#888'),
    9.5: dict(color='black'),
}

for vov, style in styles.items():
    
    plotkw = dict(linewidth=1, label=f'{vov} V', **style)
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    
    for files in ap21.filelist:
        
        file = files['templfile']
        templ = template.Template.load(file)
        
        kw = dict(timebase=1, aligned=True, randampl=False)
        y, = templ.generate(templ.template_length, [0], **kw)
        ax.plot(y, **plotkw)
        plotkw.pop('label', None)

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')

ax.legend(title='Overvoltage')

ax.set_xlabel('Sample number @ 1 GSa/s')
ax.set_ylabel('ADC digit')
ax.set_xlim(0, 1000)

fig.tight_layout()
fig.show()

figlatex.save(fig)
