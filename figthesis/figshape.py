import numpy as np
from matplotlib import pyplot as plt

import figlatex
import template
import afterpulse_tile21

styles = {
    5.5: dict(color='#f55'),
    7.5: dict(hatch='//////', facecolor='#0000'),
    9.5: dict(edgecolor='black', facecolor='#0000'),
}

fig, ax = plt.subplots(num='figshape', clear=True, figsize=[7, 3.3])

for vov, style in styles.items():
    
    ap21 = afterpulse_tile21.AfterPulseTile21(vov)
    
    templates = []
    
    for files in ap21.filelist:
        
        file = files['templfile']
        templ = template.Template.load(file)
        
        kw = dict(timebase=1, aligned=True, randampl=False)
        y, = templ.generate(templ.template_length, [0], **kw)
        templates.append(y)
    
    m = np.mean(templates, axis=0)
    s = np.std(templates, axis=0, ddof=1)
    norm = np.min(m)
    
    ax.fill_between(np.arange(len(m)), (m - s) / norm, (m + s) / norm, label=f'{vov} V', zorder=2, **style)

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')

ax.legend(title='Overvoltage')

ax.set_xlabel('Sample number after trigger @ 1 GSa/s')
ax.set_xlim(0, 1000)

fig.tight_layout()
fig.show()

figlatex.save(fig)
