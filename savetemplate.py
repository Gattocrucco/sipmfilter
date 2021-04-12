"""
Make a signal template from an LNGS wav and save it. Usage:

    savetemplate.py [filenames...]
"""

import sys
import os

from matplotlib import pyplot as plt
import numpy as np

import template
import readwav
import templateplot
import firstbelowthreshold

files = sys.argv[1:]

for source in files:
    assert source.endswith('.wav')
    destbase = 'templates/' + os.path.split(source)[1][:-4] + '-template'
    dest = destbase + '.npz'

    data = readwav.readwav(source)
    print(f'computing template...')
    if data.shape[1] == 1:
        trigger = 8969
    else:
        trigger = None
    
    fig1 = plt.figure(num='savetemplate1', clear=True)    
    templ = template.Template.from_lngs(data, 7 * 512, trigger=trigger, fig=fig1)
    print(f'write {dest}...')
    templ.save(dest)

    fig2 = plt.figure(num='savetemplate2', clear=True, figsize=[6.4, 7.1])
    templateplot.templateplot(dest, fig=fig2)
    
    for i, fig in enumerate([fig1, fig2]):
        fig.tight_layout()
        if len(files) == 1:
            fig.show()
        else:
            destplot = f'{destbase}-{i + 1}.png'
            print(f'write {destplot}...')
            fig.savefig(destplot)
