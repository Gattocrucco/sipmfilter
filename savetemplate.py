"""
Make a signal template from an LNGS wav and save it. Usage:

    savetemplate.py [filenames...]
"""

import os

from matplotlib import pyplot as plt
import numpy as np

import template
import readwav
import templateplot
import firstbelowthreshold

def templatebasepath(source):
    assert source.endswith('.wav'), source
    return 'templates/' + os.path.split(source)[1][:-4] + '-template'

def templatepath(source):
    return templatebasepath(source) + '.npz'

def defaulttrigger(source):
    return 8969

def savetemplate(source, plot='show'):
    dest = templatepath(source)
    destbase = templatebasepath(source)
    
    data = readwav.readwav(source)
    print(f'computing template...')
    if data.shape[1] == 1:
        trigger = defaulttrigger(source)
    else:
        trigger = None

    fig1 = plt.figure(num='savetemplate1', clear=True)    
    templ = template.Template.from_lngs(data, 7 * 512, trigger=trigger, fig=fig1)
    print(f'write {dest}...')
    templ.save(dest)

    fig2 = plt.figure(num='savetemplate2', clear=True, figsize=[10, 7])
    templateplot.templateplot(dest, fig=fig2)

    for i, fig in enumerate([fig1, fig2]):
        fig.tight_layout()
        if plot == 'show':
            fig.show()
        elif plot == 'save':
            destplot = f'{destbase}-{i + 1}.png'
            print(f'write {destplot}...')
            fig.savefig(destplot)
        else:
            raise KeyError(plot)

if __name__ == '__main__':
    import sys
    files = sys.argv[1:]
    for file in files:
        savetemplate(file, plot='show' if len(files) == 1 else 'save')
