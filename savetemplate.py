"""
Make a signal template from an LNGS wav and save it. Usage:

    savetemplate.py [filenames...]
"""

import sys

from matplotlib import pyplot as plt
import numpy as np

import toy
import readwav
import templateplot

files = sys.argv[1:]

for source in files:
    assert source.endswith('.wav')
    destbase = source[:-4] + '-template'
    dest = destbase + '.npz'

    data = readwav.readwav(source, mmap=False)
    print(f'computing template...')
    ignore = readwav.spurious_signals(data)
    template = toy.Template.from_lngs(data, 7 * 512, ~ignore)
    print(f'saving template to {dest}...')
    template.save(dest)

    fig = plt.figure(num='savetemplate', clear=True, figsize=[6.4, 7.1])

    templateplot.templateplot(dest, fig=fig)

    fig.tight_layout()
    if len(files) == 1:
        fig.show()
    else:
        destplot = destbase + '.png'
        print(f'save plot to {destplot}...')
        fig.savefig(destplot)
