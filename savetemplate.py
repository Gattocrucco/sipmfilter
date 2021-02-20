"""
Make a signal template from an LNGS wav and save it. Usage:

    savetemplate.py [filename]
"""

import sys
import os

from matplotlib import pyplot as plt
import numpy as np

import toy
import readwav
import templateplot

source = sys.argv[1]
assert source.endswith('.wav')
dest = source[:-4] + '-template.npz'

data = readwav.readwav(source, mmap=False)
ignore = readwav.spurious_signals(data)
template = toy.Template.from_lngs(data, 7 * 512, ~ignore)
print(f'saving template to {dest}...')
template.save(dest)

fig = plt.figure(num='savetemplate', clear=True, figsize=[6.4, 7.1])

templateplot.templateplot(dest, fig=fig)

fig.tight_layout()
fig.show()
