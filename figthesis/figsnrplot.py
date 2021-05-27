import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import fingersnr

cache = 'figthesis/figsnrplot.npz'

###########################

if not os.path.exists(cache):
    fs = fingersnr.FingerSnr()
    out = fs.snrseries(plot=False)
    print(f'write {cache}...')
    np.savez(cache, *out)

print(f'read {cache}...')
with np.load(cache) as arch:
    snrseries_output = tuple(arch.values())

fig = plt.figure(num='figsnrplot', clear=True, figsize=[9, 5.52])

fingersnr.FingerSnr.snrplot(*snrseries_output, fig, plottemplate=False)

figlatex.save(fig)
