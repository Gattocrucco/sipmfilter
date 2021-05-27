import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import fingersnr

cache = 'figthesis/figsnrmaxplot.npz'
cache_snrseries = 'figthesis/figsnrplot.npz'

###########################

if not os.path.exists(cache):

    if not os.path.exists(cache_snrseries):
        raise FileNotFoundError(f'File {cache_snrseries} missing, run figsnrplot.py')
    
    print(f'read {cache_snrseries}...')
    with np.load(cache_snrseries) as arch:
        tau, delta_ma, delta_exp, delta_mf, waveform, snr = tuple(arch.values())
    
    hint_delta_ma = delta_ma[np.arange(len(tau)), np.argmax(snr[0], axis=-1)]
    
    fs = fingersnr.FingerSnr()
    out = fs.snrmax(plot=False, hint_delta_ma=hint_delta_ma)
    print(f'write {cache}...')
    np.savez(cache, *out)

print(f'read {cache}...')
with np.load(cache) as arch:
    snrmax_output = tuple(arch.values())

fig = plt.figure(num='figsnrmaxplot', clear=True)

fingersnr.FingerSnr.snrmaxplot(*snrmax_output, fig=fig, plotoffset=False)

figlatex.save(fig)
