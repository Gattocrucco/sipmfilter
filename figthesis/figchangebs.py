import os

import numpy as np
from matplotlib import pyplot as plt

import figlatex
import fingersnr

cacheprefix = 'figthesis/figchangebs'
baselines = [8000, 1000, 200]
cache_snrseries = 'figthesis/figsnrplot.npz'

###########################

cache = lambda bs: f'{cacheprefix}-{bs}.npz'

if not all(os.path.exists(cache(bs)) for bs in baselines):

    if not os.path.exists(cache_snrseries):
        raise FileNotFoundError(f'File {cache_snrseries} missing, run figsnrplot.py')
    
    print(f'read {cache_snrseries}...')
    with np.load(cache_snrseries) as arch:
        tau, delta_ma, delta_exp, delta_mf, waveform, snr = tuple(arch.values())
    
    hint_delta_ma = delta_ma[np.arange(len(tau)), np.argmax(snr[0], axis=-1)]
    
    fs = fingersnr.FingerSnr()
    
    for bs in baselines:
        if os.path.exists(cache(bs)):
            continue
        out = fs.snrmax(plot=False, hint_delta_ma=hint_delta_ma, bslen=bs)
        print(f'write {cache(bs)}...')
        np.savez(cache(bs), *out)

snrmax_outputs = []
for bs in baselines:
    print(f'read {cache(bs)}...')
    with np.load(cache(bs)) as arch:
        snrmax_outputs.append(tuple(arch.values()))

fig = plt.figure(num='figchangebs', clear=True, figsize=[9, 5])

axs = fingersnr.FingerSnr.snrmaxplot_multiple(fig, snrmax_outputs)
for bs, ax in zip(baselines, axs[0]):
    ax.set_title(f'{bs} baseline samples')

fig.tight_layout()
fig.show()

figlatex.save(fig)
