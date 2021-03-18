"""
Plot the output from `toy1gsa.py`.

If you run this script in an IPython shell, you can call the function `doplots`
with different parameters than the defaults.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import toy
import template as _template

template = _template.Template.load('templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz')

noise_name = ['LNGS',     'white'    ]
toys = []
for inoise in range(len(noise_name)):
    filename = f'toy1gsa-{noise_name[inoise]}.npz'
    t = toy.Toy.load(filename)
    toys.append(t)

tau = toys[0].tau
snr = toys[0].snr

def isomething(a, x):
    i = np.searchsorted(a, x)
    assert a[i] == x
    return i
itau  = lambda t: isomething(tau, t)
isnr  = lambda s: min(np.searchsorted(snr, s), len(snr) - 1)

def plot_comparison(locfield='loc'):
    """
    Plot a selection of certain interesting temporal resolution curves.
    """
    r = toys[0].templocres(locfield, sampleunit=False)
    mf3072lngs = r[-1, itau(3072)]

    r = toys[1].templocres(locfield, sampleunit=False)
    mf3072white = r[-1, itau(3072)]

    r = toys[0].templocres(locfield, sampleunit=False)
    mf512lngs = r[-1, itau(512)]

    r = toys[0].templocres(locfield, sampleunit=False)
    mabestlngs = np.min(r[1], axis=0)

    r = toys[0].templocres(locfield, sampleunit=False)
    emabestlngs = np.min(r[2], axis=0)
    
    templatesnr = template.max(timebase=1) / template.noise_std
    
    lngssnrvalues = [
        float(interpolate.interp1d(snr, res)(templatesnr))
        for res in [mf3072lngs, mf3072white]
    ]

    print(f'values at maximum SNR={snr[-1]}:')
    print(f'mf512lngs  {mf512lngs[-1]:.1f} ns')
    print(f'mf3072lngs   {mf3072lngs[-1]:.1f} ns')
    print(f'mf3072white  {mf3072white[-1]:.1f} ns')

    print()
    print(f'range of mf3072 at LNGS SNR: {min(lngssnrvalues):.1f}--{max(lngssnrvalues):.1f} ns')

    fig = plt.figure('toycomparison', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)
    
    ax.set_title(f'Filter time resolution @ {toys[0].sampling_str()}')
    ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
    ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')

    kw = dict(linestyle='-')
    ax.plot(snr, emabestlngs, label='Exp. mov. avg, N best for each SNR, LNGS noise', marker='.', **kw)
    ax.plot(snr, mabestlngs, label='Moving average, N best for each SNR, LNGS noise', marker='+', **kw)
    ax.plot(snr, mf512lngs, label='Matched, N=512, LNGS noise', marker='x', **kw)
    ax.plot(snr, mf3072lngs, label='Matched, N=3072, LNGS noise', marker='^', **kw)
    ax.plot(snr, mf3072white, label='Matched, N=3072, white noise (best possible?)', marker='*', **kw)
    ax.axvline(templatesnr, 0, 0.5, linestyle='--', color='#000', label='LNGS LN SNR @ 1 GSa/s')
    ax.axhspan(0, toys[0].timebase, color='#ddd', zorder=-10, label=f'{toys[0].timebase} ns')

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.legend(title='Filter', loc='upper right', title_fontsize='large')
    ax.set_ylim(0, np.ceil(np.max(mabestlngs) / 5) * 5)

    fig.tight_layout()
    fig.show()

def doplots(locfield='loc', inoise=0):
    """
    as locfield use one of
    'loc'      : parabolic interpolation
    'locraw'   : no interpolation
    
    inoise = 0 (lngs), 1 (white)
    """
    t = toys[inoise]
    t.plot_loc_all(logscale=True, sampleunit=False, locfield=locfield)
    t.plot_loc(itau(512), np.searchsorted(snr, 4.5), locfield=locfield)
    t.plot_event(42, 3, itau(3072), np.searchsorted(snr, 4.5))
    plot_comparison(locfield)

doplots('loc')
