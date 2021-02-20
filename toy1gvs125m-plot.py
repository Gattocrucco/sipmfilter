"""
Plot the output of `toy1gvs125m.py`. Usage:

    toy1gvs125m-plot.py [<LNGS file>.wav]

The LNGS wav specified on the command line must have already been processed
with `toy1gvs125m.py`. If not specified, it is set to
nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav.

The `timebase` array hardcoded at the beginning of this script sets which
timebases used by `toy1gvs125m.py` are read.

Figures are saved in the directory `toy1gvs125m/`. If you run this script in an
IPython shell, you can then call any of the following functions to do the plots
with different parameters:

plot_noise :
    Plot a noise waveform with downsampling.
plot_templocres :
    Plot the temporal resolution for all noises and sampling frequencies.
plot_filtsnr :
    Plot the SNR after vs before filtering for all noises and sampling
    frequencies.
doplots :
    Do the generic plots for a specific noise and sampling frequency.
"""

import sys

import numpy as np
from matplotlib import pyplot as plt

import toy
import textbox

timebase = [1, 8, 16, 32]

if len(sys.argv) == 1:
    prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
else:
    source = sys.argv[1]
    suffix = '.wav'
    assert source.endswith(suffix)
    prefix = source[:-len(suffix)]

######################################

def infobox(ax, **kw):
    kwargs = dict(fontsize='medium', loc='lower left', bbox=dict(alpha=0.95))
    kwargs.update(kw)
    textbox.textbox(ax, source, **kwargs)

timebase = np.array(list(sorted(timebase)))
# index 0 = higher sampling frequency

toys = []
noise_labels = ['white', 'lngs']
noise_names  = ['white', 'LNGS']
for inoise in range(len(noise_labels)):
    toys.append([])
    for i in range(len(timebase)):
        filename = f'toy1gvs125m-{prefix}-{timebase[i]}-{noise_labels[inoise]}.npz'
        print(f'load {filename}...')
        t = toy.Toy.load(filename)
        toys[inoise].append(t)

noise = [None] * 2
noise_ratio = [None] * 2

noise[0] = [toy.WhiteNoise(timebase=t) for t in timebase]
noise_ratio[0] = [np.sqrt(timebase[0] / tb) for tb in timebase]

noise_file = f'{prefix}-noise.npz'
noise[1] = []
for i in range(len(timebase)):
    n = toy.DataCycleNoise(allow_break=True, timebase=timebase[i])
    n.load(noise_file)
    noise[1].append(n)
noise_ratio[1] = [np.std(toy.downsample(noise[1][0].noise_array, tb // timebase[0])) for i, tb in enumerate(timebase)]

for inoise in range(len(noise_labels)):
    print(f'noise_ratio ({noise_labels[inoise]}) =', ', '.join(f'{noise_ratio[inoise][i]:.3f} ({tb})' for i, tb in enumerate(timebase)))

tau = toys[0][0].tau * toys[0][0].timebase

def isomething(a, x):
    i = np.searchsorted(a, x)
    assert a[i] == x
    return i
itau = lambda t: isomething(tau, t)
isnr = lambda snr, s: min(np.searchsorted(snr, s), len(snr) - 1)

def plot_noise(inoise=1):
    """
    Compare noise at different sampling frequencies.
    
    inoise = 0 (white), 1 (lngs)
    """
    fig = plt.figure('toy1gvs125m-plot.plot_noise')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    ax.set_title(f'{noise_names[inoise]} noise')
    ax.set_xlabel(f'Time [{timebase[0]} ns]')
    
    N = 600
    
    n = noise[inoise][0].generate(1, N // timebase[0])[0]
    for itb in range(len(timebase)):
        tbr = timebase[itb] // timebase[0]
        nr = toy.downsample(n, tbr)
        x = (tbr - 1) / 2 + tbr * np.arange(len(nr))
        ax.plot(x, nr, label=toys[inoise][itb].sampling_str())
    
    ax.legend(loc='upper center')
    infobox(ax)
    ax.grid()
    
    fig.tight_layout()
    fig.show()
    
    return fig

def plot_templocres(locfield='loc', ifilter=1, tau=256):
    """
    Compare temporal resolution at different sampling frequencies and noises.
    
    locfield = 'loc' (interpolation) or 'locraw' (integer, no interpolation)
    ifilter = 0 (no filter), 1 (moving average), 2 (expmovavg), 3 (matched)
    tau is @ 1 GSa/s.
    """
    fig = plt.figure('toy1gvs125m-plot.plot_templocres', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)

    ax.set_title(f'Temporal localization resolution\n{toy.Filter.name(ifilter)}, tau={tau} ns')
    ax.set_xlabel(f'Unfiltered SNR @ {toys[0][0].sampling_str()} (avg signal peak over noise rms)')
    ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')

    for itb in range(len(timebase)):
        kw = dict(linestyle='')
        for inoise in range(len(noise)):
            t = toys[inoise][itb]
            r = t.templocres(locfield, sampleunit=False)[ifilter, itau(tau)]
            nr = noise_ratio[inoise][itb]
            snr_str = f' (SNR *= {nr:.3f})' if itb > 0 else ''
            line, = ax.plot(t.snr * nr, r, label=f'{t.sampling_str()}, {noise_names[inoise]} noise{snr_str}', marker=['.', '+'][inoise], **kw)
            kw['color'] = line.get_color()
        dark = 0.6
        light = 0.87
        color = dark + (light - dark) * itb / (len(timebase) - 1)
        ax.axhspan(0, timebase[itb], color=np.ones(3) * color, zorder=-10 - itb)

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--', color='#000')
    ax.grid(True, 'minor', linestyle=':', color='#000')
    ax.legend(loc='upper right')
    infobox(ax)
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    fig.show()
    
    return fig

def plot_filtsnr(ifilter=1, tau=256, logscale=True):
    """
    Compare filtered SNR at different sampling frequencies and noises.
    
    ifilter = 0 (no filter), 1 (moving average), 2 (expmovavg), 3 (matched)
    tau is @ 1 GSa/s.
    """
    fig = plt.figure('toy1gvs125m-plot.plot_filtsnr', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)

    ax.set_title(f'Signal to noise ratio after vs. before filtering\n{toy.Filter.name(ifilter)}, tau={tau} ns')
    ax.set_xlabel(f'Unfiltered SNR @ {toys[0][0].sampling_str()} (avg signal peak over noise rms)')
    ax.set_ylabel('Filtered SNR (median filter peak over filtered noise rms)')

    for itb in range(len(timebase)):
        kw = dict(linestyle='')
        for inoise in range(len(noise)):
            t = toys[inoise][itb]
            s = t.filteredsnr()[ifilter, itau(tau)]
            nr = noise_ratio[inoise][itb]
            snr_str = f' (SNR *= {nr:.3f})' if itb > 0 else ''
            line, = ax.plot(t.snr * nr, s, label=f'{t.sampling_str()}, {noise_names[inoise]} noise{snr_str}', marker=['.', '+'][inoise], **kw)
            kw['color'] = line.get_color()

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.legend(loc='best')
    infobox(ax, loc='lower right')
    
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.tight_layout()
    fig.show()
    
    return fig

def doplots(locfield='loc', itb=0, tau=512, ievent=42, ifilter=3, snr=3.0, inoise=0):
    """
    locfield = 'loc' (interpolation) or 'locraw' (integer, no interpolation)
    itb = 0 (high sampling frequency), 1, ...
    tau is @ 1 GSa/s
    ifilter = 0 (no filter), 1 (moving average), 2 (expmovavg), 3 (matched)
    inoise = 0 (white), 1 (lngs)
    """
    t = toys[inoise][itb]
    t.plot_loc_all(logscale=True, sampleunit=False, locfield=locfield)
    t.plot_loc(itau(tau), isnr(t.snr, snr), locfield=locfield)
    t.plot_event(ievent, ifilter, itau(tau), isnr(t.snr, snr))

fig = plot_templocres(ifilter=3, tau=2048)
fig.savefig(f'toy1gvs125m/res-{prefix}.png')
fig = plot_filtsnr(ifilter=3, tau=2048, logscale=False)
fig.savefig(f'toy1gvs125m/snr-{prefix}.png')
fig = plot_noise(inoise=1)
fig.savefig(f'toy1gvs125m/noise-{prefix}.png')
