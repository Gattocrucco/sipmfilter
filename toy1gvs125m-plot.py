import numpy as np
from matplotlib import pyplot as plt

import toy

timebase = [8, 16]

whitenoise = True

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'

######################################

timebase = list(reversed(sorted(timebase)))

toys = []
for i in range(len(timebase)):
    filename = f'toy1gvs125m-{timebase[i]}-white.npz'
    t = toy.Toy.load(filename)
    toys.append(t)

if whitenoise:
    noise = [toy.WhiteNoise(timebase=t) for t in timebase]
    noise_name = 'white'
    noise_ratio = np.sqrt(toys[1].timebase / toys[0].timebase)
else:
    noise_file = f'{prefix}-noise.npz'
    noise = []
    for i in range(len(timebase)):
        n = toy.DataCycleNoise(allow_break=True, timebase=timebase[i])
        n.load(noise_file)
        noise.append(n)
    noise_name = 'LNGS'
    noise_ratio = np.std(toy.downsample(noise[1].noise_array, toys[0].timebase // toys[1].timebase))

print(f'noise_ratio = {noise_ratio:.3f}')

tau = toys[0].tau
snr = toys[0].snr

def isomething(a, x):
    i = np.searchsorted(a, x)
    assert a[i] == x
    return i
itau = lambda t: isomething(tau, t)
isnr = lambda s: min(np.searchsorted(snr, s), len(snr) - 1)

def plot_noise():
    fig = plt.figure('toy1gvs125m-noise')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    ax.set_title(f'{noise_name} noise')
    ax.set_xlabel(f'Time [{toys[1].timebase} ns]')
    
    N = 600
    n1 = noise[1].generate(1, N // toys[1].timebase)[0]
    n0 = noise[0].generate(1, N // toys[0].timebase)[0]
    tbr = toys[0].timebase // toys[1].timebase
    n0r = toy.downsample(n1, tbr)
    ax.plot(n1, label=toys[1].sampling_str())
    x0 = (tbr - 1) / 2 + tbr * np.arange(len(n0))
    ax.plot(x0, n0, label=toys[0].sampling_str())
    ax.plot(x0, n0r, '--', label=f'{toys[0].sampling_str()}, no rescaling')
    
    ax.legend(loc='best')
    ax.grid()
    
    fig.tight_layout()
    fig.show()

def plot_comparison(locfield='loc', ifilter=1, tau=256):
    """
    Compare temporal resolution at 1 GSa/s vs. 125 MSa/s.
    
    locfield = 'loc' (interpolation) or 'locraw' (integer, no interpolation)
    ifilter = 0 (no filter), 1 (moving average), 2 (expmovavg), 3 (matched)
    tau is @ 1 GSa/s.
    """
    tau //= toys[0].timebase
    
    r0 = toys[0].templocres(locfield, sampleunit=False)[ifilter, itau(tau)]
    r1 = toys[1].templocres(locfield, sampleunit=False)[ifilter, itau(tau)]

    fig = plt.figure('toy1gvs125m', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)
    
    ax.set_title(f'Temporal localization resolution\n{toy.Filter.name(ifilter)}, tau={tau * toys[0].timebase} ns, {noise_name} noise')
    ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
    ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')

    kw = dict(linestyle='-')
    ax.plot(snr * noise_ratio, r0, label=f'{toys[0].sampling_str()} (SNR *= {noise_ratio:.3f})', marker='.', **kw)
    ax.plot(snr, r1, label=toys[1].sampling_str(), marker='+', **kw)
    ax.axhspan(0, toys[0].timebase, color='#ddd', zorder=-10)
    ax.axhspan(0, toys[1].timebase, color='#999', zorder=-9)

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.legend(title='Sampling frequency', loc='upper right', title_fontsize='medium')
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    fig.show()

def doplots(locfield='loc', itb=0, tau=64, ievent=42, ifilter=3, snr=3.0):
    """
    locfield = 'loc' (interpolation) or 'locraw' (integer, no interpolation)
    itb = 0 (125 MSa/s), 1 (1 GSa/s)
    tau is @ 125 MSa/s.
    ifilter = 0 (no filter), 1 (moving average), 2 (expmovavg), 3 (matched)
    """
    t = toys[itb]
    t.plot_loc_all(logscale=True, sampleunit=False, locfield=locfield)
    t.plot_loc(itau(tau), isnr(snr), locfield=locfield)
    t.plot_event(ievent, ifilter, itau(tau), isnr(snr))

plot_comparison()
plot_noise()
