import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import toy

template = toy.Template.load('nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz')

noise_name = ['proto0',     'LNGS',     'white'    ]
toys = []
for inoise in range(len(noise_name)):
    filename = f'toycomparison-{noise_name[inoise]}.npz'
    t = toy.Toy.load(filename)
    toys.append(t)

tau = toys[0].tau
snr = toys[0].snr
wlen = toys[0].wlen

def isomething(a, x):
    i = np.searchsorted(a, x)
    assert a[i] == x
    return i
itau  = lambda t: isomething(tau, t)
iwlen = lambda w: isomething(wlen, w)
isnr  = lambda s: min(np.searchsorted(snr, s), len(snr) - 1)

snrs = t.filteredsnr(toys[0].output)
filt_snr = snrs[1, itau(128)]
interp = interpolate.interp1d(filt_snr, snr, fill_value='extrapolate')
snr10, snr12 = interp([10, 12])

def plot_comparison(locfield='loc'):
    """
    Plot a selection of certain interesting temporal resolution curves.
    """
    r = toys[0].templocres(locfield, sampleunit=False)
    mf384proto0 = r[-1, itau(384)]

    r = toys[1].templocres(locfield, sampleunit=False)
    mf384lngs = r[-1, itau(384)]

    r = toys[2].templocres(locfield, sampleunit=False)
    mf384white = r[-1, itau(384)]

    r = toys[0].templocres(locfield, sampleunit=False)
    mf64proto0 = r[-1, itau(64)]

    r = toys[0].templocres(locfield, sampleunit=False)
    mabestproto0 = np.min(r[1], axis=0)

    r = toys[0].templocres(locfield, sampleunit=False)
    emabestproto0 = np.min(r[2], axis=0)

    lngssnrvalues = [
        float(interpolate.interp1d(snr, res)(template.snr))
        for res in [mf384proto0, mf384lngs, mf384white]
    ]

    print(f'values at maximum SNR={snr[-1]}:')
    print(f'mf64proto0  {mf64proto0[-1]:.1f} ns')
    print(f'mf384proto0 {mf384proto0[-1]:.1f} ns')
    print(f'mf384lngs   {mf384lngs[-1]:.1f} ns')
    print(f'mf384white  {mf384white[-1]:.1f} ns')

    print()
    print(f'range of mf384 at LNGS SNR: {min(lngssnrvalues):.1f}--{max(lngssnrvalues):.1f} ns')

    fig = plt.figure('toycomparison', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)
    
    ax.set_title(f'Filter time resolution @ {toys[0].sampling_str()}')
    ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
    ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')

    kw = dict(linestyle='-')
    ax.plot(snr, emabestproto0, label='Exp. mov. avg, N best for each SNR, Proto0 noise', marker='.', **kw)
    ax.plot(snr, mabestproto0, label='Moving average, N best for each SNR, Proto0 noise', marker='+', **kw)
    ax.plot(snr, mf64proto0, label='Matched, N=64, Proto0 noise (doable in FPGA?)', marker='x', **kw)
    ax.plot(snr, mf384proto0, label='Matched, N=384, Proto0 noise', marker='v', **kw)
    ax.plot(snr, mf384lngs, label='Matched, N=384, LNGS noise', marker='^', **kw)
    ax.plot(snr, mf384white, label='Matched, N=384, white noise (best possible?)', marker='*', **kw)
    ax.axvline(template.snr, 0, 0.5, linestyle='--', color='#000', label='LNGS LN SNR @ 1 GSa/s')
    ax.axvspan(snr10, snr12, color='#ccf', zorder=-11, label='Proto0 movavg128-filtered SNR 10–12')
    ax.axhspan(0, toys[0].timebase, color='#ddd', zorder=-10, label=f'{toys[0].timebase} ns')

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.legend(title='Filter', loc='upper right', title_fontsize='large')
    ax.set_ylim(0, np.ceil(np.max(mabestproto0) / 10) * 10)

    fig.tight_layout()
    fig.show()

def doplots(locfield='loc', inoise=0):
    """
    as locfield use one of
    'loc'      : parabolic interpolation
    'locraw'   : no interpolation
    'locup'    : upsampling to 1 GSa/s + interpolation
    'locupraw' : upsampling only
    
    inoise = 0 (proto0), 1 (lngs), 2 (white)
    """
    t = toys[inoise]
    t.plot_loc_all(snrspan=(snr10, snr12), logscale=True, sampleunit=False, locfield=locfield)
    t.plot_loc(itau(64), np.searchsorted(snr, 4.5), locfield=locfield)
    t.plot_event(42, 3, itau(384), np.searchsorted(snr, 4.5))
    plot_comparison(locfield)

def doplotsw(inoise=0, tau=64, wlen=128, snr=3.0, ievent=42):
    t = toys[inoise]
    t.plot_event_window(ievent, isnr(snr), itau(tau), iwlen(wlen))
    t.plot_loc_window(itau(tau), logscale=False)

doplots('loc')
doplotsw(0)
