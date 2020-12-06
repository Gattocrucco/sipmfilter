import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import toy
import readwav

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)
ignore = readwav.spurious_signals(data)
print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384]) * 8
snr = np.linspace(1.8, 6, 15)
snr2d = np.tile(snr, (len(tau), 1))

noise_LNGS = toy.DataCycleNoise(allow_break=True, timebase=1)
noise_LNGS.load('nuvhd_lf_3x_tile57_77K_64V_6VoV_1-noise.npz')
noise_white = toy.WhiteNoise(timebase=1)
noise_obj  = [noise_LNGS, noise_white]
noise_name = ['LNGS',     'white'    ]

t = toy.Toy(data, tau, mask=~ignore, snr=snr2d, bslen=1, bsoffset=1, timebase=1)

output = []
for name, noise in zip(noise_name, noise_obj):
    print(f'running with {name} noise...')
    output.append(t.run(1000, pbar=10, seed=0, noisegen=noise))

savefile = 'toy1gsa-output.npz'
print(f'saving output to {savefile}...')
kw = {
    'outlngs':     output[0][0],
    'outevlngs':   output[0][1],
    'outwhite':    output[1][0],
    'outevwhite':  output[1][1],
}
np.savez(savefile, **kw)

def itau(t):
    i = np.searchsorted(tau, t)
    assert tau[i] == t
    return i

def plot_comparison(locfield='loc'):
    """
    Plot a selection of certain interesting temporal resolution curves.
    """
    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf3072lngs = r[-1, itau(3072)]

    out = output[1][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf3072white = r[-1, itau(3072)]

    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf512lngs = r[-1, itau(512)]

    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mabestlngs = np.min(r[1], axis=0)

    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    emabestlngs = np.min(r[2], axis=0)

    lngssnrvalues = [
        float(interpolate.interp1d(snr, res)(t.template.snr))
        for res in [mf3072lngs, mf3072white]
    ]

    print(f'values at maximum SNR={snr[-1]} [ns]:')
    print(f'mf512lngs  {mf512lngs[-1]:.1f} ns')
    print(f'mf3072lngs   {mf3072lngs[-1]:.1f} ns')
    print(f'mf3072white  {mf3072white[-1]:.1f} ns')

    print()
    print(f'range of mf3072 at LNGS SNR: {min(lngssnrvalues):.1f}--{max(lngssnrvalues):.1f} ns')

    fig = plt.figure('toy1gsa', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)
    
    ax.set_title('Filter performance @ 1 GSa/s')
    ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
    ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')

    kw = dict(linestyle='-')
    ax.plot(snr, emabestlngs, label='Exp. mov. avg, N best for each SNR, LNGS noise', marker='.', **kw)
    ax.plot(snr, mabestlngs, label='Moving average, N best for each SNR, LNGS noise', marker='+', **kw)
    ax.plot(snr, mf512lngs, label='Matched, N=512, LNGS noise', marker='x', **kw)
    ax.plot(snr, mf3072lngs, label='Matched, N=3072, LNGS noise', marker='^', **kw)
    ax.plot(snr, mf3072white, label='Matched, N=3072, white noise', marker='*', **kw)
    ax.axvline(t.template.snr, 0, 0.5, linestyle='--', color='#000', label='LNGS LN SNR')
    ax.axhspan(0, 1, color='#ddd', zorder=-10, label='1 ns')

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.legend(title='Filter', loc='upper right', title_fontsize='large')
    ax.set_ylim(0, np.ceil(np.max(mabestlngs) / 10) * 10)

    fig.tight_layout()
    fig.show()

def doplots(locfield='loc'):
    """
    as locfield use one of
    'loc'      : parabolic interpolation
    'locraw'   : no interpolation
    """
    out = output[0][0]
    t.plot_loc_all(out, logscale=True, sampleunit=False, locfield=locfield)
    t.plot_loc(out, itau(512), np.searchsorted(snr, 4.5), locfield=locfield)
    t.plot_event(*output[0], 42, 3, itau(3072), np.searchsorted(snr, 4.5))
    plot_comparison(locfield)

doplots('loc')
