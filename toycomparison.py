import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import toy
import readwav

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)
ignore = readwav.spurious_signals(data)
print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

tau = np.array([4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384])
snr = np.linspace(1.8, 6, 15)
snr2d = np.tile(snr, (len(tau), 1))

noise_proto0 = toy.DataCycleNoise()
noise_proto0.load('merged_000866.npy')
noise_LNGS = toy.DataCycleNoise(allow_break=True)
noise_LNGS.load('toycomparison-lngs.npy')
noise_white = toy.WhiteNoise()
noise_obj  = [noise_proto0, noise_LNGS, noise_white]
noise_name = ['proto0',     'LNGS',     'white'    ]

t = toy.Toy(data, tau, mask=~ignore, snr=snr2d, bslen=1024, bsoffset=32)

output = []
for name, noise in zip(noise_name, noise_obj):
    print(f'running with {name} noise...')
    output.append(t.run(1000, pbar=10, seed=0, noisegen=noise, upsampling=True))
    # upsampling=True reduces speed dramatically

savefile = 'toycomparison-output.npz'
print(f'saving output to {savefile}...')
kw = {
    'outproto0':   output[0][0],
    'outevproto0': output[0][1],
    'outlngs':     output[1][0],
    'outevlngs':   output[1][1],
    'outwhite':    output[2][0],
    'outevwhite':  output[2][1],
}
np.savez(savefile, **kw)

snrs = t.filteredsnr(output[0][0])
filt_snr = snrs[1, np.searchsorted(tau, 128)]
raw_snr = t.snr[np.searchsorted(tau, 128)]
interp = interpolate.interp1d(filt_snr, raw_snr)
snr10, snr12 = interp([10, 12])

def plot_comparison(locfield='loc'):
    """
    Plot a selection of certain interesting temporal resolution curves.
    """
    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf384proto0 = r[-1, np.searchsorted(tau, 384)] * 8

    out = output[1][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf384lngs = r[-1, np.searchsorted(tau, 384)] * 8

    out = output[2][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf384white = r[-1, np.searchsorted(tau, 384)] * 8

    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mf64proto0 = r[-1, np.searchsorted(tau, 64)] * 8

    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    mabestproto0 = np.min(r[1], axis=0) * 8

    out = output[0][0]
    r = t.templocres(out['loctrue'], out[locfield])
    emabestproto0 = np.min(r[2], axis=0) * 8

    lngssnrvalues = [
        float(interpolate.interp1d(snr, res)(t.template.snr))
        for res in [mf384proto0, mf384lngs, mf384white]
    ]

    print(f'values at maximum SNR={snr[-1]} [ns]:')
    print(f'mf64proto0  {mf64proto0[-1]:.1f} ns')
    print(f'mf384proto0 {mf384proto0[-1]:.1f} ns')
    print(f'mf384lngs   {mf384lngs[-1]:.1f} ns')
    print(f'mf384white  {mf384white[-1]:.1f} ns')

    print()
    print(f'range of mf384 at LNGS SNR: {min(lngssnrvalues):.1f}--{max(lngssnrvalues):.1f} ns')

    fig = plt.figure('toycomparison', figsize=[7.14, 4.8])
    fig.clf()

    ax = fig.subplots(1, 1)

    ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
    ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')

    kw = dict(linestyle='-')
    ax.plot(snr, emabestproto0, label='Exp. mov. avg, N best for each SNR, Proto0 noise', marker='.', **kw)
    ax.plot(snr, mabestproto0, label='Moving average, N best for each SNR, Proto0 noise', marker='+', **kw)
    # ax.plot(snr, ma64proto0, label='Moving average, N=64, Proto0 noise', marker='x', **kw)
    ax.plot(snr, mf64proto0, label='Matched, N=64, Proto0 noise (doable in FPGA?)', marker='x', **kw)
    ax.plot(snr, mf384proto0, label='Matched, N=384, Proto0 noise', marker='v', **kw)
    ax.plot(snr, mf384lngs, label='Matched, N=384, LNGS noise', marker='^', **kw)
    ax.plot(snr, mf384white, label='Matched, N=384, white noise (best possible?)', marker='*', **kw)
    ax.axvline(t.template.snr, 0, 0.5, linestyle='--', color='#000', label='LNGS LN SNR')
    ax.axvspan(snr10, snr12, color='#ccf', zorder=-11, label='Proto0 movavg128-filtered SNR 10–12')
    ax.axhspan(0, 8, color='#ddd', zorder=-10, label='8 ns')

    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.legend(title='Filter', loc='upper right', title_fontsize='large')
    ax.set_ylim(0, np.ceil(np.max(mabestproto0) / 10) * 10)

    fig.tight_layout()
    fig.show()

def doplots(locfield='loc'):
    """
    as locfield use one of
    'loc'      : parabolic interpolation
    'locraw'   : no interpolation
    'locup'    : upsampling to 1 GSa/s + interpolation
    'locupraw' : upsampling only
    """
    out = output[0][0]
    t.plot_loc_all(out, snrspan=(snr10, snr12), logscale=True, sampleunit=False, locfield=locfield)
    t.plot_loc(out, np.searchsorted(tau, 64), np.searchsorted(snr, 4.5), locfield=locfield)
    t.plot_event(*output[0], 42, 3, np.searchsorted(tau, 384), np.searchsorted(snr, 4.5))
    plot_comparison(locfield)

doplots('loc')
