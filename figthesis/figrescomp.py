import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import figlatex
import toy

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
noisefile = 'merged_000886-adc_W201_Ch00.npz'
tau = [4, 8, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384]
snr = np.linspace(1.8, 6, 60)

###########################

simprefix = 'figrescomp'
names = ['proto0', 'lngs', 'white']
simfile = lambda n: f'figthesis/{simprefix}-{n}.npz'
options = dict(proto0=dict(upsampling=True))

if not all(os.path.exists(simfile(n)) for n in names):
    
    noise_proto0 = toy.DataCycleNoise(maxcycles=2, chunk_skip=1000)
    noise_proto0.load(noisefile)
    noise_lngs = toy.DataCycleNoise(maxcycles=2)
    noise_lngs.load(f'noises/{prefix}-noise.npz')
    noise_white = toy.WhiteNoise()
    noise = dict(proto0=noise_proto0, white=noise_white, lngs=noise_lngs)

    template = toy.Template.load(f'templates/{prefix}-template.npz')
    
    for n in names:
        if os.path.exists(simfile(n)):
            continue
        sim = toy.Toy(template, tau, snr, noise[n], **options.get(n, {}))
        sim.run(1000, pbar=10, seed=202102181611)
        print(f'save {simfile(n)}')
        sim.save(simfile(n))

sim = {}
for n in names:
    print(f'load {simfile(n)}')
    sim[n] = toy.Toy.load(simfile(n))
    assert np.array_equal(sim[n].tau, tau)
    assert np.array_equal(sim[n].snr, snr)

fig, ax = plt.subplots(num='figrescomp', clear=True, figsize=[6.4, 7.19])

def isomething(a, x, strict=True):
    i = min(np.searchsorted(a, x), len(a) - 1)
    assert not strict or a[i] == x
    return i
itau = lambda t: isomething(tau, t)
isnr = lambda s: isomething(snr, s, False)

snrs = sim['proto0'].snrratio()
ratio = snrs[1, itau(128)]
snr10, snr12 = np.array([10, 12]) / ratio

template = toy.Template.load(f'templates/{prefix}-template.npz')
lngssnr = template.snr
noise_lngs = toy.DataCycleNoise(timebase=1)
noise_lngs.load(f'noises/{prefix}-noise.npz')
lngssnr /= np.std(toy.downsample(noise_lngs.noise_array, 8), axis=None)

r = sim['proto0'].templocres(sampleunit=False)
mf384proto0 = r[3, itau(384)]

r = sim['proto0'].templocres(locfield='locup', sampleunit=False)
mf384proto0up = r[3, itau(384)]

r = sim['lngs'].templocres(sampleunit=False)
mf384lngs = r[3, itau(384)]

r = sim['white'].templocres(sampleunit=False)
mf384white = r[3, itau(384)]

r = sim['proto0'].templocres(sampleunit=False)
mf64proto0 = r[3, itau(64)]

r = sim['proto0'].templocres(sampleunit=False)
mabestproto0 = np.min(r[1], axis=0)

r = sim['proto0'].templocres(sampleunit=False)
emabestproto0 = np.min(r[2], axis=0)

ax.set_xlabel('SNR (before filtering)')
ax.set_ylabel('Temporal resolution [ns]')

ax.plot(snr, emabestproto0, label='Exp. mov. avg., $\\tau$ best for each SNR', color='#000', marker='.')
ax.plot(snr, mabestproto0, label='Moving average, N best for each SNR', color='#000', marker='x')
ax.plot(snr, mf64proto0, label='Cross correlation, N=64', color='#000')
ax.plot(snr, mf384proto0, label='Cross correlation, N=384', color='#f55')
ax.plot(snr, mf384proto0up, label='Cross correlation, N=384, with upsampling', color='#000', linestyle='--')
ax.plot(snr, mf384lngs, label='Cross correlation, N=384, LNGS noise', color='#000', linestyle='-.')
ax.plot(snr, mf384white, label='Cross correlation, N=384, white noise', color='#000', linestyle=':')
ax.axvspan(snr10, snr12, hatch='////', facecolor='#0000')
ax.axvline(lngssnr, color='#000', zorder=-10)
ax.axhspan(0, 8, color='#ddd')

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')
ax.legend(title='Filter', loc='best', title_fontsize='large', framealpha=0.95)
ax.set_ylim(0, np.ceil(np.max(mabestproto0) / 10) * 10)

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
