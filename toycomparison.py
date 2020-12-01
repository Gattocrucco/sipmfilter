import numpy as np
from matplotlib import pyplot as plt

import toy
import readwav

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)
ignore = readwav.spurious_signals(data)
print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

tau = np.array([4, 8, 16, 32, 40, 48, 64, 384])
snr = np.array([1.7658007 , 2.06268908, 2.35957745, 2.65646583, 2.9533542 ,
                   3.25024257, 3.54713095, 3.84401932, 4.1409077 , 4.43779607])
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
    output.append(t.run(1000, pbar=10, seed=0, noisegen=noise))

out = output[0][0]
r = t.templocres(out['loctrue'], out['loc'])
mf384proto0 = r[-1, -1]

out = output[1][0]
r = t.templocres(out['loctrue'], out['loc'])
mf384lngs = r[-1, -1]

out = output[2][0]
r = t.templocres(out['loctrue'], out['loc'])
mf384white = r[-1, -1]

out = output[0][0]
r = t.templocres(out['loctrue'], out['loc'])
mf64proto0 = r[-1, 3]

out = output[0][0]
r = t.templocres(out['loctrue'], out['loc'])
mabestproto0 = np.min(r[1], axis=0)

out = output[0][0]
r = t.templocres(out['loctrue'], out['loc'])
emabestproto0 = np.min(r[2], axis=0)

fig = plt.figure('toycomparison')
fig.clf()

ax = fig.subplots(1, 1)

ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [8 ns]')

kw = dict(linestyle='-')
ax.plot(snr, emabestproto0, label='Exp. mov. avg, N best for each SNR, Proto0 noise', marker='.', **kw)
ax.plot(snr, mabestproto0, label='Moving average, N best for each SNR, Proto0 noise', marker='+', **kw)
ax.plot(snr, mf64proto0, label='Matched, N=64, Proto0 noise (doable in FPGA?)', marker='x', **kw)
ax.plot(snr, mf384proto0, label='Matched, N=384, Proto0 noise', marker='v', **kw)
ax.plot(snr, mf384lngs, label='Matched, N=384, LNGS noise', marker='^', **kw)
ax.plot(snr, mf384white, label='Matched, N=384, white noise (best possible?)', marker='*', **kw)
ax.axvline(t.template.snr, 0, 0.5, linestyle='--', color='#000', label='LNGS LN SNR')
ax.axhspan(0, 1, color='#eee', zorder=-10)

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')
ax.legend(title='Filter', loc='upper right', title_fontsize='large')
ax.set_ylim(0, np.ceil(np.max(mf64proto0) / 2) * 2)

fig.tight_layout()
fig.show()
