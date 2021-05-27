import os

import numpy as np
from matplotlib import pyplot as plt
import tqdm
from scipy import interpolate, optimize, signal

import read
import single_filter_analysis
import integrate
import figlatex

config = [
    # label, file, maxlag [ns], repeat, point
    ('LNGS noise', 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav', 150, 10000, 0.5),
    ('Proto0 noise', 'darksidehd/merged_000886.root:57', 150, 10000, 0.5),
]

cache = 'figthesis/figautocorrlngs.npz'

###########################

def autocorr(x):
    y = x - np.mean(x)
    corr = np.correlate(y, y, 'full')
    n = len(corr) // 2
    cov = corr[n:]
    cov /= np.arange(len(x), 0, -1)
    return cov / cov[0]

if not os.path.exists(cache):
    fs = []
    timebase = []
    sdev = []
    for _, filename, maxlag, repeat, _ in config:
        
        data, freq, _ = read.read(filename, return_trigger=False)
        if '.root' in filename:
            x = data
        else:
            baseline = data[:, :8900]
            mask = np.all(baseline >= 700, axis=-1)
            x = baseline[mask]

        sdev.append(np.std(x))
        
        nlag = int(np.rint(maxlag * 1e-9 * freq))
        nlag_eff = 3 * nlag
        length = (x.shape[1] // nlag_eff) * nlag_eff
        x = x[:, :length].reshape(-1, nlag_eff)
        x = x[:repeat]

        cc = np.empty(x.shape)
        for i, a in enumerate(tqdm.tqdm(x)):
            cc[i] = autocorr(a)
        
        f = np.mean(cc, axis=0)
        fs.append(f)
        timebase.append(int(np.rint(1e9 / freq)))
    
    print(f'write {cache}...')
    np.savez(cache, *fs, timebase=timebase, sdev=sdev)

print(f'read {cache}...')
with np.load(cache) as arch:
    timebase = arch['timebase']
    sdev = arch['sdev']
    fs = [arch[f'arr_{i}'] for i in range(len(config))]

for s in sdev:
    print(f'sdev = {s:#.3g}')

fig, axs = plt.subplots(1, len(config), num='figautocorrlngs', clear=True, figsize=[9, 3], squeeze=False, sharey=True)

for (label, filename, nlag, _, point), f, tb, ax, s in zip(config, fs, timebase, axs.flat, sdev):

    fint = interpolate.interp1d(np.arange(len(f)) * tb, f, kind='linear')

    minresult = optimize.minimize_scalar(fint, (0, tb * np.argmin(f), tb * (len(f) - 1)))
    assert minresult.success
    minlag = minresult.x

    zresult = optimize.root_scalar(lambda t: fint(t) - point, bracket=(0, minlag))
    assert zresult.converged
    pointlag = zresult.root

    ax.axvline(pointlag, color='#f55', linestyle='--', label=f'{pointlag:.0f} ns, {100 * point:.0f} %', zorder=10)
    ax.axvline(minlag, color='#f55', linestyle='-', label=f'{minlag:.0f} ns, ${100 * fint(minlag):.0f}$ %', zorder=10)
    t = np.linspace(0, nlag, 1000)
    ax.plot(t, 100 * fint(t), color='black', zorder=11)

    ax.legend(loc='upper right', title=f'{label}\nsdev = {s:.2g}')
    ax.set_xlabel('Lag [ns]')
    if ax.is_first_col():
        ax.set_ylabel('Correlation [%]')
    ax.set_xlim(0, nlag)
    _, name = os.path.split(filename)
    ax.set_title(name)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
