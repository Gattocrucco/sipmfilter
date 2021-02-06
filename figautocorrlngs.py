import numpy as np
from matplotlib import pyplot as plt
import tqdm
from scipy import interpolate, optimize

import readwav
import single_filter_analysis
import integrate
import figlatex

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
nlag = 200
repeat = 10000
point = 0.5

###########################

data = readwav.readwav(filename, mmap=False)
signal = data[:, 0, :]
baseline = signal[:, :8900]
mask = np.all(baseline >= 700, axis=-1)
x = baseline[mask]

sdev = np.std(x)
print(f'sdev = {sdev:.1f} ADC')

nlag_eff = 3 * nlag
length = (x.shape[1] // nlag_eff) * nlag_eff
x = x[:, :length].reshape(-1, nlag_eff)
x = x[:repeat]

def autocorr(x):
    y = x - np.mean(x)
    corr = np.correlate(y, y, 'full')
    n = len(corr) // 2
    cov = corr[n:]
    cov /= np.arange(len(x), 0, -1)
    return cov / cov[0]

cc = np.empty(x.shape)
for i, a in enumerate(tqdm.tqdm(x)):
    cc[i] = autocorr(a)


f = np.mean(cc, axis=0)
fint = interpolate.interp1d(np.arange(len(f)), f, kind='quadratic')

minresult = optimize.minimize_scalar(fint, (0, np.argmin(f), len(f) - 1))
assert minresult.success
minlag = minresult.x

zresult = optimize.root_scalar(lambda t: fint(t) - point, bracket=(0, minlag))
assert zresult.converged
pointlag = zresult.root

fig, ax = plt.subplots(num='figautocorrlngs', clear=True, figsize=[6.4, 3.45])

ax.axvline(pointlag, color='#f55', linestyle='--', label=f'{pointlag:.0f} ns, {100 * point:.0f} %', zorder=10)
ax.axvline(minlag, color='#f55', linestyle='-', label=f'{minlag:.0f} ns, ${100 * fint(minlag):.0f}$ %', zorder=10)
t = np.linspace(0, nlag, 1000)
ax.plot(t, 100 * fint(t), color='black', zorder=11)

ax.legend(loc='best')
ax.set_xlabel('Lag [ns]')
ax.set_ylabel('Correlation [%]')
ax.set_xlim(0, nlag)
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
