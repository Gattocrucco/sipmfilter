import numpy as np
from scipy import special
from matplotlib import pyplot as plt

import figlatex

param = [0.5]
mup = 2

###########################

colors = ['black', '#f55']
linestyle = ['-', '--']

def geom(k, p):
    return p ** (k - 1) * (1 - p)

geom.pname = 'p'
geom.name = 'Geometric'

def borel(k, mu):
    kmu = k * mu
    return np.exp(-kmu) * kmu ** (k - 1) / special.factorial(k)

borel.pname = '$\\mu_B$'
borel.name = 'Borel'

@np.vectorize
def geompoisson(n, mup, p):
    z = mup * (1 - p) / p
    P = [
        np.exp(-mup),
        np.exp(-mup) * p * z,
    ]
    assert int(n) == n, n
    n = int(n)
    for n in range(2, n + 1):
        t1 = (2 * n - 2 + z) / n * p * P[n - 1]
        t2 = (2 - n) / n * p ** 2 * P[n - 2]
        P.append(t1 + t2)
    assert len(P) == max(n, 1) + 1, (len(P), n)
    return P[n]

geompoisson.name = 'Geom. Poisson'

def genpoisson(n, mup, mub):
    effmu = mup + n * mub
    return np.exp(-effmu) * mup * effmu ** (n - 1) / special.factorial(n)

genpoisson.name = 'Gen. Poisson'

def plotprob(ax, k, P, **kw):
    bins = 0.5 + np.concatenate([k[:1] - 1, k])
    x = np.repeat(bins, 2)
    y = np.pad(np.repeat(P, 2), (1, 1))
    return ax.plot(x, y, **kw)

fig, axs = plt.subplots(2, 2, num='figgeomborel', clear=True, sharex='row', figsize=[7.62, 5.4])

k = np.arange(1, 10 + 1)
n = np.arange(15 + 1)
for p, color in zip(param, colors):
    for dist, distp, style in zip([geom, borel], [geompoisson, genpoisson], linestyle):
        kw = dict(
            linestyle = style,
            color = color,
            label = f'{dist.name}, {dist.pname} = {p:.2g}',
        )
        for ax in axs[0]:
            plotprob(ax, k, dist(k, p), **kw)
        kw.update(label=f'{distp.name}, {dist.pname} = {p:.2g}, $\\mu_P$ = {mup:.2g}')
        for ax in axs[1]:
            plotprob(ax, n, distp(n, mup, p), **kw)

for ax in axs[:, 1]:
    ax.set_yscale('log')

for ax in axs[0]:
    if ax.is_first_col():
        ax.set_ylabel('$P(k)$')
    ax.set_xlabel('$k$')

for ax in axs[1]:
    if ax.is_first_col():
        ax.set_ylabel('$P(n)$')
    ax.set_xlabel('$n$')

for ax in axs.flat:
    if ax.is_first_col():
        ax.legend(fontsize='small')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

figlatex.save(fig)
