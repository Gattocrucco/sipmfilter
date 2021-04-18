import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

import figlatex

sigma = 1
c = 0.99 * sigma ** 2
deltau = sigma / 100
N = 5 * sigma / deltau
u = np.linspace(0, 5 * sigma, 1000)

###########################

def pcross0(u, sigma):
    return stats.norm.sf(u, scale=sigma)

def pcross1(u, sigma, k2):
    return np.sqrt(-k2 / (2 * np.pi)) * stats.norm.pdf(u, scale=sigma)

def pcross2(u, sigma, c, N, deltau):
    pout = 0
    for k in range(N):
        y0 = u - k * deltau
        s2 = sigma ** 2
        m = c / s2 * y0
        v = (s2 - c) * (s2 + c) / s2
        Py1y0 = stats.norm.sf(u, loc=m, scale=np.sqrt(v))
        py0 = stats.norm.pdf(y0, scale=sigma)
        pout += Py1y0 * py0 * deltau
    return pout

def ctok2(c, sigma):
    return 2 * (c - sigma ** 2)

p0 = pcross0(u, sigma)
p1 = pcross1(u, sigma, ctok2(c, sigma))
p2 = pcross2(u, sigma, c, int(N), deltau)

fig, ax = plt.subplots(num='figcrossingprob', clear=True, figsize=[6.4, 3.6])

ax.plot(u / sigma, p0, color='black', label='1) Survival function')
ax.plot(u / sigma, p1, color='#f55', label='2) Continuous')
ax.plot(u / sigma, p2, color='black', linestyle=':', label='3) Discrete')

ax.set_yscale('log')
ax.legend(loc='best')
ax.set_xlabel('Threshold [$\\sigma$]')
ax.set_ylabel('Upcrossing probability per sample')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
