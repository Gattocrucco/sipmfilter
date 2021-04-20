import numpy as np
from matplotlib import pyplot as plt
from scipy import special

n = np.arange(1, 10)
mean = 2

########################

def mean2param(mean):
    # mean geom = 1/(1-p)
    # mean borel = 1/(1-mu), the same formula
    return 1 - 1 / mean

def borel(n, mu):
    effmu = mu * n
    return np.exp(-effmu) * effmu ** (n - 1) / special.factorial(n)

def geom(n, p):
    # p is 1 - p respect to the conventional definition
    return p ** (n - 1) * (1 - p)

fig, ax = plt.subplots(num='borelgeom', clear=True)

param = mean2param(mean)
ax.plot(n, borel(n, param), '.k', label='Borel')
ax.plot(n, geom(n, param), 'xk', label='Geometric')

ax.legend()
ax.set_xlabel('$n$')
ax.set_ylabel('$P(n)$')
ax.minorticks_on()
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.tight_layout()
fig.show()
