"""
Count threshold crossings of finite difference of filtered signal. Usage:

    fdiffrate.py filename[:channel] [maxevents [length [nsamples]]]

channel = (for Proto0 root files) ADC channel or run2 tile.
maxevents = number of events read from the file, default 1000.
length = length of the moving average filter, delay of the
         difference, and dead time, in nanoseconds; default 1000.
nsamples = number of samples read per event, default all.
"""

import sys

import numpy as np
from matplotlib import pyplot as plt
import numba
from scipy import stats

import read
import textbox
import runsliced
import num2si

filename = sys.argv[1]
maxevents = 1000
length = 1000
nsamples = 0
try:
    maxevents = int(sys.argv[2])
    length = int(sys.argv[3])
    nsamples = int(sys.argv[4])
except IndexError:
    pass

data, trigger, freq, ndigit = read.read(filename, maxevents=maxevents)

nsamp = int(length * 1e-9 * freq)

nevents = len(data)
if nsamples != 0:
    assert nsamples >= 2 * nsamp
    data = data[:, :nsamples]

minthr = -ndigit
maxthr = ndigit
nthr = 1 + int((maxthr - minthr) / 0.5)

thrcounts = np.zeros(nthr, int)
varcov1k2 = np.zeros(3)

@numba.njit(cache=True)
def movavg(x, n):
    c = np.cumsum(x)
    m = c[n - 1:].astype(np.float64)
    m[1:] -= c[:-n]
    m /= n
    return m

@numba.njit(cache=True)
def accum(minthr, maxthr, thrcounts, varcov1k2, data, nsamp):
    nthr = len(thrcounts)
    step = (maxthr - minthr) / (nthr - 1)
    
    for signal in data:
        
        m = movavg(signal, nsamp)
        f = m[:-nsamp] - m[nsamp:]
        
        lasti = np.full(nthr, -(1 + nsamp))
        for i in range(len(f) - 1):
            if f[i + 1] > f[i]:
                start = int(np.ceil((f[i] - minthr) / step))
                end = 1 + int(np.floor((f[i + 1] - minthr) / step))
                
                for j in range(max(0, start), min(end, len(lasti))):
                    if i - lasti[j] >= nsamp:
                        lasti[j] = i
                        thrcounts[j] += 1
        
        x = f - np.mean(f)
        varcov1k2[0] += np.mean(x * x)
        # varcov1k2[1] += np.mean(x[1:] * x[:-1])
        
        x2 = x[2:] + x[:-2] - 2 * x[1:-1]
        varcov1k2[2] += np.mean(x[1:-1] * x2)
        
runsliced.runsliced(lambda s: accum(minthr, maxthr, thrcounts, varcov1k2, data[s], nsamp), nevents, 100)

varcov1k2 /= nevents

def upcrossings(u, var, k2):
    return 1/(2 * np.pi) * np.sqrt(-k2 / var) * np.exp(-1/2 * u**2 / var)

def deadtime(rate, deadtime):
    return rate / (1 + rate * deadtime)

# def gauss_integral(a, b, **kw):
#     dist = stats.norm(**kw)
#     return np.where(
#         (a > 0) & (b > 0),
#         dist.sf(a) - dist.sf(b),
#         dist.cdf(b) - dist.cdf(a),
#     )
#
# def upcrossings_digital(u, var, cov1, digit):
#     pout = 0
#     for k in range(100):
#         y = (np.floor(u / digit) - k) * digit
#         m = cov1 / var * y
#         # v = var - cov1 ** 2 / var
#         v = (var - cov1) * (var + cov1) / var
#         pcond = stats.norm.sf(y + (k + 1/2) * digit, loc=m, scale=np.sqrt(v))
#         # p = stats.norm.pdf(y, scale=np.sqrt(var)) * digit
#         p = gauss_integral(y - digit/2, y + digit/2, scale=np.sqrt(var))
#         pout += pcond * p
#     return pout

nsamples = data.shape[1]
effnsamples = data.shape[1] - 2 * nsamp + 1

thr = np.linspace(minthr, maxthr, nthr)

upc = upcrossings(thr, *varcov1k2[[0, 2]])
upc_dead = deadtime(upc, nsamp)
thrcounts_theory = upc_dead * nevents * effnsamples

# upc = upcrossings_digital(thr, *varcov1k2[[0, 1]], 1 / nsamp)
# thrcounts_theory2 = upc * nevents * effnsamples

fig, ax = plt.subplots(num='fdiffrate', clear=True)

cond = thrcounts_theory >= np.min(thrcounts[thrcounts > 0])
ax.plot(thr[cond], thrcounts_theory[cond], color='#f55', label='theory')

# cond = thrcounts_theory2 >= np.min(thrcounts[thrcounts > 0])
# ax.plot(thr[cond], thrcounts_theory2[cond], color='#5f5')

nz = np.flatnonzero(thrcounts)
start = max(0, nz[0] - 1)
end = min(nthr, nz[-1] + 2)
s = slice(start, end)
ax.plot(thr[s], thrcounts[s], 'k.-')

ax.set_title(filename)
ax.set_xlabel('Threshold [ADC unit]')
ax.set_ylabel('Counts')

info = f"""\
first {nevents} events
sampling frequency {num2si.num2si(freq)}Sa/s
samples/event {nsamples}
effective samples/event {effnsamples}
moving average {nsamp} ({nsamp / freq * 1e6:.2g} μs)
difference delay {nsamp} ({nsamp / freq * 1e6:.2g} μs)
dead time {nsamp} ({nsamp / freq * 1e6:.2g} μs)"""
textbox.textbox(ax, info, fontsize='small', loc='upper left')

ax.axhspan(0, nevents, color='#eee', label='1 count/event')
ax.legend(loc='upper right')

ax.set_yscale('log')
axr = ax.twinx()
axr.set_yscale(ax.get_yscale())
axr.set_ylim(np.array(ax.get_ylim()) / (nevents * effnsamples / freq))
axr.set_ylabel('Rate [cps]')

ax.minorticks_on()
ax.grid(True, 'major', linestyle='--')
ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()

# def covariance(x, y, axis=-1):
#     mx = x - np.mean(x, axis=axis, keepdims=True)
#     my = y - np.mean(y, axis=axis, keepdims=True)
#     return np.mean(mx * my, axis=axis)
#
# def autocov20(x):
#     x0 = x[..., 1:-1]
#     x2 = x[..., 2:] + x[..., :-2] - 2 * x0
#     return covariance(x0, x2)
