"""
Count threshold crossings of finite difference of filtered signal. Usage:

    fdiffrate.py filename[:channel] [maxevents [length [nsamples [veto]]]]

channel = (for Proto0 root files) ADC channel or run2 tile.
maxevents = number of events read from the file, default 1000.
length = length of the moving average filter, delay of the
         difference, and dead time, in nanoseconds; default 1000.
nsamples = number of samples read per event, default all.
veto = lower bound on values to accept events, default 0.

This file can also be imported as a module. The function is `fdiffrate`.
"""

import os

import numpy as np
import numba

import runsliced

@numba.njit(cache=True)
def movavg(x, n):
    c = np.cumsum(x)
    m = c[n - 1:].astype(np.float64)
    m[1:] -= c[:-n]
    m /= n
    return m

@numba.njit(cache=True)
def accum(minthr, maxthr, thrcounts, varcov1k2, data, nsamp, veto, vetocount):
    nthr = len(thrcounts)
    step = (maxthr - minthr) / (nthr - 1)
    
    for signal in data:
        
        if np.any(signal < veto):
            vetocount += 1
            continue
        
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

def upcrossings(u, var, k2):
    return 1/(2 * np.pi) * np.sqrt(-k2 / var) * np.exp(-1/2 * u**2 / var)

def deadtime(rate, deadtime):
    return rate / (1 + rate * deadtime)

def fdiffrate(data, nsamp, batch=100, pbar=False, thrstep=0.5, veto=None):
    """
    Count threshold crossings of filtered finite difference.
    
    Parameters
    ----------
    data : array (nevents, nsamples)
        The data. Each event is processed separately.
    nsamp : int
        The moving average length, the difference delay, and the dead time, in
        unit of samples.
    batch : int
        The number of events to process at once. `None` for all at once.
    pbar : bool
        If True, print a progressbar.
    thrstep : scalar
        The step for the threshold range, relative to an estimate of the
        standard deviation of the filter output.
    veto : int, optional
        If an event has a value below `veto`, the event is ignored.
    
    Return
    ------
    thr : array (nthr,)
        An automatically generated range of thresholds.
    thrcounts : array (nthr,)
        The count of threshold upcrossings for each threshold in `thr`.
    thrcounts_theory : ufunc scalar -> scalar
        A function mapping the threshold to the expected count. It is not based
        on a fit to thrcounts, it is a theorical formula.
    sdev : scalar
        The standard deviation of the finite difference.
    effnsamples : int
        The effective number of samples per event, for the purpose of
        computing time rates.
    nevents : int
        Returned only if `veto` is specified. The number of processed events.
    """
    nevents, nsamples = data.shape
    effnsamples = nsamples - 2 * nsamp + 1
    assert effnsamples >= 1, effnsamples

    fstd = np.sqrt(2 / nsamp) * np.std(data[0])
    thrbin = thrstep * fstd
    maxthr = 20 * fstd
    minthr = -maxthr
    nthr = 1 + int((maxthr - minthr) / thrbin)

    thrcounts = np.zeros(nthr, int)
    varcov1k2 = np.zeros(3)
    vetocount = np.array(0)
    xveto = 0 if veto is None else veto
    
    func = lambda s: accum(minthr, maxthr, thrcounts, varcov1k2, data[s], nsamp, xveto, vetocount)
    runsliced.runsliced(func, nevents, batch, pbar)
    assert xveto > 0 or vetocount == 0
    nevents -= vetocount
    varcov1k2 /= nevents
    
    thr = np.linspace(minthr, maxthr, nthr)
    var, _, k2 = varcov1k2
    sdev = np.sqrt(var)
        
    def thrcounts_theory(thr):
        upc = upcrossings(thr, var, k2)
        upc_dead = deadtime(upc, nsamp)
        return upc_dead * nevents * effnsamples

    out = thr, thrcounts, thrcounts_theory, sdev, effnsamples
    if veto is not None:
        out += (nevents,)
    return out

if __name__ == '__main__':

    import sys
    
    from matplotlib import pyplot as plt
    
    import read
    import textbox
    import num2si
    
    filename = sys.argv[1]
    maxevents = 1000
    length = 1000
    nsamples = 0
    veto = 0
    try:
        maxevents = int(sys.argv[2])
        length = int(sys.argv[3])
        nsamples = int(sys.argv[4])
        veto = int(sys.argv[5])
    except IndexError:
        pass
    except BaseException as obj:
        print(__doc__)
        raise obj

    data, freq, ndigit = read.read(filename, maxevents=maxevents, return_trigger=False)

    nsamp = int(length * 1e-9 * freq)

    if nsamples != 0:
        assert nsamples >= 2 * nsamp
        data = data[:, :nsamples]
    
    output = fdiffrate(data, nsamp, pbar=True, veto=veto)
    thr, thrcounts, thrcounts_theory, sdev, effnsamples, nevents = output
    
    thrcounts_theory = thrcounts_theory(thr)
    thrunit = sdev
    nsamples = data.shape[1]

    fig, ax = plt.subplots(num='fdiffrate', clear=True)    
    
    cond = thrcounts_theory >= np.min(thrcounts[thrcounts > 0])
    ax.plot(thr[cond] / thrunit, thrcounts_theory[cond], color='#f55', label='theory')

    nz = np.flatnonzero(thrcounts)
    start = max(0, nz[0] - 1)
    end = min(len(thr), nz[-1] + 2)
    s = slice(start, end)
    ax.plot(thr[s] / thrunit, thrcounts[s], 'k.-')
    
    ax.set_title(os.path.split(filename)[1])
    ax.set_xlabel('Threshold [Filter output sdev]')
    ax.set_ylabel('Counts')

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
    
    info = f"""\
first {nevents} events
sampling frequency {num2si.num2si(freq)}Sa/s
samples/event {nsamples}
effective samples/event {effnsamples}
moving average {nsamp} ({nsamp / freq * 1e6:.2g} μs)
difference delay {nsamp} ({nsamp / freq * 1e6:.2g} μs)
dead time {nsamp} ({nsamp / freq * 1e6:.2g} μs)
filter output sdev {sdev:.2g}
veto if any sample < {veto} (vetoed {len(data) - nevents})"""

    textbox.textbox(ax, info, fontsize='small', loc='upper left')

    fig.tight_layout()
    fig.show()

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
#
# def covariance(x, y, axis=-1):
#     mx = x - np.mean(x, axis=axis, keepdims=True)
#     my = y - np.mean(y, axis=axis, keepdims=True)
#     return np.mean(mx * my, axis=axis)
#
# def autocov20(x):
#     x0 = x[..., 1:-1]
#     x2 = x[..., 2:] + x[..., :-2] - 2 * x0
#     return covariance(x0, x2)
