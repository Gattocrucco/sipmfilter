"""
DEPRECATED, use `Template` from `toy.py`.

Make a template for the matched filter with an LNGS wav.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, signal

import integrate
from single_filter_analysis import single_filter_analysis

def make_template(data, ignore=None, length=2000, noisecorr=False, fig=None, figcov=None, norm=True):
    """
    Make a template waveform for the matched filter.
    
    Parameters
    ----------
    data : array (nevents, 2, 15001)
        As returned by readwav.readwav().
    ignore : bool array (nevents,), optional
        Flag events to be ignored.
    length : int
        Number of samples of the waveform.
    noisecorr : bool
        If True, optimize the filter for the noise spectrum.
    fig : matplotlib figure, optional
        If given, plot the waveform.
    figcov : matplotlib figure, optional
        If given, plot the covariance matrix of the noise as a bitmap.
    norm : bool
        If True, normalize the output to unit sum, so that applying it behaves
        like a weighted mean.
    
    Return
    ------
    waveform : array (length,)
        The waveform.
    """
    
    if ignore is None:
        ignore = np.zeros(len(data), bool)

    # Run a moving average filter to find and separate the signals by number of
    # photoelectrons.
    trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=1470, delta_ma=1530)
    corr_value = baseline - value[:, 0]
    snr, center, width = single_filter_analysis(corr_value[~ignore], return_full=True)
    assert snr > 15
    assert len(center) > 2
    
    # Select the data corresponding to 1 photoelectron and subtract the
    # baseline.
    lower = (center[0] + center[1]) / 2
    upper = (center[1] + center[2]) / 2
    selection = (lower < corr_value) & (corr_value < upper) & ~ignore
    t = int(np.median(trigger))
    data1pe = data[selection, 0, t:t + length] - baseline[selection].reshape(-1, 1)
    
    # Compute the waveform as the median of the signals.
    waveform = np.median(data1pe, axis=0)
    # waveform = np.mean(data1pe, axis=0)
    
    if noisecorr:
        # Select baseline data and compute covariance matrix over a slice with
        # the same length of the template.
        bsend = t - 100
        N = 2 * (length + length % 2)
        databs = data[~ignore, 0, bsend - N:bsend]
        cov = np.cov(databs, rowvar=False)
        cov = toeplitze(cov)
        s = slice(N // 4, N // 4 + length)
        cov = cov[s, s]
        # use cov(fweights=~ignore) to avoid using ~ignore
        
        # Correct the waveform.
        wnocov = waveform
        waveform = linalg.solve(cov, waveform, assume_a='pos')
        waveform *= linalg.norm(cov) / len(waveform)
    
    if fig is not None:
        axs = fig.subplots(2, 1)

        ax = axs[0]
        if noisecorr:
            ax.plot(wnocov / np.max(np.abs(wnocov)), label='assuming white noise')
            ax.plot(waveform / np.max(np.abs(waveform)), label='corrected for actual noise', zorder=-1)
        else:
            # std = np.std(data1pe, axis=0)
            # bottom = waveform - std
            # top = waveform + std
            bottom, top = np.quantile(data1pe, [0.25, 0.75], axis=0)
            ax.fill_between(np.arange(length), bottom, top, facecolor='lightgray', label='25%-75% quantiles')
            ax.plot(waveform, 'k-', label='median')
            ax.set_ylabel('ADC value')
        
        ax.grid()
        ax.legend(loc='best')
        ax.set_xlabel('Sample number')
        ax.set_title('Template for matched filter')
        
        ax = axs[1]
        
        f, s = signal.periodogram(wnocov if noisecorr else waveform, window='hann')
        ax.plot(f[1:], np.sqrt(s[1:]), label='spectrum of template')
        
        f, ss = signal.periodogram(data1pe, axis=-1)
        s = np.median(ss, axis=0)
        ax.plot(f[1:], np.sqrt(s[1:]), label='spectrum of template sources')
        
        ax.set_ylabel('Spectral density [GHz$^{-1/2}$]')
        ax.set_xlabel('Frequency [GHz]')
        ax.grid()
        ax.set_yscale('log')
        ax.legend(loc='best')
    
    if noisecorr and figcov is not None:
        ax = figcov.subplots(1, 1)
        
        m = np.max(np.abs(cov))
        ax.imshow(cov, vmin=-m, vmax=m, cmap='PiYG')
        
        ax.set_title('Noise covariance matrix')
        ax.set_xlabel('Sample number [ns]')
        ax.set_ylabel('Sample number [ns]')
    
    if norm:
        waveform /= np.sum(waveform)
    return waveform

def toeplitze(a):
    """
    Convert the matrix `a` to a Toeplitz matrix by averaging the shifted rows.
    """
    rowsum = np.zeros(len(a))
    for i in range(len(a)):
        rowsum += np.roll(a[i], -i)
    rowsum /= len(a)
    b = np.empty(a.shape)
    for i in range(len(a)):
        b[i] = np.roll(rowsum, i)
    return b
