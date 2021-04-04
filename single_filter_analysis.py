"""
Module to make a fingerplot automatically. Tuned on LNGS wav data, may have
problems on other data.
"""

from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

def plot_histogram(ax, counts, bins, **kw):
    """
    Plot an histogram.
    
    Parameters
    ----------
    ax : matplotlib axis
        The axis where the histogram is drawn.
    counts, bins : array
        The output from `np.histogram`.
    **kw :
        Keyword arguments are passed to `ax.plot`.
    
    Return
    ------
    lines : tuple
        The return value from `ax.plot`.
    """
    return ax.plot(np.pad(bins, (1, 0), 'edge'), np.pad(counts, 1), drawstyle='steps-post', **kw)

def single_filter_analysis(corr_value, fig1=None, fig2=None, return_full=False):
    """
    Do a fingerplot and compute the SNR for an array of values.
    
    Parameters
    ----------
    corr_value : 1D array
        The filter output already corrected for sign and baseline. May not
        work if there are less than 1000 values.
    fig1, fig2 : matplotlib figure objects (optional)
        If given, make a fingerplot and a plot of the peak centers and widths.
    return_full : bool
        If True return additional information.
    
    Returns
    -------
    snr : float
        The ratio of the center of the second peak over the width of the first.
    
    The following are returned if return_full=True:
    
    center : array (M,)
        The centers of the peaks.
    width : array (M,)
        The width of the peaks. Standard deviation or equivalent.
    """
    
    # Make a histogram.
    L, R = np.quantile(corr_value, [0, 1 - 200 / len(corr_value)])
    bins = np.linspace(L, R, 101)
    counts, _ = np.histogram(corr_value, bins=bins)
    
    # Add an empty bin to the left for find_peaks (it searches local maxima).
    bins = np.concatenate([[bins[0] - (bins[1] - bins[0])], bins])
    counts = np.concatenate([[0], counts])
    
    # Find peaks in the histogram.
    peaks, pp = signal.find_peaks(counts, prominence=16, height=16, distance=6)
    ph = pp['peak_heights']
    psel = np.concatenate([[True], (ph[1:] / ph[:-1]) > 1/5])
    peaks = peaks[psel]
    ph = ph[psel]
    if len(peaks) <= 1:
        if return_full:
            return 0, np.empty(0), np.empty(0)
        else:
            return 0
    
    # Take regions around the peaks.
    bins_center = (bins[1:] + bins[:-1]) / 2
    peaks_loc = bins_center[peaks]
    window_mid = (peaks_loc[1:] + peaks_loc[:-1]) / 2
    window = np.concatenate([
        # [peaks_loc[0] - 2 * (window_mid[0] - peaks_loc[0])],
        [-np.inf],
        window_mid,
        [peaks_loc[-1] + (peaks_loc[-1] - window_mid[-1])]
    ])
    
    # Compute median and interquantile range for each region.
    center, width, N = np.empty((3, len(window) - 1))
    for i in range(len(window) - 1):
        selection = (window[i] <= corr_value) & (corr_value < window[i + 1])
        values = corr_value[selection]
        center[i] = np.median(values)
        width[i] = np.diff(np.quantile(values, [0.50 - 0.68/2, 0.50 + 0.68/2]))[0] / 2
        N[i] = len(values)
    
    # Check if the positions of the peaks make sense.
    medianstd = np.sqrt(np.pi / 2) * width / np.sqrt(N)
    firstbad = np.abs(center[0]) > 5 * medianstd[0]
    secondbad = center[1] < 5 * medianstd[1]
    bad = firstbad or secondbad
    
    # Compute signal to noise ratio.
    snr = center[1] / width[0]
    
    # Figure of histogram with peaks highlighted.
    if fig1 is not None:
        ax = fig1.subplots(1, 1)
        
        ax.set_xlabel('Baseline-corrected filter output [ADC unit]')
        ax.set_ylabel('Counts per bin')

        plot_histogram(ax, counts, bins, color='black', zorder=2.1, label='histogram')
        ax.plot(peaks_loc, ph, 'o', color='#f55', zorder=2.2, label='auto-detected peaks')

        kwvline = dict(linestyle=':', color='black', label='boundaries')
        for i in range(len(window)):
            ax.axvline(window[i], **kwvline)
            kwvline.pop('label', None)

        kwvline = dict(linestyle='--', color='black', linewidth=1, label='median')
        kwvspan = dict(color='lightgray', label='symmetrized 68 % interquantile range')
        for i in range(len(center)):
            ax.axvline(center[i], **kwvline)
            ax.axvspan(center[i] - width[i], center[i] + width[i], **kwvspan)
            kwvline.pop('label', None)
            kwvspan.pop('label', None)

        ax.legend(loc='upper right')
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
    # Figure of centers and widths of peaks.
    if fig2 is not None:
        ax = fig2.subplots(2, 1, sharex=True)
        
        ax[0].set_title('Center and width of peaks in signal histogram')
        ax[0].set_ylabel('median')
        ax[1].set_ylabel('68 % half interquantile range')
        ax[1].set_xlabel('Peak number (number of photoelectrons)')

        ax[0].plot(center, '.--')
        ax[1].plot(width, '.--')

        for a in ax:
            a.grid()
    
    output = 0 if bad else snr
    if return_full:
        output = (output, center, width)
    return output
