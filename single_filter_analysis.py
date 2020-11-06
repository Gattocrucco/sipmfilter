from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

def single_filter_analysis(corr_value, fig1=None, fig2=None):
    """
    Compute the SNR i.e. ratio of signal amplitude over amplitude of noise for
    an array of filter outputs.
    
    Parameters
    ----------
    corr_value : 1D array
        The filter output already corrected for sign and baseline.
    fig1, fig2 : matplotlib figure objects (optional)
        If given, make a fingerplot and a plot of the peak centers and widths.
    
    Returns
    -------
    snr : float
        The ratio of the center of the second peak over the width of the first.
    """
    
    # Make a histogram and find the peaks in the histogram.
    L, R = np.quantile(corr_value, [0, 0.98])
    bins = np.linspace(L, R, 101)
    counts, _ = np.histogram(corr_value, bins=bins)
    peaks, pp = signal.find_peaks(counts, prominence=16, height=16, distance=6)
    ph = pp['peak_heights']
    psel = np.concatenate([[True], (ph[1:] / ph[:-1]) > 1/5])
    peaks = peaks[psel]
    ph = ph[psel]
    if len(peaks) <= 1:
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
        
        ax.set_title('Histogram of baseline-corrected and inverted signal')
        ax.set_xlabel('ADC scale')
        ax.set_ylabel('Occurences')

        ax.plot(bins[:-1], counts, drawstyle='steps-post', zorder=10, label='histogram')
        ax.plot(peaks_loc, ph, 'x')

        kwvline = dict(linestyle='--', color='black', linewidth=1, label='median')
        kwvspan = dict(color='lightgray', label='symmetrized 68 % interquantile range')
        for i in range(len(center)):
            ax.axvline(center[i], **kwvline)
            ax.axvspan(center[i] - width[i], center[i] + width[i], **kwvspan)
            kwvline.pop('label', None)
            kwvspan.pop('label', None)

        kwvline = dict(linestyle=':', color='gray', label='boundaries')
        for i in range(len(window)):
            ax.axvline(window[i], **kwvline)
            kwvline.pop('label', None)

        ax.set_xlim(L - 10, R + 10)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.legend(loc='upper right')
    
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
    
    return 0 if bad else snr
