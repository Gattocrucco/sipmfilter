import numpy as np
from matplotlib import pyplot as plt

import integrate
from single_filter_analysis import single_filter_analysis

def make_template(data, ignore=None, length=2000, fig=None):
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
    fig : matplotlib figure, optional
        If given, plot the waveform.
    
    Return
    ------
    waveform : array (length,)
        The waveform. It is normalized to unit sum.
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
    waveform, bottom, top = np.quantile(data1pe, [0.5, 0.25, 0.75], axis=0)
    # waveform = np.mean(data1pe, axis=0)
    # std = np.std(data1pe, axis=0)
    # bottom = waveform - std
    # top = waveform + std
    
    if fig is not None:
        ax = fig.subplots(1, 1)

        ax.fill_between(np.arange(length), bottom, top, facecolor='lightgray', label='25%-75% quantiles')
        ax.plot(waveform, 'k-', label='median')
        
        ax.grid()
        ax.legend(loc='best')
        ax.set_xlabel('Sample number')
        ax.set_ylabel('ADC value')
        ax.set_title('Template for matched filter (unnormalized)')
    
    return waveform / np.sum(waveform)
