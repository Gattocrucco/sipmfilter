"""
OUTDATED, replaced by `fingersnr.py` and `make_template.py`.

Compute the filtered SNR on an LNGS wav using the matched filter.

The file to read is hardcoded at the beginning of this script. Run this script
in an IPython shell and then call functions as suggested by the onscreen
instructions.

Functions
---------
make_template :
    Make a template for the matched filter.
fingerplot :
    Plot a fingerplot for a chosen filter.
snrseries :
    Compute the SNR for a range of filter length and delay from trigger.
snrplot :
    Plot the output of `snrseries`.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, optimize
import tqdm

import readwav
import integrate
from single_filter_analysis import single_filter_analysis

# Load wav file.
filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False, maxevents=None)
ignore = readwav.spurious_signals(data)
print(f'ignoring {np.sum(ignore)} events with signals in baseline zone')

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

def fingerplot(delta=0, bslen=8000):
    """
    Make a fingerplot with the matched filter.
    
    Parameters
    ----------
    delta : int
        The offset from the trigger minus the waveform length.
    bslen : int
        Number of samples used for baseline computation.
    """
    fig1 = plt.figure('fingersnrmf-fingerplot-1', figsize=[7.27, 5.73])
    fig2 = plt.figure('fingersnrmf-fingerplot-2', figsize=[6.4, 4.8])
    fig1.clf()
    fig2.clf()

    trigger, baseline, value = integrate.filter(data, bslen=bslen, delta_mf=len(waveform) + delta, waveform_mf=waveform)
    value = value[:, 0]
    corr_value = (baseline - value)[~ignore]
    snr = single_filter_analysis(corr_value, fig1, fig2)
    print(f'snr = {snr:.2f}')

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.show()
    fig2.show()

def snrseries(deltamin=-50, deltamax=50, ndelta=101, bslen=8000):
    """
    Compute SNR as a function of the offset from the trigger where the filter
    is evaluated ("delta").
    
    Parameters
    ----------
    deltamin, deltamax, ndelta: int
        The delta values where the SNR is computed is a range of ndelta values
        from deltamin to deltamax. The range is specified relative to the
        number of samples of the waveform, i.e. delta=0 -> delta=len(waveform).
    bslen : int
        The number of samples used for the baseline.
    
    Returns
    -------
    delta : array (ndelta,)
        Values of delta.
    snr : array (ndelta,)
        The SNR for each delta.
    """
    delta = np.rint(np.linspace(deltamin, deltamax, ndelta)) + len(waveform)
    start, baseline, value = integrate.filter(data, bslen=bslen, delta_mf=delta, waveform_mf=waveform)

    snr = np.empty(len(delta))

    for i in tqdm.tqdm(range(len(snr))):
        val = value[:, i]
        corr_value = (baseline - val)[~ignore]
        snr[i] = single_filter_analysis(corr_value)
    
    output = delta, snr
    snrplot(*output)
    return output

def snrplot(delta, snr):
    """
    Plot SNR as a function of delta. Called by snrseries().
    
    Parameters
    ----------
    The output from snrseries().
    
    Returns
    -------
    fig : matplotlib figure
        The figure with the plots.
    """

    fig = plt.figure('fingersnrmf-snrplot')
    fig.clf()

    ax = fig.subplots(1, 1)

    ax.plot(delta, snr, '.--')
    ax.set_title('SNR for matched filter')
    ax.set_ylabel('SNR')
    ax.set_xlabel('Offset from trigger [ns]')
    ax.grid()

    fig.tight_layout()
    fig.show()
    
    return fig

# Make the template for the matched filter and plot it.
fig = plt.figure('fingersnrmf-make_template')
fig.clf()

waveform = make_template(data, ignore=ignore, fig=fig)

fig.tight_layout()
fig.show()

# Compute the matched filter and do a fingerplot.
fingerplot()
    
# Plot the SNR as a function of delta.
print('computing snr series...')
snrseries()

print('now call interactively any of:')
print('fingerplot(<delta>, <bslen>)')
print('snrseries(<deltamin>, <deltamax>, <ndelta>, <bslen>)')
