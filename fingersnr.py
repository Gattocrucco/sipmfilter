import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, optimize
import tqdm

import readwav
import fighelp
import integrate

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)

tau = np.array([32, 64, 128, 192, 256, 320, 384, 512, 768, 1024])

# delta for moving average
delta_rel = np.linspace(0.5, 1.4, 10)
delta_off = 80 + np.linspace(-40, 40, 10)
delta = delta_off + delta_rel * tau.reshape(-1, 1)
delta = np.array(delta, int).reshape(-1)

# delta for exponential moving average
delta_rel_exp = np.linspace(0.1, 2, 10)
delta_off_exp = np.linspace(75, 500, 10)
delta_exp = delta_off_exp + delta_rel_exp * tau.reshape(-1, 1)
delta_exp = np.array(delta_exp, int).reshape(-1)

tau = np.array(tau, int)
tau = np.broadcast_to(tau.reshape(-1, 1), tau.shape + delta_rel.shape).reshape(-1)

print('computing...')
start, baseline, vma, vexp = integrate.filter(data, delta, tau, delta_exp, tau)

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, :8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

def single_filter_analysis(corr_value, fig1=None, fig2=None):
    """
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

snr = np.empty((2, len(tau)))

for i in tqdm.tqdm(range(snr.shape[1])):
    for j, value in enumerate([vma, vexp]):
        value = value[:, i]
        corr_value = (baseline - value)[~ignore]
        snr[j, i] = single_filter_analysis(corr_value)

def fun(delta, tau, useexp=False):
    delta = np.array([delta], int)
    tau = np.array([tau], int)
    start, baseline, vma, vexp = integrate.filter(data, delta, tau, delta, tau)
    value = vexp if useexp else vma
    corr_value = (baseline - value[:, 0])[~ignore]
    snr = single_filter_analysis(corr_value)
    return snr

snr = snr.reshape(2, -1, len(delta_rel))
tau = tau.reshape(-1, len(delta_rel))
delta = delta.reshape(tau.shape)
delta_exp = delta_exp.reshape(tau.shape)

fig = fighelp.figwithsize([6.4, 7.1], resetfigcount=True)

axs = fig.subplots(2, 1)

axs[0].set_title('Moving average')
axs[1].set_title('Exponential average')
axs[1].set_xlabel('Offset from trigger [samples]')

for i, ax in enumerate(axs):
    for j in range(snr.shape[1]):
        alpha = 1 - (j / snr.shape[1])
        label = f'tau = {tau[j, 0]}'
        d = delta if i == 0 else delta_exp
        ax.plot(d[j], snr[i, j], color='black', alpha=alpha, label=label)
    ax.set_ylabel('SNR')
    ax.legend(loc='best', fontsize='small')
    ax.grid()

fighelp.saveaspng(fig)

def fingerplot(tau, delta, kind='ma'):
    """
    Call this interactively after running the script in ipython
    """
    tau = np.array([tau], int)
    delta = np.array([delta], int)
    start, baseline, vma, vexp = integrate.filter(data, delta, tau, delta, tau)
    value = dict(ma=vma, exp=vexp)[kind][:, 0]
    corr_value = (baseline - value)[~ignore]
    fig1 = plt.figure('fingersnr-fingerplot-1', figsize=[7.27, 5.73])
    fig2 = plt.figure('fingersnr-fingerplot-2', figsize=[6.4, 4.8])
    fig1.clf()
    fig2.clf()
    snr = single_filter_analysis(corr_value, fig1, fig2)
    print(f'snr = {snr:.1f}')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.show()
    fig2.show()
    
print('call fingerplot(<tau>, <delta>, "ma" or "exp") interactively')

plt.show()
