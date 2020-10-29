import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, optimize
import tqdm

import readwav
import fighelp
import integrate

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)

delta_rel = np.linspace(0.9, 1.1, 10)
tau = np.linspace(500, 2000, 10)

delta = (tau.reshape(-1, 1) * delta_rel).reshape(-1)
tau = np.broadcast_to(tau.reshape(-1, 1), tau.shape + delta_rel.shape).reshape(-1)

delta = np.array(delta, int)
tau = np.array(tau, int)

print('computing...')
start, baseline, vma, vexp = integrate.filter(data, delta, tau, delta, tau)

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, :8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

def single_filter_analysis(corr_value, fig1=None, fig2=None):
    counts, bins = np.histogram(corr_value, bins=500)
    peaks, pp = signal.find_peaks(counts, prominence=16, height=16, distance=2)
    ph = pp['peak_heights']
    psel = np.concatenate([[True], (ph[1:] / ph[:-1]) > 1/5])
    peaks = peaks[psel]
    ph = ph[psel]

    bins_center = (bins[1:] + bins[:-1]) / 2
    peaks_loc = bins_center[peaks]
    window_mid = (peaks_loc[1:] + peaks_loc[:-1]) / 2
    window = np.concatenate([
        [peaks_loc[0] - (window_mid[0] - peaks_loc[0])],
        window_mid,
        [peaks_loc[-1] + (peaks_loc[-1] - window_mid[-1])]
    ])

    center, width = np.empty((2, len(window) - 1))
    for i in range(len(window) - 1):
        selection = (window[i] <= corr_value) & (corr_value < window[i + 1])
        values = corr_value[selection]
        center[i] = np.median(values)
        width[i] = np.diff(np.quantile(values, [0.50 - 0.68/2, 0.50 + 0.68/2]))[0] / 2
    
    snr = center[1] / width[0]
    
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

        ax.set_xlim(window[0] - 10, window[-1] + 10)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.legend(loc='upper right')
    
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
    
    return snr

snr = np.empty((2, len(tau)))

for i in tqdm.tqdm(range(snr.shape[1])):
    for j, value in enumerate([vma, vexp]):
        value = value[:, i]
        corr_value = (baseline - value)[~ignore]
        snr[j, i] = single_filter_analysis(corr_value)

# def fun(x, useexp):
#     delta, tau = x
#     delta = np.array([delta], int)
#     tau = np.array([tau], int)
#     start, baseline, vma, vexp = integrate.filter(data, delta, tau, delta, tau)
#     value = vexp if useexp else vma
#     corr_value = (baseline - value)[~ignore]
#     snr = single_filter_analysis(corr_value)
#     return snr

snr = snr.reshape(2, -1, len(delta_rel))
tau = tau.reshape(-1, len(delta_rel))
delta = delta.reshape(tau.shape)

fig = fighelp.figwithsize([6.4, 7.1], resetfigcount=True)

axs = fig.subplots(2, 1)

axs[0].set_title('Moving average')
axs[1].set_title('Exponential average')
axs[1].set_xlabel('Offset from trigger [samples]')

for i, ax in enumerate(axs):
    for j in range(snr.shape[1]):
        alpha = (j + 1) / snr.shape[1]
        label = f'tau = {tau[j, 0]}'
        ax.plot(delta[j], snr[i, j], color='black', alpha=alpha, label=label)
    ax.set_ylabel('SNR')
    ax.legend(loc='best', fontsize='small')

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

plt.show()
