import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import readwav
import fighelp
import integrate

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)

print('computing...')
start, baseline, vma, vexp = integrate.filter(data, *np.full((4, 1), 1000))
vma = vma[:, 0]
vexp = vexp[:, 0]

value = vexp

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, :8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

corr_value = (baseline - value)[~ignore]
counts, bins = np.histogram(corr_value, bins=1000)
peaks, pp = signal.find_peaks(counts, prominence=8, height=8)
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

fig = fighelp.figwithsize([6.4, 4.8], resetfigcount=True)

ax = fig.subplots(1, 1)
ax.set_title('Histograms of signals and baselines')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Bin count')

ax.hist(value[~ignore], bins=1000, histtype='step', label='signal')
ax.hist(baseline[~ignore], bins='auto', histtype='step', label='baseline')

ax.legend(loc='best')

fighelp.saveaspng(fig)

fig = fighelp.figwithsize([7.27, 5.73])

ax = fig.subplots(1, 1)
ax.set_title('Histogram of baseline-corrected and inverted signal')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Occurences')

corr_value = baseline - value
ax.hist(corr_value[~ignore], bins=1000, histtype='step', zorder=10, label='histogram')
ax.plot(peaks_loc, ph, 'x')

kwvline = dict(linestyle='--', color='black', linewidth=1, label='median')
kwvspan = dict(color='lightgray', label='symmetrized 68 % interquantile range')
for i in range(len(center)):
    ax.axvline(center[i], **kwvline)
    ax.axvspan(center[i] - width[i], center[i] + width[i], **kwvspan)
    kwvline.pop('label', None)
    kwvspan.pop('label', None)

kwvline = dict(linestyle=':', color='gray', label='boundaries (handpicked)')
for i in range(len(window)):
    ax.axvline(window[i], **kwvline)
    kwvline.pop('label', None)

ax.set_xlim(-15, 315)
ax.legend(loc='upper right')

fighelp.saveaspng(fig)

fig = fighelp.figwithsize([6.4, 4.8])

ax = fig.subplots(2, 1, sharex=True)
ax[0].set_title('Center and width of peaks in signal histogram')
ax[0].set_ylabel('median')
ax[1].set_ylabel('68 % half interquantile range')
ax[1].set_xlabel('Peak number (number of photoelectrons)')

ax[0].plot(center, '.--')
ax[1].plot(width, '.--')

for a in ax:
    a.grid()

fighelp.saveaspng(fig)

plt.show()
