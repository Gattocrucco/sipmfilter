from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
import numba

_figcount = 0
def figwithsize(size=None):
    global _figcount
    _figcount += 1
    fig = plt.figure(f'manine{_figcount:02d}', figsize=size)
    fig.clf()
    if size is not None:
        fig.set_size_inches(size)
    return fig

def saveaspng(fig):
    name = fig.canvas.get_window_title() + '.png'
    print(f'saving {name}...')
    fig.tight_layout()
    fig.savefig(name)

@numba.jit(cache=True, nopython=True)
def average_inadcrange(a):
    out = 0
    n = 0
    for x in a:
        if 0 <= x < 2 ** 10:
            out += x
            n += 1
    return out / n

@numba.jit(cache=True, nopython=True)
def integrate(data):
    start = np.empty(data.shape[0], dtype=np.int32)
    value = np.empty(data.shape[0])
    baseline = np.empty(data.shape[0])
    
    for i in range(data.shape[0]):
        signal = data[i, 0]
        trigger = data[i, 1]
                
        for j in range(100, len(trigger)):
            if 0 <= trigger[j] < 400:
                break
        else:
            assert False, 'no trigger start found'
        
        # Uncomment this to start from the end of the trigger square impulse.
        # for j in range(j + 1, len(trigger)):
        #     if 400 <= trigger[j] < 2 ** 10:
        #         break
        # else:
        #     assert False, 'no trigger end found'
        
        assert j + 1000 <= len(signal), 'less than 1000 samples after trigger'
        assert j >= 7000, 'less than 7000 samples before trigger'

        start[i] = j
        value[i] = average_inadcrange(signal[j:j + 1000])
        baseline[i] = average_inadcrange(signal[j - 7000:j - 100])
    
    return start, value, baseline

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True) # mmap = memory map, no RAM used
data = data.reshape(-1, 2, 15011) # the number 15011 is from dsfe/README.md
data = np.copy(data) # the file is actually loaded into memory here

print('computing...')
start, value, baseline = integrate(data)

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, 100:8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

# Boundaries for peaks in (baseline - value) histogram
# Written down by looking at the plot
window = np.array([
    -15,
    12,
    37,
    62,
    86,
    110,
    137,
    159,
    183,
    207,
    232,
    256,
    281,
    297
])

corr_value = (baseline - value)[~ignore]
center, width = np.empty((2, len(window) - 1))
for i in range(len(window) - 1):
    selection = (window[i] <= corr_value) & (corr_value < window[i + 1])
    values = corr_value[selection]
    center[i] = np.median(values)
    width[i] = np.diff(np.quantile(values, [0.50 - 0.68/2, 0.50 + 0.68/2]))[0] / 2

fig = figwithsize()

ax = fig.subplots(1, 1)
ax.set_title('Histograms of signals and baselines')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Bin count')

ax.hist(value[~ignore], bins=1000, histtype='step', label='signal')
ax.hist(baseline[~ignore], bins='auto', histtype='step', label='baseline')

ax.legend(loc='best')

saveaspng(fig)

fig = figwithsize([7.27, 5.73])

ax = fig.subplots(1, 1)
ax.set_title('Histogram of baseline-corrected and inverted signal')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Occurences')

corr_value = baseline - value
ax.hist(corr_value[~ignore], bins=1000, histtype='step', zorder=10, label='histogram')

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

saveaspng(fig)

fig = figwithsize()

ax = fig.subplots(2, 1, sharex=True)
ax[0].set_title('Center and width of peaks in signal histogram')
ax[0].set_ylabel('median')
ax[1].set_ylabel('68 % half interquantile range')
ax[1].set_xlabel('Peak number (number of photoelectrons)')

ax[0].plot(center, '.--')
ax[1].plot(width, '.--')

for a in ax:
    a.grid()

saveaspng(fig)

fig = figwithsize()

ax = fig.subplots(2, 1, sharex=True)
ax[0].set_title('Baseline < 800')
ax[1].set_title('Corresponding triggers')
ax[1].set_xlabel('Event sample number')
ax[0].set_ylabel('ADC value')
ax[1].set_ylabel('ADC value')

for i in np.argwhere(baseline < 800).reshape(-1):
    ax[0].plot(data[i, 0], ',')
    ax[1].plot(data[i, 1], ',')

ax[0].set_ylim(-50, 1050)
ax[1].set_ylim(-50, 1050)

saveaspng(fig)

fig = figwithsize()

ax = fig.subplots(2, 1, sharex=True)
ax[0].set_title('Signal 1000 samples average < 400')
ax[1].set_title('Corresponding triggers')
ax[1].set_xlabel('Event sample number')
ax[0].set_ylabel('ADC value')
ax[1].set_ylabel('ADC value')

for i in np.argwhere(value < 400).reshape(-1):
    ax[0].plot(data[i, 0], ',')
    ax[1].plot(data[i, 1], ',')

ax[0].set_ylim(-50, 1050)
ax[1].set_ylim(-50, 1050)

saveaspng(fig)

plt.show()
