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
        
        for j in range(j + 1, len(trigger)):
            if 400 <= trigger[j] < 2 ** 10:
                break
        else:
            assert False, 'no trigger end found'
        
        assert j + 1000 <= len(signal), 'less than 1000 samples after trigger'

        start[i] = j
        value[i] = average_inadcrange(signal[j:j + 1000])
        baseline[i] = average_inadcrange(signal[:j])
    
    return start, value, baseline

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True) # mmap = memory map, no RAM used
data = data.reshape(-1, 2, 15011) # the number 15011 is from dsfe/README.md
data = np.copy(data) # the file is actually loaded into memory here

print('computing...')
start, value, baseline = integrate(data)

fig = figwithsize()

ax = fig.subplots(1, 1)
ax.set_title('Histograms of signals and baselines')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Bin count')

ax.hist(value, bins=1000, histtype='step', label='signal')
ax.hist(baseline, bins='auto', histtype='step', label='baseline')

ax.legend(loc='best')

saveaspng(fig)

fig = figwithsize()

ax = fig.subplots(1, 1)
ax.set_title('Histogram of baseline-corrected and inverted signal')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Occurences')

corr_value = baseline - value
ax.hist(corr_value, bins=1000, histtype='stepfilled')

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
