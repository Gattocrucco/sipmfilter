import numpy as np
from matplotlib import pyplot as plt
import uproot
from scipy.io import wavfile
from scipy import signal
import numba

_figcount = 0
def figwithsize(size=None):
    global _figcount
    _figcount += 1
    fig = plt.figure(f'spectrum{_figcount:02d}', figsize=size)
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

filename = 'merged_000886.root'
print(f'reading {filename}...')
root = uproot.open(filename)
tree = root['midas_data']
noise = tree.array('adc_W200_Ch00')

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True) # mmap = memory map, no RAM used
data = data.reshape(-1, 2, 15011) # the number 15011 is from dsfe/README.md
data = np.copy(data) # the file is actually loaded into memory here

print('computing signal...')
start, value, baseline = integrate(data)

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, 100:8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

# Select events with no signals.
corrvalue = baseline - value
selection = ~ignore & (corrvalue < 12)
idxs = np.flatnonzero(selection)
indices0 = idxs.reshape(-1, 1)
indices1 = start[selection].reshape(-1, 1) + np.arange(1000)
nosignal = data[indices0, 0, indices1]

print('computing spectra...')
kw = dict(nperseg=2 ** 14, average='median')
f1, s1 = signal.welch(np.concatenate(noise), 125e6, **kw)
f2, s2 = signal.welch(nosignal.reshape(-1), 1e9, **kw)

fig = figwithsize([11.8,  4.8])

ax = fig.subplots(1, 1)

def norm(f, s):
    return np.sum(s) * len(f) * (f[1] - f[0])

ax.plot(f1, s1 / norm(f1, s1), label='noise file')
ax.plot(f2, s2 / norm(f2, s2) , label='no signal events')

ax.set_title('Spectra of noise and no signal events')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Normalized power spectral density [Hz$^{-1}$]')

ax.legend(loc='best')
ax.grid()
ax.set_xlim(0, min(np.max(f1), np.max(f2)))

saveaspng(fig)

plt.show()
