from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
import numba

_figcount = 0
def figwithsize(size=None):
    global _figcount
    _figcount += 1
    fig = plt.figure(f'simplefilter{_figcount:02d}', figsize=size)
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

def filter_moving_average(a, n):
    assert n >= 1
    assert len(a) > n
    s = np.cumsum(a)
    return (s[n:] - s[:-n]) / n

def trigger_under_threshold(a, thr):
    under = np.array(a < thr, dtype=np.int8)
    change = np.diff(under, prepend=0)
    indices = np.flatnonzero(change > 0)
    return indices

def dead_time_filter(indices, dead):
    diffs = np.diff(indices, prepend=-dead)
    return indices[diffs > dead]

# @numba.jit(nopython=True, cache=True)
# def trigger_under_threshold(a, thr, dead):
#     out = np.zeros(len(a), dtype=bool)
#     last = -dead
#     for i in range(1, len(a)):
#         if a[i] < thr and a[i - 1] >= thr and i - last > dead:
#             out[i] = True
#     return np.flatnonzero(out)

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True) # mmap = memory map, no RAM used
data = data.reshape(-1, 2, 15011) # the number 15011 is from dsfe/README.md
data = np.copy(data[:1000]) # the file is actually loaded into memory here

print('identify true signals with trigger...')
start, value, baseline = integrate(data)
absolute_start = start + np.arange(len(start)) * data.shape[-1]

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, 100:8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

print('search signals with filter...')
wave = data[:, 0, :]
wave[:, :20] = 880 # there's garbage at the start of each acq. event

# Apply moving average filter and 
avg_len = 1000
fwave = filter_moving_average(wave.reshape(-1), avg_len)
locs = trigger_under_threshold(fwave, 866) + avg_len

# Remove duplicate detections and spurious signals.
locs0 = locs // data.shape[-1] # event index
ignore_locs = ignore[locs0]
locs_clean = dead_time_filter(locs[~ignore_locs], 3000)

# Find detected signals which are close to a trigger impulse.
sl = np.concatenate([
    absolute_start,
    locs_clean
])
sl_locmask = np.concatenate([
    np.zeros(len(absolute_start), bool),
    np.ones(len(locs_clean), bool)
])
sl_idxs = np.argsort(sl)
sl = sl[sl_idxs]
sl_locmask = sl_locmask[sl_idxs]
sl_diff = np.diff(sl, prepend=-10000000, append=10000000)
sl_close = sl_diff < 2 * avg_len
sl_close = sl_close[1:] | sl_close[:-1]
locs_triggered = sl[sl_close & sl_locmask]

fig = figwithsize([11.8,  4.8])

ax = fig.subplots(1, 1)
ax.set_title('Filter and signal detection')
ax.set_xlabel('Sample number (ns)')
ax.set_ylabel('ADC value')

n = 100000
ax.plot(wave.reshape(-1)[:n], label='original')
ax.plot(fwave[:n], label='filtered')
ax.plot(locs_clean, np.full(locs_clean.shape, 866), 'xk', label='signal detected')
ax.plot(absolute_start, np.full(start.shape, 836), '+k', label='laser trigger')
ax.plot(locs_triggered, np.full(locs_triggered.shape, 806), '*k', label='true signals detected')

ax.legend(loc='best')
ax.set_ylim(-10, 1024)
ax.set_xlim(-1, n + 1)

saveaspng(fig)

plt.show()
