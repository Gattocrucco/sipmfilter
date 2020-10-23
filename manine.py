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
def integrate(data):
    start = np.empty(data.shape[0], dtype=np.int32)
    value = np.empty(data.shape[0], dtype=np.int32)
    
    for i in range(data.shape[0]):
        signal = data[i, 0]
        trigger = data[i, 1]
                
        for j in range(len(trigger)):
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
        value[i] = 0
        for j in range(j, j + 1000):
            if 0 <= signal[j] < 2 ** 10:
                value[i] += signal[j]
    
    return start, value

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True)
data = data.reshape(-1, 2, 15011) # the number 15011 is from dsfe/README.md
data = np.copy(data) # actually loaded into memory here

print('computing sum...')
start, value = integrate(data)

fig = figwithsize()

ax = fig.subplots(1, 1)
ax.set_title('Histogram of 1000 samples average after trigger ends')
ax.set_xlabel('ADC scale')
ax.set_ylabel('Occurences')

ax.hist(value / 1000, bins=1000, histtype='stepfilled')

saveaspng(fig)
fig.show()
