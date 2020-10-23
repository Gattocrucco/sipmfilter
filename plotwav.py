from scipy.io import wavfile
from scipy import fft, signal
import numpy as np
from matplotlib import pyplot as plt

_figcount = 0
def figwithsize(size=None):
    global _figcount
    _figcount += 1
    fig = plt.figure(f'plotwav{_figcount:02d}', figsize=size)
    fig.clf()
    if size is not None:
        fig.set_size_inches(size)
    return fig

def saveaspng(fig):
    name = fig.canvas.get_window_title() + '.png'
    print(f'saving {name}...')
    fig.tight_layout()
    fig.savefig(name)

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True)
data = data.reshape(-1, 2, 15011) # the number 15011 is from dsfe/README.md
data = np.copy(data[:1000])

signal = data[:, 0, :].reshape(-1)
trigger = data[:, 1, :].reshape(-1)

print(f'declared rate = {rate} Hz')
print(f'data type = {data.dtype}')

signalmin = np.min(signal)
signalmax = np.max(signal)
print(f'min = {signalmin}, max = {signalmax}')

print('computing global histogram...')
widesignal = np.array(signal, dtype=np.int32)
counts = np.bincount(widesignal - signalmin)

fig = figwithsize([11.8, 4.8])

ax = fig.subplots(1, 1)
ax.set_title('Histogram of all data')
ax.set_xlabel('ADC value')
ax.set_ylabel('occurences')

ax.plot(np.arange(signalmin, signalmax + 1), counts, drawstyle='steps')

ax.set_yscale('symlog')
ax.set_ylim(-1, ax.get_ylim()[1])
ax.grid()

saveaspng(fig)

fig = figwithsize([8.21, 5.09])

ax = fig.subplots(1, 1)

start = 0
s = slice(start, start + 125000)
ax.plot(signal[s], ',', color='red', label='signal')
ax.plot(trigger[s], ',', color='blue', label='trigger')
ax.set_ylim(-1, 2**10 + 1)

ax.legend(loc='best')
ax.set_title('Original signal')
ax.set_xlabel(f'Sample number (starting from {start})')
ax.set_ylabel('ADC value')

saveaspng(fig)
