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
maxsamples = 3000000
print(f'reading {filename}...')
rate, data = wavfile.read(filename, mmap=True)
data = np.copy(data[:maxsamples])

print(f'declared rate = {rate} Hz')
print(f'data type = {data.dtype}')

datamin = np.min(data)
datamax = np.max(data)
print(f'min = {datamin}, max = {datamax}')

print('computing global histogram...')
widedata = np.array(data, dtype=np.int32)
counts = np.bincount(widedata - datamin)

# I'm dropping this because it is not precise enough.
#
# print('computing trigger frequency...')
# n = 2 ** 20 # about 1 million
# datan = np.copy(data[:n])
# outofrange = (datan < 0) | (datan >= 2 ** 10)
# datan[outofrange] = 0
# f = fft.rfft(datan)
# fmag = np.abs(f)
# period_est = 30000 # period estimated by eye (number of samples)
# freq_est = int(n / period_est)
# start = freq_est - freq_est // 2
# peak_nb = fmag[start:start + freq_est] # neighboorhood of the fft peak
# freq = start + np.argmax(peak_nb)
# period = n / freq
# period_err = period * (1 / freq)
# print(f'trigger period = {period:.0f} +- {period_err:.0f} samples')

print('computing 1000 samples moving average...')


fig = figwithsize([11.8, 4.8])

ax = fig.subplots(1, 1)
ax.set_title('Histogram of all data')
ax.set_xlabel('ADC value')
ax.set_ylabel('occurences')

ax.plot(np.arange(datamin, datamax + 1), counts, drawstyle='steps')

ax.set_yscale('symlog')
ax.set_ylim(-1, ax.get_ylim()[1])
ax.grid()

saveaspng(fig)

fig = figwithsize([8.21, 5.09])

ax = fig.subplots(1, 1)

start = 0 #int(100e6)
ax.plot(data[start:start + 250000], ',', color='red')
ax.set_ylim(-1, 2**10 + 1)

ax.set_title('Original signal')
ax.set_xlabel(f'Sample number (starting from {start})')
ax.set_ylabel('ADC value')

saveaspng(fig)
