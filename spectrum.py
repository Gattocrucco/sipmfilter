import numpy as np
from matplotlib import pyplot as plt
import uproot
from scipy import signal
import readwav
import integrate
import fighelp

filename = 'merged_000886.root'
channel = 'adc_W201_Ch00'
# look at PDMadcCh.png to match the channel to the wav file
print(f'reading {filename}, channel {channel}...')
root = uproot.open(filename)
tree = root['midas_data']
noise = tree.array(channel)

filename = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
data = readwav.readwav(filename, mmap=False)

# print('computing signal...')
# start, value, baseline = integrate.integrate(data)

# Identify events with out-of-trigger signals.
baseline_zone = data[:, 0, :8900]
ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
print(f'ignoring {np.sum(ignore)} events with values < 700 in baseline zone')

# # Select events with no signals.
# corrvalue = baseline - value
# selection = ~ignore & (corrvalue < 12)
# idxs = np.flatnonzero(selection)
# indices0 = idxs.reshape(-1, 1)
# indices1 = start[selection].reshape(-1, 1) + np.arange(1000)
# nosignal = data[indices0, 0, indices1]

kw = dict()
counts = np.unique(noise.counts)
assert len(counts) == 2 and counts[0] == 0
noise_array = noise._content.reshape(-1, counts[-1])
print('computing noise file spectrum...')
f1, s1s = signal.periodogram(noise_array, 125e6, **kw)
print('computing signal file baseline spectrum...')
f2, s2s = signal.periodogram(baseline_zone[~ignore], 1e9, **kw)
s1 = np.median(s1s, axis=0)
s2 = np.median(s2s, axis=0)

s1 *= (2 / 2 ** 14) ** 2 # 2 Vpp, 14 bit
s2 *= (1 / 2 ** 10) ** 2 # 1 Vpp, 10 bit

fig = fighelp.figwithsize([11.8, 7.19], resetfigcount=True)

axs = fig.subplots(2, 1)

for i, ax in enumerate(axs):
    
    ax.plot(f1[1:] / 1e6, np.sqrt(s1[1:]) * 1e3, label='SiPM under threshold')
    sel = f2 < f1[-1] if i == 0 else slice(None)
    ax.plot(f2[sel][1:] / 1e6, np.sqrt(s2[sel][1:]) * 1e3, label='baseline before signals')

    if i == 0:
        ax.set_title('Noise spectrum')
        ax.legend(loc='best')
    if i == 1:
        ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Spectral density [Arb. un. MHz$^{-1/2}$]')
    # ax.set_ylabel('Spectral density [V MHz$^{-1/2}$]')

    ax.set_yscale('log')
    ax.grid()
    ax.set_xlim(0, f1[-1] / 1e6 if i == 0 else f2[-1] / 1e6)

fighelp.saveaspng(fig)

plt.show()
