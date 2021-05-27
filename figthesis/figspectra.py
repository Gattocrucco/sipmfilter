import os

import numpy as np
from matplotlib import pyplot as plt
import uproot
from scipy import signal

import readwav
import figlatex

npfile = 'figthesis/figspectra.npz'

def savenp():
    filename = 'darksidehd/merged_000886.root'
    channel = 'adc_W201_Ch00'
    # look at PDMadcCh.png to match the channel to the wav file
    print(f'reading {filename}, channel {channel}...')
    root = uproot.open(filename)
    tree = root['midas_data']
    noise = tree.array(channel)

    filename = 'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav'
    data = readwav.readwav(filename, mmap=False)
    ignore = readwav.spurious_signals(data)
    print(f'ignore {np.count_nonzero(ignore)} events due to spurious signals')
    baseline = data[~ignore, 0, :8900]

    kw = dict() # arguments for signal.periodogram()

    print('computing noise file spectrum...')
    counts = np.unique(noise.counts)
    assert len(counts) == 2 and counts[0] == 0
    noise_array = noise._content.reshape(-1, counts[-1])
    f1, s1s = signal.periodogram(noise_array, 125e6, **kw)
    s1 = np.median(s1s, axis=0)
    s1 *= (2 / 2 ** 14) ** 2 # 2 Vpp, 14 bit

    print('computing signal file baseline spectrum...')
    f2, s2s = signal.periodogram(baseline, 1e9, **kw)
    s2 = np.median(s2s, axis=0)
    s2 *= (1 / 2 ** 10) ** 2 # 1 Vpp, 10 bit
    
    print(f'save {npfile}')
    np.savez(npfile, f1=f1, f2=f2, s1=s1, s2=s2)

if __name__ == '__main__':
    if not os.path.exists(npfile):
        savenp()

    print(f'read {npfile}')
    arch = np.load(npfile)
    f1 = arch['f1']
    f2 = arch['f2']
    s1 = arch['s1']
    s2 = arch['s2']

    fig, axs = plt.subplots(2, 1, num='figspectra', clear=True, figsize=[9, 4.71], sharey=True)

    for i, ax in enumerate(axs):
    
        sel = f2 < f1[-1] if i == 0 else slice(None)
        ax.plot(f2[sel][1:] / 1e6, np.sqrt(s2[sel][1:]) * 1e3, label='LNGS', color='black', linestyle='--')
        ax.plot(f1[1:] / 1e6, np.sqrt(s1[1:]) * 1e3, label='Proto0', color='black')

        ax.axvspan(0, f1[-1] / 1e6, color='#ddd')

        if ax.is_first_row():
            ax.legend(loc='best')
        if ax.is_last_row():
            ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Spectral density [MHz$^{-1/2}$]')

        ax.set_yscale('log')
        ax.set_xlim(0, f1[-1] / 1e6 if i == 0 else f2[-1] / 1e6)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')

    fig.tight_layout()
    fig.show()

    figlatex.save(fig)
