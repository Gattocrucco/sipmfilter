"""
Plot the 2D histogram of a LNGS wav or Proto0 root. Usage:

    hist2d.py filename[:channel] [maxevents [length [start [trig [cmap]]]]]

channel = Proto0 run2 tile number, or tree branch (adc_Wxxx_Chxx).
maxevents = max. number of events read from the file, default 1000.
length = number of samples read per event, default 1000.
start = starting sample relative to the trigger leading edge, default 0.
trig = 'True' (default): align to the trigger, 'False': to the event start.
cmap = matplotlib colormap for the histogram, default 'magma'.
"""

import sys

import numpy as np
from matplotlib import pyplot as plt, colors
import numba

import read
import readroot
import textbox
import runsliced
import num2si

def main(argv):
    filename = argv[1]
    maxevents = 1000
    length = 1000
    start = 0
    trig = True
    cmap = 'magma'

    try:
        maxevents = int(sys.argv[2])
        length = int(sys.argv[3])
        start = int(sys.argv[4])
        trig = {'True': True, 'False': False}[sys.argv[5]]
        cmap = sys.argv[6]
    except IndexError:
        pass
    except BaseException as exc:
        print(__doc__)
        raise exc

    data, trigger, freq, ndigit = read.read(filename, maxevents)

    h = np.zeros((length, ndigit), int)

    @numba.njit(cache=True)
    def accumhist(hist, data, trigger, length, start, trig):
        for ievent, signal in enumerate(data):
            begin = start
            if trig:
                begin += trigger[ievent]
            end = begin + length
            for isample in range(max(0, begin), min(end, len(signal))):
                sample = signal[isample]
                hist[isample - begin, sample] += 1

    runsliced.runsliced(lambda s: accumhist(h, data[s], trigger[s], length, start, trig), len(data), 100)

    fig, ax = plt.subplots(num='hist2d', clear=True, figsize=[10.47, 4.8])

    if ndigit >= 2 ** 11:
        p = ndigit // 1024
        effndigit = ndigit // p * p
        effh = np.sum(h[:, :effndigit].reshape(length, -1, p), axis=-1)
    else:
        p = 1
        effndigit = ndigit
        effh = h

    im = ax.imshow(effh.T, origin='lower', cmap=cmap, norm=colors.LogNorm(), aspect='auto', extent=(-0.5+start, length-0.5+start, -0.5, effndigit-0.5))
    fig.colorbar(im, label=f'Counts per bin (1 sample x {p} digit)', fraction=0.1)

    ax.set_title(filename)
    ax.set_xlabel(f'Samples after {"trigger leading edge" if trig else "event start"} @ {num2si.num2si(freq)}Sa/s')
    ax.set_ylabel('ADC value')

    info = f"""\
    first {len(data)} events
    event length {data.shape[1]} ({data.shape[1] / freq * 1e6:.3g} μs)
    trigger median {np.median(trigger):.0f}"""
    textbox.textbox(ax, info, fontsize='medium', loc='lower right')

    if '.root' in filename:
        table = readroot.info(filename)
        info = [
            f'{col}: {table[col].values[0]}'
            for col in table.columns
            if 'run' in col
            or 'laser' in col
            or 'date' in col
            or 'tension' in col
            or 'trig' in col
        ]
        info = '\n'.join(info)
        textbox.textbox(ax, info, fontsize='x-small', loc='lower left')

    fig.tight_layout()
    return h, fig

if __name__  == '__main__':
    h, fig = main(sys.argv)
    fig.show()
