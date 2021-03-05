"""
Plot the 2D histogram of a LNGS wav or Proto0 root.

Can be used as a script or a module.
"""

import sys
import argparse

import numpy as np
from matplotlib import pyplot as plt, colors
import numba
from scipy import stats

import read
import readroot
import textbox
import runsliced
import num2si

def main(argv):
    """
    argv = command line arguments *without* program name
    """
    
    parser = argparse.ArgumentParser(prog='hist2d', description='Plot the 2D histogram of a LNGS wav or Proto0 root.')
    parser.add_argument('filename', metavar='path[:channel]', help='File to read. The channel is the tile number or tree branch.')
    parser.add_argument('-m', '--maxevents', type=int, default=1000, help='Max number of events read from the file, default 1000.')
    parser.add_argument('-l', '--length', type=int, default=0, help='Number of samples read per event, default event length.')
    parser.add_argument('-s', '--start', type=int, default=0, help='Starting sample read in each event.')
    parser.add_argument('-t', '--trigger', action='store_true', help='If specified, the starting sample is relative to the trigger.')
    parser.add_argument('-c', '--cmap', default='magma', help='matplotlib colormap for the histogram.')
    parser.add_argument('-L', '--lower', type=int, default=0, help='Lower sample value bound (inclusive).')
    parser.add_argument('-U', '--upper', type=int, default=sys.maxsize, help='Upper sample value bound (exclusive).')
    parser.add_argument('-v', '--veto', type=int, default=0, help='Lower bound on values required to accept an event, default 0.')
    
    args = parser.parse_args(argv)
        
    filename = args.filename
    maxevents = args.maxevents
    length = args.length
    start = args.start
    trig = args.trigger
    cmap = args.cmap
    lower = args.lower
    upper = args.upper
    veto = args.veto

    data, trigger, freq, ndigit = read.read(filename, maxevents)
    
    if length == 0:
        length = data.shape[1]
    upper = min(upper, ndigit)
    
    def roundp2(x, p):
        if x >= 2 ** (p + 1):
            q = x // 2 ** p
            effx = int(np.floor(x / q)) * q
        else:
            q = 1
            effx = x
        return effx, q

    efflength, q = roundp2(length, 11)
    effndigit, p = roundp2(upper - lower, 10)
    
    h = np.zeros((efflength // q, effndigit // p), int)
    vetocount = np.array(0)

    @numba.njit(cache=True)
    def accumhist(hist, q, p, data, trigger, length, start, trig, lower, upper, veto, vetocount):
        for ievent, signal in enumerate(data):
            
            begin = start
            if trig:
                begin += trigger[ievent]
            end = begin + length
            
            begin = max(0, begin)
            end = min(end, len(signal))
            
            stop = False
            for isample in range(begin, end):
                if signal[isample] < veto:
                    stop = True
            if stop:
                vetocount += 1
                continue
            
            for isample in range(begin, end):
                sample = signal[isample]
                if lower <= sample < upper:
                    bin0 = (isample - begin) // q
                    bin1 = (sample - lower) // p
                    hist[bin0, bin1] += 1

    runsliced.runsliced(lambda s: accumhist(h, q, p, data[s], trigger[s], efflength, start, trig, lower, upper, veto, vetocount), len(data), 100)

    fig, ax = plt.subplots(num='hist2d', clear=True, figsize=[10.47, 4.8])
    
    kw = dict(
        origin='lower',
        cmap=cmap,
        norm=colors.LogNorm(),
        aspect='auto',
        extent=(-0.5+start, -0.5+start+efflength, -0.5+lower, -0.5+lower+effndigit),
    )
    im = ax.imshow(h.T, **kw)
    fig.colorbar(im, label=f'Counts per bin ({q} sample x {p} digit)', fraction=0.1)

    ax.set_title(filename)
    ax.set_xlabel(f'Samples after {"trigger leading edge" if trig else "event start"} @ {num2si.num2si(freq)}Sa/s')
    ax.set_ylabel('ADC value')

    info = f"""\
first {len(data)} events
event length {data.shape[1]} ({data.shape[1] / freq * 1e6:.3g} μs)
trigger median {np.median(trigger):.0f}
veto if any sample < {veto} (vetoed {vetocount.item()})"""
    textbox.textbox(ax, info, fontsize='medium', loc='lower right', bbox=dict(alpha=0.9))

    if '.root' in filename:
        table = readroot.info(filename)
        infolist = [
            f'{col}: {table[col].values[0]}'
            for col in table.columns
            if 'run' in col
            or 'laser' in col
            or 'date' in col
            or 'tension' in col
            or 'trig' in col
            or 'Quality' in col
        ]
        inforoot = '\n'.join(infolist)
        textbox.textbox(ax, inforoot, fontsize='x-small', loc='lower left', bbox=dict(alpha=0.9))

    fig.tight_layout()

    fig2, ax = plt.subplots(num='hist2d-2', clear=True, figsize=[10.47, 4.8])
    
    counts = np.sum(h, axis=0)
    bins = -0.5 + np.linspace(lower, lower + effndigit, len(counts) + 1)
    
    val = bins[:-1] + 1/2 * np.diff(bins)
    w = counts / np.sum(counts)
    mu = np.sum(w * val)
    sigma = np.sqrt(np.sum(w * (val - mu) ** 2))
    
    nz = np.flatnonzero(counts)
    x = np.linspace(bins[nz[0]], bins[nz[-1] + 1], 1000)
    y = stats.norm.pdf(x, mu, sigma) * np.sum(counts * np.diff(bins))
    
    cond = (y >= np.min(counts[counts != 0])) & (y <= np.max(counts))
    y[~cond] = np.nan

    ax.plot(x, y, color='#f55')
    ax.plot(np.pad(bins, (1, 0), 'edge'), np.pad(counts, 1), drawstyle='steps-post', color='#000')

    ax.set_title(filename)
    ax.set_xlabel(f'Digit')
    ax.set_ylabel(f'Counts per bin ({p} digit)')
    
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

    textbox.textbox(ax, info, fontsize='medium', loc='upper right', bbox=dict(alpha=0.9))
    if '.root' in filename:
        textbox.textbox(ax, inforoot, fontsize='x-small', loc='upper left', bbox=dict(alpha=0.9))

    fig2.tight_layout()

    return h, fig, fig2

if __name__  == '__main__':
    h, fig, fig2 = main(sys.argv[1:])
    fig2.show()
    fig.show()
