"""
Plot the 2D histogram of a LNGS wav or Proto0 root.

Can be used as a script or a module.
"""

import sys
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt, colors
import numba
from scipy import stats

import read
import readroot
import textbox
import runsliced
import num2si
import npzload

def main(argv):
    """
    DEPRECATED, use Hist2D
    """
    hist = Hist2D(argv)
    return hist.h, hist.hist2d(), hist.hist1d()

@numba.njit(cache=True)
def _accumhist(hist, q, p, data, trigger, length, start, trig, lower, upper, veto, vetocount):
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
    
class Hist2D(npzload.NPZLoad):
    
    _npzload_unpack_scalars = True
    
    def _parseargv(self, argv):
        parser = argparse.ArgumentParser(prog='hist2d', description='Plot the 2D histogram of a LNGS wav or Proto0 root.')
        parser.add_argument('filename', metavar='path[:channel]', help='File to read. The channel is the tile number or tree branch.')
        parser.add_argument('-m', '--maxevents', type=int, default=1000,        help='Max number of events read from the file, default 1000.')
        parser.add_argument('-l', '--length'   , type=int, default=0,           help='Number of samples read per event, default event length.')
        parser.add_argument('-s', '--start'    , type=int, default=0,           help='Starting sample read in each event.')
        parser.add_argument('-t', '--trigger'  , action='store_true',           help='If specified, the starting sample is relative to the trigger.')
        parser.add_argument('-c', '--cmap'     , default='magma',               help='matplotlib colormap for the histogram.')
        parser.add_argument('-L', '--lower'    , type=int, default=0,           help='Lower sample value bound (inclusive).')
        parser.add_argument('-U', '--upper'    , type=int, default=sys.maxsize, help='Upper sample value bound (exclusive).')
        parser.add_argument('-v', '--veto'     , type=int, default=0,           help='Lower bound on values required to accept an event, default 0.')
    
        args = parser.parse_args(argv, namespace=self)

    def __init__(self, argv=None):
        """
        argv = command line arguments *without* program name
        """
        self._parseargv(argv)

        data, trigger, self.freq, ndigit = read.read(self.filename, self.maxevents)
        self.nevents = data.shape[0]
        self.eventlength = data.shape[1]
    
        if self.trigger and trigger is None:
            raise ValueError('can not use trigger because there\'s no trigger information')
        if trigger is None:
            trigger = np.zeros(len(data), int)
        self.hastrigger = trigger is not None
        self.triggermedian = np.median(trigger)
    
        if self.length == 0:
            self.length = data.shape[1]
        self.upper = min(self.upper, ndigit)
    
        def roundp2(x, p):
            if x >= 2 ** (p + 1):
                q = x // 2 ** p
                effx = int(np.floor(x / q)) * q
            else:
                q = 1
                effx = x
            return effx, q

        self.efflength, self.q = roundp2(self.length, 11)
        self.effndigit, self.p = roundp2(self.upper - self.lower, 10)
    
        self.h = np.zeros((self.efflength // self.q, self.effndigit // self.p), int)
        self.vetocount = np.array(0)

        func = lambda s: _accumhist(self.h, self.q, self.p, data[s], trigger[s], self.efflength, self.start, self.trigger, self.lower, self.upper, self.veto, self.vetocount)
        runsliced.runsliced(func, len(data), 100)

    def hist2d(self, fig=None, **imshowkw):
        
        if fig is None:
            fig, ax = plt.subplots(num='hist2d.Hist2D.hist2d', clear=True, figsize=[10.47, 4.8])
        else:
            ax = fig.subplots()
    
        kw = dict(
            origin='lower',
            cmap=self.cmap,
            norm=colors.LogNorm(),
            aspect='auto',
            extent=(
                -0.5 + self.start, -0.5 + self.start + self.efflength,
                -0.5 + self.lower, -0.5 + self.lower + self.effndigit,
            ),
        )
        kw.update(imshowkw)
        im = ax.imshow(self.h.T, **kw)
        fig.colorbar(im, label=f'Counts per bin ({self.q} sample x {self.p} digit)', fraction=0.1)
    
        _, name = os.path.split(self.filename)
        ax.set_title(name)
        start = "trigger leading edge" if self.trigger else "event start"
        freq = num2si.num2si(self.freq)
        ax.set_xlabel(f'Samples after {start} @ {freq}Sa/s')
        ax.set_ylabel('ADC value')
        
        self._infoboxes(ax)

        fig.tight_layout()
        return fig
    
    def _infoboxes(self, ax):
        info = [
            f'{self.nevents} events',
            f'event length {self.eventlength} ({self.eventlength / self.freq * 1e6:.3g} μs)',
        ]
        if self.hastrigger:
            info.append(f'trigger median {self.triggermedian:.0f}')
        if self.veto > 0:
            info.append(f'veto if any sample < {self.veto} (vetoed {self.vetocount})')
        textbox.textbox(ax, '\n'.join(info), fontsize='medium', loc='lower right', bbox=dict(alpha=0.9))

        if '.root' in self.filename:
            table = readroot.info(self.filename)
            info = [
                f'{col}: {table[col].values[0]}'
                for col in table.columns
                if 'run' in col
                or 'laser' in col
                or 'date' in col
                or 'tension' in col
                or 'trig' in col
                or 'Quality' in col
            ]
            info.insert(0, 'Proto0 metadata')
            textbox.textbox(ax, '\n'.join(info), fontsize='x-small', loc='lower left', bbox=dict(alpha=0.9))

    def hist1d(self, fig=None):

        if fig is None:
            fig, ax = plt.subplots(num='hist2d.Hist2D.hist1d', clear=True, figsize=[10.47, 4.8])
        else:
            ax = fig.subplots()
    
        counts = np.sum(self.h, axis=0)
        bins = -0.5 + np.linspace(self.lower, self.lower + self.effndigit, len(counts) + 1)
    
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

        _, name = os.path.split(self.filename)
        ax.set_title(name)
        ax.set_xlabel(f'Digit')
        ax.set_ylabel(f'Counts per bin ({self.p} digit)')
    
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.grid(True, 'major', linestyle='--')
        ax.grid(True, 'minor', linestyle=':')
        
        self._infoboxes(ax)

        fig.tight_layout()
        return fig

if __name__  == '__main__':
    h, fig, fig2 = main(sys.argv[1:])
    fig2.show()
    fig.show()
