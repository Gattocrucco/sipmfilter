import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

import read
import num2si
import runsliced

parser = argparse.ArgumentParser(prog='spec', description='Frequency spectrum of a LNGS wav or Proto0 root.')
parser.add_argument('filespec', metavar='path[:channel]', help='File to read. The channel is the tile number or tree branch.')
parser.add_argument('-m', '--maxevents', type=int  , default=0       , help='Max number of events read from the file, default all.')
parser.add_argument('-l', '--length'   , type=int  , default=0       , help='Number of samples read per event, default all.')
parser.add_argument('-s', '--start'    , type=int  , default=0       , help='Starting sample read in each event, default first.')
parser.add_argument('-v', '--veto'     , type=int  , default=0       , help='Lower bound on values required to accept an event, default 0.')
parser.add_argument('-w', '--window'   ,             default='boxcar', help='Windowing function, default none.')
parser.add_argument('-u', '--unit'     , type=float, default=1e6     , help='Unit of frequency, default 1e6 (MHz).')

args = parser.parse_args()

maxevents = args.maxevents if args.maxevents > 0 else None
data, trigger, freq, ndigit = read.read(args.filespec, maxevents=maxevents)

data = data[:, args.start:]
if args.length > 0:
    data = data[:, :args.length]
if args.veto > 0:
    cond = np.all(data >= args.veto, axis=-1)
    data = data[cond]

nevents, nsamples = data.shape
f = np.empty(1 + nsamples // 2)
s = np.empty((nevents, 1 + nsamples // 2))
def batch(sl):
    f[:], s[sl] = signal.periodogram(data[sl], freq, args.window)
runsliced.runsliced(batch, len(data), 100)
s = np.median(s, axis=0)

fig, ax = plt.subplots(num='spec', clear=True)

ax.plot(f / args.unit, s * args.unit, '-k')

_, filename = os.path.split(args.filespec)
ax.set_title(filename)
value, unit = num2si.num2si(args.unit).split()
if value != '1':
    unit = f'{value} {unit}'
ax.set_xlabel(f'Frequency [{unit}Hz]')
ax.set_ylabel(f'Power spectral density [{unit}Hz$^{{-1}}$]')

ax.minorticks_on()
ax.grid(which='major', linestyle='--')
ax.grid(which='minor', linestyle=':')

fig.tight_layout()
fig.show()
