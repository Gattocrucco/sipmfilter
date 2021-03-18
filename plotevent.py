import argparse

import numpy as np
from matplotlib import pyplot as plt

import toy
import read
import num2si
import textbox
import template as _template

parser = argparse.ArgumentParser(description='plot an event from an LNGS wav or Proto0 root')

parser.add_argument('filespec', metavar='file[:channel]', help='file to read, the channel is the tile or tree branch (for Proto0 files)')
parser.add_argument('-e', '--event', type=int, default=0, help='event to plot, default 0')
parser.add_argument('-l', '--length', type=int, default=100, help='filter length in samples, default 100')

args = parser.parse_args()

signal, freq, ndigit = read.read(args.filespec, firstevent=args.event, maxevents=1, return_trigger=False)

template = _template.Template.load('templates/nuvhd_lf_3x_tile57_77K_64V_6VoV_1-template.npz')
timebase = int(1e9 / freq)
print(f'timebase = {timebase}')
templ, offset = template.matched_filter_template(args.length, timebase=timebase)

filt = toy.Filter(signal, signal[0, 0])
fsignal = filt.all(templ)[:, 0]

fig, ax = plt.subplots(num='plotevent', clear=True)

ax.plot(fsignal[0], color='#f55', linewidth=1, label=toy.Filter.name(0))
for i in range(3):
    ax.plot(fsignal[i + 1], color='black', linestyle=[':', '--', '-'][i], label=toy.Filter.name(i + 1))

textbox.textbox(ax, f'filter length = {args.length} Sa ({args.length * timebase} ns)', loc='lower right', fontsize='small')

ax.legend(loc='best')

ax.set_title(f'{args.filespec}, event {args.event}')
ax.set_xlabel(f'Sample number @ {num2si.num2si(freq)}Sa/s')
ax.set_ylabel('ADC scale')

ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()
