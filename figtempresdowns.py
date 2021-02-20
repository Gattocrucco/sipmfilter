import os
import collections

import numpy as np
from matplotlib import pyplot as plt, gridspec
from scipy import interpolate

import figlatex
import toy
import textmatrix
import num2si

prefix = 'nuvhd_lf_3x_tile57_77K_64V_6VoV_1'
noisefile = 'merged_000886-adc_W201_Ch00.npz'
tau_125MSa = 256
snr_125MSa = np.logspace(np.log10(1), np.log10(6), 50)

###########################

timebase = dict(
    proto0 = [   8, 16, 32],
    lngs   = [1, 8, 16, 32],
    white  = [1, 8, 16, 32],
)
simprefix = 'figtempresdowns'
simfile = lambda n, tb: f'{simprefix}-{n}-{tb}.npz'
nicenames = dict(proto0='Proto0', lngs='LNGS', white='White')

if not all(os.path.exists(simfile(n, tb)) for n in timebase for tb in timebase[n]):
    
    template = toy.Template.load(f'{prefix}-template.npz')

    for n, timebases in timebase.items():
        
        timebases = np.sort(timebases)
        noises = {}
        noise_ratio = {}
        for itb, tb in enumerate(timebases):
            
            if n == 'proto0':
                noise = toy.DataCycleNoise(maxcycles=2, chunk_skip=1000, timebase=tb)
                noise.load(noisefile)
            elif n == 'lngs':
                noise = toy.DataCycleNoise(maxcycles=2, timebase=tb)
                noise.load(f'{prefix}-noise.npz')
            elif n == 'white':
                noise = toy.WhiteNoise(timebase=tb)
            noises[tb] = noise
            
            if itb == 0:
                basenoise = noise
                basetb = tb
            noise_ratio[tb] = np.std(toy.downsample(basenoise.generate(20, 5000), tb // basetb), axis=None)
            
        for tb in timebases:
            
            filename = simfile(n, tb)
            if os.path.exists(filename):
                continue
            
            nr = noise_ratio[tb] / noise_ratio[8]
            snr = snr_125MSa / nr
            sim = toy.Toy(template, [tau_125MSa * 8 // tb], snr, noises[tb], timebase=tb)
            sim.noise_ratio = nr
            sim.run(1000, pbar=10, seed=202102191411)
            print(f'save {filename}')
            sim.save(filename)

sim = {}
for n, timebases in timebase.items():
    sim[n] = {}
    for tb in timebases:
        filename = simfile(n, tb)
        print(f'load {filename}')
        sim[n][tb] = toy.Toy.load(filename)

def maketable():
    all_timebase = list(np.unique(np.concatenate(list(timebase.values()))))
    table = []
    def fmttb(tb):
        return '\\SI{' + num2si.num2si(1e9 / tb, format='%.3g', space='}{') + 'Sa/s}'
    table.append([fmttb(tb) for tb in all_timebase])
    for n in sim:
        row = collections.defaultdict(str)
        for tb in sim[n]:
            t = sim[n][tb]
            ratio = t.snrratio()[3, 0]
            ratio = ratio / t.noise_ratio
            row[tb] = f'{ratio:.2g}'
        table.append([nicenames[n]] + [row[tb] for tb in all_timebase])
    matrix = textmatrix.TextMatrix(table, fill_side='left')
    print(matrix.latex())
maketable()

def snrhelp(noise=None):
    fig, axs = plt.subplots(2, 1, num='snrhelp', clear=True)
    for n in sim:
        if noise is not None and n != noise:
            continue
        FSFNSNR = []
        for tb in sim[n]:
            t = sim[n][tb]
            FSFNSNR.append(t._snr_helper() + (t.noise_ratio,))
        FSFNSNR = np.array(FSFNSNR)
        for i, label in enumerate(['FS', 'FN', 'S', 'N', 'NR']):
            if 'S' in label:
                ax = axs[0]
            else:
                ax = axs[1]
            ax.plot(timebase[n], FSFNSNR[:, i], label=label + ' ' + n, marker='.')
    axs[0].plot(timebase['lngs'], [template.max(tb) for tb in timebase['lngs']])
    for ax in axs:
        ax.legend(loc='best')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        
    fig.show()

fig = plt.figure(num='figtempresdowns', clear=True, figsize=[9.17, 5.06])
gs = gridspec.GridSpec(5, 2)
axs = {}
axs['proto0'] = fig.add_subplot(gs[:, 1])
axs['lngs'] = fig.add_subplot(gs[:2, 0], sharex=axs['proto0'])
axs['white'] = fig.add_subplot(gs[-3:, 0], sharex=axs['proto0'])

plotkw = {
    1: dict(color='#f55'),
    8: dict(color='#000'),
    16: dict(color='#000', linestyle='--'),
    32: dict(color='#000', linestyle=':'),
}

for n in sim:
    ax = axs[n]
    timebases = timebase[n]
    for itb, tb in enumerate(timebases):
        
        t = sim[n][tb]
        r = t.templocres(sampleunit=False)[3, 0]
        nr = t.noise_ratio
        label = t.sampling_str()
        if tb != 8:
            label += f' (SNR $\\times$ {nr:.2g})'
        line, = ax.plot(t.snr * nr, r, label=label, **plotkw[tb])
        
    ax.axhspan(0, 8, color='#ddd')
    
    ax.legend(loc='upper right', title=f'{nicenames[n]} noise')
    
    if ax.is_last_row():
        ax.set_xlabel('SNR (before filtering) @ 125 MSa/s')
    if ax.is_first_col():
        ax.set_ylabel('Temporal resolution [ns]')
    
    ax.set_ylim(0, ax.get_ylim()[1])

    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

axs['lngs'].tick_params(labelbottom=False)

fig.tight_layout()
fig.show()

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
