import os
import re

import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from numpy.lib import recfunctions

import fdiffrate
import readroot
import read
import figlatex
import textmatrix
import uncertainties
from uncertainties import umath

length = 1000 # ns

########################

directory = 'figthesis/figfakerate'
os.makedirs(directory, exist_ok=True)

def files2specs(files):
    specs = []
    for file in files:
        if '.root' in file:
            specs += [
                f'{file}:{tile}'
                for tile in readroot.tiles()
            ]
        else:
            specs.append(file)
    return specs
    
def specs2npz(specs):
    savefiles = []
    for spec in specs:
        _, speclast = os.path.split(spec)
        savename = f'{directory}/{speclast}.npz'.replace(':', '_')
        savefiles.append(savename)
    return savefiles

def processspec(spec, savename, bound):
    data, trigger, freq, ndigit = read.read(spec)

    nsamp = int(length * 1e-9 * freq)

    usetrigger = True
    if '.root' in spec:
        table = readroot.info(spec)
        kind = table['run type'].values[0]
        if 'baseline' in kind:
            usetrigger = False
    
    if usetrigger:
        end = np.min(trigger) - int(64e-9 * freq)
        data = data[:, :end]

    nevents_noveto, nsamples = data.shape

    output = fdiffrate.fdiffrate(data, nsamp, thrstep=0.1, pbar=True, veto=bound, return_full=True)
    thr, thrcounts, thrcounts_theory, sdev, effnsamples, nevents, errsdev, k2, errk2 = output

    l = np.min(thr)
    u = np.max(thr)
    m = u - l
    x = np.linspace(l - m, u + m, 1000)

    savekw = dict(
        thr = thr,
        thrcounts = thrcounts,
        thr_theory = x,
        thrcounts_theory = thrcounts_theory(x),
        sdev = sdev,
        errsdev = errsdev,
        k2 = k2,
        errk2 = errk2,
        effnsamples = effnsamples,
        nsamples = nsamples,
        nevents = nevents,
        nsamp = nsamp,
        freq = freq,
        veto = bound,
        vetocount = nevents_noveto - nevents,
    )
    
    print(f'save {savename}...')
    np.savez(savename, **savekw)

table = [
    # title, veto, files
    ('LNGS tile 15', 860, [
        'darksidehd/LF_TILE15_77K_55V_0VoV_1.wav',
        'darksidehd/LF_TILE15_77K_59V_2VoV_1.wav',
        'darksidehd/LF_TILE15_77K_63V_4VoV_1.wav',
        'darksidehd/LF_TILE15_77K_67V_6VoV_1.wav',
        'darksidehd/LF_TILE15_77K_71V_8VoV_1.wav',
        'darksidehd/LF_TILE15_77K_73V_9VoV_1.wav',
    ]),
    ('LNGS tiles 53, 57, 59', 750, [
        'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile59_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile53_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile53_77K_66V_7VoV_1.wav',
    ]), 
    ('Proto0 all tiles', 0, [
        'darksidehd/merged_000886.root',
    ]),
]

for title, veto, files in table:
    specs = files2specs(files)
    npz = specs2npz(specs)
    for spec, savefile in zip(specs, npz):
        if not os.path.exists(savefile):
            processspec(spec, savefile, veto)

def pcross(u, sigma, fu=40e6):
    # from Sav18 pp. 97-98
    # fu = upper cutoff frequency
    R0 = 2 * fu / np.sqrt(3)
    return R0 / 2 * np.exp(-1/2 * (u / sigma) ** 2)
    
figkw = dict(
    num='figfakerate',
    clear=True,
    figsize=[8, 12],
    sharex=True,
    sharey=True,
)
fig, axs = plt.subplots(len(table), 1, **figkw)

tabular = []

for ax, (title, veto, files) in zip(axs, table):
    specs = files2specs(files)
    npz = specs2npz(specs)
    
    labeldone = False
    tile53kwstack = [
        dict(linestyle=':', marker=''),
        dict(linestyle='-', marker=''),
    ]
    for ifile, file in enumerate(npz):
        
        print(f'load {file}...')
        with np.load(file) as arch:
            
            thr = arch['thr']
            thrcounts = arch['thrcounts']
            thr_theory = arch['thr_theory']
            thrcounts_theory = arch['thrcounts_theory']
            effnsamples = arch['effnsamples']
            nevents = arch['nevents']
            freq = arch['freq']
            sdev = arch['sdev']
            k2 = arch['k2']
            errk2 = arch['errk2']
        
        filename = os.path.split(file)[1].replace('.npz', '')
        if filename.endswith('.wav'):
            setup = 'LNGS'
            tile, vov = re.search(r'\w{4}(\d\d).*?(\d)VoV', filename).groups()
            vov = f'{float(vov):.2g}'
        else:
            setup = 'Proto0'
            name, tile = re.fullmatch(r'(.*?\.root)_(\d+)', filename).groups()
            vov = '<0'
        tile = int(tile)
        
        time = nevents * effnsamples / freq
        r0 = freq * np.sqrt(-k2) / (2 * np.pi * sdev)
        u = 4
        ratetheory = r0 * np.exp(-1/2 * u ** 2)
        ratefactor = freq / (nevents * effnsamples)
        kw = dict(copy=False, assume_sorted=True)
        ratedata = ratefactor * interpolate.interp1d(thr, thrcounts, **kw)(u * sdev)
        
        tabular.append([
            setup,
            f'{tile}',
            vov,
            f'{time * 1e3:.0f}',
            f'{sdev:.1f}',
            f'{k2:#.2g}',
            f'{errk2 * np.sqrt(time * 1e9):#.2g}',
            f'{r0 * 1e-6:#.2g}',
            f'{ratetheory * 1e-3:#.2g}',
            f'{ratedata * 1e-3:#.2g}',
        ])

        nz = np.flatnonzero(thrcounts)
        start = max(0, nz[0] - 1)
        end = min(len(thr), nz[-1] + 2)
        s = slice(start, end)

        cond = thr_theory >= np.min(thr)
        cond &= thr_theory <= np.max(thr)
        cond &= thrcounts_theory >= np.min(thrcounts[thrcounts > 0])

        kwtheory = dict(color='#f55')
        kwdata = dict(color='black', linestyle='--', marker='.')
        
        if tile == 53:
            label = f'Tile {tile}'
            if vov != '<0':
                label += f' {vov} VoV'
            kwdata.update(label=label)
            kwdata.update(tile53kwstack.pop())
        elif not labeldone:
            kwtheory.update(label='Theory')
            kwdata.update(label='Data')
            labeldone = True
            
        ax.plot(thr_theory[cond] / sdev, ratefactor * thrcounts_theory[cond], **kwtheory)
        ax.plot(thr[s] / sdev, ratefactor * thrcounts[s], **kwdata)
        
        if ifile == 0:
            ax.axhspan(0, ratefactor, color='#ddd')
        
        if ifile == 0:
            thr_sav = np.linspace(2, 8, 300)
            ax.plot(thr_sav, pcross(thr_sav, 1), linewidth=5, color='black', alpha=0.5)
    
    ax.legend(loc='upper center', title=title, title_fontsize='large')

for ax in axs.flat:
    if ax.is_last_row():
        ax.set_xlabel('Threshold [$\\sigma$]')
    ax.set_ylabel('Rate [cps]')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')
    ax.set_xlim(3, 7)
    ax.set_ylim(1, 1e5)

fig.tight_layout()
fig.show()

figlatex.save(fig)

matrix = np.array(tabular)
matrix = recfunctions.unstructured_to_structured(matrix)
matrix = np.sort(matrix)
matrix = textmatrix.TextMatrix(matrix)
print(matrix.latex())
