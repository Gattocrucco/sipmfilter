import os

import tqdm
import numpy as np
from matplotlib import pyplot as plt

import fdiffrate
import readroot
import read
import figlatex
import textbox

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
    data, trigger, freq, ndigit = read.read(spec, mmap=False)

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

    cond = np.all(data >= bound, axis=-1)
    data = data[cond]

    nevents, nsamples = data.shape

    output = fdiffrate.fdiffrate(data, nsamp, thrstep=0.1)
    thr, thrcounts, thrcounts_theory, sdev, effnsamples = output

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
        effnsamples = effnsamples,
        nsamples = nsamples,
        nevents = nevents,
        nsamp = nsamp,
        freq = freq,
    )
    
    print(f'save {savename}...')
    np.savez(savename, **savekw)

table = [
    # title, veto, files
    ('Proto0 all tiles', 0, [
        'darksidehd/merged_000886.root',
    ]),
    ('LNGS tile 15', 860, [
        'darksidehd/LF_TILE15_77K_55V_0VoV_1.wav',
        'darksidehd/LF_TILE15_77K_59V_2VoV_1.wav',
        'darksidehd/LF_TILE15_77K_63V_4VoV_1.wav',
        'darksidehd/LF_TILE15_77K_67V_6VoV_1.wav',
        'darksidehd/LF_TILE15_77K_71V_8VoV_1.wav',
        'darksidehd/LF_TILE15_77K_73V_9VoV_1.wav',
    ]),
    ('LNGS tiles 53, 57, 59', 750, [
        'darksidehd/nuvhd_lf_3x_tile53_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile53_77K_66V_7VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile59_77K_64V_6VoV_1.wav',
    ]), 
]

for title, veto, files in table:
    specs = files2specs(files)
    npz = specs2npz(specs)
    for spec, savefile in zip(specs, npz):
        if not os.path.exists(npz):
            processspec(spec, savefile, veto)

figkw = dict(
    num='figfakerate',
    clear=True,
    figsize=[8, 12],
    sharex=True,
    sharey=True,
)
fig, axs = plt.subplots(len(table), 1, **figkw)

for ax, (title, veto, files) in zip(axs, table):
    specs = files2specs(files)
    npz = specs2npz(specs)
    dolegend = False
    for ifile, file in enumerate(npz):
        
        with np.load(file) as arch:
            
            thr = arch['thr']
            thrcounts = arch['thrcounts']
            thr_theory = arch['thr_theory']
            thrcounts_theory = arch['thrcounts_theory']
            effnsamples = arch['effnsamples']
            nevents = arch['nevents']
            freq = arch['freq']
            sdev = arch['sdev']

            nz = np.flatnonzero(thrcounts)
            start = max(0, nz[0] - 1)
            end = min(len(thr), nz[-1] + 2)
            s = slice(start, end)

            ratefactor = freq / (nevents * effnsamples)
            cond = thr_theory >= np.min(thr)
            cond &= thr_theory <= np.max(thr)
            cond &= thrcounts_theory >= np.min(thrcounts[thrcounts > 0])

            kw = dict(color='#f55')
            ax.plot(thr_theory[cond] / sdev, ratefactor * thrcounts_theory[cond], **kw)
            
            kw = dict(color='black', linestyle='--', marker='.')
            label = os.path.split(file)[1].replace('.npz', '')
            if label.endswith('_53') or 'tile53' in label:
                kw.update(label=label)
                dolegend = True
            ax.plot(thr[s] / sdev, ratefactor * thrcounts[s], **kw)
    
    if dolegend:
        ax.legend()
    textbox.textbox(ax, title, fontsize='medium', loc='upper center')

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

fig.savefig('../thesis/figures/' + fig.canvas.get_window_title() + '.pdf')
print(figlatex.figlatex(fig))
