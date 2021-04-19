import os

import tqdm
import numpy as np

import fdiffrate
import readroot
import read

length = 1000 # ns
bound = 860

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

files_list = [
    [
        'darksidehd/merged_000886.root',
    ], [
        'darksidehd/LF_TILE15_77K_55V_0VoV_1.wav',
        'darksidehd/LF_TILE15_77K_59V_2VoV_1.wav',
        'darksidehd/LF_TILE15_77K_63V_4VoV_1.wav',
        'darksidehd/LF_TILE15_77K_67V_6VoV_1.wav',
        'darksidehd/LF_TILE15_77K_71V_8VoV_1.wav',
        'darksidehd/LF_TILE15_77K_73V_9VoV_1.wav',
    ], [
        'darksidehd/nuvhd_lf_3x_tile53_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile53_77K_66V_7VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav',
        'darksidehd/nuvhd_lf_3x_tile59_77K_64V_6VoV_1.wav',
    ], 
]

for files in files_list:
    specs = files2specs(files)
    npz = specs2npz(specs)
    