import os
import sys

import tqdm
import numpy as np

import fdiffrate
import readroot
import read

length = 1000 # ns
wav_bound = 700 # ADC
root_bound = 15000 # ADC

########################

files = sys.argv[1:]

directory = 'fdrall'
os.makedirs(directory, exist_ok=True)

specs = []
for file in files:
    if '.root' in file:
        specs += [
            f'{file}:{tile}'
            for tile in readroot.tiles()
        ]
    else:
        specs.append(file)
   
for spec in tqdm.tqdm(specs):
    data, trigger, freq, ndigit = read.read(spec, mmap=False, quiet=True)
    
    nsamp = int(length * 1e-9 * freq)
    
    usetrigger = True
    if '.root' in spec:
        table = readroot.info(spec)
        kind = table['run type'].values[0]
        if 'baseline' in kind:
            usetrigger = False
        
    if usetrigger:
        end = np.min(trigger)
        data = data[:, :end]
    
    bound = root_bound if '.root' in spec else wav_bound
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
    
    savename = f'{directory}/{spec}.npz'.replace(':', '_')
    np.savez(savename, **savekw)
