import os
import glob
import re

import numpy as np
from scipy import stats

import afterpulse
import readwav
import toy
import uncertainties

savedir = 'afterpulse_tile15_ap'
os.makedirs(savedir, exist_ok=True)

wavfiles = list(sorted(glob.glob('darksidehd/LF_TILE15_77K_??V_?VoV_1.wav')))

vovdict = {}

for wavfile in wavfiles:
    
    _, name = os.path.split(wavfile)
    prefix, _ = os.path.splitext(name)
    if '0VoV' in wavfile:
        templfile = 'LF_TILE15_77K_59V_2VoV_1-template.npz'
    else:
        templfile = prefix + '-template.npz'
    simfile = f'{savedir}/{prefix}.npz'

    vov, = re.search(r'(\d)VoV', prefix).groups()
    vov = int(vov)
    vovdict[vov] = dict(simfile=simfile, wavfile=wavfile, templfile=templfile)

    if not os.path.exists(simfile):
    
        data = readwav.readwav(wavfile)
    
        template = toy.Template.load(templfile)
        
        kw = dict(batch=100, pbar=True)
        sim = afterpulse.AfterPulse(data, template, **kw)
    
        print(f'save {simfile}...')
        sim.save(simfile)
    
def apload(vov):
    things = vovdict[vov]
    data = readwav.readwav(things['wavfile'])
    sim = afterpulse.AfterPulse.load(things['simfile'])
    return data, sim

def savef(fig):
    if not hasattr(savef, 'figcount'):
        savef.figcount = 0
    savef.figcount += 1
    path = f'{savedir}/fig{savef.figcount:02d}.png'
    print(f'save {path}...')
    fig.savefig(path)

def upoisson(k):
    return uncertainties.ufloat(k, np.sqrt(max(k, 1)))

def ubinom(k, n):
    p = k / n
    s = np.sqrt(n * p * (1 - p))
    return uncertainties.ufloat(k, s)
    
vovdict[2].update(cut=8, l1pe=5, r1pe=14, plotr=20)
vovdict[4].update(cut=10, l1pe=10, r1pe=27, plotr=35)
vovdict[6].update(cut=10, l1pe=10, r1pe=40, plotr=50)
vovdict[8].update(cut=15, l1pe=20, r1pe=58, plotr=80)
vovdict[9].update(cut=15, l1pe=20, r1pe=70, plotr=90)

for vov, d in vovdict.items():
    if vov == 0:
        continue
    data, sim = apload(vov)
    
    mainsel = 'good&(mainpos>=0)&(length==2048)'
    fig = sim.hist('mainheight', f'{mainsel}&(mainheight<{d["plotr"]})', nbins=200)
    savef(fig)
    
    margin = 100
    ptsel = f'~saturated&(ptpos>={margin})&(ptpos<trigger-{margin})&(length==2048)'
    fig = sim.hist('ptheight', ptsel, 'log')
    savef(fig)
    
    evts = sim.eventswhere(f'{ptsel}&(ptheight>{d["cut"]})')
    for ievt in evts:
        fig = sim.plotevent(data, ievt, zoom='all')
        savef(fig)
    
    sigcount = sim.getexpr(f'count_nonzero({ptsel}&(ptheight>{d["cut"]}))')
    lowercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight<={d["cut"]})&(mainheight>{d["l1pe"]}))')
    uppercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight>{d["cut"]})&(mainheight<{d["r1pe"]}))')
    
    time = sim.getexpr(f'mean(trigger-{2*margin})', ptsel)
    nevents = sim.getexpr(f'count_nonzero({ptsel})')
    totalevt = len(sim.output)

    s = upoisson(sigcount)
    l = upoisson(lowercount)
    u = upoisson(uppercount)
    t = time * 1e-9 * ubinom(nevents, totalevt)

    r = s / t * (l + u) / u
    print(f'rate = {r:P} cps @ {vov}VoV')
    d.update(dcr=r)

