import os
import glob
import re

import numpy as np
from scipy import stats

import afterpulse
import readwav
import toy
import uncertainties

savedir = 'afterpulse_tile15'
os.makedirs(savedir, exist_ok=True)

wavfiles = list(sorted(glob.glob('darksidehd/LF_TILE15_77K_??V_?VoV_1.wav')))

vovdict = {}

for wavfile in wavfiles:
    
    path, name = os.path.split(wavfile)
    prefix = name.replace('.wav', '')
    if '0VoV' in wavfile:
        templfile = 'LF_TILE15_77K_59V_2VoV_1-template.npz'
    else:
        templfile = prefix + '-template.npz'
    simfile = f'{savedir}/{prefix}.npz'

    vov, = re.search(r'(\d)VoV', wavfile).groups()
    vov = int(vov)
    vovdict[vov] = dict(simfile=simfile, wavfile=wavfile, templfile=templfile)

    if not os.path.exists(simfile):
    
        data = readwav.readwav(wavfile)
    
        template = toy.Template.load(templfile)
        
        filtlength = 2048
        kw = dict(batch=100, pbar=True, filtlengths=filtlength, ptlength=filtlength)
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

data, sim = apload(2)
fig = sim.hist('mainheight', 'good&(mainheight<20)', nbins=200)
savef(fig)
fig = sim.hist('ptheight', '~saturated&(ptpos>=100)&(ptpos<trigger-100)', 'log')
savef(fig)
cut = 8
sigcount = sim.getexpr(f'count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut}))')
lowercount = sim.getexpr(f'count_nonzero(good&(mainheight<={cut})&(mainheight>5))')
uppercount = sim.getexpr(f'count_nonzero(good&(mainheight>{cut})&(mainheight<14))')
time = sim.getexpr('mean(trigger-200)', '~saturated&(ptpos>=100)&(ptpos<trigger-100)')
nevents = sim.getexpr('count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100))')
totalevt = len(sim.output)

s = upoisson(sigcount)
l = upoisson(lowercount)
u = upoisson(uppercount)
t = time * 1e-9 * ubinom(nevents, totalevt)

r = s / t * (l + u) / u
print(f'rate = {r:P} cps @ 2VoV')

data, sim = apload(4)
fig = sim.hist('mainheight', 'good&(mainpos>=0)&(mainheight<35)', nbins=200)
savef(fig)
fig = sim.hist('ptheight', '~saturated&(ptpos>=100)&(ptpos<trigger-100)', 'log')
savef(fig)
cut = 10
sigcount = sim.getexpr(f'count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut}))')
lowercount = sim.getexpr(f'count_nonzero(good&(mainheight<={cut})&(mainheight>10))')
uppercount = sim.getexpr(f'count_nonzero(good&(mainheight>{cut})&(mainheight<27))')
time = sim.getexpr('mean(trigger-200)', '~saturated&(ptpos>=100)&(ptpos<trigger-100)')
nevents = sim.getexpr('count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100))')
totalevt = len(sim.output)

s = upoisson(sigcount)
l = upoisson(lowercount)
u = upoisson(uppercount)
t = time * 1e-9 * ubinom(nevents, totalevt)

r = s / t * (l + u) / u
print(f'rate = {r:P} cps @ 4VoV')

data, sim = apload(6)
fig = sim.hist('mainheight', 'good&(mainpos>=0)&(mainheight<50)', nbins=200)
savef(fig)
fig = sim.hist('ptheight', '~saturated&(ptpos>=100)&(ptpos<trigger-100)', 'log')
savef(fig)
cut = 10
sigcount = sim.getexpr(f'count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut}))')
lowercount = sim.getexpr(f'count_nonzero(good&(mainheight<={cut})&(mainheight>10))')
uppercount = sim.getexpr(f'count_nonzero(good&(mainheight>{cut})&(mainheight<40))')
time = sim.getexpr('mean(trigger-200)', '~saturated&(ptpos>=100)&(ptpos<trigger-100)')
nevents = sim.getexpr('count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100))')
totalevt = len(sim.output)

s = upoisson(sigcount)
l = upoisson(lowercount)
u = upoisson(uppercount)
t = time * 1e-9 * ubinom(nevents, totalevt)

r = s / t * (l + u) / u
print(f'rate = {r:P} cps @ 6VoV')

data, sim = apload(8)
fig = sim.hist('mainheight', 'good&(mainpos>=0)&(mainheight<80)', nbins=200)
savef(fig)
fig = sim.hist('ptheight', '~saturated&(ptpos>=100)&(ptpos<trigger-100)', 'log')
savef(fig)
cut = 15
evts = sim.eventswhere(f'~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut})')
for ievt in evts:
    fig = sim.plotevent(data, ievt, zoom='all')
    savef(fig)
sigcount = sim.getexpr(f'count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut}))')
lowercount = sim.getexpr(f'count_nonzero(good&(mainheight<={cut})&(mainheight>20))')
uppercount = sim.getexpr(f'count_nonzero(good&(mainheight>{cut})&(mainheight<58))')
time = sim.getexpr('mean(trigger-200)', '~saturated&(ptpos>=100)&(ptpos<trigger-100)')
nevents = sim.getexpr('count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100))')
totalevt = len(sim.output)

s = upoisson(sigcount)
l = upoisson(lowercount)
u = upoisson(uppercount)
t = time * 1e-9 * ubinom(nevents, totalevt)

r = s / t * (l + u) / u
print(f'rate = {r:P} cps @ 8VoV')

data, sim = apload(9)
fig = sim.hist('mainheight', 'good&(mainpos>=0)&(mainheight<90)', nbins=200)
savef(fig)
fig = sim.hist('ptheight', '~saturated&(ptpos>=100)&(ptpos<trigger-100)', 'log')
savef(fig)
cut = 15
evts = sim.eventswhere(f'~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut})')
for ievt in evts:
    fig = sim.plotevent(data, ievt, zoom='all')
    savef(fig)
sigcount = sim.getexpr(f'count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100)&(ptheight>{cut}))')
lowercount = sim.getexpr(f'count_nonzero(good&(mainheight<={cut})&(mainheight>20))')
uppercount = sim.getexpr(f'count_nonzero(good&(mainheight>{cut})&(mainheight<70))')
time = sim.getexpr('mean(trigger-200)', '~saturated&(ptpos>=100)&(ptpos<trigger-100)')
nevents = sim.getexpr('count_nonzero(~saturated&(ptpos>=100)&(ptpos<trigger-100))')
totalevt = len(sim.output)

s = upoisson(sigcount)
l = upoisson(lowercount)
u = upoisson(uppercount)
t = time * 1e-9 * ubinom(nevents, totalevt)

r = s / t * (l + u) / u
print(f'rate = {r:P} cps @ 9VoV')
