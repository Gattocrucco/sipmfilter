import os
import glob
import re
import pickle

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import uncertainties
from uncertainties import unumpy

import afterpulse
import readwav
import toy

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

def savef(fig, suffix=''):
    if not hasattr(savef, 'figcount'):
        savef.figcount = 0
    savef.figcount += 1
    path = f'{savedir}/fig{savef.figcount:02d}{suffix}.png'
    print(f'save {path}...')
    fig.savefig(path)

def upoisson(k):
    return uncertainties.ufloat(k, np.sqrt(max(k, 1)))

def ubinom(k, n):
    p = k / n
    s = np.sqrt(n * p * (1 - p))
    return uncertainties.ufloat(k, s)

def usamples(x):
    return uncertainties.ufloat(np.mean(x), np.std(x, ddof=1))

def hspan(ax, y0, y1=None):
    ylim = ax.get_ylim()
    if y0 is None:
        y0 = ylim[0]
    if y1 is None:
        y1 = ylim[1]
    ax.axhspan(y0, y1, color='#0002')
    ax.set_ylim(ylim)
    
def vspan(ax, x0, x1=None):
    xlim = ax.get_xlim()
    if x0 is None:
        x0 = xlim[0]
    if x1 is None:
        x1 = xlim[1]
    ax.axvspan(x0, x1, color='#0002')
    ax.set_xlim(xlim)

# parameters:
# cut = cut on ptheight to measure dcr
# l1pe, r1pe = boundaries of 1 pe height distribution with filter length 2048
# plotr = right bound to plot distribution of main height
# length = filter length for afterpulse
# hcut = cut on minorheight
# dcut = cut on minorpos - mainpos
# npe = pe of main peak to search afterpulses
vovdict[2].update(cut= 8, l1pe= 5, r1pe=14, plotr=20, length=2048, hcut=10, dcut=500, npe=1)
vovdict[4].update(cut=10, l1pe=10, r1pe=27, plotr=35, length= 512, hcut=16, dcut=500, npe=1)
vovdict[6].update(cut=10, l1pe=10, r1pe=40, plotr=50, length= 512, hcut=25, dcut=500, npe=1)
vovdict[8].update(cut=15, l1pe=20, r1pe=58, plotr=80, length= 512, hcut=30, dcut=500, npe=1)
vovdict[9].update(cut=15, l1pe=20, r1pe=70, plotr=90, length= 512, hcut=35, dcut=500, npe=1)

for vov in vovdict:
    
    if vov == 0:
        continue
    
    d = vovdict[vov]
    d['analfile'] = f'{savedir}/dict{vov}VoV.pickle'
    globals().update(d) # YES
    if os.path.exists(analfile):
        with open(analfile, 'rb') as file:
            d.update(pickle.load(file))
        continue
    
    data, sim = apload(vov)
    
    ilength = np.searchsorted(sim.filtlengths, length)
    assert sim.filtlengths[ilength] == length
    assert npe >= 1
    suffix = f'-{vov}VoV'
    
    mainsel = 'good&(mainpos>=0)&(length==2048)'
    fig = sim.hist('mainheight', f'{mainsel}&(mainheight<{plotr})', nbins=200)
    ax, = fig.get_axes()
    vspan(ax, cut)
    savef(fig, suffix)
    
    margin = 100
    ptsel = f'~saturated&(ptpos>={margin})&(ptpos<trigger-{margin})&(length==2048)'
    fig = sim.hist('ptheight', ptsel, 'log')
    ax, = fig.get_axes()
    vspan(ax, cut)
    savef(fig, suffix)
    
    evts = sim.eventswhere(f'{ptsel}&(ptheight>{cut})')
    for ievt in evts[:3]:
        fig = sim.plotevent(data, ievt, zoom='all')
        savef(fig, suffix)
    
    sigcount = sim.getexpr(f'count_nonzero({ptsel}&(ptheight>{cut}))')
    lowercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight<={cut})&(mainheight>{l1pe}))')
    uppercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight>{cut})&(mainheight<{r1pe}))')
    
    time = sim.getexpr(f'mean(trigger-{2*margin})', ptsel)
    nevents = sim.getexpr(f'count_nonzero({ptsel})')
    totalevt = len(sim.output)

    s = upoisson(sigcount)
    l = upoisson(lowercount)
    u = upoisson(uppercount)
    t = time * 1e-9 * ubinom(nevents, totalevt)
    
    f = (l + u) / u
    r = f * s / t
    
    print(f'pre-trigger count = {sigcount} / {nevents}')
    print(f'total time = {t:P} s')
    print(f'correction factor = {f:P}')
    print(f'dcr = {r:P} cps @ {vov}VoV')
    
    d.update(dcr=r, dcrcount=s, dcrtime=t, dcrfactor=f)

    mainsel = f'good&(mainpos>=0)&(length=={length})'
    fig = sim.hist('mainheight', f'{mainsel}&(npe>={npe-1})&(npe<={npe+1})', nbins=200)
    savef(fig, suffix)

    hnpe = sim.getexpr('median(mainheight)', f'{mainsel}&(npe=={npe})')
    print(f'1 pe median height with {length} ns = {hnpe:.3g}')

    minorsel = f'{mainsel}&(minorpos>=0)'
    minornpe = f'{minorsel}&(npe=={npe})'
    fig = sim.scatter('minorpos-mainpos', 'minorheight', minornpe)
    ax, = fig.get_axes()
    hspan(ax, hcut)
    vspan(ax, dcut)
    savef(fig, suffix)

    nevents = sim.getexpr(f'count_nonzero({minornpe})')
    apcond = f'{minornpe}&(minorheight>{hcut})&(minorpos-mainpos>{dcut})'
    apcount = sim.getexpr(f'count_nonzero({apcond})')

    apevts = sim.eventswhere(apcond)
    for ievt in apevts[:3]:
        fig = sim.plotevent(data, ievt, ilength)
        savef(fig, suffix)

    fig = sim.hist('minorpos-mainpos', apcond)
    savef(fig, suffix)

    dist = sim.getexpr('minorpos-mainpos', apcond)
    m = np.mean(dist)
    s = np.std(dist)
    l = np.min(dist)
    tau = uncertainties.ufloat((m - l), s / np.sqrt(len(dist)))
    print(f'expon tau = {tau:P} ns')

    dend = sim.getexpr('max(minorpos-mainpos)', minornpe)
    factors = [
        stats.expon.cdf(dend, scale=s) - stats.expon.cdf(dcut, scale=s)
        for s in [400, 600]
    ]
    factor = 1 / usamples(factors)

    time = (dend - dcut) * 1e-9 * nevents
    bkg = vovdict[2]['dcr'] * time

    count = upoisson(apcount)
    ccount = (count - bkg) * factor
    p = ccount / nevents
    
    print(f'correction factor = {factor:P} (assuming tau a priori)')
    print(f'expected background = {bkg:P} counts in {time:.3g} s')
    print(f'ap count = {apcount} / {nevents}')
    print(f'corrected ap count = {ccount:P}')
    print(f'afterpulse probability = {p:P} @ {vov}VoV')
    
    d.update(ap=p, apcount=count, apnevents=nevents, apbkg=bkg, aptime=time, apfactor=factor)

    with open(analfile, 'wb') as file:
        print(f'save {analfile}...')
        pickle.dump(d, file)

def uerrorbar(ax, x, y, **kw):
    ym = unumpy.nominal_values(y)
    ys = unumpy.std_devs(y)
    kwargs = dict(yerr=ys)
    kwargs.update(kw)
    return ax.errorbar(x, ym, **kwargs)

vovdict.pop(0)

fig, axs = plt.subplots(1, 2, num='afterpulse_tile15_ap', clear=True, sharex=True, figsize=[9, 4.8])

axdcr, axap = axs

for ax in axs:
    if ax.is_last_row():
        ax.set_xlabel('VoV')

axdcr.set_title('DCR')
axdcr.set_ylabel('Pre-trigger rate [cps]')

axap.set_title('Afterpulse')
axap.set_ylabel('Prob. of $\\geq$1 ap after 1 pe signal [%]')

vov = list(vovdict)
dcr = np.array([d['dcr'] for d in vovdict.values()])
ap = np.array([d['ap'] for d in vovdict.values()])

kw = dict(capsize=4, linestyle='', marker='.', color='#000')
uerrorbar(axdcr, vov, dcr, **kw)
uerrorbar(axap, vov, ap * 100, **kw)

for ax in axs:
    l, u = ax.get_ylim()
    ax.set_ylim(0, u)
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()
