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
import template as _template

savedir = 'afterpulse_tile21'
os.makedirs(savedir, exist_ok=True)

wavfiles = list(sorted(glob.glob('darksidehd/LF_TILE21*.wav')))

vovdict = {}

for wavfile in wavfiles:
    
    _, name = os.path.split(wavfile)
    prefix, _ = os.path.splitext(name)
    templfile = f'templates/{prefix}-template.npz'
    simfile = f'{savedir}/{prefix}.npz'

    vbreak, vbias = re.search(r'(\d\d)V_(\d\d)VoV', prefix).groups()
    vov = (int(vbias) - int(vbreak)) / 2
    
    vd = vovdict.setdefault(vov, {})
    filelist = vd.setdefault('files', [])
    filelist.append(dict(simfile=simfile, wavfile=wavfile, templfile=templfile))

    if not os.path.exists(simfile):
    
        data = readwav.readwav(wavfile)
    
        template = _template.Template.load(templfile)
        
        kw = dict(batch=100, pbar=True, trigger=np.full(len(data), 8969))
        sim = afterpulse.AfterPulse(data, template, **kw)
    
        print(f'save {simfile}...')
        sim.save(simfile)
    
def apload(vov):
    filelist = vovdict[vov]['files']
    datalist = [
        readwav.readwav(files['wavfile'])
        for files in filelist
    ]
    simlist = [
        afterpulse.AfterPulse.load(files['simfile'])
        for files in filelist
    ]
    simcat = afterpulse.AfterPulse.concatenate(simlist)
    return datalist, simcat

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
# length = filter length for afterpulse
# hcut = cut on minorheight
# dcut, dcutr = left/right cut on minorpos - mainpos
# npe = pe of main peak to search afterpulses
vovdict[5.5].update(cut=10, l1pe=10, r1pe=42, length=512, hcut=25, dcut=500, dcutr=5500, npe=1)
vovdict[7.5].update(cut=15, l1pe=20, r1pe=58, length=512, hcut=30, dcut=500, dcutr=5500, npe=1)
vovdict[9.5].update(cut=25, l1pe=20, r1pe=90, length=512, hcut=35, dcut=500, dcutr=5500, npe=1)

for vov in vovdict:
    
    d = vovdict[vov]
    d['analfile'] = f'{savedir}/dict{vov}VoV.pickle'
    globals().update(d) # YES
    if os.path.exists(analfile):
        with open(analfile, 'rb') as file:
            d.update(pickle.load(file))
        continue
    
    datalist, sim = apload(vov)
    
    ilength = np.searchsorted(sim.filtlengths, length)
    assert sim.filtlengths[ilength] == length
    assert npe >= 1
    suffix = f'-{vov}VoV'
    
    mainsel = 'good&(mainpos>=0)&(length==2048)'
    fig = sim.hist('mainheight', f'{mainsel}&(npe>=0)&(npe<=2)', nbins=200)
    ax, = fig.get_axes()
    vspan(ax, cut)
    ax.axvline(l1pe, color='#000')
    ax.axvline(r1pe, color='#000')
    savef(fig, suffix)
    
    lmargin = 100
    rmargin = 500
    ptsel = f'(ptpos>={lmargin})&(ptpos<trigger-{rmargin})'
    fig = sim.hist('ptheight', ptsel, 'log')
    ax, = fig.get_axes()
    vspan(ax, cut)
    savef(fig, suffix)
    
    evts = sim.eventswhere(f'{ptsel}&(ptheight>{cut})')
    for ievt in evts[:3]:
        fig = sim.plotevent(datalist[sim.catindex(ievt)], ievt, zoom='all')
        savef(fig, suffix)
    
    sigcount = sim.getexpr(f'count_nonzero({ptsel}&(ptheight>{cut}))')
    lowercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight<={cut})&(mainheight>{l1pe}))')
    uppercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight>{cut})&(mainheight<{r1pe}))')
    
    time = sim.getexpr(f'mean(trigger-{lmargin}-{rmargin})', ptsel)
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
    vspan(ax, dcut, dcutr)
    savef(fig, suffix)
    
    nevents = sim.getexpr(f'count_nonzero({minornpe})')
    apcond = f'{minornpe}&(minorheight>{hcut})&(minorpos-mainpos>{dcut})&(minorpos-mainpos<{dcutr})'
    apcount = sim.getexpr(f'count_nonzero({apcond})')

    apevts = sim.eventswhere(apcond)
    for ievt in apevts[:3]:
        fig = sim.plotevent(datalist[sim.catindex(ievt)], ievt, ilength)
        savef(fig, suffix)

    fig = sim.hist('minorpos-mainpos', apcond)
    savef(fig, suffix)

    dist = sim.getexpr('minorpos-mainpos', apcond)
    m = np.mean(dist)
    s = np.std(dist)
    l = np.min(dist)
    tau = uncertainties.ufloat((m - l), s / np.sqrt(len(dist)))
    print(f'expon tau = {tau:P} ns')
    d.update(tau=tau)

    factors = [
        stats.expon.cdf(dcutr, scale=s) - stats.expon.cdf(dcut, scale=s)
        for s in [900, 1100]
    ]
    factor = 1 / usamples(factors)

    time = (dcutr - dcut) * 1e-9 * nevents
    bkg = r * time

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

fig, axs = plt.subplots(1, 2, num='afterpulse_tile21', clear=True, sharex=True, figsize=[9, 4.8])

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

vov_fbk = [
    3.993730407523511 ,  
    6.00626959247649  ,  
    7.003134796238245 ,  
    8.018808777429467 ,  
    9.015673981191222 ,  
    10.012539184952978,
]

vov_lf = [
    7.6050156739811925, 
    8.58307210031348  , 
    9.617554858934168 , 
    10.595611285266457, 
    11.630094043887148, 
    12.570532915360502,
] 

dcr_fbk = [
    0.0023342071127444575,
    0.011135620473462282 ,
    0.030621689547075892 ,
    0.11045427521349038  ,
    0.20321620635606957  ,
    0.3304608961193384   ,
]

dcr_lf = [
    0.043606772240838754,
    0.0652463247216516  ,
    0.09766680629121166 ,
    0.15917704842719693 ,
    0.2206923575829639  ,
    0.47616473894714073 ,
]

dcr_factor = 250 / 0.1 # from cps/mm^2 to cps/PDM

axdcr.plot(vov_fbk, dcr_factor * np.array(dcr_fbk), '.-', label='FBK')
axdcr.plot(vov_lf, dcr_factor * np.array(dcr_lf), '.-', label='LF')
axdcr.legend()

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
