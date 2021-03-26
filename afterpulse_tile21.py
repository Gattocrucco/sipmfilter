import os
import glob
import re
import pickle

import numpy as np
from scipy import stats, special
from matplotlib import pyplot as plt
import lsqfit
import gvar

import afterpulse
import readwav
import template as _template
import textbox

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
    return gvar.gvar(k, np.sqrt(np.maximum(k, 1)))

def ubinom(k, n):
    p = k / n
    s = np.sqrt(n * p * (1 - p))
    return gvar.gvar(k, s)

def usamples(x):
    return gvar.gvar(np.mean(x), np.std(x, ddof=1))

def hspan(ax, y0=None, y1=None):
    ylim = ax.get_ylim()
    if y0 is None:
        y0 = ylim[0]
    if y1 is None:
        y1 = ylim[1]
    ax.axhspan(y0, y1, color='#0002')
    ax.set_ylim(ylim)
    
def vspan(ax, x0=None, x1=None):
    xlim = ax.get_xlim()
    if x0 is None:
        x0 = xlim[0]
    if x1 is None:
        x1 = xlim[1]
    ax.axvspan(x0, x1, color='#0002')
    ax.set_xlim(xlim)

def vlines(ax, x, y0=None, y1=None, **kw):
    ylim = ax.get_ylim()
    if y0 is None:
        y0 = ylim[0]
    if y1 is None:
        y1 = ylim[1]
    ax.vlines(x, y0, y1, **kw)
    ax.set_ylim(ylim)

def hlines(ax, y, x0=None, x1=None, **kw):
    xlim = ax.get_xlim()
    if x0 is None:
        x0 = xlim[0]
    if x1 is None:
        x1 = xlim[1]
    ax.hlines(y, x0, x1, **kw)
    ax.set_xlim(xlim)

def fpmidpoints(x, pe):
    x = np.sort(x)
    pe = np.sort(pe)
    pepos = 1 + np.searchsorted(x, pe)
    values = []
    for start, end in zip(pepos, pepos[1:]):
        y = x[start:end]
        i = np.argmax(np.diff(y))
        values.append(np.mean(y[i:i+2]))
    assert len(values) == len(pe) - 1, len(values)
    return np.array(values)

def borelpmf(n, mu):
    return np.exp(-mu * n) * (mu * n) ** (n - 1) / special.factorial(n)

def borelcdf(n, mu):
    out = 0
    for k in range(1, n + 1):
        out += borelpmf(k, mu)
    return out

def uerrorbar(ax, x, y, **kw):
    ym = gvar.mean(y)
    ys = gvar.sdev(y)
    kwargs = dict(yerr=ys)
    kwargs.update(kw)
    return ax.errorbar(x, ym, **kwargs)

def fcn_pefit(x, p):
    mu = p['mu']
    return {
        'bins': norm * borelpmf(x, mu),
        'overflow': norm * (1 - borelcdf(x[-1], mu)),
    }

gvar.BufferDict.uniform('U', 0, 1)

# parameters:
# cut = cut on ptheight to measure dcr
# length = filter length for afterpulse
# hcut = cut on minorheight
# dcut, dcutr = left/right cut on minorpos - mainpos
# npe = pe of main peak to search afterpulses
vovdict[5.5].update(cut=10, length=512, hcut=25, dcut=500, dcutr=5500, npe=1)
vovdict[7.5].update(cut=15, length=512, hcut=30, dcut=500, dcutr=5500, npe=1)
vovdict[9.5].update(cut=25, length=512, hcut=35, dcut=500, dcutr=5500, npe=1)

for vov in vovdict:
    
    print(f'\n************** {vov} VoV **************')
    
    # load analysis file and skip if already saved
    d = vovdict[vov]
    d['analfile'] = f'{savedir}/dict{vov}VoV.pickle'
    globals().update(d) # YES
    if os.path.exists(analfile):
        with open(analfile, 'rb') as file:
            d.update(pickle.load(file))
        continue
    
    # load wav and AfterPulse object
    datalist, sim = apload(vov)
    
    # get index of the filter length to use for afterpulses
    ilength = np.searchsorted(sim.filtlengths, length)
    assert sim.filtlengths[ilength] == length
    assert npe >= 1
    suffix = f'-{vov}VoV'
    
    # recompute pe boundaries
    ptlength = sim.ptlength
    mainsel = f'good&(mainpos>=0)&(length=={ptlength})'
    maxnpe = sim.getexpr('max(npe)', f'(npe<1000)&(length=={ptlength})')
    peaks = [
        sim.getexpr('median(mainheight)', f'{mainsel}&(npe=={npe})')
        for npe in range(1 + maxnpe)
    ]
    height = sim.getexpr('mainheight', mainsel)
    boundaries = fpmidpoints(height, peaks)
    
    # plot a fingerplot
    fig = sim.hist('mainheight', mainsel, 'log', 1000)
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, suffix)
    
    # plot pre-trigger height vs. position
    lmargin = 100
    rmargin = 500
    fig = sim.scatter('ptpos', 'ptheight')
    ax, = fig.get_axes()
    trigger = sim.getexpr('trigger[0]')
    vspan(ax, lmargin, trigger - rmargin)
    hspan(ax, cut)
    hlines(ax, boundaries, linestyle=':')
    savef(fig, suffix)
    
    # plot a histogram of pre-trigger peak height
    ptsel = f'(ptpos>={lmargin})&(ptpos<trigger-{rmargin})'
    fig = sim.hist('ptheight', ptsel, 'log')
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, suffix)
    
    # counts for computing the DCR
    l1pe, r1pe = boundaries[:2]
    sigcount = sim.getexpr(f'count_nonzero({ptsel}&(ptheight>{cut}))')
    lowercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight<={cut})&(mainheight>{l1pe}))')
    uppercount = sim.getexpr(f'count_nonzero({mainsel}&(mainheight>{cut})&(mainheight<{r1pe}))')
    
    # variables to compute the rate
    time = sim.getexpr(f'mean(trigger-{lmargin}-{rmargin})', ptsel)
    nevents = sim.getexpr(f'count_nonzero({ptsel})')
    totalevt = len(sim.output)
    
    # variables with uncertainties
    s = upoisson(sigcount)
    l = upoisson(lowercount)
    u = upoisson(uppercount)
    t = time * 1e-9 * ubinom(nevents, totalevt)
    
    # correction factor and DCR
    f = (l + u) / u
    r = f * s / t
    
    # print DCR results
    print(f'pre-trigger count = {sigcount} / {nevents}')
    print(f'total time = {t} s')
    print(f'correction factor = {f}')
    print(f'dcr = {r} cps @ {vov}VoV')
    
    # save DCR results
    d.update(dcr=r, dcrcount=s, dcrtime=t, dcrfactor=f)
    
    # compute pe count of pre-trigger peaks
    ptheight = sim.getexpr('ptheight')
    ptpe = np.digitize(ptheight, boundaries)
    sim.setvar('ptnpe', ptpe)
    
    # histogram pre-trigger pe and fit borel distribution
    counts = np.bincount(sim.getexpr('ptnpe', ptsel), minlength=len(boundaries) + 1)
    assert len(counts) == len(boundaries) + 1
    prior = {
        'U(mu)': gvar.BufferDict.uniform('U', 0, 1),
    }
    x = np.arange(1, len(counts) - 1)
    y = {
        'bins': upoisson(counts[1:-1]),
        'overflow': upoisson(counts[-1]),
    }
    norm = np.sum(counts[1:])
    fit = lsqfit.nonlinear_fit((x, y), fcn_pefit, prior)
    fitdsc = fit.format(maxline=True)
    print(fitdsc)
    
    # plot histogram and fit results
    fig = sim.hist('ptnpe', f'{ptsel}&(ptnpe>0)')
    ax, = fig.get_axes()
    kw = dict(linestyle='', capsize=4, marker='.', color='k')
    uerrorbar(ax, x, y['bins'], **kw)
    uerrorbar(ax, x[-1] + 1, y['overflow'], **kw)
    yfit = fcn_pefit(x, fit.palt)
    xs = np.pad(x, (0, 1), constant_values=x[-1] + 1)
    ys = np.concatenate([yfit['bins'], [yfit['overflow']]])
    ym = gvar.mean(ys)
    ysdev = gvar.sdev(ys)
    ax.fill_between(xs, ym + ysdev, ym - ysdev, color='#0004')
    info = f"""\
chi2/dof = {fit.chi2:.1f}/{fit.dof} = {fit.chi2/fit.dof:.3g}
mu = {fit.p['mu']}"""
    textbox.textbox(ax, info, loc='center right', fontsize='small')
    savef(fig, suffix)
    
    # print and save fit results
    mu = fit.p['mu']
    print(f'dcr cross-talk mu = {mu}')
    d.update(dcrfit=fit)
    
    # plot <= 3 events with an high pre-trigger peak
    evts = sim.eventswhere(f'{ptsel}&(ptheight>{cut})')
    for ievt in evts[:3]:
        fig = sim.plotevent(datalist, ievt, zoom='all')
        savef(fig, suffix)
    
    # plot fingerplot 0-2 pe, this time with parameters for afterpulse counting
    mainsel = f'good&(mainpos>=0)&(length=={length})'
    fig = sim.hist('mainheight', f'{mainsel}&(npe>={npe-1})&(npe<={npe+1})', nbins=200)
    savef(fig, suffix)
    
    # compute median height of 1pe
    hnpe = sim.getexpr('median(mainheight)', f'{mainsel}&(npe=={npe})')
    print(f'1 pe median height with {length} ns = {hnpe:.3g}')
    
    # plot afterpulses height vs. distance from main pulse
    minorsel = f'{mainsel}&(minorpos>=0)'
    minornpe = f'{minorsel}&(npe=={npe})'
    fig = sim.scatter('minorpos-mainpos', 'minorheight', minornpe)
    ax, = fig.get_axes()
    hspan(ax, hcut)
    vspan(ax, dcut, dcutr)
    savef(fig, suffix)
    
    # counts for computing the afterpulse probability
    nevents = sim.getexpr(f'count_nonzero({minornpe})')
    apcond = f'{minornpe}&(minorheight>{hcut})&(minorpos-mainpos>{dcut})&(minorpos-mainpos<{dcutr})'
    apcount = sim.getexpr(f'count_nonzero({apcond})')
    
    # plot some events with afterpulses
    apevts = sim.eventswhere(apcond)
    for ievt in apevts[:3]:
        fig = sim.plotevent(datalist, ievt, ilength)
        savef(fig, suffix)
    
    # histogram of selected afterpulses height
    fig = sim.hist('minorpos-mainpos', apcond)
    savef(fig, suffix)

    # compute decaying time constant of afterpulses
    dist = sim.getexpr('minorpos-mainpos', apcond)
    m = np.mean(dist)
    s = np.std(dist)
    l = np.min(dist)
    tau = gvar.gvar((m - l), s / np.sqrt(len(dist)))
    print(f'expon tau = {tau} ns')
    d.update(tau=tau)
    
    # correction factor for temporal selection
    factors = [
        stats.expon.cdf(dcutr, scale=s) - stats.expon.cdf(dcut, scale=s)
        for s in [900, 1100]
    ]
    factor = 1 / usamples(factors)
    
    # expected background from DCR
    time = (dcutr - dcut) * 1e-9 * nevents
    bkg = r * time
    
    # afterpulse probability
    count = upoisson(apcount)
    ccount = (count - bkg) * factor
    p = ccount / nevents
    
    # print afterpulse results
    print(f'correction factor = {factor} (assuming tau a priori)')
    print(f'expected background = {bkg} counts in {time:.3g} s')
    print(f'ap count = {apcount} / {nevents}')
    print(f'corrected ap count = {ccount}')
    print(f'afterpulse probability = {p} @ {vov}VoV')
    
    # save afterpulse results
    d.update(ap=p, apcount=count, apnevents=nevents, apbkg=bkg, aptime=time, apfactor=factor)

    # save analysis results to file
    with open(analfile, 'wb') as file:
        print(f'save {analfile}...')
        pickle.dump(d, file)

fig, axs = plt.subplots(1, 3, num='afterpulse_tile21', clear=True, figsize=[11, 4.8])

axdcr, axct, axap = axs

for ax in axs:
    if ax.is_last_row():
        ax.set_xlabel('VoV')

axdcr.set_title('DCR')
axdcr.set_ylabel('Pre-trigger rate [cps]')

axap.set_title('Afterpulse')
axap.set_ylabel('Prob. of $\\geq$1 ap after 1 pe signal [%]')

axct.set_title('Cross talk')
axct.set_ylabel('Prob. of > 1 pe in dark counts [%]')

vov = list(vovdict)
dcr = np.array([d['dcr'] for d in vovdict.values()])
ap = np.array([d['ap'] for d in vovdict.values()])
pct = np.array([1 - np.exp(-d['dcrfit'].palt['mu']) for d in vovdict.values()])

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
uerrorbar(axct, vov, pct * 100, **kw)

for ax in axs:
    l, u = ax.get_ylim()
    ax.set_ylim(0, u)
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()
