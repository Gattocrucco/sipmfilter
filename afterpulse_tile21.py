import os
import glob
import re
import pickle
import collections

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

class SaveFigure:
    
    def __init__(self):
        self._counts = collections.defaultdict(int)

    def __call__(self, fig, prefix=''):
        self._counts[prefix] += 1
        path = f'{savedir}/{prefix}fig{self._counts[prefix]:02d}.png'
        print(f'save {path}...')
        fig.savefig(path)

savef = SaveFigure()

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

def borelpmf(n, params):
    mu = params['mu']
    return np.exp(-mu * n) * (mu * n) ** (n - 1) / special.factorial(n)

def geompmf(k, params):
    # p is 1 - p respect to the conventional definition to match the borel mu
    p = params['p']
    return p ** (k - 1) * (1 - p)

def cdf(pmf, n, params):
    out = 0
    for k in range(1, n + 1):
        out += pmf(k, params)
    return out

def uerrorbar(ax, x, y, **kw):
    ym = gvar.mean(y)
    ys = gvar.sdev(y)
    kwargs = dict(yerr=ys)
    kwargs.update(kw)
    return ax.errorbar(x, ym, **kwargs)

def exponinteg(x1, x2, scale):
    return np.exp(-x1 / scale) - np.exp(-x2 / scale)

def intbins(min, max):
    return -0.5 + np.arange(min, max + 2)

def fcn(x, p):
    pmf = x['pmf']
    center = x['center']
    norm = x['norm']
    out = dict(bins=norm * pmf(center, p))
    if x['hasoverflow']:
        out.update(overflow=norm * (1 - cdf(pmf, center[-1], p)))
    return out

def fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow=None, histcond=None):
    # histogram
    sample = sim.getexpr(expr, condexpr)
    counts, _ = np.histogram(sample, bins)
    hasoverflow = bins_overflow is not None
    if hasoverflow:
        (overflow,), _ = np.histogram(sample, bins_overflow)
    else:
        overflow = 0
    
    # add bins to the overflow bin until it has at least 3 counts
    if hasoverflow:
        lencounts = len(counts)
        for i in range(len(counts) - 1, -1, -1):
            if overflow < 3:
                overflow += counts[i]
                lencounts -= 1
            else:
                break
        counts = counts[:lencounts]
        bins = bins[:lencounts + 1]
    
    center = (bins[1:] + bins[:-1]) / 2
    norm = np.sum(counts) + overflow
    assert norm == sim.getexpr(f'count_nonzero({condexpr})')
    
    # fit
    x = dict(
        pmf = pmf,
        center = center.astype(int),
        norm = norm,
        hasoverflow = hasoverflow,
    )
    y = dict(bins=upoisson(counts))
    if hasoverflow:
        y.update(overflow=upoisson(overflow))
    fit = lsqfit.nonlinear_fit((x, y), fcn, prior)
    print(fit.format(maxline=True))
    
    # plot histogram
    if histcond is None:
        histcond = condexpr
    fig = sim.hist(expr, histcond)
    ax, = fig.get_axes()
    kw = dict(linestyle='', capsize=4, marker='.', color='k')
    uerrorbar(ax, center, y['bins'], **kw)
    if hasoverflow:
        uerrorbar(ax, center[-1] + 1, y['overflow'], **kw)

    # plot fit results
    yfit = fcn(x, fit.palt)
    if hasoverflow:
        xs = np.pad(center, (0, 1), constant_values=center[-1] + 1)
        ys = np.concatenate([yfit['bins'], [yfit['overflow']]])
    else:
        xs = center
        ys = yfit['bins']
    ym = gvar.mean(ys)
    ysdev = gvar.sdev(ys)
    ax.fill_between(xs, ym + ysdev, ym - ysdev, color='#0004')
    
    # write fit results
    info = f"""\
chi2/dof = {fit.chi2/fit.dof:.3g}
pvalue = {fit.Q:.2g}"""
    for k, v in fit.palt.items():
        info += f'\n{k} = {v}'
    for k in fit.palt.extension_keys():
        v = fit.palt[k]
        info += f'\n{k} = {v}'
    textbox.textbox(ax, info, loc='center right', fontsize='small')
    
    return fit, fig

def fitpe(sim, expr, condexpr, boundaries, kind='borel', **kw):
    if kind == 'borel':
        pmf = borelpmf
        param = 'mu'
    elif kind == 'geom':
        pmf = geompmf
        param = 'p'
    else:
        raise KeyError(kind)
    bins = intbins(1, len(boundaries) - 1)
    bins_overflow = intbins(1000, 1000)
    prior = {
        f'U({param})': gvar.BufferDict.uniform('U', 0, 1),
    }
    return fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow, **kw)

gvar.BufferDict.uniform('U', 0, 1)

# parameters:
# ptlength = length used for pre-trigger region
# aplength = filter length for afterpulse
# hcut = cut on minorheight
# dcut, dcutr = left/right cut on minorpos - mainpos
# npe = pe of main peak to search afterpulses
defaults = dict(ptlength=512, aplength=512, dcut=500, dcutr=5500, npe=1)
vovdict[5.5].update(hcut=25, **defaults)
vovdict[7.5].update(hcut=30, **defaults)
vovdict[9.5].update(hcut=35, **defaults)

for vov in vovdict:
    
    print(f'\n************** {vov} VoV **************')
    
    # load analysis file and skip if already saved
    d = vovdict[vov]
    d['analfile'] = f'{savedir}/dict{vov}VoV.pickle'
    globals().update(d) # YES
    if os.path.exists(analfile):
        with open(analfile, 'rb') as file:
            vovdict[vov] = gvar.load(file)
        continue
    prefix = f'{vov}VoV-'
    
    # load wav and AfterPulse object
    datalist, sim = apload(vov)
    
    # get index of the filter length to use for pre-trigger pulses
    ilength = np.searchsorted(sim.filtlengths, ptlength)
    assert sim.filtlengths[ilength] == ptlength
    
    # plot a fingerplot
    mainsel = f'good&(mainpos>=0)&(length=={ptlength})'
    boundaries = sim.computenpeboundaries(ilength)
    cut = boundaries[0]
    fig = sim.hist('mainheight', mainsel, 'log', 1000)
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot pre-trigger height vs. position
    lmargin = 100
    rmargin = 500
    fig = sim.scatter('ptpos', 'ptheight', f'length=={ptlength}')
    ax, = fig.get_axes()
    trigger = sim.getexpr('trigger[0]')
    vspan(ax, lmargin, trigger - rmargin)
    hspan(ax, cut)
    hlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot a histogram of pre-trigger peak height
    ptsel = f'(length=={ptlength})&(ptpos>={lmargin})&(ptpos<trigger-{rmargin})'
    fig = sim.hist('ptheight', ptsel, 'log')
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # variables to compute the dark count
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
    print(f'dcr = {r} cps')
    
    # save DCR results
    d.update(dcr=r, dcrcount=s, dcrtime=t, dcrfactor=f)
    
    # compute pe count of pre-trigger peaks
    sim.setvar('ptnpe', sim.computenpe('ptpeak'))
    
    # histogram pre-trigger pe and fit
    fit, fig = fitpe(sim, 'ptnpe', f'{ptsel}&(ptnpe>0)', boundaries, 'borel', histcond=f'{ptsel}&(ptnpe>0)&(ptnpe<1000)')
    
    # save figure and fit results
    savef(fig, prefix)
    d.update(dcrfit=fit)
    
    # plot <= 3 events with an high pre-trigger peak
    evts = sim.eventswhere(f'{ptsel}&(ptheight>{cut})')
    for ievt in evts[:3]:
        fig = sim.plotevent(datalist, ievt, zoom='all')
        savef(fig, prefix)
    
    # get index of the filter length to use for afterpulses
    ilength = np.searchsorted(sim.filtlengths, aplength)
    assert sim.filtlengths[ilength] == aplength

    # plot fingerplot 0-2 pe, this time with parameters for afterpulse counting
    mainsel = f'good&(mainpos>=0)&(length=={aplength})'
    fig = sim.hist('mainheight', f'{mainsel}&(npe>={npe-1})&(npe<={npe+1})', nbins=200)
    savef(fig, prefix)
    
    # plot afterpulses height vs. distance from main pulse
    minorsel = f'{mainsel}&(minorpos>=0)'
    minornpe = f'{minorsel}&(npe=={npe})'
    fig = sim.scatter('minorpos-mainpos', 'minorheight', minornpe)
    ax, = fig.get_axes()
    hspan(ax, hcut)
    vspan(ax, dcut, dcutr)
    savef(fig, prefix)
    
    # counts for computing the afterpulse probability
    nevents = sim.getexpr(f'count_nonzero({minornpe})')
    apcond = f'{minornpe}&(minorheight>{hcut})&(minorpos-mainpos>{dcut})&(minorpos-mainpos<{dcutr})'
    apcount = sim.getexpr(f'count_nonzero({apcond})')
    
    # histogram of selected afterpulses height
    fig = sim.hist('minorpos-mainpos', apcond)
    savef(fig, prefix)

    # plot some events with afterpulses
    apevts = sim.eventswhere(apcond)
    for ievt in apevts[:3]:
        fig = sim.plotevent(datalist, ievt, ilength)
        savef(fig, prefix)
    
    # compute decaying time constant of afterpulses
    dist = sim.getexpr('minorpos-mainpos', apcond)
    m = np.mean(dist)
    s = np.std(dist)
    l = np.min(dist)
    tau = gvar.gvar((m - l), s / np.sqrt(len(dist)))
    print(f'expon tau = {tau} ns')
    d.update(tau=tau)
    
    # correction factor for temporal selection
    factor = 1 / exponinteg(dcut, dcutr, tau)
    
    # expected background from DCR
    time = (dcutr - dcut) * 1e-9 * nevents
    bkg = r * time
    
    # afterpulse probability
    count = upoisson(apcount)
    ccount = (count - bkg) * factor
    p = ccount / nevents
    
    # print afterpulse results
    print(f'correction factor = {factor}')
    print(f'expected background = {bkg} counts in {time:.3g} s')
    print(f'ap count = {apcount} / {nevents}')
    print(f'corrected ap count = {ccount}')
    print(f'afterpulse probability = {p}')
    
    # save afterpulse results
    d.update(ap=p, apcount=count, apnevents=nevents, apbkg=bkg, aptime=time, apfactor=factor)

    # save analysis results to file
    with open(analfile, 'wb') as file:
        print(f'save {analfile}...')
        gvar.dump(d, file)

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
