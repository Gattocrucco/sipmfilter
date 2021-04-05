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

def uerrorbar(ax, x, y, **kw):
    ym = gvar.mean(y)
    ys = gvar.sdev(y)
    kwargs = dict(yerr=ys)
    kwargs.update(kw)
    return ax.errorbar(x, ym, **kwargs)

def exponinteg(x1, x2, scale):
    return np.exp(-x1 / scale) - np.exp(-x2 / scale)

def borelpmf(n, params):
    mu = params['mu']
    effmu = mu * n
    return np.exp(-effmu) * effmu ** (n - 1) / special.factorial(n)

def geompmf(k, params):
    # p is 1 - p respect to the conventional definition to match the borel mu
    p = params['p']
    return p ** (k - 1) * (1 - p)

def genpoissonpmf(k, params):
    # P(k;mu,lambda) = mu (mu + k lambda)^(k - 1) exp(-(mu + k lambda)) / k!
    mu = params['mu_poisson']
    lamda = params['mu_borel']
    effmu = mu + k * lamda
    return np.exp(-effmu) * mu * effmu ** (k - 1) / special.factorial(k)

def fcn(x, p):
    pmf = x['pmf']
    bins = x['bins']
    norm = x['norm']
    dist = np.empty(len(bins) - 1, object)
    for i, (left, right) in enumerate(zip(bins, bins[1:])):
        ints = np.arange(np.ceil(left), right)
        dist[i] = np.sum(pmf(ints, p))
    if bins[-1] == int(bins[-1]):
        dist[-1] += pmf(bins[-1], p)
    integral = np.sum(dist)
    if x['hasoverflow']:
        return dict(
            bins = norm * dist,
            overflow = norm * (1 - integral),
        )
    else:
        return dict(
            bins = norm * dist / integral,
        )

def fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow=None, histcond=None, **kw):
    # histogram
    sample = sim.getexpr(expr, condexpr)
    counts, _ = np.histogram(sample, bins)
    hasoverflow = bins_overflow is not None
    if hasoverflow:
        (overflow,), _ = np.histogram(sample, bins_overflow)
    else:
        overflow = 0
    
    if hasoverflow:
        # add bins to the overflow bin until it has at least 3 counts
        lencounts = len(counts)
        for i in range(len(counts) - 1, -1, -1):
            if overflow < 3:
                overflow += counts[i]
                lencounts -= 1
            else:
                break
        counts = counts[:lencounts]
        bins = bins[:lencounts + 1]
    else:
        # group last bins until there are at least 3 counts
        while len(counts) > 1:
            if counts[-1] < 3:
                counts[-2] += counts[-1]
                counts = counts[:-1]
                bins[-2] = bins[-1]
                bins = bins[:-1]
            else:
                break
    
    center = (bins[1:] + bins[:-1]) / 2
    wbar = np.diff(bins) / 2
    norm = np.sum(counts) + overflow
    assert norm == sim.getexpr(f'count_nonzero({condexpr})')
    
    # fit
    x = dict(
        pmf = pmf,
        bins = bins,
        norm = norm,
        hasoverflow = hasoverflow,
    )
    y = dict(bins=upoisson(counts))
    if hasoverflow:
        y.update(overflow=upoisson(overflow))
    fit = lsqfit.nonlinear_fit((x, y), fcn, prior, **kw)
    print(fit.format(maxline=True))
    
    # plot histogram
    if histcond is None:
        histcond = condexpr
    fig = sim.hist(expr, histcond)
    ax, = fig.get_axes()
    kw = dict(linestyle='', capsize=4, marker='.', color='k')
    uerrorbar(ax, center, y['bins'], xerr=wbar, **kw)
    if hasoverflow:
        uerrorbar(ax, center[-1] + 1, y['overflow'], **kw)

    # plot data and fit
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
    
    # write fit results on the plot
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

def intbins(min, max):
    return -0.5 + np.arange(min, max + 2)

def pebins(boundaries, start=1):
    return intbins(start, len(boundaries) - 1), intbins(1000, 1000)

def getkind(kind):
    if kind == 'borel':
        return borelpmf, 'mu'
    elif kind == 'geom':
        return geompmf, 'p'
    else:
        raise KeyError(kind)

def fitpe(sim, expr, condexpr, boundaries, kind='borel', overflow=True, **kw):
    pmf, param = getkind(kind)
    bins, bins_overflow = pebins(boundaries)
    if not overflow:
        bins_overflow = None
    prior = {
        f'U({param})': gvar.BufferDict.uniform('U', 0, 1),
    }
    return fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow, **kw)

def fitpepoisson(sim, expr, condexpr, boundaries, prior=None, overflow=True, **kw):
    bins, bins_overflow = pebins(boundaries, 0)
    if not overflow:
        bins_overflow = None
    prior0 = {
        'U(mu_borel)': gvar.BufferDict.uniform('U', 0, 1),
        'log(mu_poisson)': gvar.gvar(0, 1),
    }
    if prior is not None:
        prior0.update(prior)
    return fithistogram(sim, expr, condexpr, prior0, genpoissonpmf, bins, bins_overflow, **kw)

gvar.BufferDict.uniform('U', 0, 1)

def scalesdev(x, f):
    return x + (f - 1) * (x - gvar.mean(x))

# parameters:
# ptlength = length used for pre-trigger region
# aplength = filter length for afterpulse (default same as ptlength)
# dcut, dcutr = left/right cut on minorpos - mainpos
# npe = pe of main peak to search afterpulses
defaults = dict(dcut=500, dcutr=5500, npe=1)
vovdict[5.5].update(ptlength=128, **defaults)
vovdict[7.5].update(ptlength= 64, **defaults)
vovdict[9.5].update(ptlength= 64, **defaults)

for vov in vovdict:
    
    print(f'\n************** {vov} VoV **************')
    
    # load analysis file and skip if already saved
    prefix = f'{vov}VoV-'
    d = vovdict[vov]
    d['analfile'] = f'{savedir}/{prefix}archive.pickle'
    d['aplength'] = d.get('aplength', d['ptlength'])
    globals().update(d) # YES
    if os.path.exists(analfile):
        with open(analfile, 'rb') as file:
            print(f'load {analfile}...')
            vovdict[vov] = gvar.load(file)
        continue
    
    # load wav and AfterPulse object
    datalist, sim = apload(vov)
    
    
    ### DARK COUNT ###
    
    # get index of the filter length to use for pre-trigger pulses
    ilength = np.searchsorted(sim.filtlengths, ptlength)
    assert sim.filtlengths[ilength] == ptlength

    # compute dark count height cut
    mainsel = f'good&(mainpos>=0)&(length=={ptlength})'
    p01 = [
        sim.getexpr('median(mainheight)', f'{mainsel}&(npe=={npe})')
        for npe in [0, 1]
    ]
    lmargin = 100
    rmargin = 500
    ptsel = f'(length=={ptlength})&(ptpos>={lmargin})&(ptpos<trigger-{rmargin})'
    ptheight = sim.getexpr('ptheight', ptsel)
    cut, = afterpulse.maxdiff_boundaries(ptheight, p01)

    # plot a fingerplot
    boundaries = sim.computenpeboundaries(ilength)
    fig = sim.hist('mainheight', mainsel, 'log', 1000)
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot pre-trigger height vs. position
    fig = sim.scatter('ptpos', 'ptheight', f'length=={ptlength}')
    ax, = fig.get_axes()
    trigger = sim.getexpr('trigger[0]')
    vspan(ax, lmargin, trigger - rmargin)
    hspan(ax, cut)
    hlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot a histogram of pre-trigger peak height
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
    
    # fit pre-trigger pe histogram
    sim.setvar('ptnpe', sim.computenpe(sim.getexpr('ptheight')))
    fit, fig = fitpe(sim, 'ptnpe', f'{ptsel}&(ptnpe>0)', boundaries, 'borel', histcond=f'{ptsel}&(ptnpe>0)&(ptnpe<1000)')
    savef(fig, prefix)
    d.update(dcrfit=fit)
    
    # fit main peak pe histogram
    mainsel = f'~closept&(mainpos>=0)&(length=={ptlength})'
    fit, fig = fitpepoisson(sim, 'npe', f'{mainsel}&(npe<1000)', boundaries, overflow=False, histcond=f'{mainsel}&(npe<1000)')
    savef(fig, prefix)
    d.update(mainfit=fit)
    
    # plot <= 3 events with an high pre-trigger peak
    evts = sim.eventswhere(f'{ptsel}&(ptheight>{cut})')
    for ievt in evts[:3]:
        fig = sim.plotevent(datalist, ievt, ilength, zoom='all')
        savef(fig, prefix)
    
    
    ### AFTERPULSES ###
        
    # get index of the filter length to use for afterpulses
    ilength = np.searchsorted(sim.filtlengths, aplength)
    assert sim.filtlengths[ilength] == aplength

    # compute dark count height cut
    mainsel = f'good&(mainpos>=0)&(length=={aplength})'
    p01 = [
        sim.getexpr('median(mainheight)', f'{mainsel}&(npe=={npe})')
        for npe in [0, 1]
    ]
    minornpe = f'{mainsel}&(minorpos>=0)&(npe=={npe})'
    apsel = f'{minornpe}&(minorpos-mainpos>{dcut})&(minorpos-mainpos<{dcutr})'
    apheight = sim.getexpr('minorheight', apsel)
    hcut, = afterpulse.maxdiff_boundaries(apheight, p01)
    hcut = np.round(hcut, 1)

    # plot a fingerplot
    boundaries = sim.computenpeboundaries(ilength)
    fig = sim.hist('mainheight', mainsel, 'log', 1000)
    ax, = fig.get_axes()
    vspan(ax, hcut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot afterpulses height vs. distance from main pulse
    fig = sim.scatter('minorpos-mainpos', 'minorheight', minornpe)
    ax, = fig.get_axes()
    hspan(ax, hcut)
    vspan(ax, dcut, dcutr)
    hlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # counts for computing the afterpulse probability
    nevents = sim.getexpr(f'count_nonzero({minornpe})')
    apcond = f'{apsel}&(minorheight>{hcut})'
    apcount = sim.getexpr(f'count_nonzero({apcond})')
    
    # histogram of selected afterpulses delay
    fig = sim.hist('minorpos-mainpos', apcond)
    savef(fig, prefix)

    # fit afterpulses pe histogram
    sim.setvar('apnpe', sim.computenpe(sim.getexpr('minorheight')))
    fit, fig = fitpe(sim, 'apnpe', f'{apcond}&(apnpe>0)&(apnpe<1000)', boundaries, 'borel', overflow=False, histcond=f'{apcond}&(apnpe>0)&(apnpe<1000)')
    savef(fig, prefix)
    d.update(apfit=fit)

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

fig, axs = plt.subplots(2, 2, num='afterpulse_tile21', clear=True, figsize=[9, 7.1])

axdcr, axap, axpct, axnct = axs.flat

for ax in axs.flat:
    if ax.is_last_row():
        ax.set_xlabel('VoV')

axdcr.set_title('DCR')
axdcr.set_ylabel('Pre-trigger rate [cps]')

axap.set_title('Afterpulse')
axap.set_ylabel('Prob. of $\\geq$1 ap after 1 pe signal [%]')

axpct.set_title('Cross talk')
axpct.set_ylabel('Prob. of > 1 pe [%]')

axnct.set_title('Cross talk')
axnct.set_ylabel('Average excess pe')

vov = np.array(list(vovdict))
def listdict(getter):
    return np.array([getter(d) for d in vovdict.values()])
dcr = listdict(lambda d: d['dcr'])
ap = listdict(lambda d: d['ap'])
def ct(mugetter):
    pct = listdict(lambda d: 1 - np.exp(-mugetter(d)))
    nct = listdict(lambda d: 1 / (1 - mugetter(d)) - 1)
    return pct, nct
def mugetter(fitkey, param):
    def getter(d):
        fit = d[fitkey]
        mu = fit.palt[param]
        if fit.Q < 0.01:
            factor = np.sqrt(fit.chi2 / fit.dof)
            mu = scalesdev(mu, factor)
        return mu
    return getter
mus = [
    ('Dark count' , mugetter('dcrfit' , 'mu'      )),
    ('Laser'      , mugetter('mainfit', 'mu_borel')),
    ('Afterpulses', mugetter('apfit'  , 'mu'      )),
]

kw = dict(capsize=4, linestyle='', marker='.')
uerrorbar(axdcr, vov, dcr, **kw)
uerrorbar(axap, vov, ap * 100, **kw)
for i, (label, getter) in enumerate(mus):
    pct, nct = ct(getter)
    offset = 0.1 * (i/(len(mus)-1) - 1/2)
    kw.update(label=label)
    x = vov + offset
    uerrorbar(axpct, x, pct * 100, **kw)
    uerrorbar(axnct, x, nct, **kw)

axpct.legend()
axnct.legend()

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

for ax in axs.flat:
    l, u = ax.get_ylim()
    ax.set_ylim(0, u)
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()
