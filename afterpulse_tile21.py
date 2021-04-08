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
import breaklines

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

def exponbkgcdf(x, params):
    scale = params['tau']
    const = params['const']
    return 1 - np.exp(-x / scale) + const * x

def fcn(x, p):
    continuous = x['continuous']
    bins = x['bins']

    if continuous:
        cdf = x['pmf_or_cdf']
        prob = cdf(bins, p)
        dist = np.diff(prob)
        integral = prob[-1] - prob[0]
    else:
        pmf = x['pmf_or_cdf']
        dist = []
        for left, right in zip(bins, bins[1:]):
            ints = np.arange(np.ceil(left), right)
            dist.append(np.sum(pmf(ints, p)))
        if bins[-1] == int(bins[-1]):
            dist[-1] += pmf(bins[-1], p)
        dist = np.array(dist)
        integral = np.sum(dist)
   
    norm = x['norm']
    if x['hasoverflow']:
        return dict(
            bins = norm * dist,
            overflow = norm * (1 - integral),
        )
    else:
        return dict(
            bins = norm * dist / integral,
        )

def fithistogram(sim, expr, condexpr, prior, pmf_or_cdf, bins='auto', bins_overflow=None, continuous=False, **kw):
    """
    Fit an histogram.
    
    Parameters
    ----------
    sim : AfterPulse
    expr : str
    condexpr : str
        The sample is sim.getexpr(expr, condexpr).
    prior : array/dictionary of GVar
        Prior for the fit.
    pmf_or_cdf : function
        The signature must be `pmf_or_cdf(x, params)` where `params` has the
        same format of `prior`.
    bins : int, sequence, str
        Passed to np.histogram. Default 'auto'.
    bins_overflow : sequence of two elements
        If None, do not fit the overflow. It is assumed that all the
        probability mass outside of the bins is contained in the overflow bin.
    continuous : bool
        If False (default), pmf_or_cdf must be the probability distribution of
        an integer variable. If True, pmf_or_cdf must compute the cumulative
        density up to the given point.
    **kw :
        Additional keyword arguments are passed to lsqfit.nonlinear_fit.
    
    Return
    ------
    fit : lsqfit.nonlinear_fit
    fig : matplotlib figure
    """
    # histogram
    sample = sim.getexpr(expr, condexpr)
    counts, bins = np.histogram(sample, bins)
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
    
    # check total count
    norm = np.sum(counts) + overflow
    assert norm == sim.getexpr(f'count_nonzero({condexpr})')
    
    # fit
    x = dict(
        pmf_or_cdf = pmf_or_cdf,
        continuous = continuous,
        bins = bins,
        norm = norm,
        hasoverflow = hasoverflow,
    )
    y = dict(bins=upoisson(counts))
    if hasoverflow:
        y.update(overflow=upoisson(overflow))
    fit = lsqfit.nonlinear_fit((x, y), fcn, prior, **kw)
    print(fit.format(maxline=True))
    
    # plot data
    fig, ax = plt.subplots(num='afterpulse_tile21.fithistogram', clear=True)
    center = (bins[1:] + bins[:-1]) / 2
    wbar = np.diff(bins) / 2
    kw = dict(linestyle='', capsize=4, marker='.', color='k')
    uerrorbar(ax, center, y['bins'], xerr=wbar, label='histogram', **kw)

    # plot fit
    yfit = fcn(x, fit.palt)
    ys = yfit['bins']
    ym = np.repeat(gvar.mean(ys), 2)
    ysdev = np.repeat(gvar.sdev(ys), 2)
    xs = np.concatenate([bins[:1], np.repeat(bins[1:-1], 2), bins[-1:]])
    ax.fill_between(xs, ym - ysdev, ym + ysdev, color='#0004', label='fit')
    
    # plot overflow
    if hasoverflow:
        oc = center[-1] + 2 * wbar[-1]
        ys = y['overflow']
        uerrorbar(ax, oc, ys, label='overflow', **kw)
        ym = np.full(2, gvar.mean(ys))
        ysdev = np.full(2, gvar.sdev(ys))
        xs = oc + 0.8 * (bins[-2:] - center[-1])
        ax.fill_between(xs, ym - ysdev, ym + ysdev, color='#0004')
    
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
    
    # decorations
    ax.legend(loc='upper right')
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(expr)
    ax.set_ylabel('Count per bin')
    cond = breaklines.breaklines(f'Selection: {condexpr}', 40, ')', '&|')
    textbox.textbox(ax, cond, loc='upper left', fontsize='small')
    fig.tight_layout()
    
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

gvar.BufferDict.uniform('U', 0, 1)

def fitpe(sim, expr, condexpr, boundaries, kind='borel', overflow=True, **kw):
    pmf, param = getkind(kind)
    bins, bins_overflow = pebins(boundaries)
    if overflow:
        histexpr = f'where({expr}<1000,{expr},1+max({expr}[{expr}<1000]))'
    else:
        bins_overflow = None
        condexpr = f'{condexpr}&({expr}<1000)'
        histexpr = expr
    prior = {
        f'U({param})': gvar.BufferDict.uniform('U', 0, 1),
    }
    fit, fig1 = fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow, **kw)
    fig2 = sim.hist(histexpr, condexpr)
    return fit, fig2, fig1

def fitpepoisson(sim, expr, condexpr, boundaries, overflow=True, **kw):
    bins, bins_overflow = pebins(boundaries, 0)
    if overflow:
        histexpr = f'where({expr}<1000,{expr},1+max({expr}[{expr}<1000]))'
    else:
        bins_overflow = None
        condexpr = f'{condexpr}&({expr}<1000)'
        histexpr = expr
    prior = {
        'U(mu_borel)': gvar.BufferDict.uniform('U', 0, 1),
        'log(mu_poisson)': gvar.gvar(0, 1),
    }
    fit, fig1 = fithistogram(sim, expr, condexpr, prior, genpoissonpmf, bins, bins_overflow, **kw)
    fig2 = sim.hist(histexpr, condexpr)
    return fit, fig2, fig1

def fitapdecay(sim, expr, condexpr, const, **kw):
    prior = {
        'log(tau)' : gvar.gvar(np.log(1000), 1),
        'const' : const,
    }
    fit, fig1 = fithistogram(sim, expr, condexpr, prior, exponbkgcdf, continuous=True, **kw)
    fig2 = sim.hist(expr, condexpr)
    return fit, fig2, fig1

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

def analdcr(d, datalist, sim):
    # get index of the filter length to use for pre-trigger pulses
    ptlength = d['ptlength']
    ilength = np.searchsorted(sim.filtlengths, ptlength)
    assert sim.filtlengths[ilength] == ptlength

    # compute dark count height cut
    mainsel = f'good&(mainpos>=0)&(length=={ptlength})'
    p01 = [
        sim.getexpr('median(mainheight)', f'{mainsel}&(mainnpe=={npe})')
        for npe in [0, 1]
    ]
    lmargin = 100
    rmargin = 500
    ptsel = f'(length=={ptlength})&(ptApos>={lmargin})&(ptApos<trigger-{rmargin})'
    cut, = afterpulse.maxdiff_boundaries(sim.getexpr('ptAamplh', ptsel), p01)
    d.update(dcrcut=cut, dcrlmargin=lmargin, dcrrmargin=rmargin)

    # plot a fingerplot
    boundaries = sim.computenpeboundaries(ilength)
    mainsel = f'where(mainampl>=0,~saturated,good)&(mainpos>=0)&(length=={ptlength})'
    fig = sim.hist('mainamplh', mainsel, 'log', 1000)
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    prefix = d['prefix']
    savef(fig, prefix)
    
    # plot pre-trigger amplitude vs. position
    fig = sim.scatter('ptApos', 'where(ptAamplh<1000,ptAamplh,-10)', f'length=={ptlength}')
    ax, = fig.get_axes()
    trigger = sim.getexpr('median(trigger)')
    vspan(ax, lmargin, trigger - rmargin)
    hspan(ax, cut)
    hlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot a histogram of pre-trigger peak height
    fig = sim.hist('ptAamplh', ptsel, 'log')
    ax, = fig.get_axes()
    vspan(ax, cut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # variables to compute the dark count
    l1pe, r1pe = boundaries[:2]
    sigcount = sim.getexpr(f'count_nonzero({ptsel}&(ptAamplh>{cut}))')
    lowercount = sim.getexpr(f'count_nonzero({mainsel}&(mainamplh<={cut})&(mainamplh>{l1pe}))')
    uppercount = sim.getexpr(f'count_nonzero({mainsel}&(mainamplh>{cut})&(mainamplh<{r1pe}))')
    d.update(dcrl1pe=l1pe, dcrr1pe=r1pe)
    
    # variables to compute the rate
    time = sim.getexpr(f'mean(trigger-{lmargin}-{rmargin})', ptsel)
    nevents = sim.getexpr(f'count_nonzero({ptsel})')
    totalevt = len(sim.output)
    
    # variables with uncertainties
    s = upoisson(sigcount)
    l = upoisson(lowercount)
    u = upoisson(uppercount)
    t = time * 1e-9 * ubinom(nevents, totalevt)
    d.update(dcrtime=t, dcrcount=s)
    
    # correction factor and DCR
    f = (l + u) / u
    r = f * s / t
    d.update(dcr=r, dcrfactor=f)
    
    # print DCR results
    print(f'pre-trigger count = {sigcount} / {nevents}')
    print(f'total time = {t} s')
    print(f'correction factor = {f}')
    print(f'dcr = {r} cps')
    
    # fit pre-trigger pe histogram
    fit, fig1, fig2 = fitpe(sim, 'ptAnpe', f'{ptsel}&(ptAnpe>0)', boundaries, 'borel')
    savef(fig1, prefix)
    savef(fig2, prefix)
    d.update(dcrfit=fit)
    
    # fit main peak pe histogram
    expr = 'where(mainpos>=0,mainnpe,take_along_axis(mainnpe,argmax(mainpos>=0,axis=0)[None],0))'
    sim.setvar('mainnpebackup', sim.getexpr(expr))
    mainsel = f'any(mainpos>=0,0)&(length=={ptlength})'
    fit, fig1, fig2 = fitpepoisson(sim, 'mainnpebackup', mainsel, boundaries, overflow=False)
    savef(fig1, prefix)
    savef(fig2, prefix)
    d.update(mainfit=fit)
    
    # plot <= 3 events with an high pre-trigger peak
    evts = sim.eventswhere(f'{ptsel}&(ptAamplh>{cut})')
    for ievt in evts[:3]:
        fig = sim.plotevent(datalist, ievt, ilength, zoom='all')
        savef(fig, prefix)

def analap(d, datalist, sim):
        
    # get index of the filter length to use for afterpulses
    aplength = d['aplength']
    ilength = np.searchsorted(sim.filtlengths, aplength)
    assert sim.filtlengths[ilength] == aplength

    # compute afterpulses height cut
    mainsel = f'good&(mainpos>=0)&(length=={aplength})'
    p01 = [
        sim.getexpr('median(mainheight)', f'{mainsel}&(mainnpe=={npe})')
        for npe in [0, 1]
    ]
    npe = d['npe']
    minornpe = f'(length=={aplength})&(apApos>=0)&(mainnpe=={npe})'
    dcut = d['dcut']
    dcutr = d['dcutr']
    apsel = f'{minornpe}&(apApos-mainpos>{dcut})&(apApos-mainpos<{dcutr})'
    hcut, = afterpulse.maxdiff_boundaries(sim.getexpr('apAapamplh', apsel), p01)
    hcut = np.round(hcut, 1)
    d.update(apcut=hcut)

    # plot a fingerplot
    boundaries = sim.computenpeboundaries(ilength)
    mainsel = f'where(mainampl>=0,~saturated,good)&(mainpos>=0)&(length=={aplength})'
    fig = sim.hist('mainamplh', mainsel, 'log', 1000)
    ax, = fig.get_axes()
    vspan(ax, hcut)
    vlines(ax, boundaries, linestyle=':')
    prefix = d['prefix']
    savef(fig, prefix)
    
    # plot afterpulses height vs. distance from main pulse
    fig = sim.scatter('apApos-mainpos', 'apAapamplh', minornpe)
    ax, = fig.get_axes()
    hspan(ax, hcut)
    vspan(ax, dcut, dcutr)
    hlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # plot selected afterpulses height histogram
    fig = sim.hist('apAapamplh', apsel, 'log')
    ax, = fig.get_axes()
    vspan(ax, hcut)
    vlines(ax, boundaries, linestyle=':')
    savef(fig, prefix)
    
    # counts for computing the afterpulse probability
    nevents = sim.getexpr(f'count_nonzero({minornpe})')
    apcond = f'{apsel}&(apAapamplh>{hcut})'
    apcount = sim.getexpr(f'count_nonzero({apcond})')
    apcount = ubinom(apcount, nevents)
    d.update(apcount=apcount, apnevents=nevents)
    
    # fit afterpulses pe histogram
    fit, fig1, fig2 = fitpe(sim, 'apAnpe', f'{apcond}&(apAnpe>0)', boundaries, 'borel', overflow=False)
    savef(fig1, prefix)
    savef(fig2, prefix)
    d.update(apfit=fit)

    # expected background from DCR
    time = (dcutr - dcut) * 1e-9 * nevents
    bkg = d['dcr'] * time
    d.update(apbkg=bkg, aptime=time)
    
    # fit decay constant of afterpulses
    f = bkg / apcount
    const = 1 / (dcutr - dcut) * f / (1 - f)
    nbins = int(15/20 * np.sqrt(gvar.mean(apcount)))
    tau0 = 2000
    bins = -tau0 * np.log1p(-np.linspace(0, 1 - np.exp(-(dcutr - dcut) / tau0), nbins + 1))
    fit, fig1, fig2 = fitapdecay(sim, f'apApos-mainpos-{dcut}', apcond, const, bins=bins)
    savef(fig1, prefix)
    savef(fig2, prefix)
    d.update(apfittau=fit)

    # plot some events with afterpulses
    apevts = sim.eventswhere(apcond)
    for ievt in apevts[:3]:
        fig = sim.plotevent(datalist, ievt, ilength)
        savef(fig, prefix)
        
    # correction factor for temporal selection
    tau = fit.palt['tau']
    if fit.Q < 0.01:
        tau = scalesdev(tau, np.sqrt(fit.chi2 / fit.dof))
    factor = 1 / exponinteg(dcut, dcutr, tau)
    d.update(aptau=tau, apfactor=factor)
    
    # afterpulse probability
    ccount = (apcount - bkg) * factor
    p = ccount / nevents
    d.update(ap=p, apccount=ccount)
    
    # print afterpulse results
    print(f'tau = {tau}')
    print(f'correction factor = {factor}')
    print(f'expected background = {bkg} counts in {time:.3g} s')
    print(f'ap count = {apcount} / {nevents}')
    print(f'corrected ap count = {ccount}')
    print(f'afterpulse probability = {p}')
    
gvar.switch_gvar()

for vov in vovdict:
    
    print(f'\n************** {vov} VoV **************')
    
    # load analysis file and skip if already saved
    prefix = f'{vov}VoV-'
    analfile = f'{savedir}/{prefix}archive.gvar'
    d = vovdict[vov]
    d['prefix'] = prefix
    d['analfile'] = analfile
    d['aplength'] = d.get('aplength', d['ptlength'])
    if os.path.exists(analfile):
        with open(analfile, 'rb') as file:
            print(f'load {analfile}...')
            vovdict[vov] = gvar.load(file)
        continue
    
    # do analysis
    datalist, sim = apload(vov)
    print('\n******* DCR *******')
    analdcr(d, datalist, sim)
    print('\n******* Afterpulses *******')
    analap(d, datalist, sim)
    
    # save analysis results to file
    with open(analfile, 'wb') as file:
        print(f'save {analfile}...')
        gvar.dump(d, file)

fig, axs = plt.subplots(2, 3, num='afterpulse_tile21', clear=True, figsize=[11, 7.1])

(axdcr, axap, axtau), (axpct, axnct, axmu) = axs

for ax in axs.flat:
    if ax.is_last_row():
        ax.set_xlabel('VoV')

axdcr.set_title('DCR')
axdcr.set_ylabel('Pre-trigger rate [cps]')

axap.set_title('Afterpulse')
axap.set_ylabel('Prob. of $\\geq$1 ap after 1 pe signal [%]')

axtau.set_title('Afterpulse')
axtau.set_ylabel('Exponential decay constant [ns]')

axpct.set_title('Cross talk')
axpct.set_ylabel('Prob. of > 1 pe [%]')

axnct.set_title('Cross talk')
axnct.set_ylabel('Average excess pe')

axmu.set_title('Efficiency')
axmu.set_ylabel('Average detected photons')

vov = np.array(list(vovdict))
def listdict(getter):
    return np.array([getter(d) for d in vovdict.values()])
dcr = listdict(lambda d: d['dcr'])
ap = listdict(lambda d: d['ap'])
tau = listdict(lambda d: d['aptau'])
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
getter = mugetter('mainfit', 'mu_poisson')
mup = listdict(lambda d: getter(d))

kw = dict(capsize=4, linestyle='', marker='.')
uerrorbar(axdcr, vov, dcr, **kw)
uerrorbar(axap, vov, ap * 100, **kw)
uerrorbar(axtau, vov, tau, **kw)
for i, (label, getter) in enumerate(mus):
    pct, nct = ct(getter)
    offset = 0.1 * (i/(len(mus)-1) - 1/2)
    kw.update(label=label)
    x = vov + offset
    uerrorbar(axpct, x, pct * 100, **kw)
    uerrorbar(axnct, x, nct, **kw)
uerrorbar(axmu, vov, mup, **kw)

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

# axdcr.plot(vov_fbk, dcr_factor * np.array(dcr_fbk), '.-', label='FBK')
# axdcr.plot(vov_lf, dcr_factor * np.array(dcr_lf), '.-', label='LF')
# axdcr.legend()

for ax in axs.flat:
    l, u = ax.get_ylim()
    ax.set_ylim(0, u)
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()
