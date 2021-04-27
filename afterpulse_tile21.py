import os
import glob
import re
import pickle
import collections
import functools

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
import savetemplate

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

@afterpulse.figmethod
def fithistogram(sim, expr, condexpr, prior, pmf_or_cdf, bins='auto', bins_overflow=None, continuous=False, fig=None, **kw):
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
    fig : matplotlib figure, optional
        If provided, the plot is draw here.
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
    ax = fig.subplots()
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
        ys = yfit['overflow']
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
    ax.minorticks_on()
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

def pebins(boundaries, start):
    return intbins(start, len(boundaries) - 1), intbins(1000, 1000)

def getkind(kind):
    if kind == 'borel':
        return borelpmf, 'mu'
    elif kind == 'geom':
        return geompmf, 'p'
    else:
        raise KeyError(kind)

gvar.BufferDict.uniform('U', 0, 1)

def _fitpe(sim, expr, condexpr, boundaries, pmf, prior, binstart, overflow, *, fig1, fig2, **kw):
    bins, bins_overflow = pebins(boundaries, binstart)
    if overflow:
        histexpr = f'where({expr}<1000,{expr},1+max({expr}[{expr}<1000]))'
    else:
        bins_overflow = None
        condexpr = f'{condexpr}&({expr}<1000)'
        histexpr = expr
    fit, _ = fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow, fig=fig2, **kw)
    sim.hist(histexpr, condexpr, fig=fig1)
    return fit, fig1, fig2

@afterpulse.figmethod(figparams=['fig1', 'fig2'])
def fitpe(sim, expr, condexpr, boundaries, kind='borel', overflow=True, **kw):
    pmf, param = getkind(kind)
    prior = {
        f'U({param})': gvar.BufferDict.uniform('U', 0, 1),
    }
    return _fitpe(sim, expr, condexpr, boundaries, pmf, prior, 1, overflow, **kw)
    
@afterpulse.figmethod(figparams=['fig1', 'fig2'])
def fitpepoisson(sim, expr, condexpr, boundaries, overflow=True, **kw):
    prior = {
        'U(mu_borel)': gvar.BufferDict.uniform('U', 0, 1),
        'log(mu_poisson)': gvar.gvar(0, 1),
    }
    return _fitpe(sim, expr, condexpr, boundaries, genpoissonpmf, prior, 0, overflow, **kw)

@afterpulse.figmethod(figparams=['fig1', 'fig2'])
def fitapdecay(sim, expr, condexpr, const, *, fig1, fig2, **kw):
    prior = {
        'log(tau)' : gvar.gvar(np.log(1000), 1),
        'const' : const,
    }
    fit, _ = fithistogram(sim, expr, condexpr, prior, exponbkgcdf, continuous=True, fig=fig2, **kw)
    sim.hist(expr, condexpr, fig=fig1)
    return fit, fig1, fig2

def scalesdev(x, f):
    return x + (f - 1) * (x - gvar.mean(x))

class AfterPulseTile21:
    
    savedir = 'afterpulse_tile21'

    wavfiles = [
        'darksidehd/LF_TILE21_77K_54V_65VoV_1.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_2.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_3.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_4.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_5.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_6.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_7.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_8.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_9.wav',
        'darksidehd/LF_TILE21_77K_54V_65VoV_10.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_1.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_2.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_3.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_4.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_5.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_6.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_7.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_8.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_9.wav',
        'darksidehd/LF_TILE21_77K_54V_69VoV_10.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_1.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_2.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_3.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_4.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_5.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_6.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_7.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_8.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_9.wav',
        'darksidehd/LF_TILE21_77K_54V_73VoV_10.wav',
    ]
    
    # parameters:
    # ptlength = length used for pre-trigger region
    # aplength = filter length for afterpulse (default same as ptlength)
    # dcut, dcutr = left/right cut on minorpos - mainpos
    # npe = pe of main peak to search afterpulses
    # lmargin = left cut on ptpos
    # rmargin = right cut (from right margin) on ptpos
    _defaults = dict(dcut=500, dcutr=5500, npe=1, lmargin=100, rmargin=500)
    defaultparams = {
        5.5: dict(ptlength=128, **_defaults),
        7.5: dict(ptlength= 64, **_defaults),
        9.5: dict(ptlength= 64, **_defaults),
    }
    
    def __init__(self, vov, params={}):
        self.vov = vov
        self.params = dict(self.defaultparams.get(vov, {}))
        self.params.update(params)
        self.params.setdefault('aplength', self.params['ptlength'])
        self.results = {}
        self.maketemplates()
        self.processdata()
    
    @functools.cached_property
    def filelist(self):
        filelist = []
        
        for wavfile in self.wavfiles:
                
            _, name = os.path.split(wavfile)
            prefix, _ = os.path.splitext(name)

            vbreak, vbias = re.search(r'(\d+)V_(\d+)VoV', prefix).groups()
            vov = (int(vbias) - int(vbreak)) / 2
            
            if vov == self.vov:
                filelist.append(dict(
                    wavfile = wavfile,
                    simfile = f'{self.savedir}/{prefix}.npz',
                    templfile = savetemplate.templatepath(wavfile),
                ))
        
        if len(filelist) == 0:
            raise ValueError(f'no files for vov {vov}')
        
        return filelist

    def maketemplates(self):
        for files in self.filelist:
            
            wavfile = files['wavfile']
            templfile = files['templfile']
            
            if not os.path.exists(templfile):
                savetemplate.savetemplate(wavfile, plot='save')
    
    def processdata(self):
        for files in self.filelist:
            
            wavfile = files['wavfile']
            simfile = files['simfile']
            templfile = files['templfile']
            
            if not os.path.exists(simfile):
                directory, _ = os.path.split(simfile)
                os.makedirs(directory, exist_ok=True)
                
                data = readwav.readwav(wavfile)

                template = _template.Template.load(templfile)
    
                kw = dict(
                    batch = 100,
                    pbar = True,
                    trigger = np.full(len(data), savetemplate.defaulttrigger())
                )
                sim = afterpulse.AfterPulse(data, template, **kw)

                print(f'save {simfile}...')
                sim.save(simfile)
    
    @functools.cached_property
    def sim(self):
        print('load analysis files...')
        simlist = [
            afterpulse.AfterPulse.load(files['simfile'])
            for files in self.filelist
        ]
        return afterpulse.AfterPulse.concatenate(simlist)
    
    @functools.cached_property
    def datalist(self):
        return [
            readwav.readwav(files['wavfile'])
            for files in self.filelist
        ]
    
    @functools.cached_property
    def ptilength(self):
        """get index of the filter length to use for pre-trigger pulses"""
        ptlength = self.params['ptlength']
        ilength = np.searchsorted(self.sim.filtlengths, ptlength)
        assert self.sim.filtlengths[ilength] == ptlength
        return ilength
    
    @functools.cached_property
    def ptsel(self):
        lmargin = self.params['lmargin']
        rmargin = self.params['rmargin']
        ptlength = self.params['ptlength']
        return f'(length=={ptlength})&(ptApos>={lmargin})&(ptApos<trigger-{rmargin})'
    
    @functools.cached_property
    def ptcut(self):
        """compute dark count height cut"""
        ptlength = self.params['ptlength']
        mainsel = f'good&(mainpos>=0)&(length=={ptlength})'
        p01 = [
            self.sim.getexpr('median(mainheight)', f'{mainsel}&(mainnpe=={npe})')
            for npe in [0, 1]
        ]
        cut, = afterpulse.maxdiff_boundaries(self.sim.getexpr('ptAamplh', self.ptsel), p01)
        self.results.update(ptcut=cut)
        return cut
    
    @functools.cached_property
    def ptboundaries(self):
        return self.sim.computenpeboundaries(self.ptilength)
    
    @functools.cached_property
    def ptmainsel(self):
        ptlength = self.params['ptlength']
        return f'where(mainampl>=0,~saturated,good)&(mainpos>=0)&(length=={ptlength})'
    
    @afterpulse.figmethod
    def ptfingerplot(self, fig):
        """plot a fingerplot"""
        self.sim.hist('mainamplh', self.ptmainsel, 'log', 1000, fig=fig)
        ax, = fig.get_axes()
        vspan(ax, self.ptcut)
        vlines(ax, self.ptboundaries, linestyle=':')
    
    @afterpulse.figmethod
    def ptscatter(self, fig):
        """plot pre-trigger amplitude vs. position"""
        ptlength = self.params['ptlength']
        self.sim.scatter('ptApos', 'where(ptAamplh<1000,ptAamplh,-10)', f'length=={ptlength}', fig=fig)
        ax, = fig.get_axes()
        trigger = self.sim.getexpr('median(trigger)')
        lmargin = self.params['lmargin']
        rmargin = self.params['rmargin']
        vspan(ax, lmargin, trigger - rmargin)
        hspan(ax, self.ptcut)
        hlines(ax, self.ptboundaries, linestyle=':')
    
    @afterpulse.figmethod
    def pthist(self, fig):
        """plot a histogram of pre-trigger peak height"""
        self.sim.hist('ptAamplh', self.ptsel, 'log', fig=fig)
        ax, = fig.get_axes()
        vspan(ax, self.ptcut)
        vlines(ax, self.ptboundaries, linestyle=':')
        
    @functools.cached_property
    def ptfactor(self):
        """correction factor for the rate to keep into account truncation of 1
        pe distribution"""
        l1pe, r1pe = self.ptboundaries[:2]
        lowercount = self.sim.getexpr(f'count_nonzero({self.ptmainsel}&(mainamplh<={self.ptcut})&(mainamplh>{l1pe}))')
        uppercount = self.sim.getexpr(f'count_nonzero({self.ptmainsel}&(mainamplh>{self.ptcut})&(mainamplh<{r1pe}))')
        l = upoisson(lowercount)
        u = upoisson(uppercount)
        f = (l + u) / u
        self.results.update(ptl1pe=l1pe, ptr1pe=r1pe, ptfactor=f)
        return f
    
    @functools.cached_property
    def ptnevents(self):
        nevents = self.sim.getexpr(f'count_nonzero({self.ptsel})')
        totalevt = len(self.sim.output)
        n = ubinom(nevents, totalevt)
        self.results.update(ptnevents=n)
        return n
    
    @functools.cached_property
    def pttime(self):
        """total time where pre-trigger pulses are searched"""
        lmargin = self.params['lmargin']
        rmargin = self.params['rmargin']
        time = self.sim.getexpr(f'mean(trigger-{lmargin}-{rmargin})', self.ptsel)
        t = time * 1e-9 * self.ptnevents
        self.results.update(pttime=t)
        return t
    
    @functools.cached_property
    def ptcount(self):
        sigcount = self.sim.getexpr(f'count_nonzero({self.ptsel}&(ptAamplh>{self.ptcut}))')
        s = upoisson(sigcount)
        self.results.update(ptcount=s)
        return s
    
    @functools.cached_property
    def ptrate(self):
        """rate of pre-trigger pulses"""
        r = self.ptfactor * self.ptcount / self.pttime
        self.results.update(ptrate=r)
        return r
    
    @afterpulse.figmethod(figparams=['fig1', 'fig2'])
    def ptdict(self, kind='borel', *, fig1, fig2):
        """fit pre-trigger pe histogram"""
        fit, _, _ = fitpe(self.sim, 'ptAnpe', f'{self.ptsel}&(ptAnpe>0)', self.ptboundaries, kind, fig1=fig1, fig2=fig2)
        label = 'ptfit'
        if kind == 'geom':
            label += kind
        self.results[label] = fit
        return fit, fig1, fig2
    
    def defmainnpebackup(self):
        name = 'mainnpebackup'
        if name not in self.sim._variables:
            expr = 'where(mainpos>=0,mainnpe,take_along_axis(mainnpe,argmax(mainpos>=0,axis=0)[None],0))'
            value = self.sim.getexpr(expr)
            self.sim.setvar(name, value)
    
    @afterpulse.figmethod(figparams=['fig1', 'fig2'])
    def maindict(self, overflow=False, *, fig1, fig2):
        """fit main peak pe histogram"""
        self.defmainnpebackup()
        ptlength = self.params['ptlength']
        mainsel = f'any(mainpos>=0,0)&(length=={ptlength})'
        fit, _, _ = fitpepoisson(self.sim, 'mainnpebackup', mainsel, self.ptboundaries, overflow=overflow, fig1=fig1, fig2=fig2)
        label = 'mainfit'
        if overflow:
            label += 'of'
        self.results[label] = fit
        return fit, fig1, fig2
    
    @afterpulse.figmethod
    def ptevent(self, index=0, *, fig):
        """plot an event with an high pre-trigger peak"""
        evts = self.sim.eventswhere(f'{self.ptsel}&(ptAamplh>{self.ptcut})')
        ievt = evts[index]
        self.sim.plotevent(self.datalist, ievt, self.ptilength, zoom='all', fig=fig)
            
    @functools.cached_property
    def apilength(self):    
        """get index of the filter length to use for afterpulses"""
        aplength = self.params['aplength']
        ilength = np.searchsorted(self.sim.filtlengths, aplength)
        assert self.sim.filtlengths[ilength] == aplength
        return ilength
    
    @functools.cached_property
    def appresel(self):
        """afterpulse preselection (laser pe = 1 and other details)"""
        npe = self.params['npe']
        aplength = self.params['aplength']
        return f'(length=={aplength})&(apApos>=0)&(mainnpe=={npe})'
    
    @functools.cached_property
    def apsel(self):
        """afterpulse event selection (time cut, but random still included)"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        return f'{self.appresel}&(apApos-mainpos>{dcut})&(apApos-mainpos<{dcutr})'

    @functools.cached_property
    def apcut(self):
        """compute afterpulses height cut"""
        aplength = self.params['aplength']
        mainsel = f'good&(mainpos>=0)&(length=={aplength})'
        p01 = [
            self.sim.getexpr('median(mainheight)', f'{mainsel}&(mainnpe=={npe})')
            for npe in [0, 1]
        ]
        cut, = afterpulse.maxdiff_boundaries(self.sim.getexpr('apAapamplh', self.apsel), p01)
        cut = np.round(cut, 1)
        self.results.update(apcut=cut)
        return cut
    
    @functools.cached_property
    def apboundaries(self):
        """pe boundaries for afterpulse analysis"""
        return self.sim.computenpeboundaries(self.apilength)

    @afterpulse.figmethod
    def apfingerplot(self, *, fig):
        """plot a fingerplot"""
        aplength = self.params['aplength']
        mainsel = f'where(mainampl>=0,~saturated,good)&(mainpos>=0)&(length=={aplength})'
        self.sim.hist('mainamplh', mainsel, 'log', 1000, fig=fig)
        ax, = fig.get_axes()
        vspan(ax, self.apcut)
        vlines(ax, self.apboundaries, linestyle=':')
    
    @afterpulse.figmethod
    def apscatter(self, *, fig):
        """plot afterpulses height vs. distance from main pulse"""
        self.sim.scatter('apApos-mainpos', 'apAapamplh', self.appresel, fig=fig)
        ax, = fig.get_axes()
        hspan(ax, self.apcut)
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        vspan(ax, dcut, dcutr)
        hlines(ax, self.apboundaries, linestyle=':')
    
    @afterpulse.figmethod
    def aphist(self, *, fig):
        """plot selected afterpulses height histogram"""
        self.sim.hist('apAapamplh', self.apsel, 'log', fig=fig)
        ax, = fig.get_axes()
        vspan(ax, self.apcut)
        vlines(ax, self.apboundaries, linestyle=':')
    
    @functools.cached_property
    def apcond(self):
        """afterpulse selection"""
        return f'{self.apsel}&(apAapamplh>{self.apcut})'
    
    @afterpulse.figmethod(figparams=['fig1', 'fig2'])
    def apdict(self, overflow=False, kind='borel', *, fig1, fig2):
        """fit afterpulses pe histogram"""
        fit, _, _ = fitpe(self.sim, 'apAnpe', f'{self.apcond}&(apAnpe>0)', self.apboundaries, kind=kind, overflow=overflow, fig1=fig1, fig2=fig2)
        label = 'apfit'
        if overflow:
            label += 'of'
        self.results[label] = fit
        return fit, fig1, fig2
    
    @functools.cached_property
    def apnevents(self):
        nevents = self.sim.getexpr(f'count_nonzero({self.appresel})')
        self.results.update(apnevents=nevents)
        return nevents
    
    @functools.cached_property
    def apcount(self):
        """count for computing the afterpulse probability"""
        apcount = self.sim.getexpr(f'count_nonzero({self.apcond})')
        apcount = ubinom(apcount, self.apnevents)
        self.results.update(apcount=apcount)
        return apcount
    
    @functools.cached_property
    def aptime(self):
        """total time in afterpulse selection"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        time = (dcutr - dcut) * 1e-9 * self.apnevents
        self.results.update(aptime=time)
        return time
    
    @functools.cached_property
    def apbkg(self):
        """expected background from random pulses"""
        bkg = self.ptrate * self.aptime
        self.results.update(apbkg=bkg)
        return bkg
    
    @afterpulse.figmethod(figparams=['fig1', 'fig2'])
    def apfittau(self, *, fig1, fig2):
        """fit decay constant of afterpulses"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        f = self.apbkg / self.apcount
        const = 1 / (dcutr - dcut) * f / (1 - f)
        nbins = int(15/20 * np.sqrt(gvar.mean(self.apcount)))
        tau0 = 2000
        bins = -tau0 * np.log1p(-np.linspace(0, 1 - np.exp(-(dcutr - dcut) / tau0), nbins + 1))
        fit, _, _ = fitapdecay(self.sim, f'apApos-mainpos-{dcut}', self.apcond, const, bins=bins, fig1=fig1, fig2=fig2)
        self.results.update(apfittau=fit)
        return fit, fig1, fig2
    
    @functools.cached_property
    def aptau(self):
        """decay parameter of afterpulses"""
        fit = self.results['apfittau']
        tau = fit.palt['tau']
        if fit.Q < 0.01:
            tau = scalesdev(tau, np.sqrt(fit.chi2 / fit.dof))
        self.results.update(aptau=tau)
        return tau
    
    @functools.cached_property
    def apfactor(self):
        """factor to keep into account temporal cuts"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        factor = 1 / exponinteg(dcut, dcutr, self.aptau)
        self.results.update(apfactor=factor)
        return factor
    
    @functools.cached_property
    def apccount(self):
        """afterpulse count corrected for temporal cuts and background"""
        ccount = (self.apcount - self.apbkg) * self.apfactor
        self.results.update(apccount=ccount)
        return ccount
    
    @functools.cached_property
    def approb(self):
        """afterpulse probability"""
        p = self.apccount / self.apnevents
        self.results.update(approb=p)
        return p
    
    @afterpulse.figmethod
    def apevent(self, index, *, fig):
        """plot an event with a selected afterpulse"""
        evts = self.sim.eventswhere(self.apcond)
        ievt = evts[index]
        self.sim.plotevent(self.datalist, ievt, self.apilength, fig=fig)
    
    def saveresults(self, path):
        with open(path, 'wb') as file:
            self.results.update(params=self.params)
            gvar.dump(self.results, file)
    
    @staticmethod
    def loadresults(path):
        with open(path, 'rb') as file:
            return gvar.load(file)
                    
class FigureSaver:
    
    def __init__(self, prefix):
        self.count = 0
        self.prefix = prefix

    def __call__(self, fig):
        self.count += 1
        path = f'{self.prefix}fig{self.count:02d}.png'
        print(f'save {path}...')
        fig.savefig(path)

if __name__ == '__main__':
    
    gvar.switch_gvar()
    
    vovdict = {}
    
    for vov in AfterPulseTile21.defaultparams:
    
        print(f'\n************** {vov} VoV **************')
    
        # load analysis file and skip if already saved
        prefix = f'{vov}VoV-'
        analfile = f'{AfterPulseTile21.savedir}/{prefix}archive.gvar'
        if os.path.exists(analfile):
            print(f'load {analfile}...')
            vovdict[vov] = AfterPulseTile21.loadresults(analfile)
            continue
        
        savef = FigureSaver(f'{AfterPulseTile21.savedir}/{prefix}')
        anal = AfterPulseTile21(vov)
        
        print('\n******* Pre-trigger *******')
        
        savef(anal.ptfingerplot())
        savef(anal.ptscatter())
        savef(anal.pthist())
    
        for kind in ['borel', 'geom']:
            _, fig1, fig2 = anal.ptdict(kind=kind)
            savef(fig1)
            savef(fig2)
        
        for overflow in [False, True]:
            _, fig1, fig2 = anal.maindict(overflow=overflow)
            savef(fig1)
            savef(fig2)
        
        for i in range(3):
            savef(anal.ptevent(i))
        
        print(f'pre-trigger count = {anal.ptcount} / {anal.ptnevents}')
        print(f'total time = {anal.pttime} s')
        print(f'correction factor = {anal.ptfactor}')
        print(f'pre-trigger rate = {anal.ptrate} cps')
        
        print('\n******* Afterpulses *******')
        
        savef(anal.apfingerplot())
        savef(anal.apscatter())
        savef(anal.aphist())
        
        for overflow in [False, True]:
            _, fig1, fig2 = anal.apdict(overflow=overflow)
            savef(fig1)
            savef(fig2)
        
        _, fig1, fig2 = anal.apfittau()
        savef(fig1)
        savef(fig2)
        
        for i in range(3):
            savef(anal.apevent(i))
    
        print(f'tau = {anal.aptau}')
        print(f'correction factor = {anal.apfactor}')
        print(f'expected background = {anal.apbkg} counts in {anal.aptime:.3g} s')
        print(f'ap count = {anal.apcount} / {anal.apnevents}')
        print(f'corrected ap count = {anal.apccount}')
        print(f'afterpulse probability = {anal.approb}')
        
        # save analysis results to file
        print(f'save {analfile}...')
        anal.saveresults(analfile)
        vovdict[vov] = anal.results

    fig1,  axdcr                = plt.subplots(      num='afterpulse_tile21-1', clear=True, figsize=[    3.7, 3.5])
    fig2, (axap, axtau)         = plt.subplots(1, 2, num='afterpulse_tile21-2', clear=True, figsize=[2 * 3.7, 3.5])
    fig3, (axmub, axpct, axnct) = plt.subplots(1, 3, num='afterpulse_tile21-3', clear=True, figsize=[   10.5, 3.5])
    fig4,  axmu                 = plt.subplots(      num='afterpulse_tile21-4', clear=True, figsize=[    3.7, 3.5])

    figs = [fig1, fig2, fig3, fig4]

    for fig in figs:
        for ax in fig.get_axes():
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
    axmu.set_ylabel('Average detected laser photons')

    axmub.set_title('Cross talk')
    axmub.set_ylabel('Branching parameter $\\mu_B$')

    vov = np.array(list(vovdict))
    
    def listdict(getter):
        return np.array([getter(d) for d in vovdict.values()])
    
    ptrate = listdict(lambda d: d['ptrate'])
    approb = listdict(lambda d: d['approb'])
    aptau = listdict(lambda d: d['aptau'])
    
    def ct(mugetter):
        pct = listdict(lambda d: 1 - np.exp(-mugetter(d)))
        nct = listdict(lambda d: 1 / (1 - mugetter(d)) - 1)
        return pct, nct
    
    def paramgetter(fitkey, param):
        def getter(d):
            fit = d[fitkey]
            mu = fit.palt[param]
            if fit.Q < 0.01:
                factor = np.sqrt(fit.chi2 / fit.dof)
                mu = scalesdev(mu, factor)
            return mu
        return getter
    
    mus = [
        ('Dark count' , paramgetter('ptfit'  , 'mu'      )),
        ('Laser'      , paramgetter('mainfit', 'mu_borel')),
        ('Afterpulses', paramgetter('apfit'  , 'mu'      )),
    ]
    
    mup = listdict(paramgetter('mainfit', 'mu_poisson'))

    kw = dict(capsize=4, linestyle='', marker='.')
    uerrorbar(axdcr, vov, ptrate, **kw)
    uerrorbar(axap,  vov, approb * 100, **kw)
    uerrorbar(axtau, vov, aptau, **kw)
    uerrorbar(axmu,  vov, mup, **kw)
    
    for i, (label, getter) in enumerate(mus):
        mub = listdict(getter)
        pct, nct = ct(getter)
        offset = 0.1 * (i/(len(mus)-1) - 1/2)
        kw.update(label=label)
        x = vov + offset
        uerrorbar(axpct, x, pct * 100, **kw)
        uerrorbar(axnct, x, nct, **kw)
        uerrorbar(axmub, x, mub, **kw)

    axmub.legend()
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

    for fig in figs:
        for ax in fig.get_axes():
            l, u = ax.get_ylim()
            ax.set_ylim(0, u)
            ax.minorticks_on()
            ax.grid(True, 'major', linestyle='--')
            ax.grid(True, 'minor', linestyle=':')
        fig.tight_layout()

    for fig in figs:
        fig.show()
