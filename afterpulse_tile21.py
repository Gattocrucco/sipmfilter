import os
import glob
import re
import pickle
import collections
import functools
import itertools

import numpy as np
from scipy import stats, special
from matplotlib import pyplot as plt
import matplotlib
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
    margin = 1000 * (ylim[1] - ylim[0])
    if y0 is None:
        y0 = ylim[0] - margin
    if y1 is None:
        y1 = ylim[1] + margin
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
    margin = 1000 * (ylim[1] - ylim[0])
    if y0 is None:
        y0 = ylim[0] - margin
    if y1 is None:
        y1 = ylim[1] + margin
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
    mu = params['mu_borel']
    effmu = mu * n
    return np.exp(-effmu) * effmu ** (n - 1) / special.factorial(n)

def geompmf(k, params):
    # p is 1 - p respect to the conventional definition to match the borel mu
    p = params['p_geom']
    return p ** (k - 1) * (1 - p)

def genpoissonpmf(k, params):
    # P(k;mu,lambda) = mu (mu + k lambda)^(k - 1) exp(-(mu + k lambda)) / k!
    mu = params['mu_poisson']
    lamda = params['mu_borel']
    effmu = mu + k * lamda
    return np.exp(-effmu) * mu * effmu ** (k - 1) / special.factorial(k)

def geompoissonpmf_nv(k, params):
    # Nuel 2008, "Cumulative distribution function of a geometric Poisson distribution", proposition 7 (pag 5) 
    lamda = params['mu_poisson']
    theta = 1 - params['p_geom']
    z = lamda * theta / (1 - theta)
    P = [
        np.exp(-lamda),
        np.exp(-lamda) * (1 - theta) * z,
    ]
    assert int(k) == k, k
    k = int(k)
    for n in range(2, k + 1):
        t1 = (2 * n - 2 + z) / n * (1 - theta) * P[n - 1]
        t2 = (2 - n) / n * (1 - theta) ** 2 * P[n - 2]
        P.append(t1 + t2)
    assert len(P) == max(k, 1) + 1, (len(P), k)
    return P[k]

def geompoissonpmf(ks, params):
    out = None
    for i, k in np.ndenumerate(ks):
        r = geompoissonpmf_nv(k, params)
        if out is None:
            out = np.empty_like(ks, dtype=np.array(r).dtype)
        out[i] = r
    return out

def exponbkgcdf(x, params):
    scale = params['tau']
    const = params['const']
    if hasattr(scale, '__len__'):
        assert len(scale) == 2, len(scale)
        w = params['p1']
        return w * (1 - np.exp(-x / scale[0])) + (1 - w) * (1 - np.exp(-x / scale[1])) + const * x
    else:
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
def fithistogram(
    sim, expr, condexpr, prior, pmf_or_cdf,
    bins='auto',
    bins_overflow=None,
    continuous=False,
    fig=None,
    errorfactor=None,
    mincount=3,
    selection=True,
    boxloc='center right',
    fitlabel=None,
    **kw,
):
    """
    Fit an histogram.
    
    Parameters
    ----------
    sim : AfterPulse
    expr : str
    condexpr : str
        The sample is sim.getexpr(expr, condexpr).
    prior : (list of) array/dictionary of GVar
        Prior for the fit. If `prior` and `pmf_or_cdf` are lists, a fit is
        done for each pair in order.
    pmf_or_cdf : (list of) function
        The signature must be `pmf_or_cdf(x, params)` where `params` has the
        same format of `prior`.
    bins : int, sequence, str
        Passed to np.histogram. Default 'auto'.
    bins_overflow : sequence of two elements
        If None, do not fit the overflow. It is assumed that all the
        probability mass outside of the bins is contained in the overflow bin.
        The overflow bin must be to the right of the other bins.
    continuous : bool
        If False (default), pmf_or_cdf must be the probability distribution of
        an integer variable. If True, pmf_or_cdf must compute the cumulative
        density up to the given point.
    fig : matplotlib figure or axis, optional
        If provided, the plot is drawn here.
    errorfactor : 1D array, optional
        If provided, the errors on the bin counts are multiplied by these
        factors. If the array has different length than the number of bins,
        the factors are applied starting from the first element.
    mincount : int
        Bins are grouped starting from the last bin until there at least
        `mincount` elements in the last bin. (Applies to the overflow if
        present.)
    selection : bool
        If True (default), write `condexpr` on the plot.
    boxloc : str
        The location of the text box with the fit information. Default
        'center right'.
    fitlabel : (list of) str, optional
        Fit names used in the legend and text box.
    **kw :
        Additional keyword arguments are passed to lsqfit.nonlinear_fit.
    
    Return
    ------
    fit : (list of) lsqfit.nonlinear_fit
    fig : matplotlib figure
    """
    
    onefit = not hasattr(pmf_or_cdf, '__len__')
    if onefit:
        prior = [prior]
        pmf_or_cdf = [pmf_or_cdf]
        if fitlabel is not None:
            fitlabel = [fitlabel]
    
    # histogram
    sample = sim.getexpr(expr, condexpr)
    counts, bins = np.histogram(sample, bins)
    hasoverflow = bins_overflow is not None
    if hasoverflow:
        (overflow,), _ = np.histogram(sample, bins_overflow)
    else:
        overflow = 0
    
    if hasoverflow:
        # add bins to the overflow bin until it has at least `mincount` counts
        lencounts = len(counts)
        for i in range(len(counts) - 1, -1, -1):
            if overflow < mincount:
                overflow += counts[i]
                lencounts -= 1
            else:
                break
        counts = counts[:lencounts]
        bins = bins[:lencounts + 1]
    else:
        # group last bins until there are at least `mincount` counts
        while len(counts) > 1:
            if counts[-1] < mincount:
                counts[-2] += counts[-1]
                counts = counts[:-1]
                bins[-2] = bins[-1]
                bins = bins[:-1]
            else:
                break
    
    # check total count
    norm = np.sum(counts) + overflow
    assert norm == sim.getexpr(f'count_nonzero({condexpr})')
    
    # prepare data for fit
    ucounts = upoisson(counts)
    if errorfactor is not None:
        errorfactor = np.asarray(errorfactor)
        assert errorfactor.ndim == 1, errorfactor.ndim
        end = min(len(errorfactor), len(ucounts))
        ucounts[:end] = scalesdev(ucounts[:end], errorfactor[:end])
    y = dict(bins=ucounts)
    if hasoverflow:
        y.update(overflow=upoisson(overflow))

    # fit
    x = []
    fit = []
    for ifit in range(len(prior)):
        x.append(dict(
            pmf_or_cdf = pmf_or_cdf[ifit],
            continuous = continuous,
            bins = bins,
            norm = norm,
            hasoverflow = hasoverflow,
        ))
        fit.append(lsqfit.nonlinear_fit((x[ifit], y), fcn, prior[ifit], **kw))
    
    # get figure
    if hasattr(fig, 'get_axes'):
        ax = fig.subplots()
    else:
        ax = fig
        fig = ax.get_figure()
    
    # plot data
    center = (bins[1:] + bins[:-1]) / 2
    wbar = np.diff(bins) / 2
    kw = dict(linestyle='', capsize=4, marker='.', color='k')
    uerrorbar(ax, center, y['bins'], xerr=wbar, label='histogram', **kw)
    if hasoverflow:
        oc = center[-1] + 2 * wbar[-1]
        ys = y['overflow']
        uerrorbar(ax, oc, ys, label='overflow', **kw)

    # plot fit
    info = [
        f'N = {norm}'
    ]
    styles = [
        dict(color='#0004'),
        dict(color='#f004'),
    ]
    for ifit, style in zip(range(len(prior)), itertools.cycle(styles)):
        thisfit = fit[ifit]
        yfit = fcn(x[ifit], thisfit.palt)
        ys = yfit['bins']
        ym = np.repeat(gvar.mean(ys), 2)
        ysdev = np.repeat(gvar.sdev(ys), 2)
        xs = np.concatenate([bins[:1], np.repeat(bins[1:-1], 2), bins[-1:]])
        label = fitlabel[ifit] if fitlabel is not None else 'fit' if ifit == 0 else f'fit {ifit + 1}'
        style.update(zorder=1.9)
        ax.fill_between(xs, ym - ysdev, ym + ysdev, label=label, **style)
        if hasoverflow:
            ys = yfit['overflow']
            ym = np.full(2, gvar.mean(ys))
            ysdev = np.full(2, gvar.sdev(ys))
            xs = oc + 0.8 * (bins[-2:] - center[-1])
            ax.fill_between(xs, ym - ysdev, ym + ysdev, **style)
    
        # write fit results on the plot
        if fitlabel is not None or ifit > 0:
            space = 9 * ' '
            label = fitlabel[ifit] if fitlabel is not None else f'Fit {ifit + 1}:'
            info.append(space + label + space)
        info.append(f'chi2/dof (pv) = {thisfit.chi2/thisfit.dof:.2g} ({thisfit.Q:.2g})')
        p = thisfit.palt
        for k, v in p.items():
            m = p.extension_pattern.match(k)
            if m:
                f = p.extension_fcn.get(m.group(1), None)
                if f:
                    k = m.group(2)
                    v = f(v)
            if hasattr(v, '__len__'):
                for i, vv in enumerate(v):
                    info.append(f'{k}{i+1} = {vv}')
            else:
                info.append(f'{k} = {v}')
    
    textbox.textbox(ax, '\n'.join(info), loc=boxloc, fontsize='small')
    
    # decorations
    ax.legend(loc='upper right')
    ax.minorticks_on()
    ax.grid(which='major', linestyle='--')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(expr)
    ax.set_ylabel('Count per bin')
    if selection:
        cond = breaklines.breaklines(f'Selection: {condexpr}', 40, ')', '&|')
        textbox.textbox(ax, cond, loc='upper left', fontsize='small')
    fig.tight_layout()
    _, upper = ax.get_ylim()
    ax.set_ylim(0, upper)
    
    if onefit:
        fit, = fit
    return fit, fig

def intbins(min, max):
    return -0.5 + np.arange(min, max + 2)

def pebins(boundaries, start):
    return intbins(start, len(boundaries) - 1), intbins(1000, 1000)

gvar.BufferDict.uniform('U', 0, 1)

def _fitpe(sim, expr, condexpr, boundaries, pmf, prior, binstart, overflow, *, fig1, fig2, **kw):
    bins, bins_overflow = pebins(boundaries, binstart)
    value = sim.getexpr(expr)
    of = value >= 1000
    sim.setvar('overflow', of, overwrite=True)
    if overflow:
        top = 1 + np.max(value[~of])
        histexpr = f'where(overflow,{top},{expr})'
    else:
        bins_overflow = None
        condexpr = f'{condexpr}&~overflow'
        histexpr = expr
    fit, _ = fithistogram(sim, expr, condexpr, prior, pmf, bins, bins_overflow, fig=fig2, **kw)
    sim.hist(histexpr, condexpr, fig=fig1)
    return fit, fig1, fig2

@afterpulse.figmethod(figparams=['fig1', 'fig2'])
def fitpe(sim, expr, condexpr, boundaries, overflow=True, **kw):
    pmf = [
        borelpmf,
        geompmf,
    ]
    prior = [{
        'U(mu_borel)': gvar.BufferDict.uniform('U', 0, 1),
    }, {
        'U(p_geom)': gvar.BufferDict.uniform('U', 0, 1),
    }]
    return _fitpe(sim, expr, condexpr, boundaries, pmf, prior, 1, overflow, **kw)
    
@afterpulse.figmethod(figparams=['fig1', 'fig2'])
def fitpepoisson(sim, expr, condexpr, boundaries, overflow=True, **kw):
    pmf = [
        genpoissonpmf,
        geompoissonpmf,
    ]
    prior = [{
        'U(mu_borel)': gvar.BufferDict.uniform('U', 0, 1),
        'log(mu_poisson)': gvar.gvar(0, 1),
    }, {
        'U(p_geom)': gvar.BufferDict.uniform('U', 0, 1),
        'log(mu_poisson)': gvar.gvar(0, 1),
    }]
    return _fitpe(sim, expr, condexpr, boundaries, pmf, prior, 0, overflow, **kw)

@afterpulse.figmethod(figparams=['fig1', 'fig2'])
def fitapdecay(sim, expr, condexpr, const, *, fig1, fig2, **kw):
    pmf = [
        exponbkgcdf,
        exponbkgcdf,
    ]
    prior = [{
        'const': const,
        'log(tau)': gvar.gvar(np.log(1000), 1),
    }, {
        'const': const,
        'log(tau)': gvar.gvar(np.log([400, 800]), np.ones(2)),
        'p1': gvar.BufferDict.uniform('U', 0, 1)
    }]
    fit, _ = fithistogram(sim, expr, condexpr, prior, pmf, continuous=True, fig=fig2, **kw)
    sim.hist(expr, condexpr, fig=fig1)
    return fit, fig1, fig2

def scalesdev(x, f):
    return x + (f - 1) * (x - gvar.mean(x))

def figmethod(*args, figparams=['fig']):
    """
    Decorator for plotting methods of AfterPulseTile21.
    
    Assumes that the method requires a keyword argument `fig` which is a
    matplotlib figure. When `fig` is not provided or None, generate a figure
    with the method name as window title.
    
    If the original method returns None (or does not return), the decorated
    method returns the figure.
    """
    def decorator(meth):
        
        @functools.wraps(meth)
        def newmeth(self, *args, **kw):
            # TODO this should return the list of figures even when passed axes
            # maybe
            figs = []
            for i, param in enumerate(figparams):
                fig = kw.get(param, None)
                if fig is None:
                    title = meth.__qualname__
                    if len(figparams) > 1:
                        title += str(i + 1)
                    fig = plt.figure(num=title, clear=True)
                figs.append(fig)
                kw[param] = fig
            
            vovloc = kw.pop('vovloc', 'lower center')
            rt = meth(self, *args, **kw)
            
            for fig in figs:
                if hasattr(fig, 'get_axes'):
                    ax, = fig.get_axes()
                else:
                    ax = fig
                    fig = ax.get_figure()
                
                if 'event' not in fig.canvas.get_window_title() and kw.get('selection', True):
                    b, t = ax.get_ylim()
                    yscale = ax.get_yscale()
                    if yscale == 'log':
                        b = np.log(b)
                        t = np.log(t)
                    t += (t - b) / 9
                    if yscale == 'log':
                        b = np.exp(b)
                        t = np.exp(t)
                    ax.set_ylim(b, t)
                    fig.tight_layout()
                
                textbox.textbox(ax, f'{self.vov} VoV', fontsize='medium', loc=vovloc)
            
            return (fig if len(figparams) == 1 else tuple(figs)) if rt is None else rt
        
        return newmeth
    
    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    else:
        raise ValueError(len(args))

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
    _defaults = dict(dcut=300, dcutr=5500, npe=1, lmargin=100, rmargin=500)
    defaultparams = {
        5.5: dict(ptlength=128, **_defaults),
        7.5: dict(ptlength= 64, **_defaults),
        9.5: dict(ptlength= 64, **_defaults),
    }
    defaultparams[5.5].update(dcut=250)
    defaultparams[7.5].update(dcut=250)
    defaultparams[9.5].update(dcut=150)
    
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
        """
        A list of dictionaries with these keys:
        'wavfile' the source wav
        'simfile' the saved AfterPulse object file
        'templfile' the saved template file
        """
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
                    trigger = np.full(len(data), savetemplate.defaulttrigger(wavfile)),
                    filtlengths = [32, 64, 128, 256, 512, 1024, 2048],
                )
                sim = afterpulse.AfterPulse(data, template, **kw)

                print(f'save {simfile}...')
                sim.save(simfile)
    
    @functools.cached_property
    def sim(self):
        """
        The AfterPulse object.
        """
        print('load analysis files...')
        simlist = [
            afterpulse.AfterPulse.load(files['simfile'])
            for files in self.filelist
        ]
        return afterpulse.AfterPulse.concatenate(simlist)
    
    @functools.cached_property
    def datalist(self):
        """
        List of memory mapped source files.
        """
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
    
    @figmethod
    def ptfingerplot(self, fig):
        """plot a fingerplot"""
        self.sim.hist('mainamplh', self.ptmainsel, 'log', 1000, fig=fig)
        ax, = fig.get_axes()
        vspan(ax, self.ptcut)
        vlines(ax, self.ptboundaries, linestyle=':')
    
    @figmethod
    def ptscatter(self, fig, **kw):
        """plot pre-trigger amplitude vs. position"""
        ptlength = self.params['ptlength']
        self.sim.scatter('ptApos', 'where(ptAamplh<1000,ptAamplh,-10)', f'length=={ptlength}', fig=fig, **kw)
        ax, = fig.get_axes()
        trigger = self.sim.getexpr('median(trigger)')
        lmargin = self.params['lmargin']
        rmargin = self.params['rmargin']
        vspan(ax, lmargin, trigger - rmargin)
        hspan(ax, self.ptcut)
        hlines(ax, self.ptboundaries, linestyle=':')
    
    @figmethod
    def pthist(self, fig, **kw):
        """plot a histogram of pre-trigger peak height"""
        self.sim.hist('ptAamplh', self.ptsel, 'log', fig=fig, **kw)
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
        n = len(self.sim.output)
        self.results.update(ptnevents=n)
        return n
    
    @functools.cached_property
    def pttime(self):
        """total time where pre-trigger pulses are searched"""
        lmargin = self.params['lmargin']
        rmargin = self.params['rmargin']
        time = self.sim.getexpr(f'mean(trigger-{lmargin}-{rmargin})')
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
    
    @figmethod(figparams=['fig1', 'fig2'])
    def ptdict(self, *, fig1, fig2, **kw):
        """fit pre-trigger pe histogram"""
        fit, _, _ = fitpe(self.sim, 'ptAnpe', f'{self.ptsel}&(ptAamplh>{self.ptcut})', self.ptboundaries, fig1=fig1, fig2=fig2, **kw)
        label = 'ptfit'
        self.results.update(ptfit=fit[0], ptfitgeom=fit[1])
        return fit, fig1, fig2
    
    @figmethod(figparams=['fig1', 'fig2'])
    def maindict(self, overflow=False, fixzero=False, *, fig1, fig2, **kw):
        """fit main peak pe histogram"""
        ptlength = self.params['ptlength']
        mainsel = f'(mainposbackup>=0)&(length=={ptlength})'
        kwargs = dict(overflow=overflow, fig1=fig1, fig2=fig2)
        if fixzero:
            of = 'of' if overflow else ''
            fitnormal = self.results['mainfit' + of]
            factor = np.sqrt(fitnormal.chi2 / fitnormal.dof)
            kwargs.update(errorfactor=[1 / factor])
        kwargs.update(kw)
        fit, _, _ = fitpepoisson(self.sim, 'mainnpebackup', mainsel, self.ptboundaries, **kwargs)
        for i, kind in enumerate(['', 'geom']):
            label = 'mainfit'
            if overflow:
                label += 'of'
            label += kind
            if fixzero:
                label += 'fz'
            self.results[label] = fit[i]
        return fit, fig1, fig2
    
    @figmethod
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
        return f'(length=={aplength})&(apApos>=0)&(mainnpebackup=={npe})'
    
    @functools.cached_property
    def apsel(self):
        """afterpulse event selection (time cut, but random still included)"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        return f'{self.appresel}&(apApos-mainposbackup>{dcut})&(apApos-mainposbackup<{dcutr})'

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

    @figmethod
    def apfingerplot(self, *, fig):
        """plot a fingerplot"""
        aplength = self.params['aplength']
        mainsel = f'where(mainampl>=0,~saturated,good)&(mainpos>=0)&(length=={aplength})'
        self.sim.hist('mainamplh', mainsel, 'log', 1000, fig=fig)
        ax, = fig.get_axes()
        vspan(ax, self.apcut)
        vlines(ax, self.apboundaries, linestyle=':')
    
    @figmethod
    def apscatter(self, *, fig, **kw):
        """plot afterpulses height vs. distance from main pulse"""
        self.sim.scatter('apApos-mainposbackup', 'apAapamplh', self.appresel, fig=fig, **kw)
        ax, = fig.get_axes()
        hspan(ax, self.apcut)
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        vspan(ax, dcut, dcutr)
        hlines(ax, self.apboundaries, linestyle=':')
    
    @figmethod
    def aphist(self, *, fig, **kw):
        """plot selected afterpulses height histogram"""
        self.sim.hist('apAapamplh', self.apsel, 'log', fig=fig, **kw)
        ax, = fig.get_axes()
        vspan(ax, self.apcut)
        vlines(ax, self.apboundaries, linestyle=':')
    
    @functools.cached_property
    def apcond(self):
        """afterpulse selection"""
        return f'{self.apsel}&(apAapamplh>{self.apcut})'
    
    @figmethod(figparams=['fig1', 'fig2'])
    def apdict(self, overflow=False, *, fig1, fig2, **kw):
        """fit afterpulses pe histogram"""
        kwargs = dict(overflow=overflow, fig1=fig1, fig2=fig2, mincount=5)
        kwargs.update(**kw)
        fit, _, _ = fitpe(self.sim, 'apAnpe', self.apcond, self.apboundaries, **kwargs)
        for i, kind in enumerate(['', 'geom']):
            label = 'apfit'
            if overflow:
                label += 'of'
            label += kind
            self.results[label] = fit[i]
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
    
    @figmethod(figparams=['fig1', 'fig2'])
    def apfittau(self, *, fig1, fig2, **kw):
        """fit decay constant of afterpulses"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        f = self.apbkg / gvar.mean(self.apcount)
        const = 1 / (dcutr - dcut) * f / (1 - f)
        nbins = int(15/20 * np.sqrt(gvar.mean(self.apcount)))
        tau0 = 1500
        bins = -tau0 * np.log1p(-np.linspace(0, 1 - np.exp(-(dcutr - dcut) / tau0), nbins + 1))
        fit, _, _ = fitapdecay(self.sim, f'apApos-mainposbackup-{dcut}', self.apcond, const, bins=bins, fig1=fig1, fig2=fig2, **kw)
        self.results.update(apfittau=fit[0], apfittau2=fit[1])
        return fit, fig1, fig2
    
    def _apbkgfit(self, fit):
        x = fit.data[0]
        bins = x['bins']
        t = bins[-1] - bins[0]
        c = fit.palt['const']
        ct = c * t
        count = x['norm']
        bkg = count * ct / (1 + ct)
        return bkg
    
    @functools.cached_property
    def apbkgfit(self):
        fit = self.results['apfittau']
        b = self._apbkgfit(fit)
        self.results.update(apbkgfit=b)
        return b
    
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
        ccount = (self.apcount - self.apbkgfit) * self.apfactor
        self.results.update(apccount=ccount)
        return ccount
    
    @functools.cached_property
    def approb(self):
        """afterpulse probability"""
        p = self.apccount / self.apnevents
        self.results.update(approb=p)
        return p
    
    @functools.cached_property
    def apbkgfit2(self):
        fit = self.results['apfittau2']
        b = self._apbkgfit(fit)
        self.results.update(apbkgfit2=b)
        return b
        
    @functools.cached_property
    def aptau2(self):
        """decay parameters of afterpulses with two components"""
        fit = self.results['apfittau2']
        tau = fit.palt['tau']
        if fit.Q < 0.01:
            tau = scalesdev(tau, np.sqrt(fit.chi2 / fit.dof))
        self.results.update(aptau2=tau)
        return tau
    
    @functools.cached_property
    def apfactor2(self):
        """factor to keep into account temporal cuts"""
        dcut = self.params['dcut']
        dcutr = self.params['dcutr']
        tau1, tau2 = self.aptau2
        fit = self.results['apfittau2']
        w = fit.palt['p1']
        denom = w * exponinteg(dcut, dcutr, tau1)
        denom += (1 - w) * exponinteg(dcut, dcutr, tau2)
        factor = 1 / denom
        self.results.update(apfactor2=factor)
        return factor
    
    @functools.cached_property
    def apccount2(self):
        """afterpulse count corrected for temporal cuts and background"""
        ccount = (self.apcount - self.apbkgfit2) * self.apfactor2
        self.results.update(apccount2=ccount)
        return ccount
    
    @functools.cached_property
    def approb2(self):
        """afterpulse probability"""
        p = self.apccount2 / self.apnevents
        self.results.update(approb2=p)
        return p
    
    @figmethod
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
    
    @property
    def lastnamenoext(self):
        assert self.count > 0
        return f'{self.prefix}fig{self.count:02d}'
    
    def savefit(self, fit):
        onefit = not hasattr(fit, '__len__')
        if onefit:
            fit = [fit]
        path = f'{self.lastnamenoext}.txt'
        with open(path, 'w') as file:
            print(f'write {path}...')
            for i, f in enumerate(fit):
                if not onefit:
                    file.write(f'*********** Fit {i + 1} ***********\n\n')
                file.write(f.format(maxline=True))
                if i + 1 < len(fit):
                    file.write('\n')

    def __call__(self, fig):
        self.count += 1
        path = f'{self.lastnamenoext}.png'
        print(f'save {path}...')
        fig.savefig(path)

def singlevovanalysis(vov):
    print(f'\n************** {vov} VoV **************')

    # load analysis file and skip if already saved
    prefix = f'{vov}VoV-'
    analfile = f'{AfterPulseTile21.savedir}/{prefix}archive.gvar'
    if os.path.exists(analfile):
        print(f'load {analfile}...')
        return AfterPulseTile21.loadresults(analfile)
    
    savef = FigureSaver(f'{AfterPulseTile21.savedir}/{prefix}')
    anal = AfterPulseTile21(vov)
    
    print('\n******* Pre-trigger *******')
    
    savef(anal.ptfingerplot())
    savef(anal.ptscatter())
    savef(anal.pthist())

    fit, fig1, fig2 = anal.ptdict()
    savef(fig1)
    savef(fig2)
    savef.savefit(fit)
    
    for overflow in [False, True]:
        for fixzero in [False, True]:
            fit, fig1, fig2 = anal.maindict(overflow=overflow, fixzero=fixzero)
            savef(fig1)
            savef(fig2)
            savef.savefit(fit)
    
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
        fit, fig1, fig2 = anal.apdict(overflow=overflow, mincount=5)
        savef(fig1)
        savef(fig2)
        savef.savefit(fit)
    
    fit, fig1, fig2 = anal.apfittau()
    savef(fig1)
    savef(fig2)
    savef.savefit(fit)
    
    for i in range(3):
        savef(anal.apevent(i))

    print(f'tau = {anal.aptau}')
    print(f'correction factor = {anal.apfactor}')
    print(f'expected background = {anal.apbkg} counts in {anal.aptime:.3g} s')
    print(f'ap count = {anal.apcount} / {anal.apnevents}')
    print(f'corrected ap count = {anal.apccount}')
    print(f'afterpulse probability = {anal.approb}')
    print(f'afterpulse probability (two taus) = {anal.approb2}')
    
    # save analysis results to file
    print(f'save {analfile}...')
    anal.saveresults(analfile)
    return anal.results

def allvovanalysis():
    vovdict = {}
    for vov in AfterPulseTile21.defaultparams:
        vovdict[vov] = singlevovanalysis(vov)
    return vovdict

def plottermethod(meth):
    
    @functools.wraps(meth)
    def newmeth(self, *args, **kw):
        
        if len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
        elif 'ax' in kw:
            ax = kw['ax']
        else:
            fig, ax = plt.subplots(num=meth.__qualname__, clear=True)
            args = (ax,) + args
        
        meth(self, *args, **kw)
         
        _, u = ax.get_ylim()
        ax.set_ylim(0, u)
        ax.minorticks_on()
        ax.grid(True, 'major', linestyle='--')
        ax.grid(True, 'minor', linestyle=':')
        
        fig = ax.get_figure()
        return fig
    
    return newmeth

class Plotter:
    
    def __init__(self, vovdict):
        self.vovdict = vovdict
        self.errorbarkw = dict(capsize=4, linestyle='', marker='.')
        self.vov = np.array(list(vovdict))
    
    def errorbar(self, ax, uy, offset=0, **kw):
        kwargs = dict(self.errorbarkw)
        kwargs.update(kw)
        uerrorbar(ax, self.vov + offset, uy, **kwargs)
        
    def listdict(self, getter):
        return np.array([getter(d) for d in self.vovdict.values()])
    
    @staticmethod
    def paramgetter(fitkey, param):
        def getter(d):
            fit = d[fitkey]
            mu = fit.palt[param]
            if fit.Q < 0.01:
                factor = np.sqrt(fit.chi2 / fit.dof)
                mu = scalesdev(mu, factor)
            return mu
        return getter
    
    @plottermethod
    def plotptrate(self, ax):
        ptrate = self.listdict(lambda d: d['ptrate'])
        self.errorbar(ax, ptrate)
    
    @plottermethod
    def plotapprob(self, ax):
        approb = self.listdict(lambda d: d['approb'])
        self.errorbar(ax, approb * 100, -0.05, label='Single exponential', color='#f55')
        approb = self.listdict(lambda d: d['approb2'])
        self.errorbar(ax, approb * 100,  0.05, label='Double exponential', color='black')
    
    @plottermethod
    def plotaptau(self, ax):
        aptau = self.listdict(lambda d: d['aptau'])
        self.errorbar(ax, aptau, label='Single exponential', color='#f55')
        tau1, tau2 = self.listdict(lambda d: d['aptau2']).T
        self.errorbar(ax, tau1, label='Double expon., short comp.', color='black', marker='')
        self.errorbar(ax, tau2, label='Double expon., long comp.', color='black', marker='.')
    
    @plottermethod
    def plotapweight(self, ax):
        w = self.listdict(self.paramgetter('apfittau2', 'p1'))
        self.errorbar(ax, w, color='black', marker='')
    
    @plottermethod
    def plotmupoisson(self, ax, overflow=True):
        of = 'of' if overflow else ''
        mup = self.listdict(self.paramgetter(f'mainfit{of}', 'mu_poisson'))
        self.errorbar(ax, mup, -0.05, label='Borel', color='black')
        mup = self.listdict(self.paramgetter(f'mainfit{of}geom', 'mu_poisson'))
        self.errorbar(ax, mup,  0.05, label='Geometric', color='#f55')
    
    @plottermethod
    def plotdictparam(self, ax, transf=(lambda x: x), transfgeom=None, overflow=True):
        
        of = 'of' if overflow else ''
        
        muborel = [
            ('Random', self.paramgetter( 'ptfit'      , 'mu_borel')),
            ('Laser' , self.paramgetter(f'mainfit{of}', 'mu_borel')),
            ('AP'    , self.paramgetter(f'apfit{of}'  , 'mu_borel')),
        ]
        mugeom = [
            ('Random', self.paramgetter( 'ptfitgeom'      , 'p_geom')),
            ('Laser' , self.paramgetter(f'mainfit{of}geom', 'p_geom')),
            ('AP'    , self.paramgetter(f'apfit{of}geom'  , 'p_geom')),
        ]
        
        for j, (mus, color, f) in enumerate(zip([muborel, mugeom], ['#000', '#f55'], [transf, transf if transfgeom is None else transfgeom])):
            for i, ((label, getter), marker) in enumerate(zip(mus, ['>', '', '.'])):
                param = self.listdict(getter)
                param = f(param)
                n = len(mus) - 1
                N = 1
                offset = 0.15 * ((n + 1) * (j - N/2) + (i - n/2))
                self.errorbar(ax, param, offset, label=label, color=color, marker=marker)
        
        ax.legend(
            fontsize = 'small',
            ncol = 2,
            title = 1 * ' ' + 'Borel' + 14 * ' ' + 'Geometric',
            title_fontsize = 'small',
            columnspacing = 0.4,
            handletextpad = 0.2,
        )
    
    def plotdictprob(self, ax, **kw):
        self.plotdictparam(ax, lambda mu: 100 * (1 - np.exp(-mu)), transfgeom=lambda p: 100 * p, **kw)
    
    def plotdictpe(self, ax, **kw):
        self.plotdictparam(ax, lambda mu: 1 / (1 - mu) - 1, **kw)
    
def main():
    vovdict = allvovanalysis()
    plotter = Plotter(vovdict)
    
    fig1,  axdcr                = plt.subplots(      num='afterpulse_tile21-1', clear=True, figsize=[ 3.7, 3.5])
    fig2, (axap, axtau, axw)    = plt.subplots(1, 3, num='afterpulse_tile21-2', clear=True, figsize=[10.5, 3.5])
    fig3, (axpar, axpct, axnct) = plt.subplots(1, 3, num='afterpulse_tile21-3', clear=True, figsize=[ 9.0, 3.0])
    fig4,  axmu                 = plt.subplots(      num='afterpulse_tile21-4', clear=True, figsize=[ 3.7, 3.5])

    figs = [fig1, fig2, fig3, fig4]

    for fig in figs:
        for ax in fig.get_axes():
            if ax.is_last_row():
                ax.set_xlabel('VoV')

    axdcr.set_title('Random pulses rate')
    axdcr.set_ylabel('Pre-trigger rate [cps]')
    plotter.plotptrate(axdcr)

    axap.set_title('AP probability')
    axap.set_ylabel('Prob. of $\\geq$1 ap after 1 pe signal [%]')
    plotter.plotapprob(axap)
    axap.legend()

    axtau.set_title('AP tau')
    axtau.set_ylabel('Exponential decay constant [ns]')
    plotter.plotaptau(axtau)
    axtau.legend()
    
    axw.set_title('AP mixture')
    axw.set_ylabel('Weight of short component')
    plotter.plotapweight(axw)
    axw.set_ylim(0, 1)
    
    axpar.set_title('DiCT param')
    axpar.set_ylabel(f'Branching parameter')
    plotter.plotdictparam(axpar)

    axpct.set_title('DiCT prob')
    axpct.set_ylabel('Prob. of > 1 pe [%]')
    plotter.plotdictprob(axpct)

    axnct.set_title('DiCT pe')
    axnct.set_ylabel('Average excess pe')
    plotter.plotdictpe(axnct)

    axmu.set_title('Efficiency')
    axmu.set_ylabel('Average detected laser photons')
    plotter.plotmupoisson(axmu)
    axmu.legend()

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
        fig.tight_layout()
        fig.show()

if __name__ == '__main__':
    
    gvar.switch_gvar()
    
    main()
