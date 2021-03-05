"""
Module to search afterpulses in an LNGS file.

Classes
-------
AfterPulse : the main class of the module

Functions
---------
correlate : compute the cross correlation
firstbelowthreshold : to find the trigger leading edge
maxprominencedip : search the minimum with maximum negative prominence
"""

import time

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt, colors
import numba

import toy
import runsliced
import readwav
from single_filter_analysis import single_filter_analysis
import textbox

def correlate(waveform, template, method='fft', axis=-1, boundary=None):
    """
    Compute the cross correlation of two arrays.
    
    The correlation is computed with padding to the right but not to the left.
    So the first element of the cross correlation is
    
        sum(waveform[:len(template)] * template),
    
    while the last is
    
        waveform[-1] * template[0].
    
    Parameters
    ----------
    waveform : array (..., N, ...)
        The non-inverted term of the convolution.
    template : array (M,)
        The inverted term of the convolution.
    method : {'fft', 'oa'}
        Use fft (default) or overlap-add to compute the convolution.
    axis : int
        The axis of `waveform` along which the convolution is computed, default
        last.
    boundary : scalar, optional
        The padding value for `waveform`. If not specified, use the last value
        in each subarray.
    
    Return
    ------
    corr : array (..., N, ...)
        The cross correlation, with the same shape as `waveform`.
    """
    npad = len(template) - 1
    padspec = [(0, 0)] * len(waveform.shape)
    padspec[axis] = (0, npad)
    if boundary is None:
        padkw = dict(mode='edge')
    else:
        padkw = dict(mode='constant', constant_values=boundary)
    waveform_padded = np.pad(waveform, padspec, **padkw)
    
    idx = [None] * len(waveform.shape)
    idx[axis] = slice(None, None, -1)
    template_bc = template[tuple(idx)]
    
    funcs = dict(oa=signal.oaconvolve, fft=signal.fftconvolve)    
    return funcs[method](waveform_padded, template_bc, mode='valid', axes=axis)

def test_correlate():
    """
    Plot a test of `correlate`.
    """
    waveform = 1 - np.pad(np.ones(100), 100)
    waveform += 0.2 * np.random.randn(len(waveform))
    template = np.exp(-np.linspace(0, 3, 50))
    corr1 = correlate(np.repeat(waveform[None, :], 2, 0), template / np.sum(template), 'oa' , axis=1)[0, :]
    corr2 = correlate(np.repeat(waveform[:, None], 2, 1), template / np.sum(template), 'fft', axis=0)[:, 0]
    
    fig, ax = plt.subplots(num='afterpulse.test_correlate', clear=True)
    
    ax.plot(waveform, label='waveform')
    ax.plot(template, label='template')
    ax.plot(corr1, label='corr oa')
    ax.plot(corr2, label='corr fft', linestyle='--')
    
    ax.legend()
    
    fig.tight_layout()
    fig.show()

def timecorr(lenwaveform, lentemplate, method, n=100):
    """
    Time `correlate`.
    
    Parameters
    ----------
    lenwaveform : int
        Length of each waveform.
    lentemplate : int
        Length of the template.
    method : {'fft', 'oa'}
        Algorithm.
    n : int
        Number of waveforms, default 100.
    
    Return
    ------
    time : scalar
        The time, in seconds, taken by `correlate`.
    """
    waveform = np.random.randn(n, lenwaveform)
    template = np.random.randn(lentemplate)
    start = time.time()
    correlate(waveform, template, method)
    end = time.time()
    return end - start

def timecorrseries(lenwaveform, lentemplates, n=100):
    """
    Call `timecorr` for a range of values.
    
    Parameters
    ----------
    lenwaveform : int
        The length of each waveform.
    lentemplate : int
        The length of the template.
    n : int
        The number of waveforms, default 100.
    
    Return
    ------
    times : dict
        Dictionary of dictionaries with layout method -> (lentemplate -> time).
    """
    return {
        method: {
            lentemplate: timecorr(lenwaveform, lentemplate, method)
            for lentemplate in lentemplates
        } for method in ['oa', 'fft']
    }

def plot_timecorrseries(timecorrseries_output):
    """
    Plot the output of `timecorrseries`.
    """
    fig, ax = plt.subplots(num='afterpulse.plot_timecorrseries', clear=True)
    
    for method, time in timecorrseries_output.items():
        ax.plot(list(time.keys()), list(time.values()), label=method)
    
    ax.legend()
    ax.set_xlabel('Template length')
    ax.set_ylabel('Time')
    
    fig.tight_layout()
    fig.show()

@numba.njit(cache=True)
def firstbelowthreshold(events, threshold):
    """
    Find the first element below a threshold in arrays.
    
    Parameters
    ----------
    events : array (nevents, N)
        The arrays.
    threshold : scalar
        The threshold. The comparison is strict.
    
    Return
    ------
    pos : int array (nevents,)
        The index in each event of the first element below `threshold`.
    """
    output = np.full(len(events), -1)
    for ievent, event in enumerate(events):
        for isample, sample in enumerate(event):
            if sample < threshold:
                output[ievent] = isample
                break
    return output

@numba.njit(cache=True)
def maxprominencedip(events, start, top):
    """
    Find the negative peak with the maximum prominence in arrays.
    
    The prominence is measured only to the left of the peak.
    
    Parameters
    ----------
    events : array (nevents, N)
        The arrays.
    start : int array (nevents,)
        Each row of `events` is used only from the sample specified by
        `start` (inclusive).
    top : array (nevents,)
        For computing the prominence, maxima are capped at `top`.
    
    Return
    ------
    position : int array (nevents,)
        The position of the peak.
    prominence : int array (nevents,)
        The prominence of the peak.
    """
    prominence = np.zeros(len(events), events.dtype)
    position = np.full(len(events), -1)
    for ievent, event in enumerate(events):
        
        maxprom = -2 ** 20
        maxprompos = 0
        relminpos = -1
        for i in range(start[ievent] + 1, len(event) - 1):
            
            if event[i - 1] > event[i] < event[i + 1]:
                # narrow local minimum
                relmin = True
                relminpos = i
            elif event[i - 1] > event[i] == event[i + 1]:
                # possibly beginning of wide local minimum
                relminpos = i
            elif event[i - 1] == event[i] < event[i + 1] and relminpos >= 0:
                # end of wide local minimum
                relmin = True
                relminpos = (relminpos + i) // 2
            else:
                relminpos = -1
            
            if relmin:
                irev = relminpos
                maximum = event[irev]
                while irev >= start[ievent] and event[irev] >= event[relminpos] and maximum < top[ievent]:
                    if event[irev] > maximum:
                        maximum = event[irev]
                    irev -= 1

                maximum = min(maximum, top[ievent])
                prom = maximum - event[relminpos]
                if prom > maxprom:
                    maxprom = prom
                    maxprompos = relminpos
                
                relmin = False
                relminpos = -1
        
        prominence[ievent] = maxprom
        position[ievent] = maxprompos
    
    return position, prominence

def test_maxprominencedip():
    """
    Plot a random test of `maxprominencedip`.
    """
    t = np.linspace(0, 1, 1000)
    mu = np.random.uniform(0, 1, 20)
    logsigma = np.random.randn(len(mu))
    sigma = 0.2 * np.exp(logsigma)
    wf = -np.sum(np.exp(-1/2 * ((t[:, None] - mu) / sigma) ** 2), axis=-1)
    start = 500
    pos, prom = maxprominencedip(wf[None], start)
    
    fig, ax = plt.subplots(num='afterpulse.test_maxprominencedip', clear=True)
    
    ax.plot(wf)
    ax.axvline(start, linestyle='--')
    if np.all(pos >= 0):
        ax.vlines(pos, wf[pos], wf[pos] + prom)
        ax.axhline(wf[pos] + prom)
    
    fig.tight_layout()
    fig.show()

class AfterPulse(toy.NpzLoad):
    
    def __init__(self, wavdata, template, filtlengths=None, batch=10, pbar=False):
        """
        Search afterpulses in LNGS laser data.
        
        Parameters
        ----------
        wavdata : array (nevents, 2, 15001)
            Data as returned by readwav.readwav(). The first channel is the
            waveform, the second channel is the trigger.
        template : toy.Template
            A template object used for the cross correlation filter. Should be
            generated using the same wav file.
        filtlengths : array, optional
            A series of template lengths for the cross correlation filter, in
            unit of 1 GSa/s samples. If not specified, a logarithmically spaced
            range from 32 ns to 2048 ns is used.
        batch : int
            The events batch size, default 10.
        pbar : bool
            If True, print a progressbar during the computation (1 tick per
            batch). Default False.
        
        Methods
        -------
        fingerplot : plot a histogram of main peak height.
        plotevent : plot the analysis of a single event.
        getexpr : compute the value of an expression on all events.
        eventswhere : get the list of events satisfying a condition.
        hist : plot the histogram of a variable.
        scatter : plot two variables.
        hist2d : plot the histogram of two variables.
        
        Members
        -------
        filtlengths : array
            The cross correlation filter lengths.
        output : array (nevents,)
            A structured array containing the information for each event with
            these fields:
            'trigger' : int
                The index of the trigger leading edge.
            'baseline' : float
                The mean of the pre-trigger region.
            'mainpeak' : array filtlengths.shape
                A structured field for the signal identified by the trigger
                for each filter length with subfields:
                'pos' : int
                    The peak position.
                'height' : float
                    The peak height (positive) relative to the baseline.
            'minorpeak' : array filtlengths.shape
                A structured field for the maximum prominence peak after the
                main peak for each filter length with these fields:
                'pos' : int
                    The peak position.
                'height' : float
                    The peak height (positive) relative to the baseline.
                'prominence' : float
                    The prominence, computed only looking at samples to the
                    left of the peak, only back to the main peak position
                    (even if it is lower than the secondary peak), and capping
                    maxima to the baseline.
            'npe' : int
                The number of photoelectron associated to the main peak,
                estimated using the longest filter.
            'internals' : structured field
                Internals used by the object.
            'done' : bool
                True.
        templates : array filtlengths.shape
            A structured array containing the cross correlation filter
            templates with these fields:
            'template' : 1D array
                The template, padded to the right with zeros.
            'length' : int
                The length of the nonzero part of 'template'.
            'offset' : float
                The number of samples from the trigger to the beginning of
                the truncated template.
        """
        
        if filtlengths is None:
            self.filtlengths = np.logspace(5, 11, 2 * (11 - 5) + 1, base=2).astype(int)
        else:
            self.filtlengths = np.array(filtlengths, int)
        
        self.output = np.empty(len(wavdata), dtype=[
            ('trigger', int),
            ('baseline', float),
            ('mainpeak', [
                ('pos', int),
                ('height', float),
            ], self.filtlengths.shape),
            ('minorpeak', [
                ('pos', int),
                ('height', float),
                ('prominence', float),
            ], self.filtlengths.shape),
            ('npe', int),
            ('internals', [
                ('left', int, self.filtlengths.shape),
                ('right', int, self.filtlengths.shape),
                ('start', int),
            ]),
            ('done', bool),
        ])
        
        self.output['npe'] = -1
        self.output['done'] = False
        
        self._maketemplates(template)
        
        func = lambda s: self._run(wavdata[s], self.output[s])
        runsliced.runsliced(func, len(wavdata), batch, pbar)
        
        self._computenpe()
    
    def _maketemplates(self, template):
        """
        Fill the `templates` member.
        """
        self.templates = np.zeros(self.filtlengths.shape, dtype=[
            ('template', float, np.max(self.filtlengths)),
            ('length', int),
            ('offset', int),
        ])
        self.templates['length'] = self.filtlengths
        
        for entry in self.templates:
            templ, offset = template.matched_filter_template(entry['length'], timebase=1, aligned=True)
            assert len(templ) == entry['length']
            entry['template'][:len(templ)] = templ
            entry['length'] = len(templ)
            entry['offset'] = offset
    
    def _run(self, wavdata, output):
        """
        Process a batch of events, filling `output`.
        """
        trigger = firstbelowthreshold(wavdata[:, 1], 600)
        assert np.all(trigger >= 0)
        startoffset = 10
        start = np.min(trigger) - startoffset
        
        baseline = np.mean(wavdata[:, 0, :start], axis=-1)
        mean_baseline = np.mean(baseline)
        
        for ientry, entry in np.ndenumerate(self.templates):
            templ = entry['template'][:entry['length']]
            offset = entry['offset']
            
            filtered = correlate(wavdata[:, 0, start:], templ, boundary=mean_baseline)
            
            center = int(offset) + startoffset
            margin = 100
            left = max(0, center - margin)
            right = center + margin
            mainpos = left + np.argmin(filtered[:, left:right], axis=-1)
            mainheight = filtered[np.arange(len(mainpos)), mainpos]
            
            minorpos, minorprom = maxprominencedip(filtered, mainpos, baseline)
            minorheight = filtered[np.arange(len(minorpos)), minorpos]
            
            idx = (slice(None),) + ientry
            
            mainpeak_out = output['mainpeak'][idx]
            mainpeak_out['pos'] = mainpos + start
            mainpeak_out['height'] = baseline - mainheight
            
            minorpeak_out = output['minorpeak'][idx]
            minorpeak_out['pos'] = minorpos + start
            minorpeak_out['height'] = baseline - minorheight
            minorpeak_out['prominence'] = minorprom
            
            output['internals']['left'][idx] = left
            output['internals']['right'][idx] = right
        
        output['trigger'] = trigger
        output['baseline'] = baseline
        output['internals']['start'] = start
        output['done'] = True
    
    def _computenpe(self):
        """
        Compute the number of photoelectrons from a fingerplot.
        """
        ilength_flat = np.argmax(self.filtlengths)
        ilength = np.unravel_index(ilength_flat, self.filtlengths.shape)
        value = self.output['mainpeak']['height'][(slice(None),) + ilength]
        
        _, center, _ = single_filter_analysis(value, return_full=True)
        
        bins = (center[1:] + center[:-1]) / 2
        npe = np.digitize(value, bins)
        self.output['npe'] = npe
    
    def fingerplot(self):
        """
        Plot the histogram of the main peak height.
        
        The height is computed with the longest filter.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        ilength_flat = np.argmax(self.filtlengths)
        ilength = np.unravel_index(ilength_flat, self.filtlengths.shape)
        length = self.filtlengths[ilength]
        value = self.output['mainpeak']['height'][(slice(None),) + ilength]

        fig = plt.figure(num='afterpulse.AfterPulse.fingerplot', clear=True)
        
        single_filter_analysis(value, fig1=fig)
        
        ax, = fig.get_axes()
        textbox.textbox(ax, f'filter length = {length} ns', loc='center right', fontsize='small')
        
        fig.tight_layout()
        return fig
    
    def plotevent(self, wavdata, ievent, ilength):
        """
        Plot a single event.
        
        Parameters
        ----------
        wavdata : array (nevents, 2, 15001)
            The same array passed at initialization.
        ievent : int
            The event index.
        ilength : int
            The index of the filter length in `filtlengths`.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.plotevent', clear=True)
        
        wf = wavdata[ievent, 0]
        ax.plot(wf, color='#f55')
        
        entry = self.output[ievent]
        ax.axvline(entry['trigger'], color='#000', linestyle='--', label='trigger')
        
        baseline = entry['baseline']
        ax.axhline(baseline, color='#000', linestyle=':', label='baseline')
        
        template = self.templates[ilength]
        templ = template['template'][:template['length']]
        start = entry['internals']['start']
        filtered = correlate(wf[start:], templ, boundary=baseline)
        length = self.filtlengths[ilength]
        ax.plot(start + np.arange(len(filtered)), filtered, color='#000', label=f'filtered ({length} ns template)')
        
        left = entry['internals']['left'][ilength]
        right = entry['internals']['right'][ilength]
        ax.axvspan(start + left, start + right, color='#ddd', label='main peak search range')
        
        mainpeak = entry['mainpeak'][ilength]
        mainpos = mainpeak['pos']
        mainheight = mainpeak['height']
        npe = entry['npe']
        base = filtered[mainpos - start]
        ax.vlines(mainpos, base, entry['baseline'], zorder=2.1)
        markerkw = dict(linestyle='', markersize=10, markeredgecolor='#000', markerfacecolor='#fff0')
        ax.plot(mainpos, base, marker='o', label=f'main peak, h={mainheight:.1f}, npe={npe}', **markerkw)
        
        minorpeak = entry['minorpeak'][ilength]
        minorpos = minorpeak['pos']
        minorheight = minorpeak['height']
        minorprom = minorpeak['prominence']
        base = filtered[minorpos - start]
        ax.vlines(minorpos, base, base + minorprom, zorder=2.1)
        ax.hlines(base + minorprom, minorpos - 500, minorpos + 100, zorder=2.1)
        ax.plot(minorpos, base, marker='s', label=f'minor peak, h={minorheight:.1f}, prom={minorprom:.1f}', **markerkw)
        
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        
        ax.legend(fontsize='small', loc='lower right')
        textbox.textbox(ax, f'Event {ievent}', fontsize='medium', loc='upper center')
        
        ax.set_xlim(start - 500, len(wf) - 1)
        ax.set_xlabel('Sample number @ 1 GSa/s')
        
        fig.tight_layout()
        return fig

    def getexpr(self, expr, allow_numpy=True):
        """
        Evaluate an expression on all events.
        
        The expression can be any python expression involving the following
        numpy arrays:
        
            mainpos     : the index of the main peak
            mainheight  : the positive height of the main peak
            minorpos    : the index of the maximum promince peak after the main
            minorheight : ...its height
            minorprom   : ...its prominence, capped to the baseline
            npe         : the number of photoelectrons of the main peak
            trigger     : the index of the trigger leading edge
            baseline    : the value of the baseline
            length      : the cross correlation filter template length
        
        All arrays have shape (nevents,) + filtlengths.shape, although some
        of them do not vary over all axes.
        
        Parameters
        ----------
        expr : str
            The expression.
        allow_numpy : bool
            If True (default), allow numpy functions in the expression.
        
        Return
        ------
        value :
            The evaluated expression.
        """
        globals = {}
        if allow_numpy:
            globals.update({
                k: v
                for k, v in vars(np).items()
                if not k.startswith('_')
                and not k[0].isupper()
            })
        def broadcasted(x):
            x = x.reshape((-1,) + (1,) * len(self.filtlengths.shape))
            return np.broadcast_to(x, self.output['mainpeak'].shape)
        variables = dict(
            mainpos     = self.output['mainpeak']['pos'],
            mainheight  = self.output['mainpeak']['height'],
            minorpos    = self.output['minorpeak']['pos'],
            minorheight = self.output['minorpeak']['height'],
            minorprom   = self.output['minorpeak']['prominence'],
            npe         = broadcasted(self.output['npe']),
            trigger     = broadcasted(self.output['trigger']),
            baseline    = broadcasted(self.output['baseline']),
            length      = np.broadcast_to(self.filtlengths, self.output['mainpeak'].shape),
        )
        for k, v in variables.items():
            v = v.view()
            v.flags['WRITEABLE'] = False
            globals[k] = v
        return eval(expr, globals)
    
    def eventswhere(self, cond):
        """
        List the events satisfying a condition.

        For conditions that depends on filter length, the condition must be
        satisfied for at least one length.
        
        The condition must evaluate to a boolean array with shape
        (nevents,) + filtlengths.shape. If not, the behaviour is undefined.
        
        Parameters
        ----------
        cond : str
            A python expression. See the method `getexpr` for an explanation.
        
        Return
        ------
        indices : int array
            The events indices.
        """
        mask = self.getexpr(cond, allow_numpy=False)
        mask = np.any(mask, axis=tuple(range(1, len(mask.shape))))
        return np.flatnonzero(mask)
    
    def hist(self, expr, where=None, yscale='linear'):
        """
        Plot the histogram of an expression.
        
        The values are histogrammed separately for each filter length.
        
        The expression must evaluate to an array with shape (nevents,) +
        filtlengths.shape. If not, the behaviour is undefined.
        
        Parameters
        ----------
        expr : str
            A python expression. See the method `getexpr` for an explanation.
        where : str
            An expression for a boolean condition to select the values of
            `expr`.
        yscale : str
            The y scale of the plot, default 'linear'.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        values = self.getexpr(expr)
        if where is not None:
            cond = self.getexpr(where)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.hist', clear=True)
        
        for ilength, length in np.ndenumerate(self.filtlengths):
            idx = (slice(None),) + ilength
            x = values[idx]
            if where is not None:
                x = x[cond[idx]]
            if len(x) > 0:
                iflat = np.ravel_multi_index(ilength, self.filtlengths.shape)
                histkw = dict(
                    bins = self._binedges(x),
                    histtype = 'step',
                    color = '#600',
                    label = f'{length} ({len(x)})',
                    zorder = 2,
                    alpha = (1 + iflat) / self.filtlengths.size,
                )
                ax.hist(x, **histkw)
        
        ax.legend(title='Filter length (events)', fontsize='small', ncol=2, loc='upper right')
        if where is not None:
            textbox.textbox(ax, f'Selection:\n{where}', fontsize='small', loc='upper left')
        
        ax.set_xlabel(expr)
        ax.set_ylabel('Counts per bin')
        
        ax.set_yscale(yscale)
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        
        fig.tight_layout()
        return fig

    def scatter(self, xexpr, yexpr, where=None):
        """
        Plot the scatterplot of two expressions.
        
        The values are separated by filter length.
        
        The expressions must evaluate to arrays with shape (nevents,) +
        filtlengths.shape. If not, the behaviour is undefined.
        
        Parameters
        ----------
        xexpr : str
            A python expression for the x coordinate. See the method `getexpr`
            for an explanation.
        yexpr : str
            Expression for the y coordinate.
        where : str
            An expression for a boolean condition to select the values of
            `xexpr` and `yexpr`.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        xvalues = self.getexpr(xexpr)
        yvalues = self.getexpr(yexpr)
        if where is not None:
            cond = self.getexpr(where)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.scatter', clear=True)
        
        for ilength, length in np.ndenumerate(self.filtlengths):
            idx = (slice(None),) + ilength
            x = xvalues[idx]
            y = yvalues[idx]
            if where is not None:
                x = x[cond[idx]]
                y = y[cond[idx]]
            if len(x) > 0:
                iflat = np.ravel_multi_index(ilength, self.filtlengths.shape)
                plotkw = dict(
                    linestyle = '',
                    marker = '.',
                    color = '#600',
                    label = f'{length} ({len(x)})',
                    alpha = (1 + iflat) / self.filtlengths.size,
                )
                ax.plot(x, y, **plotkw)
        
        ax.legend(title='Filter length (events)', fontsize='small', ncol=2, loc='upper right')
        if where is not None:
            textbox.textbox(ax, f'Selection:\n{where}', fontsize='small', loc='upper left')
        
        ax.set_xlabel(xexpr)
        ax.set_ylabel(yexpr)
        
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        
        fig.tight_layout()
        return fig
    
    def _binedges(self, x, maxnbins='auto'):
        """
        Compute histogram bin edges for the array x.
        
        If the data type is a float, the edges are computed with the method
        'auto' of numpy.
        
        If the data type is integral, the edges are aligned to half-integer
        values. The number of bins is capped to `maxnbins`. If `maxnbins` is
        'auto' (default), it is set to the number of bins numpy would use, but
        at least 10.
        """
        bins = np.histogram_bin_edges(x, bins='auto')
        if np.issubdtype(x.dtype, np.integer):
            newbins = np.arange(np.min(x), np.max(x) + 2) - 0.5
            if maxnbins == 'auto':
                maxnbins = max(10, len(bins) - 1)
            if len(newbins) - 1 > maxnbins:
                p = int(np.ceil((len(newbins) - 1) / maxnbins))
                newbins = newbins[:-1:p]
                bins = np.pad(newbins, (0, 1), constant_values=newbins[-1] + newbins[1] - newbins[0])
                
        return bins

    def hist2d(self, xexpr, yexpr, where=None, log=True):
        """
        Plot the 2D histogram of two expressions.
        
        All filter lengths are histogrammed together.
        
        The expressions must evaluate to arrays with shape (nevents,) +
        filtlengths.shape. If not, the behaviour is undefined.
        
        Parameters
        ----------
        xexpr : str
            A python expression for the x coordinate. See the method `getexpr`
            for an explanation.
        yexpr : str
            Expression for the y coordinate.
        where : str
            An expression for a boolean condition to select the values of
            `xexpr` and `yexpr`.
        log : bool
            If True (default), the colormap is for the logarithm of the bin
            height.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        xvalues = self.getexpr(xexpr)
        yvalues = self.getexpr(yexpr)
        if where is not None:
            cond = self.getexpr(where)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.hist2d', clear=True)
        
        if where is not None:
            x = xvalues[cond]
            y = yvalues[cond]
        else:
            x = xvalues.reshape(-1)
            y = yvalues.reshape(-1)
        
        if len(x) > 0:
            maxnbins = 2048
            xbins = self._binedges(x, maxnbins)
            ybins = self._binedges(y, maxnbins)
            histkw = dict(
                cmap = 'magma',
                norm = colors.LogNorm() if log else colors.Normalize(),
                cmin = 1,
            )
            _, _, _, im = ax.hist2d(x, y, (xbins, ybins), **histkw)
            
            xstep = xbins[1] - xbins[0]
            ystep = ybins[1] - ybins[0]
            fig.colorbar(im, label=f'Counts per bin ({xstep:.3g} x {ystep:.3g})')
        
        textbox.textbox(ax, f'Selection:\n{where}\n({len(x)} entries)', fontsize='small', loc='upper left')
        
        ax.set_xlabel(xexpr)
        ax.set_ylabel(yexpr)
        
        fig.tight_layout()
        return fig
