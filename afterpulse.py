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

def correlate(waveform, template, method='fft', axis=-1, boundary=None, lpad=0):
    """
    Compute the cross correlation of two arrays.
    
    The correlation is computed with padding to the right but not to the left.
    So the first element of the cross correlation is
    
        sum(waveform[:len(template)] * template),
    
    while the last is
    
        waveform[-1] * template[0] + sum(boundary * template[1:])
    
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
    lpad : int
        The amount of padding to the left. Default 0.
    
    Return
    ------
    corr : array (..., N, ...)
        The cross correlation, with the same shape as `waveform`.
    """
    rpad = len(template) - 1
    padspec = [(0, 0)] * len(waveform.shape)
    padspec[axis] = (lpad, rpad)
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
def maxprominencedip(events, start, top, n):
    """
    Find the negative peak with the maximum prominence in arrays.
    
    For computing the prominence, maxima occuring on the border of the array
    are ignored, unless both the left and right maxima occur on the border.
        
    Parameters
    ----------
    events : array (nevents, N)
        The arrays.
    start : int array (nevents,)
        Each row of `events` is used only from the sample specified by
        `start` (inclusive).
    top : array (nevents,)
        For computing the prominence, maxima are capped at `top`.
    n : int
        The number of peaks to keep in order of prominence.
    
    Return
    ------
    position : int array (nevents, n)
        The indices of the peaks in each event, sorted along the second axis
        from lower to higher prominence. -1 for no peak found. If a local
        minimum has a flat bottom, the index of the central (rounding toward
        zero) sample is returned.
    prominence : int array (nevents, n)
        The prominence of the peaks.
    """
    
    # TODO implement using guvectorize
    
    shape = (len(events), n)
    prominence = np.full(shape, -2 ** 20, events.dtype)
    position = np.full(shape, -1)
    
    for ievent, event in enumerate(events):
        
        maxprom = prominence[ievent]
        maxprompos = position[ievent]
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
                # search for maximum before minimum position
                irev = relminpos
                lmax = event[irev]
                ilmax = irev
                maxmax = top[ievent]
                while irev >= start[ievent] and event[irev] >= event[relminpos] and lmax < maxmax:
                    if event[irev] > lmax:
                        lmax = event[irev]
                        ilmax = irev
                    irev -= 1
                lmax = min(lmax, maxmax)
                lmaxb = ilmax == start[ievent]
                
                # search for maximum after minimum position
                ifwd = relminpos
                rmax = event[ifwd]
                irmax = ifwd
                while ifwd < len(event) and event[ifwd] >= event[relminpos] and rmax < maxmax:
                    if event[ifwd] > rmax:
                        rmax = event[ifwd]
                        irmax = ifwd
                    ifwd += 1
                rmax = min(rmax, maxmax)
                rmaxb = irmax == len(event) - 1
                
                # compute prominence
                if (not rmaxb and not lmaxb) or (rmaxb and lmaxb):
                    maximum = min(lmax, rmax)
                elif rmaxb:
                    maximum = lmax
                elif lmaxb:
                    maximum = rmax
                prom = maximum - event[relminpos]
                
                # insert minimum into list sorted by prominence
                if prom > maxprom[0]:
                    for j in range(1, n):
                        if prom <= maxprom[j]:
                            break
                        else:
                            maxprom[j - 1] = maxprom[j]
                            maxprompos[j - 1] = maxprompos[j]
                    else:
                        j = n
                    maxprom[j - 1] = prom
                    maxprompos[j - 1] = relminpos
                
                # reset minimum flag
                relmin = False
                relminpos = -1
    
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
    pos, prom = maxprominencedip(wf[None], np.array([start]), np.array([0]), 2)
    
    fig, ax = plt.subplots(num='afterpulse.test_maxprominencedip', clear=True)
    
    ax.plot(wf)
    ax.axvline(start, linestyle='--')
    for i, p in zip(pos[0], prom[0]):
        print(i, p)
        if i >= 0:
            ax.vlines(i, wf[i], wf[i] + p)
            ax.axhline(wf[i] + p)
    
    fig.tight_layout()
    fig.show()

def argminrelmin(a, axis=None, out=None):
    """
    Return the index of the minimum relative minimum.
    
    A relative minimum is an element which is lower than its neigbours, or the
    central element of a series of contiguous elements which are equal to each
    other and lower than their external neighbours.
    
    If there are more relative minima with the same value, return the first. If
    there are no relative minima, return -1.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.
    """
    a = np.asarray(a)
    if axis is None:
        a = a.reshape(-1)
    else:
        a = np.moveaxis(a, axis, -1)
    return _argminrelmin(a, out=out)

@numba.guvectorize(['(f8[:],i8[:])'], '(n)->()', cache=True)
def _argminrelmin(a, out):
    idx = -1
    val = 0
    wide = -1
    for i in range(1, len(a) - 1):
        if a[i - 1] > a[i] < a[i + 1]:
            if a[i] < val or idx < 0:
                idx = i
                val = a[i]
        elif a[i - 1] > a[i] == a[i + 1]:
            wide = i
        elif wide >= 0 and a[i - 1] == a[i] < a[i + 1]:
            if a[i] < val or idx < 0:
                idx = (wide + i) // 2
                val = a[i]
        else:
            wide = -1
    out[0] = idx

def test_argminrelmin():
    """
    Test argminrelmin. Should print "29 0.0".
    """
    a = np.concatenate([
        np.linspace(-1, 1, 20),
        np.linspace(1, 0, 10),
        np.linspace(0, 1, 10),
        np.linspace(1, 0.5, 10),
        np.linspace(0.5, 1, 10),
    ])
    i = argminrelmin(a)
    print(i, a[i])

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
                    The prominence, computed only looking at samples after the
                    main peak position (even if it is lower than the secondary
                    peak), and capping maxima to the baseline.
            'minorpeak2' : array filtlengths.shape
                Analogous to the 'minorpeak' field, but for the second highest
                prominence peak.
            'npe' : int array filtlengths.shape
                The number of photoelectron associated to the main peak by each
                filter.
            'saturated' : bool
                If there's at least one sample with value 0 in the event.
            'lowbaseline' : bool
                If there's at least one sample below 700 before the trigger.
            'good' : bool
                If both 'lowbaseline' and 'saturated' are False.
            'ptpeak', 'ptpeak2' :
                Analogous to 'minorpeak' and 'minorpeak2' for the two highest
                prominence peaks in the region before the trigger, filled only
                if 'lowbaseline', searched only with the longest filter.
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
        
        peakdtype = [
            ('pos', int),
            ('height', float),
            ('prominence', float),
        ]
        
        self.output = np.empty(len(wavdata), dtype=[
            ('trigger', int),
            ('baseline', float),
            ('mainpeak', peakdtype[:-1], self.filtlengths.shape),
            ('minorpeak', peakdtype, self.filtlengths.shape),
            ('minorpeak2', peakdtype, self.filtlengths.shape),
            ('ptpeak', peakdtype),
            ('ptpeak2', peakdtype),
            ('npe', int, self.filtlengths.shape),
            ('internals', [
                # left:right is the main peak search range relative to start
                # start is the beginning of the filtered post-trigger region
                ('left', int, self.filtlengths.shape),
                ('right', int, self.filtlengths.shape),
                ('start', int),
                ('bsevent', int), # event from which the baseline was copied
                ('lpad', int), # left padding on pre-trigger region
            ]),
            ('saturated', bool),
            ('lowbaseline', bool),
            ('good', bool),
            ('done', bool),
        ])
        
        self.output['npe'] = -1
        self.output['internals']['bsevent'] = -1
        self.output['internals']['lpad'] = -1
        self.output['ptpeak'] = -1
        self.output['ptpeak2'] = -1
        self.output['done'] = False
        
        # (event from which I copy the baseline, baseline)
        self._default_baseline = (-1, template.baseline)
        
        self._maketemplates(template)
        
        func = lambda s: self._run(wavdata[s], self.output[s], s)
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
    
    def _run(self, wavdata, output, slic):
        """
        Process a batch of events, filling `output`.
        """
        # find trigger
        trigger = firstbelowthreshold(wavdata[:, 1], 600)
        assert np.all(trigger >= 0)
        startoffset = 10
        start = np.min(trigger) - startoffset
        
        # find saturated events
        saturated = np.any(wavdata[:, 0] == 0, axis=-1)
        
        # compute the baseline, handling spurious pre-trigger signals
        lowbaseline = np.any(wavdata[:, 0, :start] < 700, axis=-1)
        baseline = np.median(wavdata[:, 0, :start], axis=-1)
        okbs = np.flatnonzero(~lowbaseline)
        if len(okbs) > 0:
            ibs = okbs[-1]
            self._default_baseline = (slic.start + ibs, baseline[ibs])
        baseline[lowbaseline] = self._default_baseline[1]
        mean_baseline = np.mean(baseline)
        
        for ilength in np.ndindex(*self.templates.shape):
            templ, offset = self._mftempl(ilength)
            
            # filter the post-trigger region
            filtered = correlate(wavdata[:, 0, start:], templ, boundary=mean_baseline)
            
            # find the main peak as the minimum local minimum near the trigger
            center = int(offset) + startoffset
            margin = 80
            left = max(0, center - margin)
            right = center + margin
            searchrange = filtered[:, left:right]
            mainpos = argminrelmin(searchrange, axis=-1)
            mainheight = searchrange[np.arange(len(mainpos)), mainpos]
            
            # find two other peaks with high prominence after the main one
            minorstart = np.where(mainpos < 0, 0, mainpos + left)
            minorpos, minorprom = maxprominencedip(filtered, minorstart, baseline, 2)
            minorheight = filtered[np.arange(len(minorpos))[:, None], minorpos]
            
            idx = (slice(None),) + ilength
            
            badvalue = -1000
            
            # fill main peak
            mainpeak_out = output['mainpeak'][idx]
            bad = mainpos < 0
            mainpeak_out['pos'] = mainpos + np.where(bad, 0, start + left)
            mainpeak_out['height'] = np.where(bad, badvalue, baseline - mainheight)
            
            # fill minor peaks
            for ipeak, s in [(1, ''), (0, '2')]:
                minorpeak_out = output['minorpeak' + s][idx]
                pos = minorpos[:, ipeak]
                bad = pos < 0
                minorpeak_out['pos'] = pos + np.where(bad, 0, start)
                minorpeak_out['height'] = np.where(bad, badvalue, baseline - minorheight[:, ipeak])
                minorpeak_out['prominence'] = np.where(bad, badvalue, minorprom[:, ipeak])
            
            output['internals']['left'][idx] = left
            output['internals']['right'][idx] = right
        
        # find peaks in pre-trigger region if there are low samples
        if np.any(lowbaseline):
            templ, offset = self._mftempl()
            lpad = 100
            filtered = correlate(wavdata[lowbaseline, 0, :start], templ, boundary=mean_baseline, lpad=lpad)
            ptpos, ptprom = maxprominencedip(filtered, np.zeros(len(filtered)), baseline[lowbaseline], 2)
            ptheight = filtered[np.arange(len(ptpos))[:, None], ptpos]
            for ipeak, s in [(1, ''), (0, '2')]:
                ptpeak_out = output['ptpeak' + s]
                pos = ptpos[:, ipeak]
                bad = pos < 0
                ptpeak_out['pos'][lowbaseline] = np.where(bad, pos, np.maximum(0, pos - lpad))
                ptpeak_out['height'][lowbaseline] = np.where(bad, badvalue, baseline[lowbaseline] - ptheight[:, ipeak])
                ptpeak_out['prominence'][lowbaseline] = np.where(bad, badvalue, ptprom[:, ipeak])
            output['internals']['lpad'][lowbaseline] = lpad
        
        output['trigger'] = trigger
        output['baseline'] = baseline
        output['saturated'] = saturated
        output['lowbaseline'] = lowbaseline
        output['good'] = ~(saturated | lowbaseline)
        output['internals']['start'] = start
        output['internals']['bsevent'][lowbaseline] = self._default_baseline[0]
        output['done'] = True
    
    def _mftempl(self, ilength=None):
        """
        Return template, offset. If not specified, return the longest template.
        """
        if ilength is None:
            ilength_flat = np.argmax(self.filtlengths)
            ilength = np.unravel_index(ilength_flat, self.filtlengths.shape)
        entry = self.templates[ilength]
        templ = entry['template'][:entry['length']]
        offset = entry['offset']
        return templ, offset
    
    def _computenpe(self):
        """
        Compute the number of photoelectrons from a fingerplot.
        """
        for ilength in np.ndindex(*self.filtlengths.shape):
            idx = (slice(None),) + ilength
            value = self.output['mainpeak']['height'][idx]
            good1 = self.output['good']
            good2 = self.output['mainpeak']['pos'][idx] >= 0
            good_value = value[good1 & good2]
            
            if len(good_value) >= 100:
                _, center, _ = single_filter_analysis(good_value, return_full=True)
    
                bins = (center[1:] + center[:-1]) / 2
                npe = np.digitize(value[good2], bins)
                self.output['npe'][(good2,) + ilength] = npe
    
    def fingerplot(self, ilength=None):
        """
        Plot the histogram of the main peak height.
        
        Parameters
        ----------
        ilength : {int, tuple of int, None}, optional
            The index in `filtlengths` of the filter length. If not specified,
            use the longest filter.
            
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        if ilength is None:
            ilength_flat = np.argmax(self.filtlengths)
            ilength = np.unravel_index(ilength_flat, self.filtlengths.shape)
        elif not isinstance(ilength, tuple):
            ilength = (ilength,)
        
        length = self.filtlengths[ilength]
        value = self.output['mainpeak']['height'][(slice(None),) + ilength]
        value = value[self.output['good']]

        fig = plt.figure(num='afterpulse.AfterPulse.fingerplot', clear=True)
        
        single_filter_analysis(value, fig1=fig)
        
        ax, = fig.get_axes()
        textbox.textbox(ax, f'filter length = {length} ns', loc='center right', fontsize='small')
        
        fig.tight_layout()
        return fig
    
    def plotevent(self, wavdata, ievent, ilength=None, zoom='posttrigger'):
        """
        Plot a single event.
        
        Parameters
        ----------
        wavdata : array (nevents, 2, 15001)
            The same array passed at initialization.
        ievent : int
            The event index.
        ilength : int, optional
            The index of the filter length in `filtlengths`. If not specified,
            use the longest filter.
        zoom : {'all', 'posttrigger', 'main'}
            The x-axis extension.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        if ilength is None:
            ilength_flat = np.argmax(self.filtlengths)
            ilength = np.unravel_index(ilength_flat, self.filtlengths.shape)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.plotevent', clear=True)
        
        wf = wavdata[ievent, 0]
        ax.plot(wf, color='#f55')
        
        entry = self.output[ievent]
        ax.axvline(entry['trigger'], color='#000', linestyle='--', label='trigger')
        
        baseline = entry['baseline']
        ax.axhline(baseline, color='#000', linestyle=':', label='baseline')
        
        templ, _ = self._mftempl(ilength)
        start = entry['internals']['start']
        filtered = correlate(wf[start:], templ, boundary=baseline)
        length = self.filtlengths[ilength]
        ax.plot(start + np.arange(len(filtered)), filtered, color='#000', label=f'filtered ({length} ns template)')
        
        if entry['lowbaseline']:
            templ, _ = self._mftempl()
            lpad = entry['internals']['lpad']
            filtered = correlate(wf[:start], templ, boundary=baseline, lpad=lpad)
            ax.plot(-lpad + np.arange(len(filtered)), filtered, color='#000')
        
        left = entry['internals']['left'][ilength]
        right = entry['internals']['right'][ilength]
        base = start - 0.5
        ax.axvspan(base + left, base + right, color='#ddd', label='main peak search range')
        
        markerkw = dict(linestyle='', markersize=10, markeredgecolor='#000', markerfacecolor='#fff0')
        
        mainpeak = entry['mainpeak'][ilength]
        mainpos = mainpeak['pos']
        if mainpos >= 0:
            mainheight = mainpeak['height']
            npe = entry['npe'][ilength]
            base = baseline - mainheight
            ax.vlines(mainpos, base, baseline, zorder=2.1)
            ax.plot(mainpos, base, marker='o', label=f'main peak, h={mainheight:.1f}, npe={npe}', **markerkw)
        
        peaks = [
            # field, label, marker
            ('minorpeak' , 'minor peak'          , 's'),
            ('minorpeak2', 'third peak'          , 'v'),
            ('ptpeak'    , 'pre-trigger peak'    , '<'),
            ('ptpeak2'   , '2nd pre-trigger peak', '>'),
        ]
        for field, label, marker in peaks:
            minorpeak = entry[field]
            if minorpeak.shape:
                minorpeak = minorpeak[ilength]
            minorpos = minorpeak['pos']
            if minorpos >= 0:
                minorheight = minorpeak['height']
                minorprom = minorpeak['prominence']
                base = baseline - minorheight
                ax.vlines(minorpos, base, base + minorprom, zorder=2.1)
                ax.hlines(base + minorprom, minorpos - 100, minorpos + 100, zorder=2.1)
                ax.plot(minorpos, base, marker=marker, label=f'{label}, h={minorheight:.1f}, prom={minorprom:.1f}', **markerkw)
        
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        
        ax.legend(fontsize='small', loc='lower right')
        textbox.textbox(ax, f'Event {ievent}', fontsize='medium', loc='upper center')
        
        if zoom == 'all':
            pass
        elif zoom == 'posttrigger':
            ax.set_xlim(start - 500, len(wf) - 1)
        elif zoom == 'main':
            ax.set_xlim(start - 200, start + right + 500)
        else:
            raise KeyError(zoom)
        
        ax.set_xlabel('Sample number @ 1 GSa/s')
        
        fig.tight_layout()
        return fig

    def getexpr(self, expr, allow_numpy=True):
        """
        Evaluate an expression on all events.
        
        The expression can be any python expression involving the following
        numpy arrays:
        
            trigger     : the index of the trigger leading edge
            baseline    : the value of the baseline
            length      : the cross correlation filter template length
            saturated   : if there is a sample equal to zero in the event
            lowbaseline : if there is a sample < 700 before the trigger
            good        : if not saturated neither lowbaseline
            mainpos     : the index of the main peak
            mainheight  : the positive height of the main peak
            minorpos    : the index of the maximum promince peak after the main
            minorheight : ...its height
            minorprom   : ...its prominence, capped to the baseline, and
                          measured without crossing the main peak
            thirdpos    : the index of the second highest prominence peak
            thirdheight : as above
            thirdprom   : as above
            ptpos       : the index of the higher prominence peak before the
                          trigger, filled only if lowbaseline, searched only
                          with the longest filter
            ptheight    : as above
            ptprom      : as above
            pt2pos      : like ptpos, but the second highest
            pt2height   : as above
            pt2prom     : as above
            npe         : the number of photoelectrons of the main peak,
                          determined separately with each filter length
        
        All arrays have shape (nevents,) + filtlengths.shape, although some
        of them do not vary over all axes.
        
        When there are missing values, the indices are set to -1, while the
        heights and prominences are set to -1000.
        
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
            trigger     = broadcasted(self.output['trigger']),
            baseline    = broadcasted(self.output['baseline']),
            length      = np.broadcast_to(self.filtlengths, self.output['mainpeak'].shape),
            saturated   = broadcasted(self.output['saturated']),
            lowbaseline = broadcasted(self.output['lowbaseline']),
            good        = broadcasted(self.output['good']),
            mainpos     = self.output['mainpeak']['pos'],
            mainheight  = self.output['mainpeak']['height'],
            minorpos    = self.output['minorpeak']['pos'],
            minorheight = self.output['minorpeak']['height'],
            minorprom   = self.output['minorpeak']['prominence'],
            thirdpos    = self.output['minorpeak2']['pos'],
            thirdheight = self.output['minorpeak2']['height'],
            thirdprom   = self.output['minorpeak2']['prominence'],
            ptpos       = broadcasted(self.output['ptpeak']['pos']),
            ptheight    = broadcasted(self.output['ptpeak']['height']),
            ptprom      = broadcasted(self.output['ptpeak']['prominence']),
            pt2pos      = broadcasted(self.output['ptpeak2']['pos']),
            pt2height   = broadcasted(self.output['ptpeak2']['height']),
            pt2prom     = broadcasted(self.output['ptpeak2']['prominence']),
            npe         = self.output['npe'],
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
        mask = self.getexpr(cond)
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
            if maxnbins == 'auto':
                maxnbins = max(10, len(bins) - 1)
            bins = np.arange(np.min(x), np.max(x) + 2) - 0.5
            if len(bins) - 1 > maxnbins:
                p = int(np.ceil((len(bins) - 1) / maxnbins))
                bins = bins[:-1:p]
                bins = np.pad(bins, (0, 1), constant_values=bins[-1] + bins[1] - bins[0])
        
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
