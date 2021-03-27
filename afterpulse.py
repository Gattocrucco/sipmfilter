import functools

import numpy as np
from matplotlib import pyplot as plt, colors

import runsliced
from single_filter_analysis import single_filter_analysis
import textbox
import breaklines
import meanmedian
import correlate
import firstbelowthreshold
import argminrelmin
import maxprominencedip
import npzload

def maxdiff_boundaries(x, pe):
    """
    Compute the 'maxdiff' boudaries for AfterPulse.npeboundaries.
    
    Parameters
    ----------
    x : 1D array
        The height values.
    pe : array (M,)
        The heights of the peaks.
    
    Return
    ------
    values : array (M - 1,)
        The midpoint between the two most distant consecutive height samples
        between each pair of peaks.
    """
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

class AfterPulse(npzload.NPZLoad):
    
    def __init__(self,
        wavdata,
        template,
        filtlengths=None,
        batch=10,
        pbar=False,
        lowsample=700,
        trigger=None,
    ):
        """
        Analyze LNGS laser data.
    
        The analysis is appropriate for studying afterpulses but it can be
        used as a general purpose tool.
        
        Parameters
        ----------
        wavdata : array (nevents, nchannels, 15001)
            Data as returned by readwav.readwav(). The first channel is the
            waveform, the second channel is the trigger. If there is only one
            channel, the trigger position must be specified with `trigger`.
        template : template.Template
            A template object used for the cross correlation filter. Should be
            generated using the same wav file.
        filtlengths : array, optional
            A series of template lengths for the cross correlation filters, in
            unit of 1 GSa/s samples. If not specified, a logarithmically spaced
            range from 32 ns to 2048 ns is used.
        batch : int
            The events batch size, default 10.
        pbar : bool
            If True, print a progressbar during the computation (1 tick per
            batch). Default False.
        lowsample : scalar
            Threshold on single samples in the pre-trigger region to switch to
            a default baseline instead of computing it from the event, default
            700.
        trigger : int array (nevents,), optional
            The trigger leading edge position, used if there's no trigger
            information in `wavdata`.
        
        Methods
        -------
        filtertempl : get a cross correlation filter template.
        good : determine when the main peak height is reliable.
        npeboundaries : divide peak heights in pe bins.
        npe : assign heights to pe bins.
        fingerplot : plot fingerplot used to determine pe bins.
        computenpe : assign number of pe to a peak.
        plotevent : plot the analysis of a single event.
        getexpr : compute the value of an expression on all events.
        setvar : set a variable accessible from `getexpr`.
        eventswhere : get the list of events satisfying a condition.
        hist : plot the histogram of a variable.
        scatter : plot two variables.
        hist2d : plot the histogram of two variables.
        catindex : map event index to object in a concatenation.
        subindex : map global event index to event index in concatenated object.
    
        Class methods
        -------------
        concatenate : create an instance by concatenating instances.
        
        Members
        -------
        lowsample : scalar
            The argument given at initialization.
        filtlengths : array
            The cross correlation filter lengths.
        output : array (nevents,)
            A structured array containing the information for each event with
            these fields:
            'trigger' : int
                The index of the trigger leading edge.
            'saturated' : bool
                If there's at least one sample with value 0 in the event.
            'lowbaseline' : bool
                If there's at least one sample below `lowsample` before the
                trigger.
            'baseline' : float
                The mean of the medians of the pre-trigger region divided in 8
                parts with stride 8. Copied from another event if 'lowbaseline'.
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
            'ptpeak', 'ptpeak2' : array filtlengths.shape
                Analogous to 'minorpeak' and 'minorpeak2' for the two highest
                prominence peaks in the region before the trigger, searched
                only with the longest filter.
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
            self.filtlengths = 2 ** np.arange(5, 11 + 1)
        else:
            self.filtlengths = np.array(filtlengths, int)
            
        self.lowsample = lowsample
        
        peakdtype = [
            ('pos', int),
            ('height', float),
            ('prominence', float),
        ]
        
        self.output = np.empty(len(wavdata), dtype=[
            ('trigger', int),
            ('baseline', float),
            ('mainpeak', [
                ('pos', int),
                ('height', float),
            ], self.filtlengths.shape),
            ('minorpeak', peakdtype, self.filtlengths.shape),
            ('minorpeak2', peakdtype, self.filtlengths.shape),
            ('ptpeak', peakdtype, self.filtlengths.shape),
            ('ptpeak2', peakdtype, self.filtlengths.shape),
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
            ('done', bool),
        ])
        
        self.output['internals']['bsevent'] = -1
        self.output['done'] = False
        
        if trigger is not None:
            trigger = np.asarray(trigger)
            assert trigger.shape == self.output.shape, trigger.shape
            self.trigger = trigger
        
        # tuple (event from which I copy the baseline, baseline)
        self._default_baseline = (-1, template.baseline)
        
        self._maketemplates(template)
        
        func = lambda s: self._run(wavdata[s], self.output[s], s)
        runsliced.runsliced(func, len(wavdata), batch, pbar)
        
    def _maketemplates(self, template):
        """
        Fill the `templates` and `pttemplate` members.
        """
        def templates(lengths):
            templates = np.zeros(lengths.shape, dtype=[
                ('template', float, (np.max(lengths),)),
                ('length', int),
                ('offset', int),
            ])
            templates['length'] = lengths
            for _, entry in np.ndenumerate(templates):
                templ, offset = template.matched_filter_template(entry['length'], timebase=1, aligned=True)
                assert len(templ) == entry['length']
                entry['template'][:len(templ)] = templ
                entry['length'] = len(templ)
                entry['offset'] = offset
            return templates
        
        self.templates = templates(self.filtlengths)
    
    def _max_ilength(self):
        """
        Return index of longest filter.
        """
        ilength_flat = np.argmax(self.filtlengths)
        return np.unravel_index(ilength_flat, self.filtlengths.shape)
    
    def filtertempl(self, ilength=None):
        """
        Return a cross correlation filter template.
        
        Parameters
        ----------
        ilength : {int, tuple, None}
            The index of the filter length in `filtlengths`. If not specified,
            use the longest filter.
        
        Return
        ------
        templ : 1D array
            The normalized template.
        offset : scalar
            The offset of the template start relative to the original
            untruncated template.
        """
        if ilength is None:
            ilength = self._max_ilength()
        entry = self.templates[ilength]
        templ = entry['template'][:entry['length']]
        offset = entry['offset']
        return templ, offset
    
    def _run(self, wavdata, output, slic):
        """
        Process a batch of events, filling `output`.
        """
        # find trigger
        if wavdata.shape[1] == 2:
            trigger = firstbelowthreshold.firstbelowthreshold(wavdata[:, 1], 600)
            assert np.all(trigger >= 0)
        else:
            trigger = self.trigger[slic]
        startoffset = 10
        start = np.min(trigger) - startoffset
        
        # find saturated events
        saturated = np.any(wavdata[:, 0] == 0, axis=-1)
        
        # compute the baseline, handling spurious pre-trigger signals
        lowbaseline = np.any(wavdata[:, 0, :start] < self.lowsample, axis=-1)
        baseline = meanmedian.meanmedian(wavdata[:, 0, :start], 8)
        okbs = np.flatnonzero(~lowbaseline)
        if len(okbs) > 0:
            ibs = okbs[-1]
            self._default_baseline = (slic.start + ibs, baseline[ibs])
        baseline[lowbaseline] = self._default_baseline[1]
        mean_baseline = np.mean(baseline)
        
        for ilength in np.ndindex(*self.templates.shape):
            templ, offset = self.filtertempl(ilength)
            
            # filter the post-trigger region
            filtered = correlate.correlate(wavdata[:, 0, start:], templ, boundary=mean_baseline)
            
            # find the main peak as the minimum local minimum near the trigger
            center = int(offset) + startoffset
            margin = 80
            left = max(0, center - margin)
            right = center + margin
            searchrange = filtered[:, left:right]
            mainpos = argminrelmin.argminrelmin(searchrange, axis=-1)
            mainheight = searchrange[np.arange(len(mainpos)), mainpos]
            
            # find two other peaks with high prominence after the main one
            minorstart = np.where(mainpos < 0, 0, mainpos + left)
            minorpos, minorprom = maxprominencedip.maxprominencedip(filtered, minorstart, baseline, 2)
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
        
            # find peaks in pre-trigger region
            lpad = 100
            filtered = correlate.correlate(wavdata[:, 0, :start], templ, boundary=mean_baseline, lpad=lpad)
            ptpos, ptprom = maxprominencedip.maxprominencedip(filtered, np.zeros(len(filtered)), baseline, 2)
            ptheight = filtered[np.arange(len(ptpos))[:, None], ptpos]
            for ipeak, s in [(1, ''), (0, '2')]:
                pos = ptpos[:, ipeak]
                height = baseline - ptheight[:, ipeak]
                bad = pos < 0
            
                ptpeak_out = output['ptpeak' + s][idx]
                ptpeak_out['pos'] = np.where(bad, pos, np.maximum(0, pos - lpad))
                ptpeak_out['height'] = np.where(bad, badvalue, height)
                ptpeak_out['prominence'] = np.where(bad, badvalue, ptprom[:, ipeak])
            
        output['trigger'] = trigger
        output['baseline'] = baseline
        output['saturated'] = saturated
        output['lowbaseline'] = lowbaseline
        output['internals']['start'] = start
        output['internals']['bsevent'][lowbaseline] = self._default_baseline[0]
        output['internals']['lpad'] = lpad
        output['done'] = True
    
    def good(self, safedist=2500, safeheight=8):
        """
        Find events where the main peak height is accurate.
        
        The conditions are that there's no saturation and that there are no
        pre-trigger pulses near the main peak. The height can still be
        inaccurate due to close afterpulses.
        
        The pre-trigger pulses are identified using the longest filter.
        
        Parameters
        ----------
        safedist : scalar
            Distance from a peak in the pre-trigger region to the trigger above
            which the peak is assumed not to influence the post-trigger region,
            default 2500.
        safeheight : scalar
            The maximum height for peaks within `safedist` to be still
            considered negligible, default 8.
        
        Return
        ------
        good : bool array (nevents,)
            Where `True` the main peak height is determined accurately, apart
            from afterpulses.
        """
        good = ~self.output['saturated']
        ptlength = self.output['internals']['start']
        idx = (slice(None),) + self._max_ilength()
        for s in ['', '2']:
            peak = self.output['ptpeak' + s][idx]
            pos = peak['pos']
            height = peak['height']
            bad = pos < 0
            
            notgood = ptlength - pos < safedist
            notgood &= height > safeheight
            good &= bad | ~notgood
        
        return good
    
    def npeboundaries(self, height, plot=False, algorithm='maxdiff'):
        """
        Determine boundaries to divide peak heights by number of pe.
        
        Parameters
        ----------
        height : array (N,)
            The heights.
        plot : bool
            If True, plot the fingerplot used separate the peaks.
        algorithm : {'maxdiff', 'midpoints'}
            The algorithm used to place the boundaries. 'midpoints' uses the
            midpoints between the peaks. 'maxdiff' (default) uses the midpoint
            between the two most distant consecutive samples between two peaks.
        
        Return
        ------
        boundaries : array
            The height boundaries separating different pe. boundaries[0]
            divides 0 pe from 1 pe, boundaries[-1] divides the maximum pe
            from the overflow.
        fig : matplotlib figure
            Returned only if `plot` is True.
        """
        height = np.asarray(height)
        assert height.ndim == 1, height.ndim
        
        kw = dict(return_full=True)
        if plot:
            fig = plt.figure(num='afterpulse.AfterPulse.npeboundaries', clear=True)
            kw.update(fig1=fig)
        _, center, _ = single_filter_analysis(height, **kw)
        
        last = center[-1] + (center[-1] - center[-2])
        center = np.pad(center, (0, 1), constant_values=last)
        if algorithm == 'midpoints':
            boundaries = (center[1:] + center[:-1]) / 2
        elif algorithm == 'maxdiff':
            boundaries = maxdiff_boundaries(height, center)
            if plot:
                ax, = fig.get_axes()
                ylim = ax.get_ylim()
                ax.vlines(boundaries, *ylim, linestyle='-.', label='final boundaries')
                ax.legend(loc='upper right')
        else:
            raise KeyError(algorithm)
        
        if plot:
            return boundaries, fig
        else:
            return boundaries
    
    def npe(self, height, boundaries, overflow=1000):
        """
        Compute the number of photoelectrons from a fingerplot.
        
        Parameters
        ----------
        height : array (N,)
            The heights.
        boundaries : array (M,)
            The height boundaries separating different pe. boundaries[0]
            divides 0 pe from 1 pe, boundaries[-1] divides the maximum pe
            from the overflow.
        overflow : int
            The value used for heights after the last boundary.
        
        Return
        ------
        npe : int array (N,)
            The number of pe assigned to each height.
        """
        npe = np.digitize(height, boundaries)
        npe[npe >= len(boundaries)] = overflow
        return npe
        
    def fingerplot(self, ilength=None):
        """
        Plot the fingerplot of the main peak height used to count the pe.
        
        Parameters
        ----------
        ilength : {int, tuple of int, None}, optional
            The index in `filtlengths` of the filter length. If not specified,
            use the longest filter.
            
        Return
        ------
        fig : matplotlib figure
            The plot.
        """
        if ilength is None:
            ilength = self._max_ilength()
        elif not isinstance(ilength, tuple):
            ilength = (ilength,)
        
        idx = (slice(None),) + ilength
        peak = self.output['mainpeak'][idx]
        height = peak['height']
        good = self.getexpr('good')
        valid = peak['pos'] >= 0
        value = height[valid & good]
        
        if len(value) >= 100:
            boundaries, fig = self.npeboundaries(value, plot=True)
            ax, = fig.get_axes()
            length = self.filtlengths[ilength]
            textbox.textbox(ax, f'filter length = {length} ns', loc='center right', fontsize='small')
        
            fig.tight_layout()
            return fig
    
    def computenpe(self, peaklabel):
        """
        Compute the number of pe of peaks.
        
        Parameters
        ----------
        peaklabel : str
            One of the fields in `output` representing a peak.
        
        Return
        ------
        npe : int array filtlengths.shape + (nevents,)
            The pe assigned to each peak height. -1 when the peak is missing,
            1000 for height larger than the highest classified pe.
        """
        if not hasattr(self, '_computenpe_boundaries'):
            self._computenpe_boundaries = {}
            mainpeak = self.output['mainpeak']
            good = self.getexpr('good')
        
        boundaries = self._computenpe_boundaries
        targetpeak = self.output[peaklabel]
        
        npe = np.full(self.filtlengths.shape + self.output.shape, -1)
        
        for ilength, _ in np.ndenumerate(self.filtlengths):
            idx = (slice(None),) + ilength
            
            if ilength not in boundaries:
                peak = mainpeak[idx]
                height = peak['height']
                valid = peak['pos'] >= 0
                boundaries[ilength] = self.npeboundaries(height[valid & good])
            
            bnd = boundaries[ilength]
            peak = targetpeak[idx]
            height = peak['height']
            valid = peak['pos'] >= 0
            npe[ilength + (valid,)] = self.npe(height[valid], bnd)
        
        return npe
    
    def plotevent(self, wavdata, ievent, ilength=None, zoom='posttrigger'):
        """
        Plot a single event.
        
        Parameters
        ----------
        wavdata : array (nevents, 2, 15001) or list
            The same array passed at initialization. If the object is
            a concatenation, the data passed to the object where the the event
            originates from. Use `catindex()` to map the event to the object.
            If a list, it must be the ordered list of arrays passed at
            initialization to the concatenated objects.
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
            ilength = self._max_ilength()
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.plotevent', clear=True)
        
        if not hasattr(wavdata, 'astype'):
            wavdata = wavdata[self.catindex(ievent)]
        wf = wavdata[self.subindex(ievent), 0]
        ax.plot(wf, color='#f55')
        
        entry = self.output[ievent]
        ax.axvline(entry['trigger'], color='#000', linestyle='--', label='trigger')
        
        baseline = entry['baseline']
        ax.axhline(baseline, color='#000', linestyle=':', label='baseline')
        
        templ, _ = self.filtertempl(ilength)
        start = entry['internals']['start']
        filtered = correlate.correlate(wf[start:], templ, boundary=baseline)
        length = self.filtlengths[ilength]
        ax.plot(start + np.arange(len(filtered)), filtered, color='#000', label=f'filtered ({length} ns template)')
        
        lpad = entry['internals']['lpad']
        filtered = correlate.correlate(wf[:start], templ, boundary=baseline, lpad=lpad)
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
            npe = self.getexpr('npe')[ilength][ievent]
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
    
    @functools.cached_property
    def _variables(self):
        return dict(
            event       = "np.arange(len(self.output))",
            trigger     = "self.output['trigger']",
            baseline    = "self.output['baseline']",
            length      = "self.filtlengths[..., None]",
            saturated   = "self.output['saturated']",
            lowbaseline = "self.output['lowbaseline']",
            mainpos     = "self.output['mainpeak']['pos']",
            mainheight  = "self.output['mainpeak']['height']",
            minorpos    = "self.output['minorpeak']['pos']",
            minorheight = "self.output['minorpeak']['height']",
            minorprom   = "self.output['minorpeak']['prominence']",
            thirdpos    = "self.output['minorpeak2']['pos']",
            thirdheight = "self.output['minorpeak2']['height']",
            thirdprom   = "self.output['minorpeak2']['prominence']",
            ptpos       = "self.output['ptpeak']['pos']",
            ptheight    = "self.output['ptpeak']['height']",
            ptprom      = "self.output['ptpeak']['prominence']",
            pt2pos      = "self.output['ptpeak2']['pos']",
            pt2height   = "self.output['ptpeak2']['height']",
            pt2prom     = "self.output['ptpeak2']['prominence']",
            good        = "self.good()",
            npe         = "self.computenpe('mainpeak')",
        )
    
    def setvar(self, name, value, overwrite=False):
        """
        Set a named variable.
        
        Variables can be used in expressions with `getexpr` and other methods
        accepting expressions.
        
        Parameters
        ----------
        name : str
            The variable name.
        value : array_like
            An array. Numpy arrays are not copied.
        overwrite : bool
            If False (default), do not allow setting the value of an existing
            variable.
        """
        if not overwrite and name in self._variables:
            raise ValueError(f'variable name `{name}` already used')
        v = np.asarray(value).view()
        v.flags['WRITEABLE'] = False
        self._variables[name] = v

    @functools.cached_property
    def _exprglobals(self):
        return {
            k: v
            for k, v in vars(np).items()
            if not k.startswith('_')
            and not k[0].isupper()
        }

    def getexpr(self, expr, condexpr=None, allow_numpy=True, forcebroadcast=False):
        """
        Evaluate an expression on all events.
        
        The expression can be any python expression involving the following
        numpy arrays:
            
            event       : event index
            trigger     : the index of the trigger leading edge
            baseline    : the value of the baseline
            length      : the cross correlation filter template length
            saturated   : if there is a sample equal to zero in the event
            lowbaseline : if there is sample too low before the trigger
            good        : if not saturated and without high pre-trigger peaks
                          near the trigger
            mainpos     : the index of the main peak
            mainheight  : the positive height of the main peak
            npe         : the number of photoelectrons of the main peak,
                          determined separately with each filter length
            minorpos    : the index of the maximum promince peak after the main
            minorheight : ...its height
            minorprom   : ...its prominence, capped to the baseline, and
                          measured without crossing the main peak
            thirdpos    : the index of the second highest prominence peak
            thirdheight : as above
            thirdprom   : as above
            ptpos       : the index of the higher prominence peak before the
                          trigger
            ptheight    : as above
            ptprom      : as above
            pt2pos      : like ptpos, but the second highest
            pt2height   : as above
            pt2prom     : as above
        
        Variables which depends on the filter template length have shape
        filtlengths.shape + (nevents,), apart from the variable `length` which
        has shape filtlengths.shape + (1,). Variables which do not depend on
        filter template length have shape (nevents,).
        
        When there are missing values, the indices are set to -1, while the
        heights and prominences are set to -1000. `npe` is -1 when missing and
        1000 for the overflow bin.
        
        Additional variables can be added with `setvar`.
        
        Parameters
        ----------
        expr : str
            The expression.
        condexpr : str, optional
            An expression which must evaluate to a boolean array that can be
            broadcasted with the variables and which is used to select the
            values in all the variables prior to evaluating `expr`.
        allow_numpy : bool
            If True (default), allow numpy functions in the expression.
        forcebroadcast : bool
            If True, broadcast all variables to the shape filtlengths.shape +
            (nevents,). This is useful when using `condexpr` and mixing
            variables with different shape in `expr`. Default False.
        
        Return
        ------
        value :
            The evaluated expression.
        """
        
        class VariablesDict:
            
            def __getitem__(self2, key):
                v = self._variables[key]
                
                if isinstance(v, str):
                    expr = v
                    v = eval(expr)
                    if 'self.output[' in expr:
                        v = np.copy(v) # to make it contiguous
                        if v.shape == self.output.shape:
                            pass
                        elif v.shape == self.output.shape + self.filtlengths.shape:
                            v = np.moveaxis(v, 0, -1)
                        else:
                            raise ValueError(f'unrecognized shape {v.shape} for variable {key}')
                    else:
                        v = v.view()
                    
                    v.flags['WRITEABLE'] = False
                
                    self._variables[key] = v
                
                if forcebroadcast:
                    v = np.broadcast_to(v, self.filtlengths.shape + self.output.shape)
                
                if hasattr(self2, 'cond'):
                    v, cond = np.broadcast_arrays(v, self2.cond)
                    v = v[cond]
                
                return v
        
        globals = self._exprglobals if allow_numpy else {}
        locals = VariablesDict()
        if condexpr is not None:
            locals.cond = eval(condexpr, globals, locals)
        return eval(expr, globals, locals)
    
    def eventswhere(self, cond):
        """
        List the events satisfying a condition.

        For conditions that depends on filter length, the condition must be
        satisfied for at least one length.
        
        The condition must evaluate to a boolean array broadcastable to shape
        filtlengths.shape + (nevents,). If not, the behaviour is undefined.
        
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
        mask = np.broadcast_to(mask, self.filtlengths.shape + self.output.shape)
        mask = np.any(mask, axis=tuple(range(self.filtlengths.ndim)))
        return np.flatnonzero(mask)
    
    def hist(self, expr, where=None, yscale='linear', nbins='auto'):
        """
        Plot the histogram of an expression.
        
        The values are histogrammed separately for each filter length.
        
        The expression must evaluate to an array. If the array is not 1D, it
        must be broadcastable to a shape filtlengths.shape + (N,). If not, the
        behaviour is undefined.
        
        Parameters
        ----------
        expr : str
            A python expression. See the method `getexpr` for an explanation.
        where : str
            An expression for a boolean condition to select the values of
            `expr`. The condition is broadcasted with `expr` prior to applying
            it.
        yscale : str
            The y scale of the plot, default 'linear'.
        nbins : int, optional
            The number of bins. Computed automatically by default.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        values = self.getexpr(expr)
        if where is not None:
            cond = self.getexpr(where)
            values, cond = np.broadcast_arrays(values, cond)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.hist', clear=True)
        
        histkw = dict(
            histtype = 'step',
            color = '#600',
            zorder = 2,
        )
        
        if values.ndim == 1:
            if where is not None:
                values = values[cond]
            if len(values) > 0:
                histkw.update(
                    bins = self._binedges(values, nbins=nbins),
                )
                ax.hist(values, **histkw)
            textbox.textbox(ax, f'{len(values)} entries', fontsize='small', loc='upper right')
        else:
            xlength = []
            for ilength, length in np.ndenumerate(self.filtlengths):
                x = values[ilength]
                if where is not None:
                    x = x[cond[ilength]]
                if len(x) > 0:
                    xlength.append((x, length))
                
            for i, (x, length) in enumerate(xlength):
                histkw.update(
                    bins = self._binedges(x, nbins=nbins),
                    label = f'{length} ({len(x)})',
                    alpha = (1 + i) / len(xlength),
                )
                ax.hist(x, **histkw)
            ax.legend(title='Filter length (entries)', fontsize='small', ncol=2, loc='upper right')
        
        if where is not None:
            s = breaklines.breaklines(f'Selection: {where}', 40, ')', '&|')
            textbox.textbox(ax, s, fontsize='small', loc='upper left')
        
        ax.set_xlabel(expr)
        ax.set_ylabel('Count per bin')
        
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
        
        The expressions must evaluate to arrays. If an array is not 1D, it
        must be broadcastable to a shape filtlengths.shape + (N,). If not, the
        behaviour is undefined.
        
        Parameters
        ----------
        xexpr : str
            A python expression for the x coordinate. See the method `getexpr`
            for an explanation.
        yexpr : str
            Expression for the y coordinate.
        where : str
            An expression for a boolean condition to select the values of
            `xexpr` and `yexpr`. The condition is broadcasted with `xexpr` and
            `yexpr` prior to applying it.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        xvalues = self.getexpr(xexpr)
        yvalues = self.getexpr(yexpr)
        if where is not None:
            cond = self.getexpr(where)
            xvalues, yvalues, cond = np.broadcast_arrays(xvalues, yvalues, cond)
        else:
            xvalues, yvalues = np.broadcast_arrays(xvalues, yvalues)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.scatter', clear=True)
        
        plotkw = dict(
            linestyle = '',
            marker = '.',
            color = '#600',
        )
        
        if xvalues.ndim == 1:
            if where is not None:
                xvalues = xvalues[cond]
                yvalues = yvalues[cond]
            ax.plot(xvalues, yvalues, **plotkw)
            textbox.textbox(ax, f'{len(xvalues)} entries', fontsize='small', loc='upper right')
        else:
            xylength = []
            for ilength, length in np.ndenumerate(self.filtlengths):
                x = xvalues[ilength]
                y = yvalues[ilength]
                if where is not None:
                    x = x[cond[ilength]]
                    y = y[cond[ilength]]
                if len(x) > 0:
                    xylength.append((x, y, length))
            
            for i, (x, y, length) in xylength:
                plotkw.update(
                    label = f'{length} ({len(x)})',
                    alpha = (1 + i) / len(xylength),
                )
                ax.plot(x, y, **plotkw)
            ax.legend(title='Filter length (entries)', fontsize='small', ncol=2, loc='upper right')
        
        if where is not None:
            s = breaklines.breaklines(f'Selection: {where}', 40, ')', '&|')
            textbox.textbox(ax, s, fontsize='small', loc='upper left')
        
        ax.set_xlabel(xexpr)
        ax.set_ylabel(yexpr)
        
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        
        fig.tight_layout()
        return fig
    
    def _binedges(self, x, maxnbins='auto', nbins='auto'):
        """
        Compute histogram bin edges for the array x.
        
        If the data type is a float, the edges are computed with
        np.histogram_bin_edges(x, bins=nbins).
        
        If the data type is integral, the edges are aligned to half-integer
        values. The number of bins is capped to `maxnbins`. If `maxnbins` is
        'auto' (default), it is set to the number of bins numpy would use, but
        at least 10.
        """
        bins = np.histogram_bin_edges(x, bins=nbins)
        
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
        
        The expressions must evaluate to arrays broadcastable to shape
        filtlengths.shape + (nevents,). If not, the behaviour is undefined.
        
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
            xvalues, yvalues, cond = np.broadcast_arrays(xvalues, yvalues, cond)
        else:
            xvalues, yvalues = np.broadcast_arrays(xvalues, yvalues)
        
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
            fig.colorbar(im, label=f'Count per bin ({xstep:.3g} x {ystep:.3g})')
        
        textbox.textbox(ax, f'{len(x)} entries', fontsize='small', loc='upper right')
        if where is not None:
            s = breaklines.breaklines(f'Selection: {where}', 40, ')', '&|')
            textbox.textbox(ax, s, fontsize='small', loc='upper left')
        
        ax.set_xlabel(xexpr)
        ax.set_ylabel(yexpr)
        
        fig.tight_layout()
        return fig

    @classmethod
    def concatenate(cls, aplist):
        """
        Concatenate afterpulse objects.
        
        Parameters
        ----------
        aplist : sequence of AfterPulse instances
            The objects to concatenate. Must not be empty.
        
        Return
        ------
        self : AfterPulse
            The concatenation.
        """
        ap0 = aplist[0]
        outputs = []
        lengths = []
        for ap in aplist:
            assert np.array_equal(ap.filtlengths, ap0.filtlengths)
            outputs.append(ap.output)
            lengths += list(getattr(ap, '_catlengths', ap.output.shape))
        
        self = cls.__new__(cls)
        
        classattr = vars(cls)
        for k, v in vars(ap0).items():
            if k not in classattr and k != 'output' and not k.startswith('__'):
                setattr(self, k, v)
        
        self.output = np.concatenate(outputs)
        self._catlengths = np.array(lengths)
        assert np.sum(lengths) == len(self.output)
        
        return self
    
    def catindex(self, ievent):
        """
        Get the concatenated object index from the event index.
        
        Parameters
        ----------
        ievent : int
            The index of the event.
        
        Return
        ------
        idx : int
            Zero if the object is not the concatenation of multiple objects.
            Otherwise, the index of the position in the list passed to
            `concatenate` of the object where the requested event originates
            from. If one or more objects passed to `concatenate` where
            themselves the result of a concatenation, the indices are computed
            as if a single call to `concatenate` was done with all the original
            unconcatenated objects in order.
        """
        lengths = getattr(self, '_catlengths', self.output.shape)
        cumlen = np.pad(np.cumsum(lengths), (1, 0))
        idx = np.searchsorted(cumlen, ievent, side='right') - 1
        assert 0 <= idx < len(lengths), idx
        return idx
    
    def subindex(self, ievent):
        """
        Return the event index in a concatenated object.
        
        Parameters
        ----------
        ievent : int
            The global event index on the concatenation.
        
        Return
        ------
        jdx : int
            The event index relative to the object where the event originates.
        """
        lengths = getattr(self, '_catlengths', self.output.shape)
        cumlen = np.pad(np.cumsum(lengths), (1, 0))
        idx = np.searchsorted(cumlen, ievent, side='right') - 1
        assert 0 <= idx < len(lengths), idx
        jdx = ievent - cumlen[idx]
        assert 0 <= jdx < lengths[idx]
        return jdx
