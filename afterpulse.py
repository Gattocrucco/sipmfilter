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
import peaksampl

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

def _posampl1(x):
    return x / x[np.argmax(np.abs(x))]

class AfterPulse(npzload.NPZLoad):
    
    def __init__(self,
        wavdata,
        template,
        filtlengths=None,
        batch=10,
        pbar=False,
        lowsample=700,
        trigger=None,
        badvalue=-1000,
    ):
        """
        Analyze LNGS laser data.
    
        The analysis is appropriate for studying afterpulses but it can be
        used as a general purpose tool.
        
        Parameters
        ----------
        wavdata : array (nevents, nchannels, eventlength)
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
        badvalue : scalar
            Filler used for missing values, default -1000.
        
        Methods
        -------
        filtertempl : get a cross correlation filter template.
        closept : find pre-trigger peaks close to the trigger.
        npeboundaries : divide peak heights in pe bins.
        npe : assign heights to pe bins.
        fingerplot : plot fingerplot used to determine pe bins.
        computenpe : assign height to pe bins for all filters at once.
        computenpeboundaries : get the pe bins used by `computenpe`.
        signal : compute the filtered signal shape.
        signals : compute the filtered signal shape for all filters.
        peaksampl : compute the peak amplitude subtracting other peaks.
        apheight : correct the amplitude to be constant for afterpulses.
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
        lowsample, badvalue : scalar
            The arguments given at initialization.
        filtlengths : array
            The cross correlation filter lengths.
        eventlength : int
            The size of the last axis of `wavdata`.
        trigger : int array (nevents,)
            The `trigger` parameter, if specified.
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
                prominence peaks in the region before the trigger.
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
        template : 1D array
            The full signal template.
        """
        
        if filtlengths is None:
            self.filtlengths = 2 ** np.arange(5, 11 + 1)
        else:
            self.filtlengths = np.array(filtlengths, int)
            
        self.lowsample = lowsample
        self.eventlength = wavdata.shape[-1]
        assert badvalue < 0, badvalue
        self.badvalue = badvalue
        
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
        
        self.output.flags['WRITEABLE'] = False
        
    def _maketemplates(self, template):
        """
        Fill the `templates` and `template` members.
        """
        kw = dict(timebase=1, aligned=True)
        
        def templates(lengths):
            templates = np.zeros(lengths.shape, dtype=[
                ('template', float, (np.max(lengths),)),
                ('length', int),
                ('offset', int),
            ])
            templates['length'] = lengths
            for _, entry in np.ndenumerate(templates):
                templ, offset = template.matched_filter_template(entry['length'], **kw)
                assert len(templ) == entry['length']
                entry['template'][:len(templ)] = templ
                entry['length'] = len(templ)
                entry['offset'] = offset
            return templates
        
        self.templates = templates(self.filtlengths)
        
        kw.update(randampl=False)
        self.template, = template.generate(template.template_length, [0], **kw)
    
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
            
            # filter
            lpad = 100
            filtered = correlate.correlate(wavdata[:, 0], templ, boundary=mean_baseline, lpad=lpad)
            
            # divide the waveform in regions
            center = start + startoffset + int(offset)
            margin = 80
            left = max(start, center - margin)
            right = center + margin
            poststart = lpad + left

            # find the main peak as the minimum local minimum near the trigger
            searchrange = filtered[:, poststart:lpad + right]
            mainpos = argminrelmin.argminrelmin(searchrange, axis=-1)
            mainheight = searchrange[np.arange(len(mainpos)), mainpos]
            
            # find two other peaks with high prominence after the main one
            posttrigger = filtered[:, poststart:]
            minorstart = np.maximum(0, mainpos)
            minorend = np.full(*posttrigger.shape)
            minorpos, minorprom = maxprominencedip.maxprominencedip(posttrigger, minorstart, minorend, baseline, 2)
            minorheight = posttrigger[np.arange(len(minorpos))[:, None], minorpos]
            
            # find peaks in pre-trigger region
            ptstart = np.zeros(len(filtered))
            ptend = poststart + minorstart + 1
            ptpos, ptprom = maxprominencedip.maxprominencedip(filtered, ptstart, ptend, baseline, 2)
            ptheight = filtered[np.arange(len(ptpos))[:, None], ptpos]

            idx = np.s_[:,] + ilength
            badvalue = self.badvalue
                        
            # fill main peak
            peak = output['mainpeak'][idx]
            bad = mainpos < 0
            peak['pos'] = np.where(bad, badvalue, left + mainpos)
            peak['height'] = np.where(bad, badvalue, baseline - mainheight)
            
            # fill other peaks
            peaks = [
                ('minorpeak', minorpos, minorheight, minorprom, left ),
                ('ptpeak'   , ptpos   , ptheight   , ptprom   , -lpad),
            ]
            for prefix, peakpos, height, prom, offset in peaks:
                for ipeak, s in [(1, ''), (0, '2')]:
                    peak = output[prefix + s][idx]
                    pos = peakpos[:, ipeak]
                    bad = pos < 0
                    peak['pos'] = np.where(bad, badvalue, np.maximum(0, offset + pos))
                    peak['height'] = np.where(bad, badvalue, baseline - height[:, ipeak])
                    peak['prominence'] = np.where(bad, badvalue, prom[:, ipeak])
            
            output['internals']['left'][idx] = left
            output['internals']['right'][idx] = right
        
        output['trigger'] = trigger
        output['baseline'] = baseline
        output['saturated'] = saturated
        output['lowbaseline'] = lowbaseline
        output['internals']['start'] = start
        output['internals']['bsevent'][lowbaseline] = self._default_baseline[0]
        output['internals']['lpad'] = lpad
        output['done'] = True
    
    def closept(self, safedist=2500, safeheight=8):
        """
        Find events where there are pre-trigger peaks close to the trigger.
                
        The pre-trigger pulses are identified using the longest filter.
        
        Parameters
        ----------
        safedist : scalar
            Distance from a peak in the pre-trigger region to the trigger above
            which the peak is not considered close, default 2500.
        safeheight : scalar
            The maximum height for peaks within `safedist` to be still
            considered negligible, default 8.
        
        Return
        ------
        closept : bool array (nevents,)
            True where there's a peak.
        """
        ptlength = self.output['internals']['start']
        idx = (slice(None),) + self._max_ilength()
        closept = None
        for s in ['', '2']:
            peak = self.output['ptpeak' + s][idx]
            pos = peak['pos']
            height = peak['height']
            
            close = pos >= 0
            close &= ptlength - pos < safedist
            close &= height > safeheight
            if closept is None:
                closept = close
            else:
                closept |= close
        
        return closept
    
    @functools.cached_property
    def _closept(self):
        return self.closept()
    
    def npeboundaries(self, height, plot=False, algorithm='maxdiff'):
        """
        Determine boundaries to divide peak heights by number of pe.
        
        Parameters
        ----------
        height : 1D array
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
        
        if algorithm == 'midpoints':
            last = center[-1] + (center[-1] - center[-2])
            center = np.pad(center, (0, 1), constant_values=last)
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
        
        idx = np.s_[:,] + ilength
        peak = self.output['mainpeak'][idx]
        height = peak['height']
        good = ~self.output['saturated'] & ~self._closept
        valid = peak['pos'] >= 0
        value = height[valid & good]
        
        if len(value) >= 100:
            boundaries, fig = self.npeboundaries(value, plot=True)
            ax, = fig.get_axes()
            length = self.filtlengths[ilength]
            textbox.textbox(ax, f'filter length = {length} ns', loc='center right', fontsize='small')
        
            fig.tight_layout()
            return fig
    
    @functools.cached_property
    def _computenpe_boundaries(self):
        boundaries = np.empty(self.filtlengths.shape, object)
        mainpeak = self.output['mainpeak']
        good = ~self.output['saturated'] & ~self._closept
        for ilength, _ in np.ndenumerate(self.filtlengths):
            idx = np.s_[:,] + ilength
            peak = mainpeak[idx]
            height = peak['height']
            valid = peak['pos'] >= 0
            boundaries[ilength] = self.npeboundaries(height[valid & good])
        return boundaries
    
    def computenpe(self, height):
        """
        Compute the number of pe of peaks.
        
        The pe bin boundaries are computed automatically from laser peaks
        fingerplots.
        
        Parameters
        ----------
        height : array filtlengths.shape + (nevents,)
            The height of a peak.
        
        Return
        ------
        npe : int array filtlengths.shape + (nevents,)
            The pe assigned to each peak height. 1000 for height larger than
            the highest classified pe. Negative heights are assigned 0 pe.
        """
        boundaries = self._computenpe_boundaries
        npe = np.empty(self.filtlengths.shape + self.output.shape, int)
        for ilength, _ in np.ndenumerate(self.filtlengths):
            npe[ilength] = self.npe(height[ilength], boundaries[ilength])
        return npe
    
    def computenpeboundaries(self, ilength):
        """
        Get the pe height bins used by `computenpe`.
        
        Parameters
        ----------
        ilength : {int, tuple}
            The index of the filter length in `filtlengths`.
        
        Return
        ------
        boundaries : 1D array
            See `npeboundaries` for a detailed description.
        """
        return self._computenpe_boundaries[ilength]
    
    def signal(self, ilength):
        """
        Filter the signal template.
        
        Parameters
        ----------
        ilength : {int, tuple of ints}
            The index of the filter length.
        
        Return
        ------
        s : 1D array
            The filtered signal waveform. It is positive and the amplitude is 1.
        """
        templ, _ = self.filtertempl(ilength)
        s = np.correlate(self.template, templ, 'full')
        return _posampl1(s)
    
    def signals(self):
        """
        Filter the signal template with all filters.
        
        Equivalent to calling `signal` for each filter length.
        
        Return
        ------
        signals : array filtlengths.shape + (N,)
            An array with the filtered signals. The length of the last axis
            is the maximum signal length. Shorter signals are padded with zeros.
        """
        maxflen = np.max(self.filtlengths)
        maxlen = len(self.template) + maxflen - 1
        signal = np.zeros(self.filtlengths.shape + (maxlen,))
        for ilength, _ in np.ndenumerate(self.filtlengths):
            s = self.signal(ilength)
            signal[ilength][:len(s)] = s
        return signal
    
    def peaksampl(self, minheight='auto', minprom=0, fillignore=None):
        """
        Compute the amplitude of filtered signals.
        
        The amplitude is the height that the peak in the filter output would
        have if there were not other signals nearby.
        
        Parameters
        ----------
        minheight : {array_like, 'auto'}
            Minimum height of a peak to be considered a signal. If 'auto'
            (default), use the boundary between 0 and 1 pe returned by
            `computenpeboundaries`. If an array it must broadcast against
            `filtlengths`.
        minprom : array_like
            The minimum prominence (does not apply to the laser peak). Default
            0.
        fillignore : scalar, optional
            The value used for ignored peaks, `badvalue` if not specified.
        
        Return
        ------
        ampl : array filtlengths.shape + (nevents, 5)
            The amplitude (positive). The last axis runs over peaks 'mainpeak',
            'minorpeak', 'minorpeak2', 'ptpeak', 'ptpeak2'.
        """
        if minheight == 'auto':
            minheight = np.reshape([
                self.computenpeboundaries(ilength)[0]
                for ilength, _ in np.ndenumerate(self.filtlengths)
            ], self.filtlengths.shape)
        
        minheight = np.broadcast_to(minheight, self.filtlengths.shape)
        minprom = np.broadcast_to(minprom, self.filtlengths.shape)
        
        signal = self.signals()[..., None, :]
        
        peaks = ['mainpeak', 'minorpeak', 'minorpeak2', 'ptpeak', 'ptpeak2']
            
        pos, height, prom = [
            np.stack([
                np.moveaxis(self.output[peak][label], 0, -1)
                if not (label == 'prominence' and peak == 'mainpeak')
                else np.broadcast_to(np.inf, self.filtlengths.shape + self.output.shape)
                for peak in peaks
            ], axis=-1)
            for label in ['pos', 'height', 'prominence']
        ]

        missing = pos < 0
        ignore  = height < minheight[..., None, None]
        ignore |=   prom <   minprom[..., None, None]

        cond = missing | low
        height[cond] = 0
        pos[cond] = 0
        ampl = peaksampl.peaksampl(signal, height, pos)

        ampl[ignore] = self.badvalue if fillignore is None else fillignore
        ampl[missing] = self.badvalue
        
        return ampl
    
    @functools.cached_property
    def _peaksampl(self):
        return self.peaksampl()
    
    def apheight(self, ampl):
        """
        Compute the afterpulse height.
        
        The amplitude of the post-trigger peaks is added to the main peak tail,
        but the waveform used is unfiltered. The main peak amplitude is fixed
        to 1 pe.
        
        The normalization is still that of the filter output.
        
        Parameters
        ----------
        ampl : array filtlengths.shape + (nevents, 5)
            The amplitude as returned by `peaksampl`.
        
        Return
        ------
        height : array filtlengths.shape + (nevents, 2)
            The last axis correspond to peaks 'minorpeak' and 'minorpeak2'.
        """        
        mainpos, pos1, pos2 = [
            np.moveaxis(self.output[peak]['pos'], 0, -1)
            for peak in ['mainpeak', 'minorpeak', 'minorpeak2']
        ]
        
        peampl = np.empty(self.filtlengths.shape)
        for ilength, _ in np.ndenumerate(self.filtlengths):
            height = ampl[ilength + np.s_[:, 0]]
            l, r = self.computenpeboundaries(ilength)[:2]
            cond = (height >= l) & (height < r)
            peampl[ilength] = np.median(height[cond])
        
        signal = _posampl1(self.template)
        signal = np.pad(signal, 1)
        signal = peampl[..., None] * signal
        signal = signal[..., None, :]
        offset = np.argmax(signal, axis=-1)
        
        height = np.empty(self.filtlengths.shape + self.output.shape + (2,))
        for i, pos in enumerate([pos1, pos2]):
            indices = pos - mainpos + offset
            indices = np.maximum(indices, 0)
            indices = np.minimum(indices, signal.shape[-1] - 1)
            base = np.take_along_axis(signal, indices[..., None], -1)
            amp = ampl[..., i + 1]
            height[..., i] = amp + np.squeeze(base, -1)
            ignore = (pos < 0) | (mainpos < 0) | (amp < 0)
            height[..., i][ignore] = self.badvalue
        
        return height
    
    @functools.cached_property
    def _apheight(self):
        return self.apheight(self._peaksampl)
    
    def plotevent(self, wavdata, ievent, ilength=None, zoom='posttrigger', debug=False):
        """
        Plot a single event.
        
        Parameters
        ----------
        wavdata : array (nevents, 2, 15001) or list
            The same array passed at initialization. If the object is
            a concatenation, the data passed to the object where the event
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
        debug : bool
            If False (default), reduce the amount of information showed.
        
        Return
        ------
        fig : matplotlib figure
            The figure.
        """
        if ilength is None:
            ilength = self._max_ilength()
        elif not isinstance(ilength, tuple):
            ilength = (ilength,)
        
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
        lpad = entry['internals']['lpad']
        filtered = correlate.correlate(wf, templ, boundary=baseline, lpad=lpad)
        length = self.filtlengths[ilength]
        ax.plot(-lpad + np.arange(len(filtered)), filtered, color='#000', label=f'filtered ({length} ns templ)')
                
        left = entry['internals']['left'][ilength]
        right = entry['internals']['right'][ilength]
        offset = -0.5
        ax.axvspan(offset + left, offset + right, color='#ddd', label='laser peak search range')
        
        markerkw = dict(
            linestyle = '',
            markersize = 10,
            markeredgecolor = '#000',
            markerfacecolor = '#fff0',
        )
        
        peaks = [
            # field, label, marker
            ('mainpeak'  , 'laser'       , 'o'),
            ('minorpeak' , '1st posttrig', 's'),
            ('minorpeak2', '2nd posttrig', 'v'),
            ('ptpeak'    , '1st pretrig' , '<'),
            ('ptpeak2'   , '2nd pretrig' , '>'),
        ]
        
        if zoom == 'all':
            xlim = (-lpad, len(wf) - 1)
        elif zoom == 'posttrigger':
            xlim = (left - 500, len(wf) - 1)
        elif zoom == 'main':
            xlim = (left - 200, right + 500)
        else:
            raise KeyError(zoom)
        
        boundaries = self.computenpeboundaries(ilength)
        
        for ipeak, (field, label, marker) in enumerate(peaks):
            peak = entry[field][ilength]
            pos = peak['pos']
            ampl = self._peaksampl[ilength + (ievent, ipeak)]
            if pos < 0:
                continue
            if not debug and (ampl < 0 or not xlim[0] <= pos <= xlim[1]):
                continue
            height = peak['height']
            labels = [label]
            if ampl >= 0:
                npe = np.digitize(ampl, boundaries)
                pe = '' if npe == len(boundaries) else f' ({npe} pe)'
                labels.append(f'a={ampl:.3g}{pe}')
            if debug or ampl < 0:
                labels.append(f'h={height:.3g}')
            if debug and 'prominence' in peak.dtype.names:
                prom = peak['prominence']
                labels.append(f'p={prom:.3g}')
            top = baseline - height
            if ampl >= 0:
                base = top + ampl
                ticks = base - boundaries[:npe + 1]
                ax.vlines(pos, min(top, ticks[-1]), base, zorder=2.1)
                ax.plot(pos, base, 'ok', zorder=2.1)
                wtick = 0.015 * (xlim[1] - xlim[0])
                ax.hlines(ticks, pos - wtick, pos + wtick, zorder=2.1)
            ax.plot(pos, top, label=', '.join(labels), marker=marker, **markerkw)
        
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--')
        ax.grid(which='minor', linestyle=':')
        
        ax.legend(fontsize='small', loc='lower right')
        textbox.textbox(ax, f'Event {ievent}', fontsize='medium', loc='upper center')
        
        ax.set_xlim(xlim)
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
            mainampl    = "self._peaksampl[..., 0]",
            npe         = "self.computenpe(self._peaksampl[..., 0])",
            minorpos    = "self.output['minorpeak']['pos']",
            minorheight = "self.output['minorpeak']['height']",
            minorprom   = "self.output['minorpeak']['prominence']",
            minorampl   = "self._peaksampl[..., 1]",
            minorapampl = "self._apheight[..., 0]",
            thirdpos    = "self.output['minorpeak2']['pos']",
            thirdheight = "self.output['minorpeak2']['height']",
            thirdprom   = "self.output['minorpeak2']['prominence']",
            thirdampl   = "self._peaksampl[..., 2]",
            thirdapampl = "self._apheight[..., 1]",
            ptpos       = "self.output['ptpeak']['pos']",
            ptheight    = "self.output['ptpeak']['height']",
            ptprom      = "self.output['ptpeak']['prominence']",
            ptampl      = "self._peaksampl[..., 3]",
            pt2pos      = "self.output['ptpeak2']['pos']",
            pt2height   = "self.output['ptpeak2']['height']",
            pt2prom     = "self.output['ptpeak2']['prominence']",
            pt2ampl     = "self._peaksampl[..., 4]",
            closept     = "self._closept",
            good        = "~self.output['saturated'] & ~self._closept",
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
            closept     : if there are pre-trigger peaks near the trigger
            good        : ~closept & ~saturated
            mainpos     : the index of the main peak
            mainheight  : the positive height of the main peak
            mainampl    : the height corrected for other peaks of the main peak
            npe         : the number of photoelectrons of the main peak,
                          determined separately with each filter length
            minorpos    : the index of the maximum promince peak after the main
            minorheight : ...its height
            minorprom   : ...its prominence, capped to the baseline, and
                          measured without crossing the main peak
            minorampl   : ...its corrected height
            minorapampl : the ampl plus the height of a 1 pe signal located
                          at mainpos, should not depend on delay for
                          afterpulses
            thirdpos    : the index of the second highest prominence peak
            thirdheight : as above
            thirdprom   : as above
            thirdampl   : as above
            thirdapampl : as above
            ptpos       : the index of the higher prominence peak before the
                          trigger
            ptheight    : as above
            ptprom      : as above
            ptampl      : as above
            pt2pos      : like ptpos, but the second highest
            pt2height   : as above
            pt2prom     : as above
            pt2ampl     : as above
        
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
                    if expr.startswith('self.output[') and expr.endswith(']'):
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
        if values.dtype == bool:
            values = values.astype('i1')
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
            
            for i, (x, y, length) in enumerate(xylength):
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
        
        The `output` member is the concatenation of the corresponding members
        of the concatenated object. Other members are set to the ones of the
        first object in the concatenation, without copying.
        
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
            if k not in classattr and not k.startswith('__'):
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
