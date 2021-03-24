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

class AfterPulse(npzload.NPZLoad):
    
    def __init__(self,
        wavdata,
        template,
        filtlengths=None,
        batch=10,
        pbar=False,
        ptlength=2048,
        lowsample=700,
        safedist=2500,
        safeheight=8,
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
            A series of template lengths for the cross correlation filters used
            in the post-trigger region, in unit of 1 GSa/s samples. If not
            specified, a logarithmically spaced range from 32 ns to 2048 ns is
            used.
        batch : int
            The events batch size, default 10.
        pbar : bool
            If True, print a progressbar during the computation (1 tick per
            batch). Default False.
        ptlength : int
            Length of the cross correlation filter template used in the
            pre-trigger region, default 2048.
        lowsample : scalar
            Threshold on single samples in the pre-trigger region to switch to
            a default baseline instead of computing it from the event, default
            700.
        safedist : scalar
            Distance from a peak in the pre-trigger region to the trigger above
            which the peak is assumed not to influence the post-trigger region,
            default 2500.
        safeheight : scalar
            The maximum height for peaks within `safedist` to be still
            considered negligible, default 8.
        trigger : int array (nevents,), optional
            The trigger leading edge position, used if there's no trigger
            information in `wavdata`.
        
        Methods
        -------
        fingerplot : plot a histogram of main peak height.
        plotevent : plot the analysis of a single event.
        getexpr : compute the value of an expression on all events.
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
        ptlength, lowsample, safedist, safeheight : scalar
            The arguments given at initialization.
        filtlengths : array
            The post-trigger cross correlation filter lengths.
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
                'npe' : int
                    The (likely) number of photoelectrons.
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
            'ptpeak', 'ptpeak2' :
                Analogous to 'minorpeak' and 'minorpeak2' for the two highest
                prominence peaks in the region before the trigger, searched
                only with the longest filter.
            'good' : bool
                If 'saturated' is False and the two detected peaks before the
                trigger either are at least `safedist` samples before the
                trigger or have height not above `safeheight`.
            'internals' : structured field
                Internals used by the object.
            'done' : bool
                True.
        templates : array filtlengths.shape
            A structured array containing the post-trigger cross correlation
            filter templates with these fields:
            'template' : 1D array
                The template, padded to the right with zeros.
            'length' : int
                The length of the nonzero part of 'template'.
            'offset' : float
                The number of samples from the trigger to the beginning of
                the truncated template.
        pttemplate : 0d array
            An array like `templates` with the pre-trigger template.
        """
        
        if filtlengths is None:
            self.filtlengths = np.logspace(5, 11, 2 * (11 - 5) + 1, base=2).astype(int)
        else:
            self.filtlengths = np.array(filtlengths, int)
            
        self.ptlength = ptlength
        self.lowsample = lowsample
        self.safedist = safedist
        self.safeheight = safeheight
        
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
                ('npe', int),
            ], self.filtlengths.shape),
            ('minorpeak', peakdtype, self.filtlengths.shape),
            ('minorpeak2', peakdtype, self.filtlengths.shape),
            ('ptpeak', peakdtype),
            ('ptpeak2', peakdtype),
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
        
        self.output['mainpeak']['npe'] = -1
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
        
        self._computegood()
        self._computenpe()
    
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
        self.pttemplate = templates(np.array(self.ptlength))
    
    def _mftempl(self, ilength=None):
        """
        Return template, offset. If length not specified, return pre-trigger
        template.
        """
        if ilength is None:
            entry = self.pttemplate[()]
        else:
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
            templ, offset = self._mftempl(ilength)
            
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
        templ, offset = self._mftempl()
        lpad = 100
        filtered = correlate.correlate(wavdata[:, 0, :start], templ, boundary=mean_baseline, lpad=lpad)
        ptpos, ptprom = maxprominencedip.maxprominencedip(filtered, np.zeros(len(filtered)), baseline, 2)
        ptheight = filtered[np.arange(len(ptpos))[:, None], ptpos]
        for ipeak, s in [(1, ''), (0, '2')]:
            pos = ptpos[:, ipeak]
            height = baseline - ptheight[:, ipeak]
            bad = pos < 0
            
            ptpeak_out = output['ptpeak' + s]
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
    
    def _computegood(self):
        """
        Fill the `good` field.
        """
        good = ~self.output['saturated']
        ptlength = self.output['internals']['start']
        for s in ['', '2']:
            peak = self.output['ptpeak' + s]
            pos = peak['pos']
            height = peak['height']
            bad = pos < 0
            
            notgood = ptlength - pos < self.safedist
            notgood &= height > self.safeheight
            good &= bad | ~notgood
        
        self.output['good'] = good
    
    def _fingerplot(self, ilengths, writenpe=False, fig=None):
        """
        Shared implementation of `_computenpe` and `fingerplot`.
        """
        for ilength in ilengths:
            idx = (slice(None),) + ilength
            value = self.output['mainpeak']['height'][idx]
            good1 = self.output['good']
            good2 = self.output['mainpeak']['pos'][idx] >= 0
            good_value = value[good1 & good2]
            
            if len(good_value) >= 100:
        
                _, center, _ = single_filter_analysis(good_value, return_full=True, fig1=fig)
                
                if writenpe:
                    bins = (center[1:] + center[:-1]) / 2
                    npe = np.digitize(value[good2], bins)
                    self.output['mainpeak']['npe'][(good2,) + ilength] = npe
    
    def _computenpe(self):
        """
        Compute the number of photoelectrons from a fingerplot.
        """
        self._fingerplot(np.ndindex(*self.filtlengths.shape), writenpe=True)
    
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
        
        fig = plt.figure(num='afterpulse.AfterPulse.fingerplot', clear=True)
        
        self._fingerplot([ilength], fig=fig)
        
        axs = fig.get_axes()
        if len(axs) > 0:
            ax, = axs
            length = self.filtlengths[ilength]
            textbox.textbox(ax, f'filter length = {length} ns', loc='center right', fontsize='small')
        
        fig.tight_layout()
        return fig
    
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
            ilength_flat = np.argmax(self.filtlengths)
            ilength = np.unravel_index(ilength_flat, self.filtlengths.shape)
        
        fig, ax = plt.subplots(num='afterpulse.AfterPulse.plotevent', clear=True)
        
        if not hasattr(wavdata, 'astype'):
            wavdata = wavdata[self.catindex(ievent)]
        wf = wavdata[self.subindex(ievent), 0]
        ax.plot(wf, color='#f55')
        
        entry = self.output[ievent]
        ax.axvline(entry['trigger'], color='#000', linestyle='--', label='trigger')
        
        baseline = entry['baseline']
        ax.axhline(baseline, color='#000', linestyle=':', label='baseline')
        
        templ, _ = self._mftempl(ilength)
        start = entry['internals']['start']
        filtered = correlate.correlate(wf[start:], templ, boundary=baseline)
        length = self.filtlengths[ilength]
        ax.plot(start + np.arange(len(filtered)), filtered, color='#000', label=f'filtered ({length} ns template)')
        
        templ, _ = self._mftempl()
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
            npe = entry['mainpeak']['npe'][ilength]
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
        heights and prominences are set to -1000.
        
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
        if allow_numpy:
            globals = {
                k: v
                for k, v in vars(np).items()
                if not k.startswith('_')
                and not k[0].isupper()
            }
        else:
            globals = {}

        variables = dict(
            event       = np.arange(len(self.output)),
            trigger     = self.output['trigger'],
            baseline    = self.output['baseline'],
            length      = self.filtlengths,
            saturated   = self.output['saturated'],
            lowbaseline = self.output['lowbaseline'],
            good        = self.output['good'],
            mainpos     = self.output['mainpeak']['pos'],
            mainheight  = self.output['mainpeak']['height'],
            npe         = self.output['mainpeak']['npe'],
            minorpos    = self.output['minorpeak']['pos'],
            minorheight = self.output['minorpeak']['height'],
            minorprom   = self.output['minorpeak']['prominence'],
            thirdpos    = self.output['minorpeak2']['pos'],
            thirdheight = self.output['minorpeak2']['height'],
            thirdprom   = self.output['minorpeak2']['prominence'],
            ptpos       = self.output['ptpeak']['pos'],
            ptheight    = self.output['ptpeak']['height'],
            ptprom      = self.output['ptpeak']['prominence'],
            pt2pos      = self.output['ptpeak2']['pos'],
            pt2height   = self.output['ptpeak2']['height'],
            pt2prom     = self.output['ptpeak2']['prominence'],
        )
        
        class VariablesDict(dict):
            
            def __getitem__(self2, key):
                v = super().__getitem__(key)
                v = v.view()
                v.flags['WRITEABLE'] = False
                
                if v.shape == self.filtlengths.shape:
                    v.shape = self.filtlengths.shape + (1,)
                elif v.shape == self.output.shape:
                    pass
                elif v.shape == self.output.shape + self.filtlengths.shape:
                    v = np.moveaxis(v, 0, -1)
                else:
                    raise ValueError(f'unrecognized shape {v.shape} for variable {key}')
                
                if forcebroadcast:
                    v = np.broadcast_to(v, self.filtlengths.shape + self.output.shape)
                
                if hasattr(self2, 'cond'):
                    v, cond = np.broadcast_arrays(v, self2.cond)
                    v = v[cond]
                
                return v
        
        locals = VariablesDict(variables)
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
        mask = np.any(mask, axis=tuple(range(len(self.filtlengths.shape))))
        return np.flatnonzero(mask)
    
    def hist(self, expr, where=None, yscale='linear', nbins='auto'):
        """
        Plot the histogram of an expression.
        
        The values are histogrammed separately for each filter length.
        
        The expression must evaluate to an array broadcastable to shape
        filtlengths.shape + (nevents,). If not, the behaviour is undefined.
        
        Parameters
        ----------
        expr : str
            A python expression. See the method `getexpr` for an explanation.
        where : str
            An expression for a boolean condition to select the values of
            `expr`.
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
        
        if values.ndim == 1:
            if where is not None:
                values = values[cond]
            if len(values) > 0:
                histkw = dict(
                    bins = self._binedges(values, nbins=nbins),
                    histtype = 'step',
                    color = '#600',
                    zorder = 2,
                )
                ax.hist(values, **histkw)
            textbox.textbox(ax, f'{len(values)} events', fontsize='small', loc='upper right')
        else:
            for ilength, length in np.ndenumerate(self.filtlengths):
                x = values[ilength]
                if where is not None:
                    x = x[cond[ilength]]
                if len(x) > 0:
                    iflat = np.ravel_multi_index(ilength, self.filtlengths.shape) if ilength else 0
                    histkw = dict(
                        bins = self._binedges(x, nbins=nbins),
                        histtype = 'step',
                        color = '#600',
                        label = f'{length} ({len(x)})',
                        zorder = 2,
                        alpha = (1 + iflat) / self.filtlengths.size,
                    )
                    ax.hist(x, **histkw)
            ax.legend(title='Filter length (events)', fontsize='small', ncol=2, loc='upper right')
        
        if where is not None:
            s = breaklines.breaklines(f'Selection: {where}', 40, ')', '&|')
            textbox.textbox(ax, s, fontsize='small', loc='upper left')
        
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
            textbox.textbox(ax, f'{len(xvalues)} events', fontsize='small', loc='upper right')
        else:
            for ilength, length in np.ndenumerate(self.filtlengths):
                x = xvalues[ilength]
                y = yvalues[ilength]
                if where is not None:
                    x = x[cond[ilength]]
                    y = y[cond[ilength]]
                if len(x) > 0:
                    iflat = np.ravel_multi_index(ilength, self.filtlengths.shape)
                    plotkw.update(
                        label = f'{length} ({len(x)})',
                        alpha = (1 + iflat) / self.filtlengths.size,
                    )
                    ax.plot(x, y, **plotkw)
            ax.legend(title='Filter length (events)', fontsize='small', ncol=2, loc='upper right')
        
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
            fig.colorbar(im, label=f'Counts per bin ({xstep:.3g} x {ystep:.3g})')
        
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
            assert ap.ptlength == ap0.ptlength
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
