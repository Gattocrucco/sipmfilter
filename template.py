import numpy as np
import numba
from scipy import signal

import npzload
import firstbelowthreshold
from single_filter_analysis import single_filter_analysis
import argminrelmin

@numba.njit(cache=True)
def meanat(x, t, l):
    """
    x : array (N, M)
    t : int array (N,)
    l : int
    out : array (N,)
        The mean of x from t to t+l.
    """
    out = np.empty(len(x))
    for i, a in enumerate(x):
        out[i] = np.mean(a[t[i]:t[i] + l])
    return out

@numba.njit(cache=True)
def vecmeanat(x, x0, mask, t, l):
    """
    x : array (N, M)
    x0 : array (N,)
    mask : bool array (N,)
    t : int array (N,)
    l : int
    out : array (l,)
        The mean of x - x0 from t to t+l, only where mask.
    """
    out = np.zeros(l)
    count = 0
    for i, a in enumerate(x):
        if mask[i]:
            out += a[t[i]:t[i] + l] - x0[i]
            count += 1
    out /= count
    return out

class Template(npzload.NPZLoad):
    """
    Class to make a signal template.
    
    Class methods
    -------------
    load : load a template from a file.
    from_lngs : make a template from LNGS data.
    
    Methods
    -------
    generate : generate signal waveforms from the template.
    matched_filter_template : make a template for the matched filter.
    maxoffset : return the position of the peak of the template.
    max : amplitude of the template.
    save : save the template to file.
    
    Properties
    ----------
    template_length : the length of the 1 GSa/s template.
    
    Instance variables
    ------------------
    baseline : scalar
        The average baseline.
    """
    
    def __init__(self):
        raise NotImplementedError('use a class method to construct Template objects')
    
    @property
    def template_length(self):
        """
        The length, in ns, of the 1 GSa/s template used to generate waveforms.
        """
        return self.templates.shape[1]
    
    @classmethod
    def from_lngs(cls, data, length, mask=None, trigger=None):
        """
        Compute a template from 1 p.e. signals in an LNGS wav.
        
        Parameters
        ----------
        data : array (nevents, nchannels, 15001)
            Wav data as read by readwav.readwav(). If it has two channels, the
            second channel is the trigger. If there is only one channel,
            specify the trigger position with `trigger`.
        length : int
            Number of samples of the template (@ 1 GSa/s), starting from the
            beginning of the trigger impulse.
        mask : bool array (nevents,), optional
            Mask for the `data` array.
        trigger : int, optional
            Position of the trigger start in the events. If not specified,
            the trigger is read from the second channel of `data`. If specified,
            it supersedes `data` even with two channels.
        
        Return
        ------
        self : Template
            A template object.
        """
        self = cls.__new__(cls)
        
        if mask is None:
            mask = np.ones(len(data), bool)
            
        # Find the trigger.
        if trigger is None:
            hastrigger = True
            trigger = firstbelowthreshold.firstbelowthreshold(data[:, 1], 600)
        else:
            hastrigger = False
            trigger = np.full(len(data), trigger)
        
        # Find spurious signals.
        baseline_zone = data[:, 0, :np.min(trigger) - 100]
        spurious = firstbelowthreshold.firstbelowthreshold(baseline_zone, 700) >= 0
        mask = mask & ~spurious
        
        # Count photoelectrons using the average.
        baseline = np.mean(baseline_zone, axis=-1)
        value = meanat(data[:, 0], trigger, 1500)
        corr_value = baseline - value
        snr, center, width = single_filter_analysis(corr_value[mask], return_full=True)
        minsnr = 5
        assert snr >= minsnr, f'SNR = {snr:.3g} < {minsnr}'
        assert len(center) > 2
    
        # Select the data corresponding to 1 photoelectron.
        lower = (center[0] + center[1]) / 2
        upper = (center[1] + center[2]) / 2
        selection = (lower < corr_value) & (corr_value < upper) & mask
    
        # Compute the waveform as the mean of the signals.
        mtrig = np.full(len(trigger), np.median(trigger))
        template = vecmeanat(data[:, 0], baseline, selection, mtrig, length)
        
        # Do it with alignment.
        if hastrigger:
            start = trigger
        else:
            delta = 100
            t = trigger[0]
            filtered = signal.fftconvolve(data[selection, 0, t - delta:t + delta + length], -template[None, ::-1], axes=-1, mode='valid')
            indices = np.flatnonzero(selection)
            assert filtered.shape == (len(indices), 2 * delta + 1)
            idx = argminrelmin.argminrelmin(filtered, axis=-1)
            selection[indices] &= idx >= 0
            start = np.zeros(len(data), int)
            start[indices] = t - delta + idx
        template_aligned = vecmeanat(data[:, 0], baseline, selection, start, length)
            
        self.templates = np.stack([template, template_aligned])
        self.template_rel_std = np.sqrt(width[1] ** 2 - width[0] ** 2) / center[1]
        self.template_N = np.sum(selection)
        
        # For the moving average.
        self._cumsum_templates = np.pad(np.cumsum(self.templates, axis=1), [(0,0),(1,0)])
        
        # Compute the noise standard deviation.
        STDs = np.std(baseline_zone, axis=1)
        self.noise_std = np.mean(STDs[mask])
        
        # Compute the baseline distribution.
        bs = baseline[mask]
        self.baseline = np.mean(bs)
        self.baseline_std = np.std(bs)
        
        # Save aligned template start distribution.
        trigarray = start[selection]
        self.start_min = np.min(trigarray)
        self.start_max = np.max(trigarray)
        self.start_median = np.median(trigarray)
        
        return self
    
    def _ma_template(self, n, idx=0):
        """apply a n-moving average to the 1 GSa/s template"""
        cs = self._cumsum_templates[idx]
        x = (cs[n:] - cs[:-n]) / n
        return x
    
    def generate(self, event_length, signal_loc, generator=None, randampl=True, timebase=8, aligned=False):
        """
        Simulate signals.
        
        Parameters
        ----------
        event_length : int
            Number of samples in each event.
        signal_loc : float array (nevents,)
            An array of positions of the signal in each event. The position is
            in unit of samples but it can be non-integer. The position is the
            beginning of the template.
        generator : np.random.Generator, optional
            Random number generator.
        randampl : bool
            If True (default), vary the amplitude of signals.
        timebase : int
            Duration of a sample in nanoseconds. Default is 8 i.e. 125 MSa/s.
        aligned : bool
            If True, use the template made with trigger alignment.
        
        Return
        ------
        out : array (nevents, event_length)
            Simulated signals.
        """
        # TODO alignment bug: when signal_loc=[0] the template starts from
        # the second sample.
        
        signal_loc = np.asarray(signal_loc)
        nevents = len(signal_loc)
        
        out = np.zeros((nevents, event_length))
        
        loc_int = np.array(np.floor(signal_loc), int)
        loc_ns = np.array(np.floor(signal_loc * timebase), int) % timebase
        loc_subns = (signal_loc * timebase) % 1
                
        idx = 1 if aligned else 0
        templ = self._ma_template(timebase, idx)
        tlen = ((len(templ) - 1) // timebase) * timebase

        indices0 = np.arange(nevents)[:, None]
        indices1 = loc_int[:, None] + np.arange(tlen // timebase)
        
        tindices = 1 - loc_ns[:, None] + np.arange(0, tlen, timebase)
        weight = loc_subns[:, None]
        out[indices0, indices1] = (1 - weight) * templ[tindices] + weight * templ[tindices - 1]
        
        if randampl:
            if generator is None:
                generator = np.random.default_rng()
            out *= 1 + self.template_rel_std * generator.standard_normal((nevents, 1))
        
        return out
    
    def max(self, timebase=8, aligned=False):
        """
        Compute the average maximum amplitude of the signals generated by
        `generate` if the signal location varies randomly.
        
        Parameters
        ----------
        timebase : int
            Duration of a sample in nanoseconds. Default is 8 i.e. 125 MSa/s.
        aligned : bool
            If True, use the template made with trigger alignment.
    
        Return
        ------
        ampl : scalar
            The average maximum signal amplitude (positive).
        """
        event_length = 2 + self.templates.shape[1] // timebase
        signal_loc = np.linspace(0, 1, 101)[:-1]
        signals = self.generate(event_length, signal_loc, randampl=False, timebase=timebase, aligned=aligned)
        return np.mean(np.max(np.abs(signals), axis=1))
    
    def maxoffset(self, timebase=8, aligned=False):
        """
        Time from the start of the template to the maximum.
        
        Parameters
        ----------
        timebase : int
            The unit of time in nanoseconds.
        aligned : bool
            If True, use the template made with trigger alignment.
        """
        idx = 1 if aligned else 0
        return np.argmax(np.abs(self.templates[idx])) / timebase
    
    def matched_filter_template(self, length, norm=True, timebase=8, aligned=False):
        """
        Return a template for the matched filter. The template is chosen to
        maximize its vector norm.
        
        Parameters
        ----------
        length : int
            Number of samples of the template.
        norm : bool, optional
            If True (default) the template is normalized to unit sum, so that
            the output from the matched filter is comparable to the output from
            a moving average filter.
        timebase : int
            The original template is at 1 GSa/s. The returned template is
            downsampled by this factor. Default is 8 (125 MSa/s).
        aligned : bool
            If True, use the template made with trigger alignment.
        
        Return
        ------
        template : float array (length,)
            The template.
        offset : float
            The offset in unit of sample number of the returned template from
            the beginning of the template used to generate the fake signals to
            the beginning of the returned template.
        """
        len1ghz = timebase * length
        assert len1ghz <= self.templates.shape[1]
        idx = 1 if aligned else 0
        template = self.templates[idx]
        cs = np.pad(np.cumsum(template ** 2), (1, 0))
        s = cs[len1ghz:] - cs[:-len1ghz] # s[j] = sum(template[j:j+len1ghz]**2)
        offset1ghz = np.argmax(s)
        offset = offset1ghz / timebase
        template = template[offset1ghz:offset1ghz + len1ghz]
        template = np.mean(template.reshape(-1, timebase), axis=1)
        if norm:
            template /= np.sum(template)
        return template, offset
