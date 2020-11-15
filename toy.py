import numpy as np
import numba

import integrate
from single_filter_analysis import single_filter_analysis

class Noise:
    
    def generate(self, nevents, event_length, generator=None):
        """
        Generate noise with unitary variance.
        
        Parameters
        ----------
        nevents : int
            Number of events i.e. independent chunks of simulated data.
        event_length : int
            Number of samples of each event (@ 125 MSa/s).
        generator : np.random.Generator, optional
            Random number generator.
        
        Return
        ------
        events : array (nevents, event_length)
            Simulated noise.
        """
        if generator is None:
            generator = np.random.default_rng()
        return generator.standard_normal((nevents, event_length))
    
    def compute_spectrum(self, proto0_root_file):
        pass
    
    def save(self, filename):
        pass
    
    def load(self, filename):
        pass

class Template:
    
    def make(self, data, length, mask=None):
        """
        Compute a template to be used for simulating signals and for the matched
        filter. The template is computed as the mean of 1 photoelectron signals.
        
        Parameters
        ----------
        data : array (nevents, 2, 15001)
            LNGS data as read by readwav.readwav().
        length : int
            Number of samples of the template (@ 1 GSa/s), starting from the
            beginning of the trigger impulse.
        mask : bool array (nevents,), optional
            Mask for the data array.
        """
        
        if mask is None:
            mask = np.ones(len(data), bool)
        
        # Run a moving average filter to find and separate the signals by
        # number of photoelectrons.
        trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=1470, delta_ma=1530)
        corr_value = baseline - value[:, 0]
        snr, center, width = single_filter_analysis(corr_value[mask], return_full=True)
        assert snr > 15
        assert len(center) > 2
    
        # Select the data corresponding to 1 photoelectron and subtract the
        # baseline.
        lower = (center[0] + center[1]) / 2
        upper = (center[1] + center[2]) / 2
        selection = (lower < corr_value) & (corr_value < upper) & mask
        t = int(np.median(trigger))
        data1pe = data[selection, 0, t:t + length] - baseline[selection].reshape(-1, 1)
    
        # Compute the waveform as the mean of the signals.
        self.template = np.mean(data1pe, axis=0)
        self.template_rel_std = np.std(np.mean(data1pe, axis=1)) / np.mean(self.template)
        self.template_N = np.sum(selection)
        
        # TODO smooth and window the template
        
        # Compute the baseline distribution.
        bs = baseline[mask]
        self.baseline = np.mean(bs)
        self.baseline_std = np.std(bs)
        
        # Compute the noise standard deviation.
        STDs = np.std(data[:, 0, :t - 100], axis=1)
        self.noise_std = np.sum(STDs, where=mask) / np.count_nonzero(mask)
    
    def generate(self, event_length, signal_loc, generator=None, baseline=True, randampl=True):
        """
        Simulate signals, including the baseline.
        
        Parameters
        ----------
        event_length : int
            Number of samples in each event (@ 125 MSa/s).
        signal_loc : float array (nevents,)
            An array of positions of the signal in each event. The position is
            in unit of samples but it can be non-integer. The position is the
            beginning of the template.
        generator : np.random.Generator, optional
            Random number generator.
        baseline : bool
            If True (default), add a baseline to the signal.
        randampl : bool
            If True (default), vary the amplitude of signals.
        
        Return
        ------
        out : array (nevents, event_length)
            Simulated signals.
        """
        nevents = len(signal_loc)
        
        if generator is None:
            generator = np.random.default_rng()
        
        # Convert the template from 1 GSa/s to 125 MSa/s.
        # TODO interpolation, use Gaussian processes, or something quick from
        # scipy
        tlen1ghz = (len(self.template) // 8) * 8
        template = np.mean(self.template[:tlen1ghz].reshape(-1, 8), axis=1)
        
        out = np.zeros((nevents, event_length))
        
        indices0 = np.arange(nevents)[:, None]
        indices1 = np.array(np.rint(signal_loc), int)[:, None] + np.arange(len(template))
        out[indices0, indices1] = template
        if randampl:
            out *= 1 + self.template_rel_std * generator.standard_normal((nevents, 1))
        if baseline:
            out += self.baseline
            out += self.baseline_std * generator.standard_normal((nevents, 1))
        
        # TODO digitalization
        return out
    
    @property
    def maximum(self):
        """
        Return the average maximum amplitude of the signals generated by
        `generate`.
        """
        return np.max(np.abs(self.template))
    
    def matched_filter_template(self, length, norm=True):
        """
        Return a template for the matched filter. The template is chosen to
        maximize its vector norm.
        
        Parameters
        ----------
        length : int
            Number of samples of the template @ 125 MSa/s.
        norm : bool, optional
            If True (default) the template is normalized to unit sum, so that
            the output from the matched filter is comparable to the output from
            a moving average filter.
        
        Return
        ------
        template : float array (length,)
            The template.
        offset : float
            The offset in unit of 8 nanoseconds from the beginning of the
            template used to generate the fake signals to the beginning of the
            returned template.
        """
        len1ghz = 8 * length
        assert len1ghz <= len(self.template)
        cs = np.concatenate([[0], np.cumsum(self.template ** 2)])
        s = cs[len1ghz:] - cs[:-len1ghz] # s[j] = sum(template[j:j+len1ghz])
        offset1ghz = np.argmax(s)
        offset = offset1ghz / 8
        template = self.template[offset1ghz:offset1ghz + len1ghz]
        template = np.mean(template.reshape(-1, 8), axis=1)
        if norm:
            template /= np.sum(template)
        return template, offset

class Filter:
    
    def __init__(self, events, boundary=0):
        """
        Class to apply various filters to the same piece of data.
        
        Parameters
        ----------
        events : array (nevents, event_length)
            An array of events. The filter is applied separately to each event.
        boundary : scalar
            The past boundary condition, it is like each event has an infinite
            series of samples with value `boundary` before sample 0.
        
        Methods
        -------
        The filter methods are:
            moving_average
            exponential_moving_average
            matched
        
        Each method has the following return signature:
        
        filtered : float array (nevents, event_length)
            filtered[:, i] is the filter output after reading sample
            events[:, i].
        
        and accepts an optional parameter `out` where the output is written to.
        The method `all` computes all the filters in one array.
        
        """
        self.events = events
        self.boundary = boundary
    
    def _add_boundary(self, length):
        shape = self.events.shape
        events = np.empty((shape[0], length + shape[1]))
        events[:, :length] = self.boundary
        events[:, length:] = self.events
        return events
    
    def _out(self, out):
        if out is None:
            return np.empty(self.events.shape)
        else:
            return out
    
    def moving_average(self, nsamples, out=None):
        """
        Parameters
        ----------
        nsamples : int
            The number of averaged samples.
        """
        events = self._add_boundary(nsamples)
        cs = np.cumsum(events, axis=1)
        out = self._out(out)
        out[...] = cs[:, nsamples:]
        out -= cs[:, :-nsamples]
        out /= nsamples
        return out
    
    def exponential_moving_average(self, tau, out=None):
        """
        Parameters
        ----------
        tau : float
            The exponential scale parameter. The filter is
                
                y[i] = a * y[i - 1] + (1 - a) * x[i]
            
            where a = 1 - 1/tau.
        """
        out = self._out(out)
        a = 1 - 1 / tau # Taylor of exp(-1/tau)
        _exponential_moving_average(self.events, a, self.boundary, out)
        return out
    
    def matched(self, template, out=None):
        """
        Parameters
        ----------
        template : 1D array
            The array which is cross-correlated with the signal.
        """
        events = self._add_boundary(len(template) - 1)
        out = self._out(out)
        _correlate(events, template, out)
        return out
    
    def all(self, template):
        """
        Convenience method to run all the filters.
        
        Parameters
        ----------
        template : 1D array
            The template for the matched filter. The averaged number of samples
            and the exponential length scale of the moving average and the
            exponential moving average filters respectively is the length of the
            template.
        
        Return
        ------
        out : float array (4, nevents, event_length)
            The first axis is over filters in this order:
                no filter
                moving average
                exponential moving average
                matched filter
        """
        out = np.empty((4,) + self.events.shape)
        out[0] = self.events
        self.moving_average(len(template), out[1])
        self.exponential_moving_average(len(template), out[2])
        self.matched(template, out[3])
        return out

@numba.jit(cache=True, nopython=True, parallel=True)
def _exponential_moving_average(events, a, boundary, out):
    for i in range(len(events)):
        out[i, 0] = a * boundary + (1 - a) * events[i, 0]
        for j in range(1, events.shape[1]):
            out[i, j] = a * out[i, j - 1] + (1 - a) * events[i, j]

@numba.jit(cache=True, nopython=True, parallel=True)
def _correlate(events, template, out):
    for i in range(len(events)):
        for j in range(out.shape[1]):
            out[i, j] = np.dot(events[i, j:j + len(template)], template)

@numba.jit(cache=True, nopython=True, parallel=True)
def apply_threshold(events, threshold):
    """
    For each event, find the first sample below a threshold
    
    Parameters
    ----------
    events : array (nevents, event_length)
    threshold : scalar
    
    Return
    ------
    signal_indices : int array (nevents, 3)
        The last axis is (threshold crossed, maximum, threshold crossed)
    signal_found : bool array (nevents,)
        True iff the threshold has been crossed.
    """
    nevents, event_length = events.shape
    
    signal_indices = np.zeros((nevents, 3), int)
    signal_found = np.zeros(nevents, bool)
    
    for i in numba.prange(nevents):
        event = events[i]
        
        for j in range(event_length):
            if event[j] < threshold:
                start = j
                break
        else:
            break
        
        for j in range(start + 1, event_length):
            if event[j] > threshold:
                break
        end = j
        
        signal_found[i] = True
        signal_indices[i, 0] = start
        signal_indices[i, 1] = start + np.argmax(event[start:end])
        signal_indices[i, 2] = end
    
    return signal_indices, signal_found

def min_snr_ratio(data, tau, mask=None, nnoise=128, generator=None):
    """
    Compute the signal amplitude over noise rms needed to obtain a given
    filtered signal amplitude over filtered noise rms.
    
    Parameters
    ----------
    data : array (nevents, 2, 15001)
        LNGS data as read by readwav.readwav().
    tau : array (ntau,)
        The length parameter for filters @ 125 MSa/s.
    mask : bool array (nevents,), optional
        A mask for the first axis of `data`.
    nnoise : int
        Approximate effective sample size of generated noise to compute the
        filtered noise standard deviation. The default should give a result
        with at least 3 % precision.
    
    Return
    ------
    snrs : float array (4, ntau)
        The required unfiltered SNR if the filtered SNR is 1. Multiply it by the
        target filtered SNR. The first axis is over filters as in
        toy.Filter.all().
    """
    
    # the following @ 125 MSa/s
    dead_time = 3 * np.max(tau)
    template_length = 512
    noise_event_length = dead_time + nnoise * np.max(tau)
    signal_event_length = dead_time + max(template_length, np.max(tau))
    
    # The noise event length is somewhat high because the filtered noise has
    # an O(tau) autocorrelation length.

    noise = Noise().generate(1, noise_event_length, generator)
    template = Template()
    template.make(data, template_length * 8, mask)
    signal = template.generate(signal_event_length, [dead_time], generator, False, False)
    filt_noise = Filter(noise)
    filt_signal = Filter(signal)

    snrs = np.empty((len(tau), 4))
    for i, t in enumerate(tau):
        mftemp, mfoffset = template.matched_filter_template(t)
        fnoise = filt_noise.all(mftemp)
        fsignal = filt_signal.all(mftemp)
        fnrms = np.std(fnoise[:, 0, dead_time:], axis=-1)
        fsmin = np.abs(np.min(fsignal[:, 0, dead_time:], axis=-1))
        snrs0 = template.maximum
        snrs[i] = snrs0 * fnrms / fsmin

    return snrs

class Toy:
    
    def __init__(self, nevents, event_length, baseline_length, snr_array, threshold_array, tau_array, signal):
        pass
