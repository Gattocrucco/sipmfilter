import numpy as np
from numpy.lib import format as nplf
import numba
import tqdm

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
        
        # TODO this method should be a __init__ really man
        
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
        Average maximum amplitude of the signals generated by `generate`.
        """
        return np.max(np.abs(self.template))
    
    @property
    def snr(self):
        """
        Average peak signal amplitude over noise rms.
        """
        return self.maximum / self.noise_std
    
    def maxoffset(self):
        """
        Time from the start of the template to the maximum, in units of 8 ns.
        """
        return np.argmax(np.abs(self.template)) / 8
    
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

@numba.jit(cache=True, nopython=True)
def _exponential_moving_average(events, a, boundary, out):
    for i in numba.prange(len(events)):
        out[i, 0] = a * boundary + (1 - a) * events[i, 0]
        for j in range(1, events.shape[1]):
            out[i, j] = a * out[i, j - 1] + (1 - a) * events[i, j]

@numba.jit(cache=True, nopython=True)
def _correlate(events, template, out):
    for i in numba.prange(len(events)):
        for j in range(out.shape[1]):
            out[i, j] = np.dot(events[i, j:j + len(template)], template)

@numba.jit(cache=True, nopython=True)
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
        The last axis is (threshold crossed down, minimum, threshold crossed up)
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
        signal_indices[i, 1] = start + np.argmin(event[start:end])
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
    generator : np.random.Generator, optional
        A random number generator used to simulate the noise.
    
    Return
    ------
    snrs : float array (ntau, 4)
        The required unfiltered SNR if the filtered SNR is 1. Multiply it by the
        target filtered SNR. The second axis is over filters as in
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
    
    @staticmethod
    def makesnr(data, mask, tau, generator, template, nsnr, min_filtered_snr=6, max_snr_ratio=1.4):
        """
        Generate the `snr` argument for the initialization of Toy().
        
        Parameters
        ----------
        data : array (nevents, 2, 15001)
            LNGS data as read by readwav.readwav().
        mask : bool array (nevents,)
            Mask for `data`.
        tau : array (ntau,)
            The values of filter length parameters.
        generator : np.random.Generator
            Random generator used to generate noise.
        template : toy.Template
            A template object used to get the signal to noise ratio of 1 p.e.
            signals.
        nsnr : int
            The number of snr values in each range.
        min_filtered_snr : scalar
            The minimum unfiltered snr is chosen to obtain this minimum
            filtered snr with the moving average filter.
        max_snr_ratio : scalar
            The maximum snr is the snr obtained from the template object times
            this constant.
        
        Return
        ------
        snr : array (ntau, nsnr)
            The unfiltered snr ranges for each tau.
            
        """
        snrs = min_snr_ratio(data, tau, mask, generator=generator)
        snr0 = template.snr
        return np.array([
            np.linspace(snrs[i, 1] * min_filtered_snr, snr0 * max_snr_ratio, nsnr)
            for i in range(len(snrs))
        ])
    
    def __init__(self, data, mask, tau, snr=10, bslen=1024, bsoffset=32):
        """
        A toy object simulates 1 p.e. signals with noise, each signal in a
        separate "event", and localize the signal with filters, for a range of
        values of the filter parameters and the SNR.
        
        The time base is 125 Msa/s.
        
        Parameters
        ----------
        data : array (N, 2, 15001)
            LNGS data as read by readwav.readwav().
        mask : bool array (N,) or None
            Mask for the `data` array.
        tau : array (ntau,)
            The values of the filter length parameters.
        snr : int or array (ntau, nsnr)
            If an integer, for each tau value an appropriate range of SNR values
            is generated and the number of such values is `snr`. If an array, it
            must already contain such SNR values.
        bslen : int
            The number of samples (@ 125 MSa/s) used to compute the baseline.
        bsoffset : int
            The offset (number of samples) between the last sample used to
            compute the baseline and the sample `tau` samples before the one
            where the minimum value of the filter in the event is achieved.
        
        Methods
        -------
        run : generate the events
        
        Members
        -------
        snr : array (ntau, nsnr)
            The ranges of SNR values used for each tau.
        
        Static methods
        --------------
        makesnr : used to generate automatically `snr` when not given.
        
        """
        generator = np.random.default_rng(202011231516)
        
        if mask is None:
            mask = np.ones(len(data), bool)
        
        self.template = Template()
        template_length = max(np.max(tau) + 32, 256) # @ 125 MSa/s
        self.template.make(data, template_length * 8, mask)

        if np.isscalar(snr):
            snr = Toy.makesnr(data, mask, tau, generator, self.template, snr)
        else:
            assert snr.shape == (len(tau), snr.shape[1])
        
        self.template_length = template_length
        self.bslen = bslen
        self.bsoffset = bsoffset
        self.tau = tau
        self.snr = snr

    def run(self, nevents, outfile=None, bslen=None, bsoffset=None, pbar=None, seed=0):
        """
        Simulate signals and localize them.
        
        Parameters
        ----------
        nevents : int
            The number of events. Each event contains one and only one signal.
        outfile : str, optional
            If given, the array containing the simulated signals is memmapped
            into this file.
        bslen, bsoffset : int, optional
            Override the values given at initialization for the baseline.
        pbar : int, optional
            If given, a progress bar is shown that ticks every `pbar` events.
        seed : int
            Seed for the random number generator.
        
        Return
        ------
        output : array (nevents,)
            A structured numpy array with these fields:
            'loctrue', float :
                The sample number where the signal is generated. See
                toy.Template.generate.
            'sigma', float, (ntau, nsnr) :
                The standard deviation of the noise, for each value of tau
                and SNR. Actually the same for each event.
            'baseline', float, (4, ntau, nsnr) :
                The computed baseline for each filter (no filter, moving
                average, exponential moving average, matched filter), tau
                and SNR value.
            'loc', float, (4, ntau, nsnr) :
                Localized signal start. It is not calibrated, it is just
                corrected to be roughly the same as `loctrue`.
            'value', float, (4, ntau, nsnr) :
                The filtered value at the minimum (signals are negative)
                corrected for the baseline and sign.
        output_events : array (nevents,)
            A structured numpy array with these fields:
            'signal', float, (event_length,) :
                The signal and baseline without noise. `event_length` is decided
                automatically to fit the baseline and signal.
            'noise', float, (event_length,) :
                The zero-centered noise with unitary rms. The complete event is
                obtained with signal + sigma * noise, with sigma from the
                `output` array.
            'loctrue', float :
                A copy of the field in `output`.
        """
        generator = np.random.default_rng(0)
        
        bslen = self.bslen if bslen is None else bslen
        bsoffset = self.bsoffset if bsoffset is None else bsoffset

        margin = 64
        event_length = bslen + bsoffset + self.template_length + margin

        output = np.empty(nevents, dtype=[
            ('loctrue', float),
            ('sigma', float, self.snr.shape),
            ('baseline', float, (4,) + self.snr.shape),
            ('loc', float, (4,) + self.snr.shape),
            ('value', float, (4,) + self.snr.shape)
        ])
        
        dtype = [
            ('signal', float, event_length),
            ('noise', float, event_length),
            ('loctrue', float)
        ]
        if outfile is None:
            output_event = np.empty(nevents, dtype)
        else:
            output_event = nplf.open_memmap(outfile, mode='w+', shape=(nevents,), dtype=dtype)
        
        if pbar is None:
            n = nevents
        else:
            n = pbar
        it = range(nevents // n + bool(nevents % n))
        if pbar is not None:
            it = tqdm.tqdm(it)
        
        for i in it:
            start = i * n
            end = min((i + 1) * n, nevents)
            s = slice(start, end)
            self._run(output[s], output_event[s], bslen, bsoffset, generator, margin)

        return output, output_event

    def _run(self, output, output_event, bslen, bsoffset, generator, margin):
        nevents = len(output)
        event_length = output_event.dtype.fields['signal'][0].shape[0]
            
        signal_loc = bslen + bsoffset + margin // 2 + np.zeros(nevents)
        simulated_signal = self.template.generate(event_length, signal_loc, generator)
        simulated_noise = Noise().generate(nevents, event_length, generator)

        filt_noise = Filter(simulated_noise)
        filt_signal = Filter(simulated_signal, self.template.baseline)

        bs_noise = filt_noise.moving_average(bslen)
        bs_signal = filt_signal.moving_average(bslen)

        minima = np.empty((4,) + self.snr.shape + (nevents,), int)
        minval = np.empty(minima.shape)
        mfoffset = np.empty(len(self.tau))
        
        for i in range(len(self.tau)):
            mf_templ, mf_offset = self.template.matched_filter_template(self.tau[i])
            mfoffset[i] = mf_offset
            
            noise = filt_noise.all(mf_templ)
            signal = filt_signal.all(mf_templ)
            
            sigma = self.template.maximum / self.snr[i]
            sim = signal[:, None, :, :] + sigma[None, :, None, None] * noise[:, None, :, :]
            assert sim.shape == (4, self.snr.shape[1], nevents, event_length)
    
            minima[:, i] = np.argmin(sim, axis=-1)
            minval[:, i] = np.min(sim, axis=-1)
    
        idx0 = np.arange(nevents)
        idx1 = minima - bsoffset - self.tau[:, None, None]
        sigma = self.template.maximum / self.snr[..., None]
        baseline = bs_signal[idx0, idx1] + sigma * bs_noise[idx0, idx1]

        val = baseline - minval
        loc = np.asarray(minima, float)
        loc -= self.template.maxoffset()
        loc[1:] -= self.tau[:, None, None]
        loc[3] += mfoffset[:, None, None]
        
        output['loctrue'] = signal_loc
        output['sigma'] = np.moveaxis(sigma, -1, 0)
        output['baseline'] = np.moveaxis(baseline, -1, 0)
        output['loc'] = np.moveaxis(loc, -1, 0)
        output['value'] = np.moveaxis(val, -1, 0)
        
        output_event['signal'] = simulated_signal
        output_event['noise'] = simulated_noise
        output_event['loctrue'] = signal_loc
