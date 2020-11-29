"""
Module to run a simulation of SiPM signals and test various filters.

Classes
-------
Toy : the main class to run the simulations
Noise : abstract class to generate noise, see concrete subclasses
Template : class to make a signal template and get other properties
Filter : class to apply filters

Functions
---------
apply_threshold : function to find where a threshold is crossed
min_snr_ratio : function to compute the unfiltered-to-filtered SNR ratio
"""

import numpy as np
from numpy.lib import format as nplf
import numba
import tqdm
from matplotlib import pyplot as plt
import uproot
import abc

import integrate
from single_filter_analysis import single_filter_analysis

class Noise(metaclass=abc.ABCMeta):
    """
    Abstract base class for generating noise for simulations.
    
    Concrete subclasses
    -------------------
    WhiteNoise
    DataCycleNoise
    
    Methods
    -------
    generate : generate an array of noise
    """
    
    @abc.abstractmethod
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
        pass

class LoadableNoise(Noise):
    """
    Abstract subclass of Noise defining interface to load and save noise
    information from a file.
    
    Methods
    -------
    save
    load
    """

    @abc.abstractmethod
    def save(self, filename):
        """
        Save the noise information to a file that can be later reloaded with
        `load`.
        
        Parameters
        ----------
        filename : str
            The file path.
        """
        pass
        
    @abc.abstractmethod
    def load(self, filename):
        """
        Load the noise information from a file that was written by `save`.
        
        Parameters
        ----------
        filename : str
            The file path.
        """
        pass
        
class DataCycleNoise(LoadableNoise):
    """
    Class to generate noise cycling through actual noise data.
    
    Methods
    -------
    generate
    load_proto0_root_file
    save
    load
    
    Properties
    ----------
    noise_array
    """
    
    def __init__(self):
        self.cycle = 0

    @property
    def noise_array(self):
        """
        An array (nchunks, chunk_length) of noise. `generate` uses a chunk for
        each event generated, cycling in order through chunks.
        """
        if not hasattr(self, '_noise_array'):
            raise RuntimeError('Noise information not loaded')
        return self._noise_array
    
    @noise_array.setter
    def noise_array(self, val):
        mean = np.mean(val, axis=1, keepdims=True)
        std = np.std(val, axis=1, keepdims=True)
        self._noise_array = (val - mean) / std
        
    def generate(self, nevents, event_length, generator=None):
        maxlen = self.noise_array.shape[1]
        if event_length > maxlen:
            raise ValueError(f'Event length {event_length} > maximum {maxlen}')
        
        indices = (1 + self.cycle + np.arange(nevents)) % len(self.noise_array)
        self.cycle = indices[-1]
        return self.noise_array[:, :event_length][indices]
    
    def load_proto0_root_file(self, filename, channel):
        """
        Load noise from a specific file that Simone Stracka gave me. It's not
        on the repository.
        
        Parameters
        ----------
        filename : str
            File name.
        channel : str
            ADC channel to read. Must choose the correct one depending on the
            SiPM used.
        """
        root = uproot.open(filename)
        tree = root['midas_data']
        noise = tree.array(channel)
        counts = np.unique(noise.counts)
        assert len(counts) == 2 and counts[0] == 0, 'inconsistent array lengths'
        self.noise_array = noise._content.reshape(-1, counts[-1])
    
    def save(self, filename):
        np.save(filename, self.noise_array)
    
    def load(self, filename):
        self.noise_array = np.load(filename)

class WhiteNoise(Noise):
    """
    Class to generate white noise.
    
    Methods
    -------
    generate
    """
    
    def generate(self, nevents, event_length, generator=None):
        if generator is None:
            generator = np.random.default_rng()
        return generator.standard_normal(size=(nevents, event_length))

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
        
        # Convert the template from 1 GSa/s to 125 MSa/s. Make a template for
        # each possible 1 GSa/s -> 125 GSa/s sample alignment.
        tlen1ghz = (len(self.template) // 8) * 8
        padded_template = np.concatenate([np.zeros(7), self.template])
        self.templates125 = np.stack([
            np.mean(padded_template[i:tlen1ghz + i].reshape(-1, 8), axis=1)
            for i in range(7, -1, -1)
        ])
        
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
        tlen = self.templates125.shape[1]
        
        if generator is None:
            generator = np.random.default_rng()
                
        out = np.zeros((nevents, event_length))
        
        loc_int = np.array(np.floor(signal_loc), int)
        loc_subsample = np.array(np.floor((signal_loc % 1) * 8), int)
                
        for i in range(8):
            selection = loc_subsample == i
            idx = np.flatnonzero(selection)
            indices0 = idx[:, None]
            indices1 = loc_int[idx][:, None] + np.arange(tlen)
            out[indices0, indices1] = self.templates125[i]
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
    
    @staticmethod
    def name(ifilter):
        """
        Return the name of a filter based on the indexing used in Filter.all().
        """
        names = [
            'No filter',
            'Moving average',
            'Exponential moving average',
            'Matched filter'
        ]
        return names[ifilter]

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

def min_snr_ratio(data, tau, mask=None, nnoise=128, generator=None, noisegen=WhiteNoise()):
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
        A random number generator.
    noisegen : Noise
        An instance of a subclass of Noise. Default white noise.
    
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

    noise = noisegen.generate(1, noise_event_length, generator)
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
    def makesnr(data, mask, tau, template, nsnr, min_filtered_snr=6, max_snr_ratio=1.4, **kw):
        """
        Generate the `snr` argument for the initialization of Toy(). For each
        tau, the range of SNR has a minimum which is chosen to get a target
        filtered SNR with the moving average and a maximum which is a multiple
        of the maximum SNR observed in LNGS data.
        
        Parameters
        ----------
        data : array (nevents, 2, 15001)
            LNGS data as read by readwav.readwav().
        tau : array (ntau,)
            The values of filter length parameters.
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
        
        Additional keyword arguments are passed to `min_snr_ratio`.
        
        Return
        ------
        snr : array (ntau, nsnr)
            The unfiltered snr ranges for each tau.
            
        """
        snrs = min_snr_ratio(data, tau, **kw)
        snr0 = template.snr
        return np.array([
            np.linspace(snrs[i, 1] * min_filtered_snr, snr0 * max_snr_ratio, nsnr)
            for i in range(len(snrs))
        ])
    
    def __init__(self, data, tau, mask=None, snr=10, bslen=1024, bsoffset=32, noisegen=WhiteNoise()):
        """
        A Toy object simulates 1 p.e. signals with noise, each signal in a
        separate "event", and localize the signal with filters, for a range of
        values of the filter parameters and the SNR.
        
        The time base is 125 Msa/s.
        
        Parameters
        ----------
        data : array (N, 2, 15001)
            LNGS data as read by readwav.readwav().
        tau : array (ntau,)
            The values of the filter length parameters.
        mask : bool array (N,), optional
            Mask for the `data` array.
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
        noisegen : Noise
            An instance of a subclass of Noise. Default white noise.
        
        Methods
        -------
        run : generate the events
        plot_event : plot a single event
        plot_loc_all : plot temporal localization precision vs. parameters
        plot_loc : plot temporal localization histogram
        plot_val : plot filtered signal value histogram
        
        Members
        -------
        tau : array (ntau,)
            The values of the length parameters of the filters.
        snr : array (ntau, nsnr)
            The ranges of SNR values used for each tau.
        
        Static methods
        --------------
        makesnr : used to generate automatically `snr` when not given.
        
        """
        generator = np.random.default_rng(202011231516)
        
        if mask is None:
            mask = np.ones(len(data), bool)
        
        template_length = max(np.max(tau) + 32, 256) # @ 125 MSa/s
        self.template = Template()
        self.template.make(data, template_length * 8, mask)

        if np.isscalar(snr):
            snr = Toy.makesnr(data, tau, self.template, snr, mask=mask, generator=generator, noisegen=noisegen)
        else:
            assert snr.shape == (len(tau), snr.shape[1])
        
        self.template_length = template_length
        self.bslen = bslen
        self.bsoffset = bsoffset
        self.tau = tau
        self.snr = snr
        self.noisegen = noisegen

    def run(self, nevents, outfile=None, bslen=None, bsoffset=None, pbar=None, seed=0, noisegen=None):
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
            Seed for the random number generator (default 0).
        noisegen : Noise, optional
            Overrides the noise generator object given at initialization
            (default white noise).
        
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
        generator = np.random.default_rng(seed)
        
        bslen = self.bslen if bslen is None else bslen
        bsoffset = self.bsoffset if bsoffset is None else bsoffset
        noisegen = self.noisegen if noisegen is None else noisegen

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
            self._run(output[s], output_event[s], bslen, bsoffset, generator, margin, noisegen)

        return output, output_event

    def _run(self, output, output_event, bslen, bsoffset, generator, margin, noisegen):
        nevents = len(output)
        event_length = output_event.dtype.fields['signal'][0].shape[0]
        
        signal_loc = generator.uniform(size=nevents)
        signal_loc += bslen + bsoffset + margin // 2
        simulated_signal = self.template.generate(event_length, signal_loc, generator)
        simulated_noise = noisegen.generate(nevents, event_length, generator)

        filt_noise = Filter(simulated_noise)
        filt_signal = Filter(simulated_signal, self.template.baseline)

        bs_noise = filt_noise.moving_average(bslen)
        bs_signal = filt_signal.moving_average(bslen)

        minima = np.empty((4,) + self.snr.shape + (nevents,))
        minval = np.empty(minima.shape)
        mfoffset = np.empty(len(self.tau))
        
        indices = (
            np.arange(4)[:, None, None],
            np.arange(self.snr.shape[1])[None, :, None],
            np.arange(nevents)[None, None, :]
        ) # used in the cycle
        
        for i in range(len(self.tau)):
            mf_templ, mf_offset = self.template.matched_filter_template(self.tau[i])
            mfoffset[i] = mf_offset
            
            noise = filt_noise.all(mf_templ)
            signal = filt_signal.all(mf_templ)
            
            sigma = self.template.maximum / self.snr[i]
            sim = signal[:, None, :, :] + sigma[None, :, None, None] * noise[:, None, :, :]
            assert sim.shape == (4, self.snr.shape[1], nevents, event_length)
            
            # interpolate the minimum with a parabola
            x0 = np.argmin(sim, axis=-1)
            xp = np.minimum(x0 + 1, event_length - 1)
            xm = np.maximum(x0 - 1, 0)
            
            y0 = sim[indices + (x0,)]
            yp = sim[indices + (xp,)]
            ym = sim[indices + (xm,)]
            
            num = yp - ym
            denom = yp + ym - 2 * y0
    
            minima[:, i] = x0 - 1/2 * num / denom
            minval[:, i] = y0 - 1/8 * num ** 2 / denom
    
        idx0 = np.arange(nevents)
        idx1 = np.array(np.rint(minima), int) - bsoffset - self.tau[:, None, None]
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
    
    def plot_event(self, output, output_event, ievent, ifilter, itau, isnr):
        """
        Plot a simulated event.
        
        Parameters
        ----------
        output, output_event : array (nevents,)
            The output from Toy.run().
        ievent, ifilter, itau, isnr : int
            The indices indicating the event, filter, tau and SNR respectively.
        
        Return
        ------
        fig : matplotlib.figure.Figure
            A figure with a single plot.
        """
        out = output[ievent]
        oute = output_event[ievent]
        
        tau = self.tau[itau]
        snr = self.snr[itau, isnr]
        sigma = out['sigma'][itau, isnr]
        unfilt = oute['signal'] + sigma * oute['noise']
        templ, _ = self.template.matched_filter_template(tau)
        sim = Filter(unfilt[None, :], self.template.baseline).all(templ)[ifilter, 0]
        
        fig = plt.figure('toy.Toy.plot_event')
        fig.clf()
    
        ax = fig.subplots(1, 1)
    
        ax.plot(unfilt, label=f'signal (snr = {snr:.2f})')
        ax.plot(sim, label=f'{Filter.name(ifilter)} (tau = {tau})')
        
        ax.axvline(out['loctrue'], label='signal template start', color='black')
        ax.axvline(out['loc'][ifilter, itau, isnr], label='localization (uncalib.)', color='red', linestyle='--')
        ax.axhline(out['baseline'][ifilter, itau, isnr], label='baseline', color='black', linestyle='--')
        
        ax.grid()
        ax.legend(loc='best', fontsize='small')
        ax.set_title(f'Event {ievent}')
        ax.set_xlabel('Sample number @ 125 MSa/s')
        ax.set_ylabel('ADC scale [10 bit]')
    
        fig.tight_layout()
        fig.show()
        
        return fig
    
    def templocres(self, output):
        """
        Compute the temporal localization resolution as the half interquantile
        range from 0.16 to 0.84, which is approximately a standard deviation for
        the normal distribution, of the localization error, in units of samples
        (8 ns).
        
        Parameters
        ----------
        output : array (nevents,)
            The first output from Toy.run().
        
        Return
        ------
        error : array (4, ntau, nsnr)
            The temporal localization error for each filter, tau and SNR.
        """
        loctrue = output['loctrue'][:, None, None, None]
        loc = output['loc']
        quantiles = np.quantile(loc - loctrue, [0.5 - 0.68/2, 0.5 + 0.68/2], axis=0)
        error = (quantiles[1] - quantiles[0]) / 2
        assert error.shape == (4,) + self.snr.shape
        return error

    def plot_loc_all(self, output, logscale=True):
        """
        Plot temporal localization precision vs filter, filter length, SNR.
        
        Parameters
        ----------
        output : array (nevents,)
            The first output from Toy.run().
        logscale : bool
            If True (default), use a vertical logarithmic scale instead of a
            linear one.
        
        Return
        ------
        fig : matplotlib.figure.Figure
            A figure with an axis for each filter.
        """
        width = self.templocres(output)
    
        fig = plt.figure('toy.Toy.plot_loc_all', figsize=[10.95,  7.19])
        fig.clf()

        axs = fig.subplots(2, 2, sharex=True, sharey=True).reshape(-1)
        
        linekw = dict(color='#600000')
        for ifilter, ax in enumerate(axs):
            if ifilter > 0:
                for itau in range(len(self.tau)):
                    alpha = (itau + 1) / len(self.tau)
                    tauname = 'tau' if ifilter == 2 else 'nsamples'
                    label = f'{tauname} = {self.tau[itau]}'
                    ax.plot(self.snr[itau], width[ifilter, itau], alpha=alpha, label=label, **linekw)
                ax.legend(loc='upper right', fontsize='small', title='(@ 125 MSa/s)')
            else:
                x = self.snr.reshape(-1)
                y = width[0].reshape(-1)
                isort = np.argsort(x)
                ax.plot(x[isort], y[isort], **linekw)
        
            if ax.is_last_row():
                ax.set_xlabel('Unfiltered SNR\n(avg signal peak over noise rms)')
            if ax.is_first_col():
                ax.set_ylabel('half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [8 ns]')
            ax.grid(True, which='major', axis='both', linestyle='--')
            if logscale:
                ax.grid(True, which='minor', axis='y', linestyle=':')
            ax.set_title(Filter.name(ifilter))
        
        if logscale:
            # axs[0].set_yscale('symlog', linthreshy=1, linscaley=0.5)
            axs[0].set_yscale('log')
        else:
            lims = axs[0].get_ylim()
            axs[0].set_ylim(min(0, lims[0]), lims[1])
        lims = axs[0].get_xlim()
        axs[0].set_xlim(lims[0], lims[1] + (lims[1] - lims[0]) * 0.4)
    
        fig.tight_layout()
        fig.show()
        
        return fig

    def plot_loc(self, output, itau, isnr):
        """
        Plot temporal localization histograms for all filters.
        
        Parameters
        ----------
        output : array (nevents,)
            The first output from Toy.run().
        itau, isnr : int
            Indices indicating the tau and SNR values to use.
        
        Return
        ------
        fig : matplotlib.figure.Figure
            A figure with an axis for each filter.
        """
        tau = self.tau[itau]
        snr = self.snr[itau, isnr]
        
        fig = plt.figure('toy.Toy.plot_loc', figsize=[10.95,  7.19])
        fig.clf()

        axs = fig.subplots(2, 2).reshape(-1)
    
        for ifilter, ax in enumerate(axs):
            data = output['loc'][:, ifilter, itau, isnr] - output['loctrue']
            if ifilter == 0:
                label = f'SNR={snr:.2f}'
            elif ifilter == 2:
                label = f'$\\tau$={tau}'
            else:
                label = f'Nsamples={tau}'
            h, _, _ = ax.hist(data, bins='auto', histtype='step')
            left, center, right = np.quantile(data, [0.5 - 0.68/2, 0.5, 0.5 + 0.68/2])
            xerr = [[center - left], [right - center]]
            ax.errorbar(center, np.max(h) / 2, xerr=xerr, capsize=4, fmt='k.', label='"$\\pm 1 \\sigma$" quantiles')
            if ax.is_last_row():
                ax.set_xlabel('Temporal localization error (uncalib.) [8 ns]')
            if ax.is_first_col():
                ax.set_ylabel('Bin count')
            ax.grid()
            ax.legend(loc='best')
            ax.set_title(f'{Filter.name(ifilter)} ({label})')
        
        fig.tight_layout()
        fig.show()

        return fig

    def plot_val(self, output, itau, isnr):
        """
        Plot a histogram of the baseline corrected filtered value at signal
        detection point for each filter.
        
        Parameters
        ----------
        output : array (nevents,)
            The first output from Toy.run().
        itau, isnr : int
            Indices indicating the tau and SNR values to use.
        
        Return
        ------
        fig : matplotlib.figure.Figure
            A figure with an axis for each filter.
        """
        fig = plt.figure('toy.Toy.plot_val', figsize=[10.95,  7.19])
        fig.clf()
        
        tau = self.tau[itau]
        snr = self.snr[itau, isnr]
        noise = self.noisegen.generate(1, len(output) + tau)
        sigma = output['sigma'][0, itau, isnr]
        filt = Filter(sigma * noise)
        templ, _ = self.template.matched_filter_template(tau)
        fn = filt.all(templ)[:, 0, tau:]

        axs = fig.subplots(2, 2).reshape(-1)
    
        for ifilter, ax in enumerate(axs):

            if ifilter == 0:
                label1 = 'filtered signal+noise\n(min. in each event)'
                label2 = 'filtered noise'
                legendtitle = f'Unfiltered SNR = {snr:.2f}'
            else:
                tauname = 'tau' if ifilter == 2 else 'nsamples'
                label1 = f'{tauname} = {tau}'
                label2 = None
                legendtitle = '(@ 125 MSa/s)'
            
            data = output['value'][:, ifilter, itau, isnr]
            ax.hist(data, bins='auto', histtype='step', label=label1)
            ax.hist(fn[ifilter], bins='auto', histtype='step', label=label2)
            
            if ax.is_last_row():
                ax.set_xlabel('Filter output [ADC 10 bit]')
            if ax.is_first_col():
                ax.set_ylabel('Bin count')
            ax.legend(loc='best', title=legendtitle)
            ax.grid()
            ax.set_title(Filter.name(ifilter))

        fig.tight_layout()
        fig.show()
        
        return fig
