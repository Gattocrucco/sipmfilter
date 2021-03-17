"""
Module to run a simulation of photodetection signals and test various filters.

Classes
-------
Toy : the main class to run the simulations
Noise : abstract class to generate noise, see concrete subclasses
WhiteNoise : generate white noise
DataCycleNoise : generate noise copying it from a source array
Template : class to make a signal template and get other properties
Filter : class to apply filters
NpzLoad : class to load/save objects to npz files

Functions
---------
apply_threshold : find where a threshold is crossed
min_snr_ratio : (BROKEN) compute the unfiltered-to-filtered SNR ratio
downsample : downsample by averaging in groups
"""

import abc

import numpy as np
from numpy.lib import format as nplf
from matplotlib import pyplot as plt
import numba
import tqdm
import uproot

import integrate
from single_filter_analysis import single_filter_analysis
import readwav
import textbox

def downsample(a, n, axis=-1, dtype=None):
    """
    Downsample an array by averaging nearby elements.
    
    Parameters
    ----------
    a : array
        The array to downsample.
    n : int
        The number of averaged elements per group.
    axis : int
        The axis along which the averaging is computed. Default last.
    dtype : data-type, optional
        The data type of the output.
    
    Return
    ------
    da : array
        The downsampled array. The shape is the same as `a` apart from the
        specified axis, which has size a.shape[axis] // n.
    """
    if n == 1:
        return np.asarray(a, dtype=dtype)
    
    length = a.shape[axis]
    axis %= len(a.shape)
    trunc_length = length - length % n
    
    idx = (slice(None),) * axis
    idx += (slice(0, trunc_length),)
    idx += (slice(None),) * (len(a.shape) - axis - 1)
    
    shape = tuple(a.shape[i] for i in range(axis))
    shape += (trunc_length // n, n)
    shape += tuple(a.shape[i] for i in range(axis + 1, len(a.shape)))
    
    return np.mean(np.reshape(a[idx], shape), axis=axis + 1, dtype=dtype)

class NpzLoad:
    """
    DEPRECATED, used npzload.NPZLoad.
    
    Superclass for adding automatic save/load from npz files. Only scalar/array
    instance variables are considered.
    """
    
    def save(self, filename):
        """
        Save the object to file as a `.npz` archive.
        """
        classdir = dir(type(self))
        variables = {
            n : x
            for n, x in vars(self).items()
            if n not in classdir
            and not n.startswith('__')
            and (np.isscalar(x) or isinstance(x, np.ndarray))
        }
        np.savez(filename, **variables)
    
    @classmethod
    def load(cls, filename):
        """
        Return an instance loading the object from a file which was written by
        `save`.
        """
        self = cls.__new__(cls)
        arch = np.load(filename)
        for n, x in arch.items():
            # if x.shape == ():
            #     x = x.item()
            setattr(self, n, x)
        arch.close()
        return self

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
    
    def __init__(self, timebase=8):
        """
        Parameters
        ----------
        timebase : int
            The duration of samples in nanoseconds. Default is 8, i.e. the
            sampling frequency of the waveform returned by `generate` is
            125 MSa/s.
        """
        self.timebase = timebase
    
    @abc.abstractmethod
    def generate(self, nevents, event_length, generator=None):
        """
        Generate noise with unitary variance.
        
        Parameters
        ----------
        nevents : int
            Number of events i.e. independent chunks of simulated data.
        event_length : int
            Number of samples of each event.
        generator : np.random.Generator, optional
            Random number generator.
        
        Return
        ------
        events : array (nevents, event_length)
            Simulated noise.
        """
        pass
      
class DataCycleNoise(Noise):
    
    def __init__(self, timebase=8, allow_break=False, chunk_skip=None, maxcycles=None):
        """
        Class to generate noise cycling through actual noise data.
    
        Parameters
        ----------
        timebase : int
            The duration of samples in nanoseconds. Default is 8, i.e. the
            sampling frequency of the waveform returned by `generate` is
            125 MSa/s.
        allow_break : bool
            Default False. If True, the event length can be longer than the
            noise chunks obtained from data, but there may be breaks in the
            events where one sample is not properly correlated with the next.
        chunk_skip : int, optional
            By default each noise event is copied from a different noise data
            chunk to avoid correlations between events. If chunk_skip is
            specified, multiple events can be taken from the same chunk,
            skipping chunk_skip samples between each event.
        maxcycles : int, optional
            By default `generate` can reuse the same chunk for different events
            if there's not enough noise data for all events. maxcycles, if
            specified, set the maximum number of times any chunk can be reused.
            Once the limit is hit an exception is raised and the counter is
            reset.

        Members
        -------
        allow_break : bool
            The parameter given at initialization. Can be changed directly.
        
        Methods
        -------
        generate
        load_proto0_root_file
        load_LNGS_wav
        save
        load
    
        Properties
        ----------
        noise_array
        """
        self.timebase = timebase
        self.allow_break = allow_break
        self.chunk_skip = chunk_skip
        self.maxcycles = maxcycles

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
        nchunks, chunklen = self.noise_array.shape
        
        if not self.allow_break and event_length > chunklen:
            raise ValueError(f'Event length {event_length} > maximum {chunklen}')
        
        if event_length > chunklen // 2 or self.chunk_skip is None:
            # each event uses one or more chunks
            
            chperev = int(np.ceil(event_length / chunklen))
            assert chperev <= nchunks
            newnchunks = (nchunks // chperev) * chperev
            noise_array = self.noise_array[:newnchunks].reshape(newnchunks // chperev, chperev * chunklen)
        
            cycle = int(np.ceil(self.cycle / chperev))
            nextcycle = (cycle + nevents) * chperev
            
            indices = (cycle + np.arange(nevents)) % len(noise_array)
            events = noise_array[:, :event_length][indices]
        
        else:
            # each chunk is used for one or more events
            
            effevlen = event_length + self.chunk_skip
            evperch = (chunklen + self.chunk_skip) // effevlen
            
            cycle = self.cycle * evperch
            nextcycle = int(np.ceil((cycle + nevents) / evperch))
            
            flatindices = cycle + np.arange(nevents)
            indices0 = (flatindices // evperch) % nchunks
            indices0 = indices0[:, None]
            indices1 = (flatindices % evperch) * effevlen
            indices1 = indices1[:, None] + np.arange(event_length)
            
            events = self.noise_array[indices0, indices1]
        
        if self.maxcycles is not None and nextcycle > self.maxcycles * nchunks:
            self.cycle = 0
            ncycles = int(np.ceil(nextcycle / nchunks))
            raise RuntimeError(f'Data cycled {ncycles} times > limit {self.maxcycles}')
        else:
            self.cycle = nextcycle
        
        assert events.shape == (nevents, event_length)
        
        return events
    
    def load_proto0_root_file(self, filename, channel, maxevents=None):
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
        maxevents : int, optional
            The maximum number of events loaded from the wav file.
        """
        if self.timebase % 8 != 0:
            raise RuntimeError(f'can not load `{filename}` data at 125 MSa/s with timebase={self.timebase}')
        root = uproot.open(filename)
        tree = root['midas_data']
        
        if maxevents is None:
            maxevents = 2 ** 31
        nsamples = tree.array('nsamples')
        nz = np.flatnonzero(nsamples)
        if maxevents < len(nz):
            maxevents = nz[maxevents]
        entrystop = min(tree.numentries, maxevents)
        
        noise = tree.array(channel, entrystop=entrystop)
        counts = np.unique(noise.counts)
        assert len(counts) == 2 and counts[0] == 0, 'inconsistent array lengths'
        array = noise._content.reshape(-1, counts[-1])
        
        self.noise_array = downsample(array, self.timebase // 8, dtype=np.float32)
    
    def load_LNGS_wav(self, filename, maxevents=None):
        """
        Load noise from a LNGS wav file. THERE SOME ASSUMPTIONS HERE ON WHAT'S
        IN THE FILE, RECHECK IT WORKS IF I CHANGE THE FILE.
        
        Parameters
        ----------
        filename : str
            The wav file path.
        maxevents : int, optional
            The maximum number of events loaded from the wav file.
        """
        data = readwav.readwav(filename, maxevents=maxevents, mmap=False, quiet=True)
        baseline_zone = data[:, 0, :8900]
        ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
        downsampled = downsample(baseline_zone, self.timebase, dtype=np.float32)
        self.noise_array = downsampled[~ignore]

    def save(self, filename):
        np.savez(filename, noise_array=self.noise_array, timebase=self.timebase)
    
    def load(self, filename):
        arch = np.load(filename)
        try:
            timebase = arch['timebase'].item()
            if self.timebase % timebase != 0:
                raise ValueError(f'timebase={self.timebase} in object not a multiple of timebase={timebase} in file `{filename}`')
            self.noise_array = downsample(arch['noise_array'], self.timebase // timebase)
        finally:
            arch.close()

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
    
class Template(NpzLoad):
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
    save : save the template to file.
    
    Properties
    ----------
    maximum : peak amplitude of the template.
    snr : SNR observed in the LNGS data used to make the template.
    ready : True if the template object can be used.
    template_length : the length of the 1 GSa/s template.
    
    Instance variables
    ------------------
    baseline : scalar
        The average baseline.
    template : 1D array
        The source 1 GSa/s template used by methods.
    """
    
    def __init__(self):
        raise NotImplementedError('use a class method to construct Template objects')
    
    @property
    def ready(self):
        """
        True if the template object has been filled either with `make` or
        `load`.
        """
        return hasattr(self, 'template')
    
    @property
    def template_length(self):
        """
        The length, in ns, of the 1 GSa/s template used to generate waveforms.
        """
        return len(self.template)
    
    @classmethod
    def from_lngs(cls, data, length, mask=None):
        """
        Compute a template from 1 p.e. signals in an LNGS wav.
        
        Parameters
        ----------
        data : array (nevents, 2, 15001)
            Wav data as read by readwav.readwav().
        length : int
            Number of samples of the template (@ 1 GSa/s), starting from the
            beginning of the trigger impulse.
        mask : bool array (nevents,), optional
            Mask for the `data` array.
        
        Return
        ------
        self : Template
            A template object.
        """
        self = cls.__new__(cls)
        
        if mask is None:
            mask = np.ones(len(data), bool)
        
        # Run a moving average filter to find and separate the signals by
        # number of photoelectrons.
        trigger, baseline, value = integrate.filter(data, bslen=8000, length_ma=1470, delta_ma=1530)
        corr_value = baseline - value[:, 0]
        snr, center, width = single_filter_analysis(corr_value[mask], return_full=True)
        minsnr = 5
        assert snr >= minsnr, f'SNR = {snr:.3g} < {minsnr}'
        assert len(center) > 2
    
        # Select the data corresponding to 1 photoelectron and subtract the
        # baseline.
        lower = (center[0] + center[1]) / 2
        upper = (center[1] + center[2]) / 2
        selection = (lower < corr_value) & (corr_value < upper) & mask
        t = int(np.median(trigger))
        data1pe = data[selection, 0, t:t + length] - baseline[selection].reshape(-1, 1)
    
        # Compute the waveform as the mean of the signals.
        template = np.mean(data1pe, axis=0)
        self.template_rel_std = np.std(np.mean(data1pe, axis=1)) / np.mean(template)
        self.template_rel_std_alt = np.sqrt(width[1] ** 2 - width[0] ** 2) / center[1]
        self.template_N = np.sum(selection)
        
        # Repeat but aligning the data to the trigger for each event instead of
        # to the median of all triggers.
        indices0 = np.flatnonzero(selection)[:, None]
        indices2 = trigger[selection, None] + np.arange(length)
        data1pe = data[indices0, 0, indices2] - baseline[selection][:, None]
        template_aligned = np.mean(data1pe, axis=0)
        
        # Save variables.
        self.templates = np.stack([template, template_aligned])
        self.template = template # for backward compatibility
        
        # For the moving average.
        self._cumsum_templates = np.pad(np.cumsum(self.templates, axis=1), [(0,0),(1,0)])
        
        # Compute the noise standard deviation.
        STDs = np.std(data[:, 0, :t - 100], axis=1)
        self.noise_std = np.sum(STDs, where=mask) / np.count_nonzero(mask)
        
        # Compute the baseline distribution.
        bs = baseline[mask]
        self.baseline = np.mean(bs)
        self.baseline_std = np.std(bs)
        self.baseline_std_alt = np.sqrt(np.var(bs) - self.noise_std ** 2 / 8000)
        
        # Save trigger distribution.
        self.trigger_min = np.min(trigger)
        self.trigger_max = np.max(trigger)
        self.trigger_median = np.median(trigger)
        
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
        signal_loc = np.asarray(signal_loc)
        nevents = len(signal_loc)
        
        if generator is None:
            generator = np.random.default_rng()
            
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
            out *= 1 + self.template_rel_std * generator.standard_normal((nevents, 1))
        # if baseline:
        #     out += self.baseline
        #     if randbaseline:
        #         out += self.baseline_std * generator.standard_normal((nevents, 1))
        
        return out
    
    @property
    def maximum(self):
        """
        DEPRECATED, use max().
        
        Maximum amplitude of the original internal template used by
        generate() (the unaligned one).
        """
        return np.max(np.abs(self.template))
    
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
    
    @property
    def snr(self):
        """
        DEPRECATED
        
        Average peak signal amplitude over noise rms at 1 GSa/s.
        """
        return self.maximum / self.noise_std
    
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

class Filter:
    
    def __init__(self, events, boundary=0, rmargin=0):
        """
        Class to apply various filters to the same piece of data.
        
        Parameters
        ----------
        events : array (nevents, event_length)
            An array of events. The filter is applied separately to each event.
        boundary : scalar
            The past boundary condition, it is like each event has an infinite
            series of samples with value `boundary` before sample 0.
        rmargin : int
            The signals are prolonged this much in the future filling with the
            value `boundary`. Default 0.
        
        Methods
        -------
        The filter methods are:
            moving_average
            exponential_moving_average
            matched
        
        Each method has the following return signature:
        
        filtered : float array (nevents, event_length + margin)
            filtered[:, i] is the filter output after reading sample
            events[:, i].
        
        and accepts an optional parameter `out` where the output is written to.
        The method `all` computes all the filters in one array.
        
        """
        self.events = events
        self.boundary = boundary
        self.rmargin = rmargin
    
    def _add_boundary(self, left, right=0):
        nevents, event_length = self.events.shape
        events = np.empty((nevents, left + event_length + right))
        events[:, :left] = self.boundary
        events[:, left:left + event_length] = self.events
        events[:, left + event_length:] = self.boundary
        return events
    
    def _out(self, out):
        nevents, event_length = self.events.shape
        if out is None:
            return np.empty((nevents, event_length + self.rmargin))
        else:
            return out
    
    def moving_average(self, nsamples, out=None):
        """
        Parameters
        ----------
        nsamples : int
            The number of averaged samples.
        """
        events = self._add_boundary(nsamples, self.rmargin)
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
        events = self._add_boundary(len(template) - 1, self.rmargin)
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
    def name(ifilter, short=False):
        """
        Return the name of a filter based on the indexing used in Filter.all().
        """
        names = [
            ('No filter'                 , 'unfiltered'),
            ('Moving average'            , 'movavg'    ),
            ('Exponential moving average', 'expmovavg' ),
            ('Cross correlation'         , 'crosscorr' ),
        ]
        return names[ifilter][short]
    
    @staticmethod
    def tauname(ifilter):
        """
        Return the name of the filter length parameter (uses LaTeX).
        """
        names = [None, 'N', '$\\tau$', 'N']
        return names[ifilter]

@numba.jit(cache=True, nopython=True)
def _exponential_moving_average(events, a, boundary, out):
    for i in numba.prange(len(events)):
        out[i, 0] = a * boundary + (1 - a) * events[i, 0]
        for j in range(1, events.shape[1]):
            out[i, j] = a * out[i, j - 1] + (1 - a) * events[i, j]
        for j in range(events.shape[1], out.shape[1]):
            out[i, j] = a * out[i, j - 1] + (1 - a) * boundary

@numba.jit(cache=True, nopython=True)
def _correlate(events, template, out):
    # TODO should use np.correlate which uses fft
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
    BROKEN
    
    Compute the signal amplitude over noise rms needed to obtain a given
    filtered signal amplitude over filtered noise rms, i.e. the ratio
    "unfiltered SNR" over "filtered SNR".
    
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

def run_sliced(fun, ntot, n=None):
    """
    Run a cycle which calls a given function with a progressing slice as sole
    argument until a range is covered, printing a progressbar.
    
    Parameters
    ----------
    fun : function
        A function with a single parameter which is a slice object.
    ntot : int
        The end of the range covered by the sequence of slices.
    n : int, optional
        The length of each slice (the last slice may be shorter). If None, the
        function is called once with the slice 0:ntot.
    """
    if n is None:
        fun(slice(0, ntot))
    else:
        for i in tqdm.tqdm(range(ntot // n + bool(ntot % n))):
            start = i * n
            end = min((i + 1) * n, ntot)
            s = slice(start, end)
            fun(s)

class Toy(NpzLoad):
        
    def __init__(self, template, tau, snr, noisegen=None, timebase=8, upsampling=False):
        """
        A Toy object simulates 1 p.e. signals with noise, each signal in a
        separate "event", and localize the signal with filters, for a range of
        values of the filter parameters and the SNR.
        
        Parameters
        ----------
        template : toy.Template
            A template object.
        tau : array (ntau,)
            Values of the filters length parameter, in number of samples.
        snr : array (nsnr,)
            SNR values. The SNR is the average signal peak height @ 1 GSa/s
            over the noise RMS @ chosen timebase.
        noisegen : Noise
            An instance of a subclass of Noise. Default white noise.
        timebase : int
            The duration of a sample in nanoseconds. Default is 8 i.e.
            125 MSa/s.
        upsampling : bool
            Default False. If True, compute the temporal localization with
            upsampling to 1 GSa/s.
        
        Methods
        -------
        run : generate the events
        window_center : helper function to make an argument to run_window
        run_window : extract windows from the events generated
        
        sampling_str : string describing the sampling frequency
        mftempl : get a matched filter template.
        templocres : compute temporal localization resolution
        filteredsnr : compute the SNR with signal-within-noise amplitude
        snrratio : compute the ratio SNR after over before filtering

        plot_event : plot a single event
        plot_loc_all : plot temporal localization resolution vs. parameters
        plot_loc : plot temporal localization histogram
        plot_val : plot filtered signal value histogram
        
        plot_event_window : plot an event with the windowed matched filter
        plot_loc_window : plot temporal localization resolution after windowing
        
        save : save to file.
        
        Class methods
        -------------
        load : load from file. The loaded object can do plots but not run again.
        
        Members
        -------
        template : toy.Template
            The template object.
        tau : array (ntau,)
            The values of the length parameters of the filters.
        snr : array (nsnr,)
            SNR values.
        sigma : array (nsnr,)
            The standard deviation of the noise for each SNR.
        event_length : int
            The number of samples in each simulated event.
        output : array (nevents,)
            A structured numpy array with these fields, filled after calling
            `run`:
            'filter_start', int, (ntau,) :
                The sample number where the filtering starts, the same for each
                event.
            'filter_skip', int, (ntau,) :
                The number of initial samples from the filtered zone not used
                to compute results, the same for each event.
            'loctrue', float :
                The sample number where the signal is generated. See
                toy.Template.generate.
            'loc', float, (4, ntau, nsnr) :
                Signal localization. It is not calibrated. Computed with
                parabolic interpolation.
            'locraw' :
                Like 'loc' but without interpolation.
            'locup' :
                Like 'loc' but with upsampling. Present only if upsampling=True.
            'locupraw' :
                Like 'locup' but without interpolation.
            'value', float, (4, ntau, nsnr) :
                The filtered value at the minimum, with inverted sign.
            'valueclean', float, (4, ntau) :
                Like 'value' but filtering only the signal without noise.
            'filtnoisesdev', float, (4, ntau) :
                The standard deviation of filtered unitary variance noise for
                each filter.
        output_event : array (nevents,)
            A structured numpy array with these fields, filled after calling
            `run`:
            'signal', float, (event_length,) :
                The signal and baseline without noise.
            'noise', float, (event_length,) :
                The zero-centered noise with unitary rms. The complete event is
                obtained with signal + sigma * noise.
        output_window : array (nevents,)
            A structured numpy array with these fields, filled after calling
            `run_window`:
            'wstart', int, (ncenter, nwin) :
                The sample number where the window starts.
            'wloc', float, (ncenter, nwin, ntau, nsnr) :
                The localization from the window (the sample number is still
                relative to the whole event).
        wlen : array (nwin,)
            The window lengths used by `run_window`.
        wlmargin : array (nwin,)
            The left margin of the windows used by `run_window`.
        wcenter : array (ncenter,)
            The `wcenter` argument to `run_window` (see `window_center`).
        """        
        assert template.ready
        tau = np.asarray(tau)
        assert template.template_length >= np.max(tau) * timebase
        snr = np.asarray(snr)
        if noisegen is None:
            noisegen = WhiteNoise(timebase=timebase)
        assert timebase == noisegen.timebase, 'timebase of Noise object must match timebase of Toy object'
        
        self.template = template
        self.tau = tau
        self.snr = snr
        self.noisegen = noisegen
        self.timebase = timebase
        self.upsampling = upsampling
        
        self.sigma = template.maximum / snr
                
        self.margin = 512 // timebase
        self.event_length = np.max(tau) + template.template_length // timebase + self.margin
        
        self.templs = self._make_templ_array(timebase)
        self.uptempls = self._make_templ_array(1)
            
    def _make_templ_array(self, timebase):
        maxlen = np.max(self.tau) * self.timebase // timebase
        templs = np.zeros(len(self.tau), dtype=[
            ('length', int),
            ('offset', float),
            ('template', float, maxlen)
        ])
        
        for i in range(len(self.tau)):
            length = self.tau[i] * self.timebase // timebase
            templ, offset = self.template.matched_filter_template(length, timebase=timebase)
            assert len(templ) == length
            templs[i]['length'] = length
            templs[i]['offset'] = offset
            templs[i]['template'][:length] = templ
        
        return templs
    
    def mftempl(self, itau, upsampling=False):
        """
        Return a matched filter template used.
        
        Parameters
        ----------
        itau : int
            The index into the tau array for the filter length.
        upsampling : bool
            Default False. If True, return the template at 1 GSa/s.
        
        Return
        ------
        template : array
            The template. It is normalized to unit sum.
        """
        templs = self.uptempls if upsampling else self.templs
        templ = templs[itau]
        return templ['template'][:templ['length']]
    
    def run(self, nevents, pbar=None, seed=0):
        """
        Simulate signals and localize them. After running this function, the
        members `output` and `output_event` are filled.
        
        Parameters
        ----------
        nevents : int
            The number of events. Each event contains one and only one signal.
        pbar : int, optional
            If given, a progress bar is shown that ticks every `pbar` events.
        seed : int
            Seed for the random number generator (default 0).        
        """
        generator = np.random.default_rng(seed)
        
        ntau = len(self.tau)
        nsnr = len(self.snr)
        
        dtype = [
            ('filter_start', int, (ntau,)),
            ('filter_skip', int, (ntau,)),
            ('loctrue', float),
            ('loc', float, (4, ntau, nsnr)),
            ('locraw', int, (4, ntau, nsnr)),
            ('value', float, (4, ntau, nsnr)),
            ('valueclean', float, (4, ntau)),
            ('filtnoisesdev', float, (4, ntau)),
        ]
        # if self.dobaseline:
        #     dtype.append(('baseline', float, (4, ntau, nsnr)))
        if self.upsampling:
            dtype.append(('locup', float, (4, ntau, nsnr)))
            dtype.append(('locupraw', float, (4, ntau, nsnr)))
        output = np.empty(nevents, dtype=dtype)
        
        dtype = [
            ('signal', float, self.event_length),
            ('noise', float, self.event_length),
        ]
        output_event = np.empty(nevents, dtype)
                
        def fun(s):
            self._run(output[s], output_event[s], generator)
        run_sliced(fun, nevents, pbar)

        self.output = output
        self.output_event = output_event

    def _run(self, output, output_event, generator):
        # Get sizes of things.
        nevents = len(output)
        ntau = len(self.tau)
        nsnr = len(self.snr)
        event_length = self.event_length
        upsampling = self.upsampling
        timebase = self.timebase
        template_length = self.template.template_length // timebase
        margin = self.margin
        
        # Generate signal and noise arrays.
        signal_loc_offset = event_length - template_length - margin // 2
        signal_loc_cont = generator.uniform(size=nevents)
        signal_loc = signal_loc_offset + signal_loc_cont
        simulated_signal = self.template.generate(event_length, signal_loc, generator, timebase=timebase)
        simulated_noise = self.noisegen.generate(nevents, event_length, generator)
        
        # # Filter with a moving average to compute the baseline.
        # if self.dobaseline:
        #     filt_noise = Filter(simulated_noise)
        #     filt_signal = Filter(simulated_signal, self.template.baseline)
        #     bs_noise = filt_noise.moving_average(self.bslen)
        #     bs_signal = filt_signal.moving_average(self.bslen)
        
        # Upsampled signal and noise (without baseline zone).
        if upsampling:
            upskip = signal_loc_offset - margin // 2
            simulated_noise_up = np.repeat(simulated_noise[:, upskip:], timebase, axis=-1)
            simulated_signal_up = np.repeat(simulated_signal[:, upskip:], timebase, axis=-1)
            filt_noise_up = Filter(simulated_noise_up)
            filt_signal_up = Filter(simulated_signal_up)
        
        # Arrays filled in the cycle over tau.
        minima = np.empty((4, 4, ntau, nsnr, nevents))
        # first axis = (loc, locraw, locup, locupraw)
        minval = np.empty_like(minima)
        sdev = np.empty((4, ntau, nevents))
        cleanval = np.empty_like(sdev)
        filter_start, filter_skip = np.empty((2, ntau), int)
        
        # Indices used for the interpolation.
        indices = tuple(np.ogrid[:4, :nsnr, :nevents])
        indices_clean = tuple(np.ogrid[:4, :nevents])
        
        for itau in range(ntau):
            tau = self.tau[itau]
            
            # Get the matched filter template.
            mf_templ = self.mftempl(itau)
            
            # Filter objects.
            skip = signal_loc_offset - margin // 2 - tau
            filt_noise = Filter(simulated_noise[:, skip:])
            filt_signal = Filter(simulated_signal[:, skip:])
            filter_start[itau] = skip
            
            # Filter the signal and noise separately.
            noise = filt_noise.all(mf_templ)[..., tau:]
            signal = filt_signal.all(mf_templ)[..., tau:]
            skip += tau
            filter_skip[itau] = tau
            
            # Compute the standard deviation of the filtered noise.
            sdev[:, itau] = np.std(noise, axis=-1)
            
            # Combine the noise and signal with the given SNR.
            sim = signal[:, None, :, :] + self.sigma[None, :, None, None] * noise[:, None, :, :]
            assert sim.shape == (4, nsnr, nevents, event_length - skip)
            
            # Interpolate the minimum with a parabola, without noise.
            x0 = np.argmin(signal, axis=-1)
            xp = np.minimum(x0 + 1, event_length - skip - 1)
            xm = np.maximum(x0 - 1, 0)
            
            y0 = signal[indices_clean + (x0,)]
            yp = signal[indices_clean + (xp,)]
            ym = signal[indices_clean + (xm,)]
            
            num = yp - ym
            denom = yp + ym - 2 * y0
    
            cleanval[:, itau] = y0 - 1/8 * num ** 2 / denom
            
            # Interpolate the minimum with a parabola.
            x0 = np.argmin(sim, axis=-1)
            xp = np.minimum(x0 + 1, event_length - skip - 1)
            xm = np.maximum(x0 - 1, 0)
            
            y0 = sim[indices + (x0,)]
            yp = sim[indices + (xp,)]
            ym = sim[indices + (xm,)]
            
            num = yp - ym
            denom = yp + ym - 2 * y0
            
            # TODO handle the case denom == 0
            minima[0, :, itau] = x0 - 1/2 * num / denom + skip
            minval[0, :, itau] = y0 - 1/8 * num ** 2 / denom
            
            minima[1, :, itau] = x0 + skip
            minval[1, :, itau] = y0
            
            if upsampling:
                # Get the matched filter template.
                mf_templ = self.mftempl(itau, True)
    
                # Filter the signal and noise separately.
                noise = filt_noise_up.all(mf_templ)
                signal = filt_signal_up.all(mf_templ)
                        
                # Combine the noise and signal with the given SNR.
                sim = signal[:, None, :, :] + self.sigma[None, :, None, None] * noise[:, None, :, :]
                assert sim.shape == (4, nsnr, nevents, (event_length - upskip) * timebase)
            
                # Interpolate the minimum with a parabola.
                x0 = np.argmin(sim, axis=-1)
                xp = np.minimum(x0 + 1, (event_length - upskip) * timebase - 1)
                xm = np.maximum(x0 - 1, 0)
            
                y0 = sim[indices + (x0,)]
                yp = sim[indices + (xp,)]
                ym = sim[indices + (xm,)]
            
                num = yp - ym
                denom = yp + ym - 2 * y0
    
                # TODO handle the case denom == 0
                minima[2, :, itau] = (x0 - 1/2 * num / denom) / timebase + upskip
                minval[2, :, itau] = y0 - 1/8 * num ** 2 / denom

                minima[3, :, itau] = x0 / timebase + upskip
                minval[3, :, itau] = y0

        # # Compute the baseline.
        # idx0 = np.arange(nevents)
        # idx1 = np.array(minima[1], int) - bsoffset - self.tau[:, None, None]
        # baseline = bs_signal[idx0, idx1] + self.sigma[None, :, None] * bs_noise[idx0, idx1]
                
        # Align approximately the localization.
        # loc = np.asarray(minima, float)
        # loc[[0, 2], :3] -= self.template.maxoffset(timebase)
        # loc[:, 1:] -= self.tau[:, None, None]
        # loc[[0, 2], 3] += self.templs['offset'][:, None, None]
        
        # Write results in the output arrays.
        output['loctrue'] = signal_loc
        output['loc'] = np.moveaxis(minima[0], -1, 0)
        output['locraw'] = np.moveaxis(minima[1], -1, 0)
        if upsampling:
            output['locup'] = np.moveaxis(minima[2], -1, 0)
            output['locupraw'] = np.moveaxis(minima[3], -1, 0)
        output['value'] = np.moveaxis(-minval[0], -1, 0)
        output['valueclean'] = np.moveaxis(-cleanval, -1, 0)
        output['filtnoisesdev'] = np.moveaxis(sdev, -1, 0)
        output['filter_start'] = filter_start
        output['filter_skip'] = filter_skip
        
        output_event['signal'] = simulated_signal
        output_event['noise'] = simulated_noise
    
    def window_center(self, ifilter, isnr, itau='best'):
        """
        Calibrate the temporal localization of a given filter centering the
        median on the true location and return it in a format suitable as input
        to run_window().
        
        Parameters
        ----------
        ifilter : int array (ncenter,)
            The indices of the filter, see Filter.all.
        isnr : int array (ncenter,)
            The indices of the SNR.
        itau : int array (ncenter,) or str
            The indices of tau. If `best` (default), take the best temporal
            resolution for each filter and SNR.
        
        Return
        ------
        wcenter : array (1 + ncenter,)
            A structured numpy array with these fields:
                'ifilter', int
                'itau', int
                'isnr', int
                'center', int, (nevents,)
                    The calibrated temporal localization for given filter, tau,
                    snr, rounded to nearest integer.
            
            The first entry in the array has the indices fields set to 9999 and
            the 'center' field set to the true signal location.
        """
        locfield = 'loc'
        ifilter = np.asarray(ifilter)
        isnr = np.asarray(isnr)
        
        loctrue = self.output['loctrue']
        locall = self.output[locfield]
        
        if isinstance(itau, str) and itau == 'best':
            res = self.templocres(locfield)
            itau2d = np.argmin(res, axis=1)
            itau = itau2d[ifilter, isnr]
        else:
            itau = np.asarray(itau)
        
        loc = locall[:, ifilter, itau, isnr].T
        corr = np.median(loc - loctrue, axis=-1)
        center = loc - corr[:, None]
        
        wcenter = np.empty(1 + len(itau), dtype=[
            ('ifilter', int),
            ('itau', int),
            ('isnr', int),
            ('center', int, (len(loctrue),))
        ])
        wcenter[0] = (9999, 9999, 9999, np.rint(loctrue))
        wcenter[1:]['ifilter'] = ifilter
        wcenter[1:]['itau'] = itau
        wcenter[1:]['isnr'] = isnr
        wcenter[1:]['center'] = np.rint(center)
        
        return wcenter
    
    def run_window(self, wlen, wlmargin, wcenter=None, pbar=None):
        """
        Extract a subset of samples from simulated signals and relocalize the
        signal filtering only in the window with the matched filter.
                
        This function sets the member `output_window`.

        Parameters
        ----------
        wlen : int array (nwin,)
            Lengths for windows that are extracted around the localized signal.
        wlmargin : int array (nwin,)
            The window starts `wlmargin` samples before the signal template
            start.
        wcenter : array (ncenter,), optional
            If specified, the window position is relative to the specified
            sample for each event instead of the signal template start.
            Use the method window_center() to compute this array.
        pbar : int, optional
            If given, a progress bar is shown that ticks every `pbar` events.
        
        """
        
        # TODO For speed I should run a single tau cross correlation filter
        # on each window instead of all taus.

        wlen = np.asarray(wlen)
        wlmargin = np.asarray(wlmargin)
        if wcenter is None:
            i = np.empty(0, int)
            wcenter = self.window_center(i, i, i)
        
        nwin, = wlen.shape
        assert wlmargin.shape == (nwin,)
        nevents = len(self.output)
        ncenter, = wcenter.shape
                
        output = np.empty(nevents, dtype=[
            ('wstart', int, (ncenter, nwin)),
            ('wloc', float, (ncenter, nwin, len(self.tau), len(self.snr)))
        ])
        
        def fun(s):
            self._run_window(output[s], self.output[s], self.output_event[s], wlen, wlmargin, wcenter['center'][:, s])
        run_sliced(fun, len(self.output), pbar)

        self.output_window = output
        self.wlen = wlen
        self.wlmargin = wlmargin
        self.wcenter = wcenter

    def _run_window(self, output, run_output, run_output_event, wlen, wlmargin, wcenter):
        
        # Get lengths of things.
        ntau = len(self.tau)
        nwin = len(wlen)
        nevents = len(output)
        nsnr = len(self.snr)
        ncenter = len(wcenter)
        event_length = self.event_length
        
        # Extract simulated signals.
        simulated_signal = run_output_event['signal']
        simulated_noise = run_output_event['noise']
        signal_loc = run_output['loctrue']
        sigma = self.sigma
        
        # Things used in the cycle.
        minima = np.empty((ncenter, nwin, ntau, nsnr, nevents))
        wstart = wcenter[:, None, :] - wlmargin[None, :, None]
        assert wstart.shape == (ncenter, nwin, nevents)
        indices = np.ix_(np.arange(ncenter), np.arange(nsnr), np.arange(nevents))
        
        for i, wlen in enumerate(wlen):
            
            # Extract window.
            idx0 = np.arange(nevents)[None, :, None]
            idx1 = wstart[:, i, :, None] + np.arange(wlen)
            idx1 = np.minimum(idx1, event_length - 1)
            idx1 = np.maximum(idx1, 0)
            wsignal = simulated_signal[idx0, idx1]
            wnoise = simulated_noise[idx0, idx1]
            assert wsignal.shape == (ncenter, nevents, wlen)
            
            # Make filter objects.
            rmargin = max(0, np.max(self.tau) - wlen) + 64
            filt_wnoise = Filter(wnoise.reshape(-1, wlen), 0, rmargin)
            filt_wsignal = Filter(wsignal.reshape(-1, wlen), 0, rmargin)
            
            for j, tau in enumerate(self.tau):
                
                # Run the matched filter.
                templ = self.mftempl(j)
                noise = filt_wnoise.matched(templ).reshape(ncenter, nevents, wlen + rmargin)
                signal = filt_wsignal.matched(templ).reshape(ncenter, nevents, wlen + rmargin)
                sim = signal[:, None] + sigma[None, :, None, None] * noise[:, None]
                assert sim.shape == (ncenter, nsnr, nevents, wlen + rmargin)
        
                # Interpolate the minimum with a parabola.
                x0 = np.argmin(sim, axis=-1)
                xp = np.minimum(x0 + 1, wlen + rmargin - 1)
                xm = np.maximum(x0 - 1, 0)
            
                y0 = sim[indices + (x0,)]
                yp = sim[indices + (xp,)]
                ym = sim[indices + (xm,)]
            
                num = yp - ym
                denom = yp + ym - 2 * y0
    
                minima[:, i, j] = x0 - 1/2 * num / denom
        
        # Align temporal localization.
        wloc = minima # shape == (ncenter, nwin, ntau, nsnr, nevents)
        wloc += wstart[:, :, None, None, :]
        # wloc -= self.tau[:, None, None]
        # wloc += self.templs['offset'][:, None, None]
        
        # Save results.
        output['wstart'] = np.moveaxis(wstart, -1, 0)
        output['wloc'] = np.moveaxis(wloc, -1, 0)
    
    def sampling_str(self):
        """
        A string representing the sampling frequency.
        """
        if self.timebase > 1:
            freq = 1000 / self.timebase
            return f'{freq:.3g} MSa/s'
        else:
            return '1 GSa/s'
    
    def plot_event(self, ievent, ifilter, itau, isnr, ax=None):
        """
        Plot a simulated event.
        
        Parameters
        ----------
        ievent, ifilter, itau, isnr : int
            The indices indicating the event, filter, tau and SNR respectively.
            Set ifilter=None to plot all filters.
        ax : matplotlib axis, optional
            If provided, draw the plot on the given axis. Otherwise (default)
            make a new figure and show it.
        
        Return
        ------
        fig : matplotlib figure or None
            The figure object, if ax is not specified.
        """
        # Get output for the event.
        out = self.output[ievent]
        oute = self.output_event[ievent]
        
        tau = self.tau[itau]
        snr = self.snr[isnr]
        sigma = self.sigma[isnr]
        fstart = out['filter_start'][itau]
        fskip = out['filter_skip'][itau]
        
        unfilt = oute['signal'] + sigma * oute['noise']
        templ = self.mftempl(itau)
        sim = Filter(unfilt[None, fstart:]).all(templ)[:, 0]
        
        if ax is None:
            fig, ax = plt.subplots(num='toy.Toy.plot_event', clear=True)
            returnfig = True
        else:
            returnfig = False
    
        ax.plot(unfilt, label=f'Signal+noise (SNR={snr:.3g})', color='#f55')
        
        if ifilter is None:
            ifilter = [1, 2, 3]
        else:
            ifilter = [ifilter]
        kw = [
            None,
            dict(linestyle=':'),
            dict(linestyle='--'),
            dict(linestyle='-'),
        ]
        for ifilt in ifilter:
            label = f'{Filter.name(ifilt)} ({Filter.tauname(ifilt)}={tau})'
            x = np.arange(fstart, len(unfilt))
            y = sim[ifilt]
            ax.plot(x, y, label=label, color='black', **kw[ifilt])
            
            loc = out['loc'][ifilt, itau, isnr]
            val = -out['value'][ifilt, itau, isnr]
            ax.plot(loc, val, marker='o', color='black')
        
        ax.axvspan(fstart + fskip, len(unfilt), label='Samples used for localization', color='#ddd')
        ax.axvline(out['loctrue'], label='Signal template start', color='black', linestyle='-.')
        # ax.axhline(out['baseline'][ifilter, itau, isnr], label='baseline', color='black', linestyle='--')        
        
        ax.legend(loc='upper left', fontsize='small', title=f'Event {ievent}')
        ax.set_xlabel(f'Sample number @ {self.sampling_str()}')

        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        # ax.set_xlim(0, len(unfilt))
    
        if returnfig:
            fig.tight_layout()
            fig.show()
            return fig
    
    def _win_center_str(self, icenter):
        """
        Return a string describing the filter used to center the windows.
        """
        if icenter == 0:
            return 'signal template start'
        jfilter, jtau, jsnr, _ = self.wcenter[icenter]
        r = self.templocres()[jfilter, jtau, jsnr]
        tauname = 'N' if jfilter != 2 else '$\\tau$'
        taustr = f', {tauname}={self.tau[jtau]}' if jfilter != 0 else ''
        return f'{Filter.name(jfilter, short=True)}{taustr}, SNR={self.snr[jsnr]:.3g} ($\\sigma$ = {r:.2g} Sa)'
    
    def plot_event_window(self, ievent, isnr, itau, iwlen, icenter=0, ax=None):
        """
        Plot a simulated event with localization from window.
        
        Parameters
        ----------
        ievent, isnr, itau, iwlen : int
            The indices indicating the event, SNR, tau and window length.
        icenter : int
            The index indicating the window centering. Default 0 i.e. centering
            on the signal template start.
        ax : matplotlib axis, optional
            If provided, draw the plot on the given axis. Otherwise (default)
            make a new figure and show it.
        
        Return
        ------
        fig : matplotlib figure or None
            The figure object, if ax is not specified.
        """
        out = self.output[ievent]
        oute = self.output_event[ievent]
        outw = self.output_window[ievent]
        
        fstart = out['filter_start'][itau]

        tau = self.tau[itau]
        wlen = self.wlen[iwlen]
        wlmargin = self.wlmargin[iwlen]
        wstart = outw['wstart'][icenter, iwlen]
        snr = self.snr[isnr]
        sigma = self.sigma[isnr]
        
        unfilt = oute['signal'] + sigma * oute['noise']
        templ = self.mftempl(itau)
        filt = Filter(unfilt[None, fstart:])
        sim = filt.matched(templ)[0]
        wfilt = Filter(unfilt[None, wstart:wstart + wlen], 0, len(unfilt) - wstart - wlen)
        wsim = wfilt.matched(templ)[0]
        
        if ax is None:
            fig, ax = plt.subplots(num='toy.Toy.plot_event_window', clear=True)
            returnfig = True
        else:
            returnfig = False
    
        ax.plot(unfilt, label=f'Signal+noise (SNR={snr:.3g})', color='#f55')
        label = f'{Filter.name(3)} ({Filter.tauname(3)}={tau})'
        ax.plot(fstart + np.arange(len(sim)), sim, color='#888', linewidth=3, label=label)
        ax.plot(wstart + np.arange(len(wsim)), wsim, label='Same filter on window only', linestyle='-', color='black', zorder=10)
        
        ax.axvline(out['loctrue'], label='Signal template start', color='black', linestyle='-.')
        window_label = f'Window ($-${wlmargin}+{wlen - wlmargin}), centered with\n'
        window_label += self._win_center_str(icenter)
        ax.axvspan(wstart, wstart + wlen, label=window_label, color='#ddd')
        # ax.axhline(self.template.baseline, label='baseline', color='black', linestyle='--')
        
        ax.legend(loc='best', fontsize='small', title=f'Event {ievent}')
        ax.set_xlabel(f'Sample number @ {self.sampling_str()}')
        
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        # ax.set_xlim(0, len(unfilt))

        if returnfig:
            fig.tight_layout()
            fig.show()
            return fig

    def templocres(self, locfield='loc', sampleunit=True):
        """
        Compute the temporal localization resolution as the half interquantile
        range from 0.16 to 0.84, which is approximately a standard deviation for
        the normal distribution, of the localization error, in units of samples.
        
        Parameters
        ----------
        locfield : str or array
            The field to read from the `output` member for the location.
            Default 'loc'. If an array, it is used directly.
        sampleunit : bool
            If True (default) use the sample duration as time unit, otherwise
            use nanoseconds.
        
        Return
        ------
        error : array
            The temporal localization error. The shape depends on the input.
        """
        if isinstance(locfield, str):
            loc = self.output[locfield]
        else:
            loc = locfield
        shape = (-1,) + (1,) * (len(loc.shape) - 1)
        loctrue = self.output['loctrue'].reshape(shape)
        quantiles = np.quantile(loc - loctrue, [0.5 - 0.68/2, 0.5 + 0.68/2], axis=0)
        x = (quantiles[1] - quantiles[0]) / 2
        if not sampleunit:
            x *= self.timebase
        return x
    
    def filteredsnr(self):
        """
        Compute the SNR after filtering, i.e. the median peak filter output
        over the filtered noise rms.
        
        Return
        ------
        fsnr : array (4, ntau, nsnr)
            The filtered SNR.
        """
        val = np.median(self.output['value'], axis=0)
        width = np.median(self.output['filtnoisesdev'], axis=0)[:, :, None] * self.sigma[None, None, :]
        return val / width

    def snrratio(self):
        """
        Compute the ratio SNR after filtering over SNR before filtering,
        where the SNR is the median peak signal amplitude over the noise rms.
        
        Return
        ------
        fsnr : array (4, ntau)
            The SNR ratio for the various filters.
        """
        # nofilter = np.median(self.output['valueclean'][:, 0], axis=0)
        # assert np.all(nofilter[0] == nofilter)
        # S = nofilter[0]
        # TODO for the SNR before filtering I can use the output_event array
        ampl = self.sigma * self.snr
        assert np.allclose(ampl[0], ampl)
        S = ampl[0]
        N = 1
        SNR = S / N
        FS = np.median(self.output['valueclean'], axis=0)
        FN = np.median(self.output['filtnoisesdev'], axis=0)
        FSNR = FS / FN
        return FSNR / SNR
        
    def _snr_helper(self, ifilter=3, itau=0):
        nofilter = np.median(self.output['valueclean'][:, 0], axis=0)
        assert np.all(nofilter[itau] == nofilter)
        S = nofilter[itau]
        N = 1
        FS = np.median(self.output['valueclean'], axis=0)[ifilter, itau]
        FN = np.median(self.output['filtnoisesdev'], axis=0)[ifilter, itau]
        return FS, FN, S, N

    def plot_loc_all(self, logscale=True, sampleunit=True, snrspan=None, locfield='loc', axs=None):
        """
        Plot temporal localization resolution vs filter, filter length, SNR.
        
        Parameters
        ----------
        logscale : bool
            If True (default), use a vertical logarithmic scale instead of a
            linear one.
        sampleunit : bool
            If True (default) use the sample duration as time unit, otherwise
            use nanoseconds.
        snrspan : tuple of two scalars
            Values to plot a vertical span between two SNR values.
        locfield : str
            The field in `output` used as temporal localization. Default 'loc'.
        axs : 2x2 array of matplotlib axes, optional
            If provided, draw the plot on the given axes. Otherwise (default)
            make a new figure and show it.
        
        Return
        ------
        fig : matplotlib figure or None
            The figure object, if axs is not specified.
        """
        tau = self.tau
        snr = self.snr
        output = self.output
        
        width = self.templocres(locfield, sampleunit)
        
        if axs is None:
            fig, axs = plt.subplots(2, 2, num='toy.Toy.plot_loc_all', figsize=[10.95, 7.19], clear=True)
            returnfig = True
        else:
            returnfig = False

        axs = axs.reshape(-1)
        
        linekw = dict(color='#600')
        for ifilter, ax in enumerate(axs):
            if ifilter > 0:
                lines = []
                for itau in range(len(tau)):
                    alpha = (itau + 1) / len(tau)
                    labeltau = tau[itau]
                    line, = ax.plot(snr, width[ifilter, itau], alpha=alpha, label=f'{labeltau}', **linekw)
                    lines.append(line)
            else:
                ax.plot(snr, width[0, 0], **linekw)
        
            if not sampleunit:
                ax.axhspan(0, self.timebase, zorder=-10, color='#ddd')
            
            if snrspan is not None:
                ax.axvspan(*snrspan, color='#ccf', zorder=-11)
            
            if ax.is_last_row():
                ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
            if ax.is_first_col():
                unitname = f'{self.timebase} ns' if sampleunit else 'ns'
                ax.set_ylabel(f'half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [{unitname}]')
            
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
            
            textbox.textbox(ax, Filter.name(ifilter), loc='upper center', fontsize='medium')
        
        labels = [line.get_label() for line in lines]
        legendtitle = 'N or $\\tau$ [samples]'
        axs[0].legend(lines, labels, ncol=3, fontsize='small', title=legendtitle, loc='best', framealpha=0.9)
        
        if logscale:
            axs[0].set_yscale('log')
        else:
            lims = axs[0].get_ylim()
            axs[0].set_ylim(min(0, lims[0]), lims[1])
        
        if returnfig:
            fig.tight_layout()
            fig.show()
            return fig

    def plot_loc_window(self, itau, icenter=0, logscale=True, ax=None):
        """
        Plot temporal localization precision for matched filter on windows
        for each SNR and window length at fixed filter length.
        
        Parameters
        ----------
        itau : int
            The index of the filter length.
        icenter : int
            The index of the window centering. Default 0 i.e. centering on
            signal template start.
        logscale : bool
            If True (default), use a vertical logarithmic scale instead of a
            linear one.
        ax : matplotlib axis, optional
            If provided, draw the plot on the given axis. Otherwise (default)
            make a new figure and show it.
        
        Return
        ------
        fig : matplotlib figure or None
            The figure object, if ax is not specified.
        """
        tau = self.tau[itau]
        snr = self.snr
        wlen = self.wlen
        wmargin = self.wlmargin
        
        output = self.output
        output_window = self.output_window
        
        fstart = output[0]['filter_start'][itau]
        wlen_orig = self.event_length - fstart
        wmargin_orig = int(np.rint(np.mean(output['loctrue'] - fstart)))
        
        width = self.templocres(output_window['wloc'], sampleunit=False)[icenter, :, itau]
        width_orig = self.templocres(sampleunit=False)[3, itau]
        
        if ax is None:
            fig, ax = plt.subplots(num='toy.Toy.plot_loc_window', clear=True)
            returnfig = True
        else:
            returnfig = False
        
        for iwlen, wl in enumerate(wlen):
            alpha = (iwlen + 1) / len(wlen)
            lenus = wl * self.timebase / 1000
            label = f'$-${wmargin[iwlen]}+{wl - wmargin[iwlen]} ({lenus:.1f} $\\mu$s)'
            ax.plot(snr, width[iwlen], alpha=alpha, label=label, color='#600000')
        ax.plot(snr, width_orig, 'k.', label=f'$-${wmargin_orig}+{wlen_orig - wmargin_orig}')
        ax.legend(loc='best', title='Window [samples]\n$-$L+R of center')
        
        ax.axhspan(0, self.timebase, color='#ddd', zorder=-10)

        if ax.is_last_row():
            ax.set_xlabel('Unfiltered SNR (avg signal peak over noise rms)')
        if ax.is_first_col():
            ax.set_ylabel('Half "$\\pm 1 \\sigma$" interquantile range of\ntemporal localization error [ns]')
        ax.grid(True, which='major', axis='both', linestyle='--')
        ax.grid(True, which='minor', axis='both', linestyle=':')
        ax.minorticks_on()
        title = f'{Filter.name(3)} (N={tau}), window centered\nwith '
        title += self._win_center_str(icenter)
        textbox.textbox(ax, title, fontsize='small', loc='upper center')
        
        if logscale:
            ax.set_yscale('log')
        
        if returnfig:
            fig.tight_layout()
            fig.show()
            return fig

    def plot_loc(self, itau, isnr, locfield='loc', axs=None, center=False):
        """
        Plot temporal localization histograms for all filters.
        
        Parameters
        ----------
        itau, isnr : int
            Indices of the tau and SNR values to use.
        locfield : str
            The field in `output` used as temporal localization. Default 'loc'.
        axs : 2x2 array of matplotlib axes, optional
            If provided, draw the plot on the given axes. Otherwise (default)
            make a new figure and show it.
        center : bool
            If True, calibrate the localization such that the median of the
            localization error is zero. Default (False) use the filter minimum
            as-is.
        
        Return
        ------
        fig : matplotlib figure or None
            The figure object, if axs is not specified.
        """
        tau = self.tau[itau]
        snr = self.snr[isnr]
        output = self.output
        
        if axs is None:
            fig, axs = plt.subplots(2, 2, num='toy.Toy.plot_loc', clear=True, figsize=[10.95,  7.19])
            returnfig = True
        else:
            returnfig = False

        axs = axs.reshape(-1)
    
        for ifilter, ax in enumerate(axs):
            data = output[locfield][:, ifilter, itau, isnr] - output['loctrue']
            if center:
                data -= np.median(data)
            h, _, _ = ax.hist(data, bins='auto', histtype='step', color='#f55', zorder=10)
            
            left, cent, right = np.quantile(data, [0.5 - 0.68/2, 0.5, 0.5 + 0.68/2])
            xerr = [[cent - left], [right - cent]]
            ax.errorbar(cent, np.max(h) / 2, xerr=xerr, capsize=4, fmt='k.', label='"$\\pm 1 \\sigma$" quantiles', zorder=11)
            
            if ax.is_last_row():
                if center:
                    xlabel = 'Temporal localization error [samples]'
                else:
                    xlabel = 'Uncalibrated temporal\nlocalization error [samples]'
                ax.set_xlabel(xlabel)
            if ax.is_first_col():
                ax.set_ylabel('Counts per bin')
            if ifilter == 0:
                ax.legend(loc='center right')
            
            if ifilter == 0:
                label = f'SNR={snr:.3g}'
            else:
                label = Filter.tauname(ifilter) + f'={tau}'
            textbox.textbox(ax, f'{Filter.name(ifilter)} ({label})', loc='upper center', fontsize='medium', zorder=12)
            
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
        
        if returnfig:
            fig.tight_layout()
            fig.show()
            return fig

    def plot_val(self, itau, isnr):
        """
        Plot a histogram of the baseline corrected filtered value at signal
        detection point for each filter.
        
        Parameters
        ----------
        itau, isnr : int
            Indices indicating the tau and SNR values to use.
        
        Return
        ------
        fig : matplotlib.figure.Figure
            A figure with an axis for each filter.
        """
        
        tau = self.tau[itau]
        snr = self.snr[isnr]
        sigma = self.sigma[isnr]

        fig = plt.figure('toy.Toy.plot_val', figsize=[10.95,  7.19])
        fig.clf()
        
        noise = self.noisegen.generate(1, len(output) + tau)
        filt = Filter(sigma * noise)
        templ = self.mftempl(itau)
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
                legendtitle = f'(@ {self.sampling_str()})'
            
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
