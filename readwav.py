from scipy.io import wavfile
import numpy as np

def readwav(filename, maxevents=None, mmap=True, quiet=False):
    """
    Read a wav file, one of those like nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav.
    
    Parameters
    ----------
    maxevents : int
        Maximum number of events read from the file. Default no limit.
    mmap : bool
        If True, the array is memory mapped i.e. it is not actually on RAM.
    quiet : bool
        Default False. If True, do not print a log message.
    
    Return
    ------
    data : array (nevents, nchannels=2, nsamples=15001)
        The first channel is the signal and the second channel is the trigger.
    """
    if not quiet:
        print(f'reading {filename}...')
    _, data = wavfile.read(filename, mmap=True)
    # mmap = memory map, no RAM used
    data = data.reshape(-1, 30022)[:, 20:].reshape(-1, 2, 15001)
    # The number 30022 is from dsfe/README.md, the 20 sample gap is from
    # looking at the output of dsfe/readwav.
    if maxevents is not None:
        data = data[:maxevents]
    if not mmap:
        data = np.copy(data)
    return data

def spurious_signals(data):
    """
    Identify events with out-of-trigger signals.
    
    Parameters
    ----------
    data : array (N, 2, 15001)
        The output from readwav().
    
    Return
    ------
    ignore : bool array (N,)
        Flag events with a sample less than 700 in the region from sample 0
        to sample 8900.
    """
    baseline_zone = data[:, 0, :8900]
    ignore = np.any((0 <= baseline_zone) & (baseline_zone < 700), axis=-1)
    return ignore
