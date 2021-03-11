"""
Module to read an LNGS wav. Functions:

readwav :
    Read an LNGS wav.
spurious_signals :
    Determine which events have spurious signals before the trigger.
firstbelowthreshold :
    As the name suggests. Use it to find the trigger leading edge.
"""

import os

from scipy.io import wavfile
import numpy as np
import numba

def readwav(filename, maxevents=None, mmap=True, quiet=False, swapch='auto'):
    """
    Read an LNGS laser wav.
    
    Parameters
    ----------
    filename : str
        The file path to read.
    maxevents : int
        Maximum number of events read from the file. Default no limit.
    mmap : bool
        If True, the array is memory mapped i.e. it is not actually on RAM.
    quiet : bool
        Default False. If True, do not print a log message.
    swapch : {'auto', bool}
        If False, assume the first channel in the wav file is the signal and
        the second is the trigger. If True, the opposite. If 'auto' (default),
        decide based on the file name. Applies only if there are two channels.
        
    Return
    ------
    data : array (nevents, nchannels, nsamples)
        When there are two channels, the first channel is the signal and the
        second channel is the trigger.
    """
    if swapch == 'auto':
        _, name = os.path.split(filename)
        swapch = name.startswith('LF_TILE')
    
    if not quiet:
        print(f'reading {filename}...')
    _, data = wavfile.read(filename, mmap=True)
    
    # parse metadata
    assert len(data.shape) == 1, data.shape
    assert len(data) >= 2, len(data)
    assert data[0] == -1, data[0]
    metasize = data[1]
    assert metasize == 20, metasize
    assert len(data) >= metasize, len(data)
    metadata = data[:metasize]
    size = metadata[6]
    assert size == 15001, size
    nchannels = metadata[10]
    assert nchannels == 1 or nchannels == 2, nchannels
    
    # divide data in events
    eventsize = metasize + nchannels * size
    assert len(data) % eventsize == 0, (len(data), eventsize)
    data.shape = (len(data) // eventsize, eventsize)
    
    if maxevents is not None:
        data = data[:maxevents]
    
    # divide channels
    data = data[:, metasize:]
    data.shape = (data.shape[0], nchannels, size)
    
    if nchannels == 2 and swapch:
        data = data[:, ::-1, :]
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

def first_nonzero(cond):
    """
    DEPRECATED, use firstbelowthreshold.
    
    Find the first nonzero element in each row of a matrix.
    
    Parameters
    ----------
    cond : array (N, M)
        The matrix.
    
    Return
    ------
    idx : array (N,)
        The index of the first nonzero element of `cond` in each row.
    """
    i0, i1 = np.nonzero(cond)
    i0, indices = np.unique(i0, return_index=True)
    idx = np.full(*cond.shape)
    idx[i0] = i1[indices]
    return idx

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
        The index in each event of the first element below `threshold`, N if
        there's none.
    """
    output = np.empty(len(events), np.intp)
    for ievent, event in enumerate(events):
        for isample, sample in enumerate(event):
            if sample < threshold:
                output[ievent] = isample
                break
        else:
            output[ievent] = len(event)
    return output
