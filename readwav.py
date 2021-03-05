"""
Module to read an LNGS wav. Functions:

readwav :
    Read an LNGS wav.
spurious_signals :
    Determine which events have spurious signals before the trigger.
first_nonzero :
    Find the first nonzero value in each row of a matrix; use it to find the
    trigger leading edge.
"""

import os

from scipy.io import wavfile
import numpy as np

def readwav(filename, maxevents=None, mmap=True, quiet=False, swapch='auto'):
    """
    Read an LNGS wav, one of those like nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav.
    
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
    swapch : bool
        If False, assume the first channel in the wav file is the signal and
        the second is the trigger. If True, the opposite. If 'auto' (default),
        decide based on the file name.
    
    Return
    ------
    data : array (nevents, nchannels=2, nsamples=15001)
        The first channel is the signal and the second channel is the trigger.
    """
    if swapch == 'auto':
        _, name = os.path.split(filename)
        swapch = name.startswith('LF_TILE')
    
    if not quiet:
        print(f'reading {filename}...')
    _, data = wavfile.read(filename, mmap=True)
    # mmap = memory map, no RAM used
    
    eventsize = 30022
    eventgap = 20
    # The number 30022 is from dsfe/README.md, the 20 sample gap is from
    # looking at the output of dsfe/readwav.
    data.shape = (len(data) // eventsize, eventsize)
    
    if maxevents is not None:
        data = data[:maxevents]
    
    data = data[:, eventgap:]
    data.shape = (data.shape[0], 2, (eventsize - eventgap) // 2)
    
    if swapch:
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
