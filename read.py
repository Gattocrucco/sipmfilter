import os

import numpy as np

import readwav
import readroot

def read(filespec, maxevents=None, quiet=False, mmap=True, swapch='auto'):
    """
    Read an LNGS laser wav or a Proto0 root.
    
    Parameters
    ----------
    filespec : str
        The file path. For root files, there must be an additional channel
        specification separated by a column. Examples:
        'XXX.root:adc_W200_Ch00', 'XXX.root:57'.
    maxevents : int
        The maximum number of events read from the file.
    quiet : bool
        If True, do not print 'reading <file>...'.
    mmap : bool
        (wav-specific) return a memmap to the file.
    swapch : str, bool
        (wav-specific) If True the signal and trigger channels are swapped in
        the file. 'auto' (default) tries to deduce it from the file name.
    
    Return
    ------
    array : int array (nevents, nsamples)
        The array with the signal waveforms.
    trigger : array (nevents,)
        The trigger position for each event in samples.
    freq : scalar
        The sampling frequency in samples per second.
    ndigit : int
        The number of ADC values.
    """
    path, last = os.path.split(filespec)
    comp = last.split(':')
    if len(comp) == 1 or comp[1] == '':
        name = comp[0]
        ch = None
    else:
        name = ':'.join(comp[:-1])
        ch = comp[-1]
    
    if name.endswith('.wav'):
        data = readwav.readwav(filespec, maxevents=maxevents, quiet=quiet, mmap=mmap, swapch=swapch)
        array = data[:, 0]
        trigger = readwav.first_nonzero(data[:, 1] < 600)
        freq = 1e9
        ndigit = 2 ** 10
    
    elif name.endswith('.root'):
        if ch is None:
            raise ValueError(f'missing channel for file {name}')
        if not ch.startswith('adc'):
            ch = int(ch)
        if path != '':
            path += '/'
        array, trigger, freq = readroot.readroot(path + name, ch, maxevents=maxevents, quiet=quiet)
        trigger = np.full(array.shape[0], trigger)
        ndigit = 2 ** 14
    
    else:
        raise ValueError(f'unrecognized extension for file {name}')
    
    return array, trigger, freq, ndigit
