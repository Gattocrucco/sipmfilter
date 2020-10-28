from scipy.io import wavfile
import numpy as np

def readwav(filename, maxevents=None, mmap=True):
    """
    Read a wav file, one of those like nuvhd_lf_3x_tile57_77K_64V_6VoV_1.wav,
    return an array with shape (nevents, nchannels=2, nsamples=15001), where
    the first channel is the signal and the second channel is the trigger.
    
    Parameters
    ----------
    maxevents : int
        Maximum number of events read from the file. Default no limit.
    mmap : bool
        If True, the array is memory mapped i.e. it is not actually on RAM.
    """
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
