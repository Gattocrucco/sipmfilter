import numba
import numpy as np

@numba.jit(cache=True, nopython=True)
def integrate(data):
    """
    Take data from wav file and add compute a 1000 samples average of the signal
    after each trigger pulse start.
    
    Parameters
    ----------
    data : array with shape (nevents, 2, 15001)
        Wav data as read by readwav.py.
    
    Returns
    -------
    start : array with shape (nevents,)
        The index, relative to each event, where the integration starts.
    value : array with shape (nevents,)
        The average of the "signal" region.
    baseline : array with shape (nevents,)
        The average of the region starting 7000 samples before the start and
        ending 100 samples before.
    """
    start = np.empty(data.shape[0], dtype=np.int32)
    value = np.empty(data.shape[0])
    baseline = np.empty(data.shape[0])
    
    for i in range(data.shape[0]):
        signal = data[i, 0]
        trigger = data[i, 1]
                
        for j in range(len(trigger)):
            if trigger[j] < 400:
                break
        else:
            assert False, 'no trigger start found'
        
        # Uncomment this to start from the end of the trigger square impulse.
        # for j in range(j + 1, len(trigger)):
        #     if 400 <= trigger[j] < 2 ** 10:
        #         break
        # else:
        #     assert False, 'no trigger end found'
        
        assert j + 1000 <= len(signal), 'less than 1000 samples after trigger'
        assert j >= 7000, 'less than 7000 samples before trigger'

        start[i] = j
        value[i] = np.mean(signal[j:j + 1000])
        baseline[i] = np.mean(signal[j - 7000:j - 100])
    
    return start, value, baseline
