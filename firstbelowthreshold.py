import numpy as np
import numba

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
        The index in each event of the first element below `threshold`. -1
        where the threshold is not crossed.
    """
    output = np.full(len(events), -1)
    for ievent, event in enumerate(events):
        for isample, sample in enumerate(event):
            if sample < threshold:
                output[ievent] = isample
                break
    return output
