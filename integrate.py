import numba
import numpy as np

@numba.jit(cache=True, nopython=True)
def filter(data, delta_ma, length_ma, delta_exp, tau_exp, bslen=6900):
    """
    Parameters
    ----------
    data : array (nevents, 2, 15001)
        As read by readwav.py.
    delta_ma : array (N,)
        Sample where the moving average filter is evaluated relative to the
        start of the trigger impulse.
    length_ma : array (N,)
        Number of averaged samples.
    delta_exp : array (M,)
        Sample where the exponential filter is evaluated relative to the
        start of the trigger impulse.
    tau_exp : array (M,)
        Scale parameter of the exponential filter.
    bslen : int
        The number of samples used for the baseline.
    
    Returns
    -------
    trigger : array (nevents,)
        Within-event sample where the trigger fires.
    baseline : array (nevents,)
        Average of the region before the trigger impulse.
    ma : array (nevents, N)
        Value of the moving average filter.
    exp : array (nevents, M)
        Value of the exponential average filter.
    """
    nevents = data.shape[0]
    N = len(delta_ma)
    M = len(delta_exp)
    
    trigger = np.empty(nevents, np.int32)
    baseline = np.empty(nevents)
    ma = np.empty((nevents, N))
    exp = np.empty((nevents, M))
    
    for i in range(nevents):
        wsig, wtrig = data[i]
        
        trigger[i] = _trigger(wtrig)
        baseline[i] = _baseline(wsig, trigger[i], bslen)
        for j in range(N):
            ma[i, j] = _filter_ma(wsig, trigger[i] + delta_ma[j], length_ma[j])
        for j in range(M):
            exp[i, j] = _filter_exp(wsig, trigger[i] + delta_exp[j], tau_exp[j])
    
    return trigger, baseline, ma, exp

@numba.jit(cache=True, nopython=True)
def _trigger(x):
    for i in range(len(x)):
        if x[i] < 400:
            break
    else:
        assert False, 'no trigger found'
    return i

@numba.jit(cache=True, nopython=True)
def _baseline(x, t, l):
    L = l + 100
    assert t >= L, 'not enough samples for baseline before trigger'
    return np.mean(x[t - L:t - 100])

@numba.jit(cache=True, nopython=True)
def _filter_ma(x, t, l):
    return np.mean(x[t - l + 1:t + 1])

@numba.jit(cache=True, nopython=True)
def _filter_exp(x, t, l):
    lamda = 1 - 1/l
    out = x[0]
    for i in range(1, t + 1):
        out = lamda * out + (1 - lamda) * x[i]
    return out

@numba.jit(cache=True, nopython=True)
def integrate(data, bslen=6900):
    """
    OLD FUNCTION, USE `filter` INSTEAD
    
    Take data from wav file and compute a 1000 samples average of the signal
    after each trigger pulse start.
    
    Parameters
    ----------
    data : array with shape (nevents, 2, 15001)
        Wav data as read by readwav.py.
    bslen : int
        The number of samples used for the baseline.
    
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
        bsstart = bslen + 100
        assert j >= bsstart, 'not enough samples for baseline before trigger'

        start[i] = j
        value[i] = np.mean(signal[j:j + 1000])
        baseline[i] = np.mean(signal[j - bsstart:j - 100])
    
    return start, value, baseline
