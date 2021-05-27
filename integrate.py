"""
Module to filter an LNGS wav. Functions:

make_start_mf :
    Compute the truncation alignment of a matched filter template.
filter :
    Filter an LNGS wav computing the filter at a single point a fixed delay
    after the trigger, also compute the trigger position and the baseline.
"""

import numba
import numpy as np

_asarray1d = lambda a, d: np.asarray(a, d).reshape(-1)

def _asarray1d_fullifnone(a, v, d):
    if a is None:
        return np.full(a.size, v, d)
    else:
        return _asarray1d(a, d)

_emptyint = np.empty(0, int)

def filter(data, bslen=6900, delta_ma=None, length_ma=None, delta_exp=None, tau_exp=None, delta_mf=None, waveform_mf=None, length_mf=None, start_mf=None):
    """
    Filter LNGS wav data.
    
    Parameters
    ----------
    data : array (nevents, 2, 15001)
        As read by readwav.py.
    bslen : int
        The number of samples used for the baseline.
    delta_ma : array (N,) or scalar
        Sample where the moving average filter is evaluated relative to the
        start of the trigger impulse.
    length_ma : array (N,) or scalar
        Number of averaged samples.
    delta_exp : array (M,) or scalar
        Sample where the exponential filter is evaluated relative to the start
        of the trigger impulse.
    tau_exp : array (M,) or scalar
        Scale parameter of the exponential filter.
    delta_mf : array (K,) or scalar
        Sample where the matched filter is evaluated relative to the start of
        the trigger impulse.
    waveform_mf : 1D array
        The waveform correlated to compute the matched filter.
    length_mf, start_mf : array (K,) or scalar
        The waveform is truncated to the slice start_mf:start_mf+length_mf.
        After the truncation, the waveform is normalized to unity. These
        parameters can be None even if delta_mf and waveform_mf are specified.
        delta_mf is still relative to the beginning of the untruncated waveform.
    
    Returns
    -------
    trigger : array (nevents,)
        Within-event sample where the trigger fires.
    baseline : array (nevents,)
        Average of the region before the trigger impulse.
    
    Each of the following is returned only if its corresponding parameters are
    specified:
    
    ma : array (nevents, N)
        Value of the moving average filter.
    exp : array (nevents, M)
        Value of the exponential moving average filter.
    mf : array (nevents, K)
        Value of the matched filter.
    """
    bslen = int(bslen)
    assert bslen > 0
    
    compute_ma = delta_ma is not None and length_ma is not None
    if compute_ma:
        delta_ma = _asarray1d(delta_ma, int)
        length_ma = _asarray1d(length_ma, int)
    else:
        delta_ma = _emptyint
        length_ma = _emptyint

    compute_exp = delta_exp is not None and tau_exp is not None
    if compute_exp:
        delta_exp = _asarray1d(delta_exp, int)
        tau_exp = _asarray1d(tau_exp, int)
    else:
        delta_exp = _emptyint
        tau_exp = _emptyint
    
    compute_mf = delta_mf is not None and waveform_mf is not None
    if compute_mf:
        delta_mf = _asarray1d(delta_mf, int)
        waveform_mf = np.asarray(waveform_mf, float)
        length_mf = _asarray1d_fullifnone(length_mf, len(waveform_mf), int)
        start_mf = _asarray1d_fullifnone(start_mf, 0, int)
    else:
        delta_mf = _emptyint
        waveform_mf = np.empty(0, float)
        length_mf = _emptyint
        start_mf = _emptyint
    
    start, baseline, ma, exp, mf = _filter(data, bslen, delta_ma, length_ma, delta_exp, tau_exp, delta_mf, waveform_mf, length_mf, start_mf)
    
    output = (start, baseline)
    if compute_ma:
        output += (ma,)
    if compute_exp:
        output += (exp,)
    if compute_mf:
        output += (mf,)
    return output

@numba.jit(cache=True, nopython=True)
def _filter(data, bslen, delta_ma, length_ma, delta_exp, tau_exp, delta_mf, waveform_mf, length_mf, start_mf):
    """
    Compiled internal for filter(). Parameters are the same but non optional.
    
    Returns
    -------
    trigger : array (nevents,)
        Within-event sample where the trigger fires.
    baseline : array (nevents,)
        Average of the region before the trigger impulse.
    ma : array (nevents, N)
        Value of the moving average filter.
    exp : array (nevents, M)
        Value of the exponential moving average filter.
    mf : array (nevents, K)
        Value of the matched filter.
    """
    nevents = data.shape[0]
    N = len(delta_ma)
    M = len(delta_exp)
    K = len(delta_mf)
    
    trigger = np.empty(nevents, np.int32)
    baseline = np.empty(nevents)
    ma = np.empty((nevents, N))
    exp = np.empty((nevents, M))
    mf = np.empty((nevents, K))
    
    for i in range(nevents):
        wsig, wtrig = data[i]
        
        trigger[i] = _trigger(wtrig)
        baseline[i] = _baseline(wsig, trigger[i], bslen)
        for j in range(N):
            ma[i, j] = _filter_ma(wsig, trigger[i] + delta_ma[j], length_ma[j])
        for j in range(M):
            exp[i, j] = _filter_exp(wsig, trigger[i] + delta_exp[j], tau_exp[j])
        for j in range(K):
            w = waveform_mf[start_mf[j] : start_mf[j] + length_mf[j]]
            t = trigger[i] + delta_mf[j] + start_mf[j]
            mf[i, j] = _filter_matched(wsig, t, w)
    
    return trigger, baseline, ma, exp, mf

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
    lamda = 1 - 1/l # taylor of exp(-1/l)
    out = x[0]
    for i in range(1, t + 1):
        out = lamda * out + (1 - lamda) * x[i]
    return out

@numba.jit(cache=True, nopython=True)
def _filter_matched(x, t, w):
    return np.sum(x[t - len(w) + 1:t + 1] * w) / np.sum(w)

def make_start_mf(waveform_mf, length_mf):
    """
    Find the optimal start_mf parameter for filter() given the waveform and the
    truncation length, in the sense of finding the slice that maximizes the
    sum of squares of the slice.
    
    Parameters
    ----------
    waveform_mf, length_mf : arrays
        See filter().
    
    Return
    ------
    start_mf : array
        See filter().
    """
    waveform_mf = _asarray1d(waveform_mf, float)
    length_mf = _asarray1d(length_mf, int)

    start_mf = np.empty(len(length_mf), int)
    cs = np.concatenate([[0], np.cumsum(waveform_mf ** 2)])
    for i, l in enumerate(length_mf):
        assert l <= len(waveform_mf)
        s = cs[l:] - cs[:-l] # s[j] = sum(waveform_mf[j:j+l] ** 2)
        start_mf[i] = np.argmax(s)
    
    return start_mf

@numba.jit(cache=True, nopython=True)
def integrate(data, bslen=6900):
    """
    DEPRECATED, use `filter`
    
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
