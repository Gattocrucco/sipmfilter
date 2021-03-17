import numpy as np
from scipy import signal

def correlate(waveform, template, method='fft', axis=-1, boundary=None, lpad=0):
    """
    Compute the cross correlation of two arrays.
    
    The correlation is computed with padding to the right but not to the left.
    So the first element of the cross correlation is
    
        sum(waveform[:len(template)] * template),
    
    while the last is
    
        waveform[-1] * template[0] + sum(boundary * template[1:])
    
    Parameters
    ----------
    waveform : array (..., N, ...)
        The non-inverted term of the convolution.
    template : array (M,)
        The inverted term of the convolution.
    method : {'fft', 'oa'}
        Use fft (default) or overlap-add to compute the convolution.
    axis : int
        The axis of `waveform` along which the convolution is computed, default
        last.
    boundary : scalar, optional
        The padding value for `waveform`. If not specified, use the last value
        in each subarray.
    lpad : int
        The amount of padding to the left. Default 0.
    
    Return
    ------
    corr : array (..., N, ...)
        The cross correlation, with the same shape as `waveform`.
    """
    rpad = len(template) - 1
    padspec = [(0, 0)] * len(waveform.shape)
    padspec[axis] = (lpad, rpad)
    if boundary is None:
        padkw = dict(mode='edge')
    else:
        padkw = dict(mode='constant', constant_values=boundary)
    waveform_padded = np.pad(waveform, padspec, **padkw)
    
    idx = [None] * len(waveform.shape)
    idx[axis] = slice(None, None, -1)
    template_bc = template[tuple(idx)]
    
    funcs = dict(oa=signal.oaconvolve, fft=signal.fftconvolve)    
    return funcs[method](waveform_padded, template_bc, mode='valid', axes=axis)

def test_correlate():
    """
    Plot a test of `correlate`.
    """
    waveform = 1 - np.pad(np.ones(100), 100)
    waveform += 0.2 * np.random.randn(len(waveform))
    template = np.exp(-np.linspace(0, 3, 50))
    corr1 = correlate(np.repeat(waveform[None, :], 2, 0), template / np.sum(template), 'oa' , axis=1)[0, :]
    corr2 = correlate(np.repeat(waveform[:, None], 2, 1), template / np.sum(template), 'fft', axis=0)[:, 0]
    
    fig, ax = plt.subplots(num='correlate.test_correlate', clear=True)
    
    ax.plot(waveform, label='waveform')
    ax.plot(template, label='template')
    ax.plot(corr1, label='corr oa')
    ax.plot(corr2, label='corr fft', linestyle='--')
    
    ax.legend()
    
    fig.tight_layout()
    fig.show()

def timecorr(lenwaveform, lentemplate, method, n=100):
    """
    Time `correlate`.
    
    Parameters
    ----------
    lenwaveform : int
        Length of each waveform.
    lentemplate : int
        Length of the template.
    method : {'fft', 'oa'}
        Algorithm.
    n : int
        Number of waveforms, default 100.
    
    Return
    ------
    time : scalar
        The time, in seconds, taken by `correlate`.
    """
    waveform = np.random.randn(n, lenwaveform)
    template = np.random.randn(lentemplate)
    start = time.time()
    correlate(waveform, template, method)
    end = time.time()
    return end - start

def timecorrseries(lenwaveform, lentemplates, n=100):
    """
    Call `timecorr` for a range of values.
    
    Parameters
    ----------
    lenwaveform : int
        The length of each waveform.
    lentemplate : int
        The length of the template.
    n : int
        The number of waveforms, default 100.
    
    Return
    ------
    times : dict
        Dictionary of dictionaries with layout method -> (lentemplate -> time).
    """
    return {
        method: {
            lentemplate: timecorr(lenwaveform, lentemplate, method)
            for lentemplate in lentemplates
        } for method in ['oa', 'fft']
    }

def plot_timecorrseries(timecorrseries_output):
    """
    Plot the output of `timecorrseries`.
    """
    fig, ax = plt.subplots(num='correlate.plot_timecorrseries', clear=True)
    
    for method, time in timecorrseries_output.items():
        ax.plot(list(time.keys()), list(time.values()), label=method)
    
    ax.legend()
    ax.set_xlabel('Template length')
    ax.set_ylabel('Time')
    
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    