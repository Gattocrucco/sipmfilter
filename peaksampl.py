import numpy as np

def peaksampl(y, z, t, yout=0, positive=True):
    """
    Get peak amplitudes given their sum.
    
    This assumes that the position of the signals is given by peaks positions
    even when they are summed.
    
    Parameters
    ----------
    y : array (..., M,)
        The single signal shape.
    z : array (..., N,)
        The peak height in the sum of the signals for each peak.
    t : int array (..., N,)
        The indices of the peaks in the sum.
    yout : scalar
        The value of the signal outside the provided values, default 0.
    positive : bool
        If False, the signal is negative, i.e. the peaks in z are minima
        corresponding to the minimum of y. Default True.
    
    Return
    ------
    a : array (..., N),
        The amplitudes such that z_i = sum_j a_j * y[t_i - t_j].
        Broadcasted along non-last axis.
    """
    y = np.asarray(y)
    z = np.asarray(z)
    t = np.asarray(t)
    
    y = np.pad(y, [(0, 0)] * (y.ndim - 1) + [(1, 1)], constant_values=yout)
    offset = (np.argmax if positive else np.argmin)(y, axis=-1)

    indices = t[..., :, None] - t[..., None, :] + offset[..., None, None]
    indices = np.minimum(indices, y.shape[-1] - 1)
    indices = np.maximum(indices, 0)
    
    N = t.shape[-1]
    indices = indices.reshape(indices.shape[:-2] + (N * N,))
    n = max(y.ndim, indices.ndim)
    y       = np.expand_dims(y      , tuple(range(n -       y.ndim)))
    indices = np.expand_dims(indices, tuple(range(n - indices.ndim)))
    y = np.take_along_axis(y, indices, -1)
    y = y.reshape(y.shape[:-1] + (N, N))
    
    z = z[..., None]
    n = max(y.ndim, z.ndim)
    y = np.expand_dims(y, tuple(range(n - y.ndim)))
    z = np.expand_dims(z, tuple(range(n - z.ndim)))

    a = np.linalg.solve(y, z)
    return np.squeeze(a, -1)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy import signal
    
    y = np.exp(-np.linspace(0, 10, 1000) / 10)
    i = np.arange(1, 1000)
    t0 = np.array([10, 340, 523])
    a0 = np.array([3, 2, 1])
    indices = i - t0[:, None]
    z = np.take(y, indices, mode='clip') * a0[:, None]
    z = np.where((indices < 0) | (indices >= len(y)), 0, z)
    z = np.sum(z, axis=0)
    t, = signal.argrelmax(z)
    assert len(t) == len(t0)
    a = peaksampl(y, z[t], t)
    
    fig, ax = plt.subplots(num='peaksampl', clear=True)
    
    ax.plot(z, color='#f55')
    ax.vlines(t0, 0, a0, color='gray', zorder=3)
    ax.vlines(t, 0, a, linestyle='--', zorder=3)
    
    ax.grid('major', linestyle='--')
    
    fig.tight_layout()
    fig.show()
