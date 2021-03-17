import numpy as np

def meanmedian(x, n, axis=-1):
    """
    Compute the mean of medians over interlaced subarrays.
    
    Example: meanmedian(x, 2) == mean([median(x[::2]), median(x[1::2])]).
    
    Parameters
    ----------
    x : array
        The array.
    n : int
        The number of subarrays `x` is divided into.
    axis : int
        The axis along which the operation is applied, default last.
    
    Return
    ------
    m : array
        Array with the same shape of `x` but with the specified axis removed.
    """
    axis %= x.ndim
    length = x.shape[axis]
    trunclen = length // n * n
    index = axis * (slice(None),) + (slice(None, trunclen),) + (x.ndim - axis - 1) * (slice(None),)
    shape = x.shape[:axis] + (length // n, n) + x.shape[axis + 1:]
    return np.mean(np.median(x[index].reshape(shape), axis=axis), axis=axis)

def test_meanmedian():
    x = np.random.randn(999)
    m1 = meanmedian(x, 3)
    m2 = np.mean([np.median(x[k::3]) for k in range(3)])
    assert m1 == m2

if __name__ == '__main__':
    
    test_meanmedian()
