import numpy as np

# TODO put the downcasting condition in its own function, write an upcast
# function, share the code using an internal function with a boolean parameter.

def downcast(dtype, *shorttypes):
    """
    Downcast a numpy data type, in the sense of converting it to a similar type
    but of smaller size. Works recursively for structured/array data types.
    
    Parameters
    ----------
    dtype : numpy data type
        The data type to downcast.
    *shorttypes : numpy data types
        The types that the dtype can be downcasted to.
    
    Return
    ------
    dtype : numpy data type
        The downcasted data type. Fields and shapes are preserved, but not the
        memory layout.
    
    Examples
    --------
    >>> downcast('f8', 'f4')    # shorter floating type
    dtype('float32')
    >>> downcast('f8', 'i4')    # no downcasting from floating to integer
    dtype('float64')
    >>> downcast('f4', 'f8')    # no upcasting
    dtype('float32')
    >>> downcast('S4', 'S2')    # strings are truncated
    dtype('S2')
    >>> downcast('f8,i8', 'f4', 'i4')           # structured data type
    dtype([('f0', '<f4'), ('f1', '<i4')])
    >>> x = np.zeros(5, [('a', float), ('b', float)])
    >>> x
    array([(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
          dtype=[('a', '<f8'), ('b', '<f8')])
    >>> x.astype(downcast(x.dtype, 'f4'))       # downcast an array
    array([(0., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
          dtype=[('a', '<f4'), ('b', '<f4')])
    """
    dtype = np.dtype(dtype)
    for shorttype in shorttypes:
        # TODO move the cycle inside a last <else> in _downcast
        shorttype = np.dtype(shorttype)
        dtype = _downcast(dtype, shorttype)
    return dtype

def _downcast(dtype, shorttype):
    if dtype.names is not None:
        return np.dtype([
            (name, _downcast(field[0], shorttype))
            for name, field in dtype.fields.items()
        ])
    elif dtype.subdtype is not None:
        # TODO maybe I first have to check for subdtype, then names, check
        # if the current implementation fails on names with shape
        return np.dtype((_downcast(dtype.base, shorttype), dtype.shape))
    elif np.can_cast(shorttype, dtype, 'safe') and np.can_cast(dtype, shorttype, 'same_kind') and dtype.itemsize > shorttype.itemsize:
        return shorttype
    else:
        return dtype
