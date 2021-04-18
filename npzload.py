import numpy as np

import downcast as _downcast

class NPZLoad:
    """
    Superclass for adding automatic serialization to/from npz files.
    
    Only scalar/array instance variables are serialized.
    
    Instance methods
    ----------------
    save : save the object to a file.
    
    Class methods
    -------------
    load : read an instance form a file.
    
    Attributes
    ----------
    _npzload_unpack_scalars : bool
        If True, 0d arrays are converted to scalars when loading. Default False.
    """
    
    def save(self, filename, compress=False, downcast=None):
        """
        Save the object to file as a `.npz` archive.
        
        Parameters
        ----------
        filename : str
            The file path to save to.
        compress : bool
            If True, compress the npz archive (slow). Default False.
        downcast : numpy data type or tuple of numpy data types, optional
            A list of "short" data types. Arrays (but not scalars) with a
            data type compatible with one in the list, but larger, are
            casted to the short type. Applies also to structured arrays.
        """
        classvars = vars(type(self))
        variables = {
            n: x
            for n, x in vars(self).items()
            if n not in classvars
            and not n.startswith('__')
            and (np.isscalar(x) or isinstance(x, np.ndarray))
        }
        
        if downcast is not None:
            if not isinstance(downcast, tuple):
                downcast = (downcast,)
            for n, x in variables.items():
                if hasattr(x, 'dtype'):
                    variables[n] = np.asarray(x, _downcast.downcast(x.dtype, *downcast))
        
        fun = np.savez_compressed if compress else np.savez
        fun(filename, **variables)
        
    _npzload_unpack_scalars = False
    
    @classmethod
    def load(cls, filename):
        """
        Return an instance loading the object from a file which was written by
        `save`.
        """
        self = cls.__new__(cls)
        with np.load(filename) as arch:
            for n, x in arch.items():
                if x.shape == () and self._npzload_unpack_scalars:
                    x = x.item()
                setattr(self, n, x)
        return self
