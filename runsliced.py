import tqdm

def runsliced(fun, ntot, n=None):
    """
    Run a cycle which calls a given function with a progressing slice as sole
    argument until a range is covered, printing a progressbar.
    
    Parameters
    ----------
    fun : function
        A function with a single parameter which is a slice object.
    ntot : int
        The end of the range covered by the sequence of slices.
    n : int, optional
        The length of each slice (the last slice may be shorter). If None, the
        function is called once with the slice 0:ntot.
    """
    if n is None:
        fun(slice(0, ntot))
    else:
        for i in tqdm.tqdm(range(ntot // n + bool(ntot % n))):
            start = i * n
            end = min((i + 1) * n, ntot)
            s = slice(start, end)
            fun(s)
