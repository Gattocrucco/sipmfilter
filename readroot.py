import uproot
import numpy as np

def readroot(filename, channel, maxevents=None, quiet=False):
    """
    Read a proto0 root file.
    
    Parameters
    ----------
    filename : str
        The file path.
    channel : str, int
        The channel to read (example: 'adc_W200_Ch00'). If an integer, the
        Proto0 run2 tile number.
    maxevents : int, optional
        The number of nonempty entries to read from the file.
    quiet : bool
        If True, do not print 'reading <file>...'.
    
    Return
    ------
    branch : array (nevents, length)
        The array of nonempty events in the tree branch.
    """
    if isinstance(channel, int):
        channel = tilerun2ch(channel)
    
    if not quiet:
        print(f'reading {filename}, channel {channel}...')
    
    root = uproot.open(filename)
    tree = root['midas_data']
    
    nsamples = tree.array('nsamples')
    un, indices = np.unique(nsamples, return_inverse=True)
    assert len(un) == 2 and un[0] == 0
    
    entrystop = len(nsamples)
    if maxevents == 0:
        return np.empty((0, un[1]))
    elif maxevents is not None:
        idx = np.flatnonzero(indices)
        if maxevents < len(idx):
            entrystop = idx[maxevents - 1] + 1
    
    branch = tree.array(channel, entrystop=entrystop)
    array = branch._content.reshape(-1, un[-1])
    
    assert np.array_equal(branch.counts, nsamples[:entrystop])
    assert maxevents is None or len(array) <= maxevents
    
    root.close()
    return array

def tilerun2ch(tile):
    """
    Map Proto0 run2 tile number to ADC channel.
    """
    # from PDMadcCh.png
    channels = {
        31: 'adc_W200_Ch00',
        32: 'adc_W200_Ch02',
        39: 'adc_W200_Ch04',
        64: 'adc_W200_Ch06',
        55: 'adc_W200_Ch08',
        
        30: 'adc_W200_Ch10',
        59: 'adc_W200_Ch12',
        57: 'adc_W201_Ch00',
        37: 'adc_W201_Ch02',
        29: 'adc_W201_Ch04',
        
        38: 'adc_W201_Ch06',
        36: 'adc_W201_Ch08',
        58: 'adc_W201_Ch10',
        62: 'adc_W202_Ch00',
        60: 'adc_W202_Ch02',
        
        41: 'adc_W202_Ch04',
        61: 'adc_W202_Ch06',
        66: 'adc_W202_Ch08',
        63: 'adc_W202_Ch10',
        52: 'adc_W203_Ch00',
        
        34: 'adc_W203_Ch02',
        53: 'adc_W203_Ch04',
        54: 'adc_W203_Ch06',
        65: 'adc_W203_Ch08',
        42: 'adc_W203_Ch10',
    }
    return channels[tile]
