import re
import os

import uproot
import numpy as np
import pandas

def readroot(filename, channel, maxevents=None, quiet=False, firstevent=None):
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
    firstevent : int, optional
        The first non-empty event to read.
    
    Return
    ------
    branch : array (nevents, length)
        The array of nonempty events in the tree branch.
    trigger : int
        The trigger position in samples from the start of the event.
    freq : scalar
        The sampling frequency in samples per second.
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
    idx = np.flatnonzero(indices)
    
    entrystart = 0
    if firstevent is not None and len(idx) > 0:
        entrystart = idx[0]
    
    entrystop = len(nsamples)
    if maxevents == 0:
        entrystop = 0
    elif maxevents is not None and maxevents < len(idx):
        entrystop = idx[maxevents - 1] + 1
    
    branch = tree.array(channel, entrystop=entrystop)
    array = branch._content.reshape(-1, un[1])
    
    assert np.array_equal(branch.counts, nsamples[:entrystop])
    assert maxevents is None or len(array) <= maxevents
    
    root.close()

    table = info(filename)
    ttrig = table['trigtime (µs)'].values[0]
    ttotal = table['Time gate  (µs)'].values[0]
    trigger = int(ttrig * array.shape[1] / ttotal)
    freq = array.shape[1] / (ttotal * 1e-6)
    
    return array, trigger, freq

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

def tiles():
    """
    List of Proto0 run2 tiles.
    """
    return list(channels.keys())

def tilerun2ch(tile):
    """
    Map Proto0 run2 tile number to ADC channel.
    """
    return channels[tile]

def info(filename):
    """
    Return the info table row for a Proto0 root file.
    
    Parameters
    ----------
    filename : str
        The file path.
    
    Return
    ------
    row : pandas.DataFrame
        A dataframe with a single row.
    """

    _, name = os.path.split(filename)
    regexp = r'.*?(\d+).*?\.root(:.*|\Z)'
    x = re.fullmatch(regexp, name)
    num = int(x.groups()[0])
    
    table = pandas.read_csv('DS_proto_runs_nov_2019.csv')
    idx = np.flatnonzero(table['run'].values == num)
    assert len(idx) > 0, f'index {num} of file {name} missing in csv'
    assert len(idx) == 1
    
    return table[table['run'] == num]
