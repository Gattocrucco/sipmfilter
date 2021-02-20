"""
Module to create automatically named figures for a script. Functions:

figwithsize :
    Create or clear a figure named <name of the script><incrementing counter>.
saveaspng :
    Activate tight layout and save a figure to a file with the same name.
"""

from matplotlib import pyplot as plt
import sys

_figcount = 0

def figwithsize(size=None, resetfigcount=False):
    """
    Create a matplotlib figure with automatic naming.
    
    The name is the running script name plus a two-digit counter.
    
    Parameters
    ----------
    size : sequence, optional
        A sequence [width, height] with the dimensions of the figure in inches.
        The dimensions are enforced even if the figure already exists.
    resetfigcount : bool
        If True, reset the counter to 0. Default False. Set to True in the
        first invocation in the script.
    
    Return
    ------
    fig : matplotlib figure
        A possibly new matplotlib figure.
    """
    global _figcount
    if resetfigcount:
        _figcount = 0
    _figcount += 1
    scriptname = '.'.join(sys.argv[0].split('.')[:-1])
    fig = plt.figure(f'{scriptname}{_figcount:02d}', figsize=size)
    fig.clf()
    if size is not None:
        fig.set_size_inches(size)
    return fig

def saveaspng(fig):
    """
    Save a matplotlib figure to file.
    
    The file name is taken from the window title. The format is PNG. The tight
    layout is activated before saving the figure.
    """
    name = fig.canvas.get_window_title() + '.png'
    print(f'saving {name}...')
    fig.tight_layout()
    fig.savefig(name)
