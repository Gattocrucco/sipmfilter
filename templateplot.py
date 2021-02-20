"""
Plot a template object saved by `savetemplate.py`. Usage:

    templateplot.py something-template.npz
"""

import os

from matplotlib import pyplot as plt
import numpy as np

import toy
import textbox

def templateplot(dest, fig=None):
    """
    Plot a template saved by `savetemplate.py`.
    
    Parameters
    ----------
    dest : str
        The file path of the template file.
    fig : matplotlib figure, optional
        A clean figure where the plot is drawn. If not specified, a new one
        is created.
    
    Return
    ------
    fig : matplotlib figure
        The figure where the plot is drawn.
    """
    
    suffix = '-template.npz'
    assert dest.endswith(suffix)
    source = dest[:-len(suffix)] + '.wav'

    template = toy.Template.load(dest)
    
    if fig is None:
        fig, axs = plt.subplots(3, 1, num='templateplot', clear=True, figsize=[6.4, 7.1])
    else:
        axs = fig.subplots(3, 1)

    ax = axs[0]

    ax.set_title(os.path.split(source)[1])
    textbox.textbox(ax, 'Full template @ 1 GSa/s', fontsize='medium', loc='lower center')
    ax.plot(template.template)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

    ax = axs[1]

    textbox.textbox(ax, 'Cross corr. filter templates @ 125 MSa/s', fontsize='medium', loc='lower center')
    template_offset = [
        template.matched_filter_template(length, norm=False)
        for length in [4, 8, 16, 32, 64]
    ]
    for i, (y, offset) in enumerate(reversed(template_offset)):
        kw = dict(linewidth=i + 1, color='#060', alpha=(i + 1) / len(template_offset))
        ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

    ax.legend(title='Template length', loc='best', fontsize='small')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

    ax = axs[2]

    textbox.textbox(ax, 'Cross corr. filter templates @ 1 GSa/s', fontsize='medium', loc='lower center')
    template_offset = [
        template.matched_filter_template(length, norm=False, timebase=1)
        for length in np.array([2, 4, 8, 16, 32]) * 8
    ]
    for i, (y, offset) in enumerate(reversed(template_offset)):
        kw = dict(linewidth=i + 1, color='#060', alpha=(i + 1) / len(template_offset))
        ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

    ax.legend(title='Template length', loc='best', fontsize='small')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    
    return fig

if __name__ == '__main__':
    import sys
    
    dest = sys.argv[1]
    
    fig = templateplot(dest)

    fig.tight_layout()
    fig.show()
