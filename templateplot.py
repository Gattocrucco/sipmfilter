"""
Plot a template object saved by `savetemplate.py`. Usage:

    templateplot.py something-template.npz

Can be imported as a module, the function is `templateplot`.
"""

import os

from matplotlib import pyplot as plt
import numpy as np

import template
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
    path, name = os.path.split(dest)
    source = 'darksidehd/' + name[:-len(suffix)] + '.wav'

    templ = template.Template.load(dest)
    
    if fig is None:
        fig, axs = plt.subplots(2, 2, num='templateplot', clear=True, figsize=[10, 7.1])
    else:
        axs = fig.subplots(2, 2)

    for ax in axs.flat[:3]:
        textbox.textbox(ax, 'Full template @ 1 GSa/s', fontsize='medium', loc='lower center')
        ax.plot(templ.templates[0], color='#f55', label='Aligned to event window')
        ax.plot(templ.templates[1], color='#000', linestyle=':', label='Aligned with trigger/filter')
        ax.legend(loc='upper right')
    
    ax = axs.flat[1]
    ax.set_xlim(0, 200)
    
    ax = axs.flat[2]
    x = 200
    margin = 10
    y = templ.templates[0, x - margin:x + margin]
    ax.set_xlim(x - margin, x + margin)
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)

    ax = axs.flat[3]

    textbox.textbox(ax, 'Cross corr. filter templates @ 125 MSa/s', fontsize='medium', loc='lower center')
    template_offset = [
        templ.matched_filter_template(length, norm=False, aligned=True)
        for length in [4, 8, 16, 32, 64]
    ]
    for i, (y, offset) in enumerate(reversed(template_offset)):
        kw = dict(linewidth=i + 1, color='#060', alpha=(i + 1) / len(template_offset))
        ax.plot(np.arange(len(y)) + offset, y, label=str(len(y)), **kw)

    ax.legend(title='Template length', loc='best', fontsize='small')

    for ax in axs.flat:
        if ax.is_first_row():
            ax.set_title(os.path.split(source)[1])
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
