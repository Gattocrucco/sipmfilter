"""
Module with a function to generate the LaTeX command to include a matplotlib
figure.
"""

import os

from matplotlib import pyplot as plt

def _parsefigs(figs=None):
    if figs is None:
        figs = plt.gcf()
    if not hasattr(figs, '__len__'):
        figs = [figs]
    for i, fig in enumerate(figs):
        if not hasattr(fig, '__len__'):
            figs[i] = [fig]
    return figs

def save(figs=None, path='../thesis/figures'):
    """
    Print the LaTeX command to include a figure, and save it in a directory.
    """
    figs = _parsefigs(figs)
    print(figlatex(figs))
    
    if path != '' and not os.path.isdir(path):
        print(f'figlatex: warning: {path} is not a directory, skip saving')
        return
    
    for row in figs:
        for fig in row:
            options = dict(
                saveaspng = False,
            )
            options.update(getattr(fig, 'figlatex_options', {}))
            kw = {}
            ext = '.pdf'
            if options['saveaspng']:
                ext = '.png'
                kw.update(dpi=3 * fig.dpi)
            file = os.path.join(path, fig.canvas.get_window_title() + ext)
            print(f'save {file}...')
            fig.savefig(file, **kw)

def figlatex(figs=None, indent=' '*4):
    """
    Generate the LaTeX command to include a matplotlib figure.
    
    The figure width is chosen to have the same font size as the LaTeX
    document, assuming matplotlib and LaTeX defaults are in effect. The figure
    is centered even if it overflows the text column width.
    
    Parameters
    ----------
    figs : (list of) matplotlib figure, optional
        If not specified, the current figure is used. If an list of figures,
        the layout is a single column. If a list of lists, each sublist is a
        row.
    indent : str
        A string prepended to each line of the output, default 4 spaces.
    
    Return
    ------
    command : str
        The LaTeX code to include the figure, assuming the file name is the
        window title of the figure. Paste it between \\begin{figure} ...
        \\end{figure}.
    """
    figs = _parsefigs(figs)
    
    lines = []
    for row in figs:
        line = indent + '\\widecenter{'
        for fig in row:
            title = fig.canvas.get_window_title()
            line += '\\includempl{' + title + '}'
        line += '}'
        lines.append(line)
    return '\n\n'.join(lines)
