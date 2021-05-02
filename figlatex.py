"""
Module with a function to generate the LaTeX command to include a matplotlib
figure.
"""

import os

from matplotlib import pyplot as plt

def save(fig=None, path='../thesis/figures'):
    """
    Print the LaTeX command to include a figure, and save it in a directory.
    """
    if fig is None:
        fig = plt.gcf()
    print(figlatex(fig))
    if path != '' and not os.path.isdir(path):
        print(f'figlatex: warning: {path} is not a directory, skip saving')
        return
    file = os.path.join(path, fig.canvas.get_window_title() + '.pdf')
    print(f'save {file}...')
    fig.savefig(file)

def figlatex(fig=None, indent=' '*4):
    """
    Generate the LaTeX command to include a matplotlib figure.
    
    The figure width is chosen to have the same font size as the LaTeX
    document, assuming matplotlib and LaTeX defaults are in effect. The figure
    is centered even if it overflows the text column width.
    
    Parameters
    ----------
    fig : matplotlib figure, optional
        If not specified, the current figure is used.
    indent : str
        A string prepended to each line of the output, default 4 spaces.
    
    Return
    ------
    command : str
        The LaTeX code to include the figure, assuming the file name is the
        window title of the figure. Paste it between \\begin{figure} ...
        \\end{figure}.
    """
    if fig is None:
        fig = plt.gcf()
    title = fig.canvas.get_window_title()
    text_width = 6.4
    width, _ = fig.get_size_inches()
    relwidth = width / text_width
    lines = [
        f'\\includegraphics[width={relwidth:.2f}\\textwidth]{{{title}}}'
    ]
    if relwidth < 1:
        lines.insert(0, '\\centering')
    else:
        lines.insert(0, f'\\hspace{{{(1 - relwidth) / 2:.2f}\\textwidth}}')
        lines.insert(0, '\\mbox{')
        lines.append('}')
    return indent + f'\n{indent}'.join(lines)
