from matplotlib import pyplot as plt

def figlatex(fig=None, indent=' '*4):
    if fig is None:
        fig = plt.gcf()
    title = fig.canvas.get_window_title()
    text_width = 6.4
    width, _ = fig.get_size_inches()
    relwidth = width / text_width
    if relwidth < 1:
        align = '\\centering'
    else:
        align = f'\\hspace{{{(1 - relwidth) / 2:.2f}\\textwidth}}'
    return f'{indent}{align}\n{indent}\\includegraphics[width={relwidth:.2f}\\textwidth]{{{title}}}'
