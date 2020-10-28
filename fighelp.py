from matplotlib import pyplot as plt
import sys

_figcount = 0


def figwithsize(size=None, resetfigcount=False):
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
    name = fig.canvas.get_window_title() + '.png'
    print(f'saving {name}...')
    fig.tight_layout()
    fig.savefig(name)
