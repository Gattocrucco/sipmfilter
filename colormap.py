from matplotlib import colors
import colorspacious
import numpy as np

def uniform(rgb0=[1, 0, 0], N=256, lrange=(0, 100)):
    """
    Make a perceptually uniform colormap with a single color.
    
    Parameters
    ----------
    rgb0 : sequence
        The three RGB components in range [0, 1] of the color. The chroma and
        hue are taken from this color. Default red.
    N : int
        The number of steps of the colormap. Default 256.
    lrange : sequence
        Two values for the start and end luminosity in range [0, 100].
        Default (0, 100).
    
    Return
    ------
    cmap : matplotlib.colors.ListedColormap
        A new colormap.
    """
    jch, lab = np.zeros((2, N, 3))
    
    # luminosity
    lab[:, 0] = np.linspace(*lrange, N)
    jch[:, 0] = colorspacious.cspace_convert(lab, 'CAM02-UCS', 'JCh')[:, 0]
    
    # chroma
    c0 = colorspacious.cspace_convert([0, 0, 0], 'sRGB1', 'JCh')[1]
    c1 = colorspacious.cspace_convert(rgb0, 'sRGB1', 'JCh')[1]
    c2 = colorspacious.cspace_convert([1, 1, 1], 'sRGB1', 'JCh')[1]
    jch[:, 1] = np.concatenate([
        np.linspace(c0, c1, N // 2),
        np.linspace(c1, c2, N - N // 2 + 1)[1:]
    ])
    
    # hue
    jch[:, 2] = colorspacious.cspace_convert(rgb0, 'sRGB1', 'JCh')[2]

    rgb = colorspacious.cspace_convert(jch, 'JCh', 'sRGB1')
    rgb = np.clip(rgb, 0, 1)
    return colors.ListedColormap(rgb)
