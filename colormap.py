from matplotlib import colors as _colors
import colorspacious
import numpy as np
from scipy import interpolate
    
def uniform(colors=['black', '#f55', 'white'], N=256, lrange=(0, 100)):
    """
    Make a colormap with monotonically increasing luminosity.
    
    Parameters
    ----------
    colors : sequence of matplotlib colors
        The sequence of evenly spaced colors. The luminosity will be modified
        to increase linearly from the first color to the last.
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
    
    # TODO This is not uniformly spaced because say the distance from black to
    # f55 is different from the difference from f55 to white. I should solve
    # a system of equations for the partition keeping into account that the
    # luminosity is bound to the partition and appears in the distance. With
    # black and white as extrema anyway this correction is small because they
    # both have a, b ≈ 0.
    
    rgb0 = np.array([_colors.to_rgb(color) for color in colors])
    lab0 = colorspacious.cspace_convert(rgb0, 'sRGB1', 'CAM02-UCS')
    
    lab = np.zeros((N, 3))
    lab[:, 0] = np.linspace(*lrange, N)
    
    x = np.linspace(0, 1, len(lab0))
    newx = np.linspace(0, 1, N)
    kw = dict(axis=0, assume_sorted=True, copy=False)
    lab[:, 1:] = interpolate.interp1d(x, lab0[:, 1:], **kw)(newx)
    
    rgb = colorspacious.cspace_convert(lab, 'CAM02-UCS', 'sRGB1')
    rgb = np.clip(rgb, 0, 1)
    return _colors.ListedColormap(rgb)

def plotcmap(ax, cmap, N=512, **kw):
    img = np.linspace(0, 1, N)[None]
    d = 1 / N
    return ax.imshow(img, cmap=cmap, aspect='auto', extent=(0, 1, 1, 0), **kw)
