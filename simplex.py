from matplotlib import pyplot as plt
import numpy as np

def spherical_to_rect(cosphi, theta):
    """
    cosphi = n-2 values in (-1, 1)
    theta = angle in (0, 2π)
    return n coordinates in (-1, 1)
    """
    sinphi = np.sqrt(np.clip(1 - cosphi * cosphi, 0, 1))
    cumsin = np.cumprod(sinphi, axis=-1)
    padspec = [(0, 0)] * (cumsin.ndim - 1) + [(1, 0)]
    cumsin = np.pad(cumsin, padspec, constant_values=1)
    return np.concatenate([
        cumsin[..., :-1] * cosphi,
        cumsin[..., -1:] * np.cos(theta)[..., None],
        cumsin[..., -1:] * np.sin(theta)[..., None],
    ], axis=-1)

def sample_sphere(ndim, size):
    # this is wrong above 2 dimensions
    if not isinstance(size, tuple):
        size = (size,)
    cosphi = np.random.uniform(-1, 1, size + (ndim - 2,))
    theta = np.random.uniform(0, 2 * np.pi, size)
    return spherical_to_rect(cosphi, theta)

def plot_sphere(n=2, squared=False):
    assert n >= 2, n
    coord = sample_sphere(n, 1000000)
    if squared:
        coord = coord ** 2
    fig, ax = plt.subplots(num='simplex.plot_circ', clear=True)
    _, _, _, im = ax.hist2d(*coord.T[:2], 30, cmap='magma', cmin=1)
    fig.colorbar(im)
    fig.tight_layout()
    fig.show()
    return fig
