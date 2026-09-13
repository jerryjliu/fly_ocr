"""Label-free input calibration; internal connectome edges are never modified."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def balanced_retina(uv):
    """Assign unique sample sites to a full rectangular grid at minimum squared displacement.

    Receptors sharing a source site still share an image location. This is an
    engineering input adapter, not a reconstruction of biological visual fields.
    """
    uv = np.asarray(uv, np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2 or not len(uv) or not np.isfinite(uv).all():
        raise ValueError("Expected finite retinal coordinates")
    sites, inverse = np.unique(uv, axis=0, return_inverse=True)
    n = len(sites)
    height = max(d for d in range(1, int(np.sqrt(n))+1) if n % d == 0)
    width = n // height
    # The retained graph has 825 unique sites, giving a 33 by 25 grid.
    xs = np.linspace(0, 1, width) if width > 1 else np.array([.5])
    ys = np.linspace(0, 1, height) if height > 1 else np.array([.5])
    target = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    cost = ((sites[:, None, :]-target[None, :, :])**2).sum(2)
    rows, columns = linear_sum_assignment(cost)
    mapped = np.empty_like(sites)
    mapped[rows] = target[columns]
    return mapped[inverse].astype(np.float32), {
        "method": "minimum-displacement rectangular sampling grid v1",
        "learned": False, "uses_labels": False, "unique_sites": n,
        "grid_width": width, "grid_height": height,
        "rms_displacement": float(np.sqrt(cost[rows, columns].mean())),
        "scope": "Engineered pixel-to-retina calibration; not biological visual-field reconstruction",
    }


def sampling_support(uv, size=48):
    """Pixels that can influence at least one bilinear sample."""
    support = np.zeros((size, size), bool)
    for x, y in np.asarray(uv)*(size-1):
        for yy in {int(np.floor(y)), int(np.ceil(y))}:
            for xx in {int(np.floor(x)), int(np.ceil(x))}:
                support[yy, xx] = True
    return support
