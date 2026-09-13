from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def normalize_glyph(image: Image.Image, size=48, occupancy=.72) -> Image.Image:
    """Tight ink crop, aspect-preserving fit and padding; no character identity."""
    gray = np.asarray(image.convert("L"))
    ys, xs = np.where(gray < 180)
    canvas = Image.new("L", (size, size), 255)
    if not len(xs):
        return canvas
    box = image.convert("L").crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))
    scale = size*occupancy/max(box.size)
    box = box.resize((max(1, round(box.width*scale)), max(1, round(box.height*scale))), Image.Resampling.LANCZOS)
    canvas.paste(box, ((size-box.width)//2, (size-box.height)//2))
    return canvas


def sample_retina(image, uv):
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2 or not np.isfinite(gray).all():
        raise ValueError("Expected a finite grayscale image")
    gray = np.clip(gray/255., 0., 1.)
    values = map_coordinates(gray, [uv[:, 1]*(gray.shape[0]-1), uv[:, 0]*(gray.shape[1]-1)], order=1, mode="nearest")
    return np.ascontiguousarray(values, dtype=np.float32)
