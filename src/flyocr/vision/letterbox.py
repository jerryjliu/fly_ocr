"""Line-relative glyph pixels retain case, ascenders, and descenders.

Only the raster enters the circuit. No line metrics enter the learned readout.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

NORMALIZATION = "line-box-v1"


def letterbox(image, baseline, cap_height, size=48, occupancy=.58):
    if not np.isfinite([baseline, cap_height, occupancy]).all() or cap_height <= 0:
        raise ValueError("Invalid line geometry")
    gray = np.asarray(image.convert("L"))
    ys, xs = np.where(gray < 180)
    canvas = Image.new("L", (size, size), 255)
    if not len(xs):
        return canvas
    box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
    glyph = image.convert("L").crop(box)
    scale = size*occupancy/cap_height
    glyph = glyph.resize((max(1, round(glyph.width*scale)), max(1, round(glyph.height*scale))), Image.Resampling.LANCZOS)
    canvas.paste(glyph, ((size-glyph.width)//2, round(size*.79+(box[1]-baseline)*scale)))
    return canvas


def render_letter(char, font_path, rng, size=48):
    font = ImageFont.truetype(str(font_path), int(rng.integers(36, 68)))
    try:
        axes = font.get_variation_axes(); values = [a["default"] for a in axes]
        for i, a in enumerate(axes):
            if b"Weight" in a["name"]:
                values[i] = float(rng.uniform(max(300, a["minimum"]), min(700, a["maximum"])))
        font.set_variation_by_axes(values)
    except (OSError, AttributeError):
        pass
    cap = font.getbbox("H", anchor="ls")
    cap_height = cap[3]-cap[1]
    image = Image.new("L", (140, 140), 255)
    draw = ImageDraw.Draw(image)
    box = font.getbbox(char, anchor="ls")
    baseline = 85
    draw.text((70-(box[2]+box[0])/2, baseline), char, font=font, anchor="ls", fill=int(rng.integers(0, 45)))
    image = image.rotate(float(rng.uniform(-1.5, 1.5)), center=(70, baseline), resample=Image.Resampling.BICUBIC, fillcolor=255)
    if rng.random() < .3:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.1, .4))))
    # Jitter the estimated line geometry, with identical distributions per class.
    return letterbox(image, baseline+float(rng.uniform(-1, 1)), cap_height*float(rng.uniform(.94, 1.06)),
                     size=size, occupancy=float(rng.uniform(.54, .62)))


def estimate_line_geometry(boxes):
    """Estimate a shared baseline and cap height without recognized characters."""
    if not boxes:
        raise ValueError("A text line has no glyphs")
    b = np.asarray(boxes)
    heights = b[:, 3]-b[:, 1]
    tall = b[heights >= np.percentile(heights, 75)*.65]
    baseline = float(np.median(tall[:, 3]))
    cap_height = max(1., baseline-float(np.min(tall[:, 1])))
    return baseline, cap_height
