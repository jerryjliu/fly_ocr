"""Synthetic printed glyphs with disjoint font families and reproducible labels."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from flyocr.common import read_json, save_json, digest_file, identity
from flyocr.vision.retina import normalize_glyph

DIGITS = "0123456789"
NUMERIC = "0123456789,.-()$"
ALPHANUMERIC = NUMERIC + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def render_glyph(char, font_path, rng, size=48):
    # Jitter the source raster before canonical normalization. Every class has
    # the same distribution of nuisance variables; source identity stays split.
    font = ImageFont.truetype(str(font_path), int(rng.integers(30, 62)))
    try:
        axes = font.get_variation_axes()
        values = [a["default"] for a in axes]
        for i, a in enumerate(axes):
            if b"Weight" in a["name"]:
                values[i] = float(rng.uniform(max(300, a["minimum"]), min(700, a["maximum"])))
        font.set_variation_by_axes(values)
    except (OSError, AttributeError):
        pass
    image = Image.new("L", (100, 100), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), char, font=font)
    draw.text((50-(bbox[2]-bbox[0])/2-bbox[0], 50-(bbox[3]-bbox[1])/2-bbox[1]), char, font=font, fill=int(rng.integers(0, 45)))
    image = image.rotate(float(rng.uniform(-2.0, 2.0)), resample=Image.Resampling.BICUBIC, fillcolor=255)
    if rng.random() < .3:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.1, .45))))
    image = normalize_glyph(image, size=size, occupancy=float(rng.uniform(.65, .80)))
    # Small translations leave the shape intact while testing positional tolerance.
    shifted = Image.new("L", (size, size), 255)
    shifted.paste(image, (int(rng.integers(-1, 2)), int(rng.integers(-1, 2))))
    return shifted


def generate(output, font_manifest=Path("assets/fonts/manifest.json"), chars=DIGITS,
             per_class=None, seed=20260912, normalization="ink-box-v1"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fonts = read_json(font_manifest)
    per_class = per_class or {"train": 100, "validation": 20, "test": 50}
    seen = {}
    for f in fonts["fonts"]:
        if f["family"] in seen and seen[f["family"]] != f["split"]:
            raise ValueError("Font-family leakage")
        seen[f["family"]] = f["split"]
        if digest_file(Path(f["file"])) != f["sha256"]:
            raise ValueError("Font changed")
    if normalization == "line-box-v1":
        from flyocr.vision.letterbox import render_letter
        renderer = render_letter
    elif normalization == "ink-box-v1":
        renderer = render_glyph
    else:
        raise ValueError("Unknown glyph normalization")
    config = {"characters": chars, "per_class": per_class, "seed": seed,
              "font_manifest_sha256": digest_file(font_manifest), "normalization": normalization}
    for split_index, (split, n) in enumerate(per_class.items()):
        pool = [f for f in fonts["fonts"] if f["split"] == split]
        if not pool:
            raise ValueError(f"No fonts for {split}")
        rng = np.random.default_rng(seed+split_index)
        images, labels, entries = [], [], []
        for label, char in enumerate(chars):
            for k in range(n):
                f = pool[k % len(pool)]
                im = np.asarray(renderer(char, f["file"], rng), dtype=np.uint8)
                images.append(im)
                labels.append(label)
                entries.append({"family": f["family"], "character": char,
                                "image_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                                "source_group": f"{split}:{f['family']}:{label}:{k}"})
        order = rng.permutation(len(images))
        np.savez_compressed(output/(split+".npz"), images=np.array(images)[order], labels=np.array(labels, np.int16)[order])
        save_json(output/(split+".json"), [entries[i] for i in order])
    config["corpus_id"] = identity({**config, "splits": {s: digest_file(output/(s+".npz")) for s in per_class}})
    save_json(output/"manifest.json", config)
    return config


def contact_sheet(corpus, split, path, n=100):
    a = np.load(Path(corpus)/(split+".npz"))
    images, labels = a["images"][:n], a["labels"][:n]
    chars = read_json(Path(corpus)/"manifest.json")["characters"]
    sheet = Image.new("RGB", (640, ((len(images)+9)//10)*72), "#e9e9e9")
    draw = ImageDraw.Draw(sheet)
    for i, (im, y) in enumerate(zip(images, labels)):
        x, row = (i % 10)*64, (i//10)*72
        sheet.paste(Image.fromarray(im), (x+8, row))
        draw.text((x+26, row+50), chars[int(y)], fill="black")
    sheet.save(path)
