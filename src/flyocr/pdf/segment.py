"""Geometry-only segmentation for an upright, single numeric column."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from flyocr.vision.retina import normalize_glyph
from flyocr.common import save_json, digest_file


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def segment(image):
    gray = np.asarray(image.convert("L"))
    ink = gray < 160
    # Long horizontal rules carry no character label information.
    rules = ink.sum(1) >= .8*ink.shape[1]
    ink[rules] = False
    gray = gray.copy(); gray[rules] = 255
    bands = runs(ink.any(1))
    if not bands: return [], Image.fromarray(gray)
    # Join separated dot/comma pixels with their text baseline at raster scale.
    typical_height = float(np.percentile([b-a for a, b in bands], 75))
    merged = []
    for a, b in bands:
        if merged and a-merged[-1][1] <= max(1, round(typical_height*.18)):
            merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    rows = []
    for y0, y1 in merged:
        columns = runs(ink[y0:y1].any(0))
        boxes = []
        for x0, x1 in columns:
            ys = np.flatnonzero(ink[y0:y1, x0:x1].any(1))
            if not len(ys): continue
            box = [int(x0), int(y0+ys[0]), int(x1), int(y0+ys[-1]+1)]
            if int(ink[box[1]:box[3], box[0]:box[2]].sum()) < 2: continue
            boxes.append(box)
        if boxes:
            rows.append({"row_id": f"r{len(rows)+1:02d}", "box": [min(b[0] for b in boxes), int(y0), max(b[2] for b in boxes), int(y1)], "glyph_boxes": boxes})
    return rows, Image.fromarray(gray)


def prepare(image_path, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    original = Image.open(image_path).convert("L")
    rows, cleaned = segment(original)
    overlay = original.convert("RGB"); draw = ImageDraw.Draw(overlay)
    glyph_dir = output/"glyphs"; glyph_dir.mkdir(exist_ok=True)
    for row in rows:
        row["glyphs"] = []
        for i, box in enumerate(row["glyph_boxes"]):
            gid = f"{row['row_id']}-g{i+1:02d}"
            glyph = normalize_glyph(cleaned.crop(box))
            path = glyph_dir/(gid+".png"); glyph.save(path)
            row["glyphs"].append({"id": gid, "box": box, "image": f"glyphs/{gid}.png", "sha256": digest_file(path)})
            draw.rectangle([box[0]-1, box[1]-1, box[2], box[3]], outline="#e14732", width=1)
    original.save(output/"crop.png"); overlay.save(output/"segmentation.png")
    report = {"format": 1, "crop_sha256": digest_file(output/"crop.png"), "method": "pixel projections, horizontal rule removal, no OCR or embedded text",
              "scope": "upright isolated numeric column", "rows": rows}
    save_json(output/"segmentation.json", report)
    return report
