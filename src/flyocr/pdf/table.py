"""Pixel-only structure for an upright, whitespace-separated numeric table.

No OCR, expected values, column count, or headers enter this stage. Wide blank
vertical strips define columns; overlapping text baselines define rows. This is
deliberately a limited layout heuristic, not a general document table detector.
"""
from pathlib import Path
import csv
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw
from flyocr.common import save_json, digest_file
from flyocr.pdf.segment import runs, segment
from flyocr.vision.retina import normalize_glyph


def segment_table(image):
    gray = np.array(image.convert("L"))
    ink = gray < 160
    # Remove rules that span a substantial fraction of the selected table.
    rule_width = max(20, round(gray.shape[1] * .15))
    rules = ndimage.binary_opening(ink, structure=np.ones((1, rule_width)))
    ink[rules] = False
    gray[rules] = 255
    components, _ = ndimage.label(ink)
    heights = [s[0].stop-s[0].start for s in ndimage.find_objects(components)
               if s is not None and s[0].stop-s[0].start >= 5]
    if not heights:
        return [], Image.fromarray(gray), {"n_rows": 0, "n_columns": 0, "column_bounds": []}
    height = float(np.median(heights))
    xs = np.flatnonzero(ink.any(0))
    left, right = int(xs[0]), int(xs[-1]+1)
    # A column's underline can span an accounting symbol and its distant value.
    # Keep that pair together. Ignore full-table rules, which delimit no columns.
    same_cell = np.zeros(gray.shape[1], dtype=bool)
    for row in rules:
        for a, b in runs(row):
            if b-a < .9*gray.shape[1]:
                same_cell[a:b] = True
    gaps = []
    for a, b in runs(~ink.any(0)):
        if a <= left or b >= right or b-a < max(5, .8*height):
            continue
        candidates = runs(~same_cell[a:b])
        if candidates:
            lo, hi = max(candidates, key=lambda pair: pair[1]-pair[0])
            if hi-lo >= max(5, .4*height):
                gaps.append((a+lo, a+hi))
    boundaries = [0] + [int((a+b)//2) for a, b in gaps] + [gray.shape[1]]
    candidates = []
    for col, (x0, x1) in enumerate(zip(boundaries, boundaries[1:])):
        local_rows, _ = segment(Image.fromarray(gray[:, x0:x1]))
        for r in local_rows:
            boxes = [[b[0]+x0, b[1], b[2]+x0, b[3]] for b in r["glyph_boxes"]]
            candidates.append({"column": col, "center": (r["box"][1]+r["box"][3])/2,
                               "glyph_boxes": boxes})
    # Align independently segmented columns without assuming equal row spacing.
    bands = []
    for candidate in sorted(candidates, key=lambda r: r["center"]):
        if (bands and abs(candidate["center"]-np.median([c["center"] for c in bands[-1]])) <= .5*height
                and candidate["column"] not in {c["column"] for c in bands[-1]}):
            bands[-1].append(candidate)
        else:
            bands.append([candidate])
    rows = []
    for ri, band in enumerate(bands):
        for item in sorted(band, key=lambda r: r["column"]):
            boxes = item["glyph_boxes"]
            rows.append({"row_id": f"r{ri+1:02d}-c{item['column']+1:02d}",
                         "table_row": ri, "table_column": item["column"],
                         "box": [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                 max(b[2] for b in boxes), max(b[3] for b in boxes)],
                         "glyph_boxes": boxes})
    return rows, Image.fromarray(gray), {"n_rows": len(bands), "n_columns": len(boundaries)-1,
        "column_bounds": [[int(a), int(b)] for a, b in zip(boundaries, boundaries[1:])],
        "method": "horizontal rule removal, whitespace columns constrained by cell rules, baseline alignment",
        "scope": "manually selected numeric region; no semantic labels or merged-cell inference"}


def prepare_table(image_path, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    original = Image.open(image_path).convert("L")
    rows, cleaned, layout = segment_table(original)
    overlay = original.convert("RGB"); draw = ImageDraw.Draw(overlay)
    (output/"glyphs").mkdir(exist_ok=True)
    for row in rows:
        row["glyphs"] = []
        for i, box in enumerate(row["glyph_boxes"]):
            gid = f"{row['row_id']}-g{i+1:02d}"
            path = output/"glyphs"/(gid+".png")
            normalize_glyph(cleaned.crop(box)).save(path)
            row["glyphs"].append({"id": gid, "box": box, "image": f"glyphs/{gid}.png", "sha256": digest_file(path)})
            draw.rectangle(box, outline="#e14732", width=1)
    for x0, _ in layout["column_bounds"][1:]:
        draw.line((x0, 0, x0, original.height), fill="#168aba", width=2)
    original.save(output/"crop.png"); overlay.save(output/"segmentation.png")
    report = {"format": 1, "crop_sha256": digest_file(output/"crop.png"),
              "method": layout["method"] if rows else "blank", "layout": layout, "rows": rows}
    save_json(output/"segmentation.json", report)
    return report


def write_table(run, output):
    """Assemble raw recognizer output, preserving empty positions and strings."""
    layout = run["layout"]
    matrix = [["" for _ in range(layout["n_columns"])] for _ in range(layout["n_rows"])]
    for cell in run["rows"]:
        matrix[cell["table_row"]][cell["table_column"]] = cell["raw"]
    save_json(Path(output)/"table.json", {"artifact_id": run["artifact_id"],
        "crop_sha256": run["crop_sha256"], "layout": layout, "cells": run["rows"], "matrix": matrix,
        "disclosure": "Raw strings, no numeric repair, semantic headers, or accounting interpretation."})
    with (Path(output)/"table.csv").open("w", newline="") as f:
        csv.writer(f).writerows(matrix)
