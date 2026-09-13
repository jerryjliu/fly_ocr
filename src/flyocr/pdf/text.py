"""Upright printed text: pixel rows, cautious bridge splitting, and word gaps."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from flyocr.pdf.segment import segment
from flyocr.vision.letterbox import letterbox, estimate_line_geometry
from flyocr.common import save_json, digest_file


def split_bridges(gray, boxes):
    """Split a narrow upper bridge near the middle of a merged glyph box.

    This handles some tightly kerned pairs. It is not a general ligature or
    cursive segmenter. No predicted class or dictionary enters the decision.
    """
    output = []
    for b in boxes:
        ink = np.asarray(gray.crop(b)) < 160
        h, w = ink.shape
        projection = ink.sum(0)
        candidates = []
        if w >= .75*h and h >= 10:
            for x in range(max(2,round(w*.35)), min(w-2,round(w*.65))+1):
                ys = np.flatnonzero(ink[:,x])
                if 0 < len(ys) <= 2 and float(ys.mean()) < .45*h:
                    left_y = np.flatnonzero(ink[:,:x].any(1))
                    right_y = np.flatnonzero(ink[:,x:].any(1))
                    # A normal n/m arch spans parts with the same top height.
                    # A tightly kerned r+t has two distinct top heights.
                    if len(left_y) and len(right_y) and abs(int(left_y[0])-int(right_y[0])) >= .15*h:
                        candidates.append(x)
        if candidates:
            cut = min(candidates,key=lambda x:(projection[x],abs(x-w/2)))
            pieces = [(0,cut),(cut,w)]
        else:
            pieces = [(0,w)]
        for x0,x1 in pieces:
            ys,xs = np.where(ink[:,x0:x1])
            if len(xs):output.append([b[0]+x0+int(xs.min()),b[1]+int(ys.min()),b[0]+x0+int(xs.max())+1,b[1]+int(ys.max())+1])
    return output


def prepare_text(image_path, output):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/"glyphs").mkdir(exist_ok=True)
    original = Image.open(image_path).convert("L")
    rows,cleaned = segment(original)
    overlay = original.convert("RGB"); draw = ImageDraw.Draw(overlay)
    for row in rows:
        row["glyph_boxes"] = split_bridges(cleaned,row["glyph_boxes"])
        baseline,cap_height = estimate_line_geometry(row["glyph_boxes"])
        gaps = [b[0]-a[2] for a,b in zip(row["glyph_boxes"],row["glyph_boxes"][1:]) if b[0]>a[2]]
        word_gap = max(.28*cap_height,1.8*float(np.median(gaps))) if gaps else .28*cap_height
        row["baseline"] = baseline; row["cap_height"] = cap_height; row["glyphs"] = []
        previous = None
        for i,box in enumerate(row["glyph_boxes"]):
            gid = f"{row['row_id']}-g{i+1:02d}"; path = output/"glyphs"/(gid+".png")
            letterbox(cleaned.crop(box),baseline-box[1],cap_height).save(path)
            prefix = " " if previous is not None and box[0]-previous[2] > word_gap else ""
            row["glyphs"].append({"id":gid,"box":box,"image":f"glyphs/{gid}.png","sha256":digest_file(path),"prefix":prefix})
            previous=box
            draw.rectangle(box,outline="#e14732",width=1)
    original.save(output/"crop.png"); overlay.save(output/"segmentation.png")
    ys,xs=np.where(np.asarray(original)<180)
    ink_box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else [0,0,original.width,original.height]
    result = {"format":1,"normalization":"line-box-v1","crop_sha256":digest_file(output/"crop.png"),
        "source_ink_box":ink_box,
        "method":"pixel projections, cautious narrow bridge splitting, estimated baseline, geometric word gaps",
        "scope":"upright printed text region; no OCR, dictionary, or embedded PDF text","rows":rows}
    save_json(output/"segmentation.json",result)
    return result
