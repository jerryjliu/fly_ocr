from PIL import Image, ImageDraw, ImageFont
import numpy as np
from flyocr.pdf.segment import segment
from flyocr.pdf.render import render
from flyocr.eval.metrics import edit_distance


def test_pixel_segmentation_retains_numeric_punctuation():
    image = Image.new("L", (450, 280), 255)
    font = ImageFont.truetype("assets/fonts/lato/Lato-Regular.ttf", 40)
    draw = ImageDraw.Draw(image)
    lines = ["$12,345", "(67.89)", "-0.10"]
    for i, line in enumerate(lines):
        draw.text((20, 20+i*80), line, font=font, fill=0)
        draw.line((0, 82+i*80, 449, 82+i*80), fill=0, width=2)
    rows, cleaned = segment(image)
    assert len(rows) == 3
    assert [len(r["glyph_boxes"]) for r in rows] == list(map(len, lines))
    assert segment(Image.new("L", (100, 100), 255))[0] == []


def test_edit_distance_counts_missing_punctuation():
    assert edit_distance("(4,901)", "4,901") == 2
    assert edit_distance("13.70", "13,70") == 1
