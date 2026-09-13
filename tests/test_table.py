import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flyocr.pdf.table import segment_table, write_table
from flyocr.common import read_json


def test_table_geometry_preserves_missing_cell_and_accounting_punctuation(tmp_path):
    image = Image.new("L", (750, 310), 255)
    font = ImageFont.truetype("assets/fonts/lato/Lato-Regular.ttf", 32)
    draw = ImageDraw.Draw(image)
    values = [["$1,203", "(23)", "0"], ["3.50", "", "(100)"], ["45", "7,234", "9.01"]]
    for r, row in enumerate(values):
        for col, value in enumerate(row):
            draw.text((210+240*col, 20+r*100), value, anchor="rt", font=font, fill=0)
            draw.line((15+240*col, 78+r*100, 225+240*col, 78+r*100), fill=0, width=2)
    cells, _, layout = segment_table(image)
    assert (layout["n_rows"], layout["n_columns"]) == (3, 3)
    assert {(c["table_row"], c["table_column"]) for c in cells} == {(r,c) for r in range(3) for c in range(3) if values[r][c]}
    for c in cells:
        assert len(c["glyph_boxes"]) == len(values[c["table_row"]][c["table_column"]])
        c["raw"] = values[c["table_row"]][c["table_column"]]
    run = {"artifact_id":"fixture", "crop_sha256":"fixture", "layout":layout, "rows":cells}
    write_table(run, tmp_path)
    assert read_json(tmp_path/"table.json")["matrix"] == values
    assert list(csv.reader((tmp_path/"table.csv").open())) == values
    assert segment_table(Image.new("L", (200, 200), 255))[2]["n_rows"] == 0


def test_accounting_rule_keeps_distant_currency_sign_in_cell():
    image = Image.new("L", (600, 180), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("assets/fonts/lato/Lato-Regular.ttf", 30)
    for x in [0, 300]:
        draw.text((x+10, 20), "$", font=font, fill=0)
        draw.text((x+260, 20), "1,234", anchor="rt", font=font, fill=0)
        draw.text((x+260, 110), "567", anchor="rt", font=font, fill=0)
        draw.line((x+5, 75, x+270, 75), fill=0, width=2)
    cells, _, layout = segment_table(image)
    assert (layout["n_rows"], layout["n_columns"]) == (2, 2)
    assert len(cells[0]["glyph_boxes"]) == 6
