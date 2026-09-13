"""Render, record, and separately score the declared additional PDF examples."""
import argparse
from pathlib import Path
import shutil
from PIL import Image
from flyocr.common import read_json, save_json, digest_file
from flyocr.pdf.render import render
from flyocr.demo.record import record
from flyocr.demo.schema import verify
from flyocr.eval.pdf import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--example", help="One example id; default all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    specs = read_json(root/"examples/microsoft-2025/more-examples.json")
    pdf = root/"data/source/microsoft-2025.pdf"
    if digest_file(pdf) != specs["pdf_sha256"]:
        raise ValueError("PDF source checksum differs")
    for item in specs["examples"]:
        name = item["id"]
        if args.example and args.example != name:
            continue
        raster = root/"data/more-examples"/name
        crop = render(pdf, raster, item["page"], 300, item["region"])
        if item.get("rotation"):
            im = Image.open(crop)
            im.rotate(item["rotation"], resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255).save(crop)
        # Source-page context is for the presentation, not a model input.
        page = Image.open(raster/"page.png"); page.thumbnail((650, 850))
        page.save(raster/"source-page.png")
        if args.prepare_only:
            print("Prepared", name, flush=True)
            continue
        output = root/"demo/examples"/name
        result = record(crop, root/"data/graph", root/"artifacts/numeric", output, table=item["table"])
        # Save compact event JSON: the same measurements, easier to distribute.
        import json
        (output/"run.json").write_text(json.dumps(result, separators=(",", ":")))
        shutil.copy(raster/"source-page.png", output/"source-page.png")
        save_json(output/"source.json", {**item, "pdf_sha256": specs["pdf_sha256"],
            "source_url": specs["source_url"], "prepared_crop_sha256": digest_file(crop),
            "selection": specs["selection"], "crop_operation": "Poppler 300 dpi, normalized region, optional declared rotation"})
        save_json(output/"verification.json", verify(output, root/"artifacts/numeric"))
        # Evaluation labels enter only after the entire run has been saved.
        truth = root/"examples/microsoft-2025"/(name+"-truth.json")
        score = evaluate(output/"run.json", truth, root/"reports/more-examples"/(name+".json"))
        print(name, score["correct_cells"], "/", score["n_cells"], "exact cells;", score.get("predicted_shape"), flush=True)


if __name__ == "__main__":
    main()
