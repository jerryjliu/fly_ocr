"""Records predictions from pixels. Deliberately has no evaluation-truth argument."""
import csv
import time
from pathlib import Path
import numpy as np
from PIL import Image
from flyocr.pdf.segment import prepare
from flyocr.readout.predict import Recognizer
from flyocr.common import save_json, digest_file


def record(image, graph, artifact, output, table=False):
    output = Path(output)
    recognizer = Recognizer(graph, artifact)
    text_mode = recognizer.card.get("normalization") == "line-box-v1"
    if text_mode and table:
        raise ValueError("The letter checkpoint currently accepts text regions; use the numeric checkpoint for --table")
    if table:
        from flyocr.pdf.table import prepare_table
        segmentation = prepare_table(image, output)
    elif text_mode:
        from flyocr.pdf.text import prepare_text
        segmentation = prepare_text(image, output)
    else:
        segmentation = prepare(image, output)
    brain = recognizer.brain
    events, rows = [], []
    for row in segmentation["rows"]:
        text = ""
        for glyph in row["glyphs"]:
            begin = time.perf_counter()
            result = recognizer.recognize(np.asarray(Image.open(output/glyph["image"]).convert("L")))
            wall = time.perf_counter()-begin
            event = {**glyph, **result, "index": len(events), "row_id": row["row_id"],
                     "wall_seconds": wall, "simulated_ms": brain.config.duration_ms}
            events.append(event); text += glyph.get("prefix", "")+result["character"]
        rows.append({"row_id": row["row_id"], "box": row["box"], "raw": text,
                     **({"table_row": row["table_row"], "table_column": row["table_column"]} if table else {})})
        print(f"{row['row_id']}: {text}", flush=True)
    report = {"format": 1, "mode": "recorded inference", "artifact_id": recognizer.card["artifact_id"],
        "model_id": brain.model_id, "crop_sha256": digest_file(output/"crop.png"),
        "characters": recognizer.card["characters"], "nodes": brain.n, "edges": len(brain.graph["post"]),
        "feature_neuron_ids": [str(i) for i in brain.graph["ids"][recognizer.neurons]],
        "feature_cell_types": brain.graph["cell_type"][recognizer.neurons].tolist(),
        "retina_uv": brain.graph["uv"].tolist(), "bin_ms": brain.config.duration_ms/brain.config.bins,
        "events": events, "rows": rows, "wall_seconds": sum(e["wall_seconds"] for e in events),
        "disclosure": "Fixed connectome model + trained " + ("small neural" if recognizer.card.get("readout_type") == "mlp" else "linear") + " decoder. Approximate retinal mapping. Closed declared alphabet. Fly motion is a presentation effect. Binned activity, not exact spike timestamps."}
    if text_mode:
        report["normalization"] = "line-box-v1"
        report["source_ink_box"] = segmentation["source_ink_box"]
        report["readout_type"] = recognizer.card.get("readout_type", "linear")
        report["spaces"] = "Pixel word gaps; spaces are not output classes or language-model corrections"
    if "input_projection" in recognizer.card:
        report["input_projection"] = recognizer.card["input_projection"]
        report["readout_architecture"] = recognizer.card["readout_architecture"]
    from flyocr.vision.atlas import load_atlas
    atlas = load_atlas(graph)
    if atlas is not None:
        report["retinal_atlas"] = atlas
    if table:
        from flyocr.pdf.table import write_table
        report["layout"] = segmentation["layout"]
        write_table(report, output)
    save_json(output/"run.json", report)
    with (output/"predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "raw"]); writer.writeheader()
        writer.writerows({"row_id": r["row_id"], "raw": r["raw"]} for r in rows)
    return report
