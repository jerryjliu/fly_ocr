from pathlib import Path
import csv
import tempfile
import numpy as np
from PIL import Image
from flyocr.common import read_json, digest_file, identity
from flyocr.readout.train import probabilities
from flyocr.vision.retina import sample_retina


def verify(directory, artifact):
    directory, artifact = Path(directory), Path(artifact)
    run = read_json(directory/"run.json"); card = read_json(artifact/"model-card.json")
    def require(condition):
        if not condition: raise ValueError("Replay/model provenance or prediction mismatch")
    require(identity({k:v for k,v in card.items() if k != "artifact_id"}) == card["artifact_id"])
    require(run["artifact_id"] == card["artifact_id"])
    require(digest_file(artifact/"readout.npz") == card["readout_sha256"])
    require(digest_file(directory/"crop.png") == run["crop_sha256"])
    if "retinal_atlas" in run:
        from flyocr.vision.atlas import validate_atlas
        validate_atlas(run["retinal_atlas"], len(run["retina_uv"]), card.get("graph_id"))
    if "input_projection" in card:
        require(run.get("input_projection") == card["input_projection"])
        require(digest_file(artifact/"retina-uv.npy") == card["input_projection"]["sha256"])
        require(np.array_equal(run["retina_uv"], np.load(artifact/"retina-uv.npy", allow_pickle=False)))
        require(run.get("readout_architecture") == card["readout_architecture"])
    if card.get("normalization") == "line-box-v1":
        from flyocr.pdf.text import prepare_text
        require(run.get("normalization") == "line-box-v1")
        require(digest_file(Path(__file__).parents[1]/"vision/letterbox.py") == card["normalization_sha256"])
        # Spaces and glyph boxes are preprocessing results, not neural classes.
        # Recompute them from source pixels so they cannot silently be repaired.
        with tempfile.TemporaryDirectory() as temp:
            segmentation = prepare_text(directory/"crop.png",temp)
            require(run.get("source_ink_box") == segmentation["source_ink_box"])
            expected = [g for r in segmentation["rows"] for g in r["glyphs"]]
            require(len(expected) == len(run["events"]))
            for glyph,event in zip(expected,run["events"]):
                require(all(glyph[k] == event[k] for k in ("id","box","prefix")))
                # PNG compression bytes can differ across zlib/Pillow builds.
                # Compare regenerated content; the saved file's original hash
                # remains independently checked below for every event.
                with Image.open(Path(temp)/glyph["image"]) as regenerated, Image.open(directory/event["image"]) as saved:
                    require(np.array_equal(np.asarray(regenerated.convert("L")), np.asarray(saved.convert("L"))))
    head = np.load(artifact/"readout.npz", allow_pickle=False)
    seen, rows = set(), {}
    for index, e in enumerate(run["events"]):
        require(e["index"] == index and e["id"] not in seen)
        seen.add(e["id"])
        require(digest_file(directory/e["image"]) == e["sha256"])
        if "input_projection" in card:
            pixels = np.asarray(Image.open(directory/e["image"]).convert("L"))
            require(np.array_equal(sample_retina(pixels, np.asarray(run["retina_uv"], np.float32)), e["retina"]))
        features = np.array(e["counts"]).reshape(1, -1)
        p = probabilities(head, features)[0]
        require(np.allclose(p, e["probabilities"], rtol=1e-10, atol=1e-12))
        character = card["characters"][int(head["classes"][p.argmax()])] if p.max() >= card["reject_threshold"] else "?"
        require(character == e["character"])
        prefix = e.get("prefix", "")
        require(prefix in ("", " "))
        rows[e["row_id"]] = rows.get(e["row_id"], "")+prefix+character
    require(all(rows.get(r["row_id"], "") == r["raw"] for r in run["rows"]))
    if "layout" in run:
        nr, nc = run["layout"]["n_rows"], run["layout"]["n_columns"]
        require(type(nr) is int and type(nc) is int and nr >= 0 and nc >= 0)
        matrix = [["" for _ in range(nc)] for _ in range(nr)]
        occupied = set()
        for cell in run["rows"]:
            r, c = cell["table_row"], cell["table_column"]
            require(0 <= r < nr and 0 <= c < nc and (r, c) not in occupied)
            require(cell["row_id"] == f"r{r+1:02d}-c{c+1:02d}")
            occupied.add((r, c)); matrix[r][c] = cell["raw"]
        table = read_json(directory/"table.json")
        require(table["artifact_id"] == run["artifact_id"] and table["crop_sha256"] == run["crop_sha256"])
        require(table["layout"] == run["layout"] and table["cells"] == run["rows"] and table["matrix"] == matrix)
        with (directory/"table.csv").open(newline="") as f:
            require(list(csv.reader(f)) == matrix)
    return {"verified_events": len(seen), "verified_rows": len(rows), "artifact_id": card["artifact_id"], "network_required": False, "graph_required": False}
