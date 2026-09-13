"""Rebuild a display-only atlas and attach it to existing, unaltered inference."""
from pathlib import Path
import numpy as np
import pandas as pd
from flyocr.common import read_json, save_json, identity
from flyocr.vision.atlas import make_atlas, load_atlas
from flyocr.vision.retina import sample_retina
from PIL import Image

graph = Path("data/graph")
ids = np.load(graph/"ids.npy", allow_pickle=False)
retina = np.load(graph/"retina.npy", allow_pickle=False)
nodes = pd.read_feather("data/source/annotations.feather").set_index("bodyId")
atlas = make_atlas(ids, retina, np.load(graph/"uv.npy", allow_pickle=False),
                   nodes.loc[ids[retina], "rootSide"].to_numpy(),
                   read_json(graph/"manifest.json")["graph_id"])
save_json(graph/"retinal-atlas.json", atlas)
save_json("artifacts/retinal-atlas.json", atlas)
assert load_atlas(graph) == atlas
results = []
for path in [Path("demo/run/run.json"), *sorted(Path("demo/examples").glob("*/run.json"))]:
    run = read_json(path)
    before = identity({k:v for k,v in run.items() if k != "retinal_atlas"})
    for event in run["events"]:
        pixels = np.asarray(Image.open(path.parent/event["image"]).convert("L"))
        assert np.array_equal(sample_retina(pixels, np.asarray(run["retina_uv"], np.float32)), event["retina"])
    run["retinal_atlas"] = atlas
    save_json(path, run)
    after = identity({k:v for k,v in read_json(path).items() if k != "retinal_atlas"})
    assert before == after
    results.append({"run": str(path), "events": len(run["events"]), "inference_unchanged": before == after})
save_json("reports/retinal-atlas/verification.json", {"atlas_id": atlas["atlas_id"], "runs": results,
          "retained_receptors_by_eye": {s:int((nodes.loc[ids[retina], "rootSide"] == s).sum()) for s in ["L", "R"]},
          "display_only": True, "accuracy_change": 0})
print("Attached verified atlas to", len(results), "runs; all recorded inference unchanged")
