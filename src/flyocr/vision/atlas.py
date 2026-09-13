"""Display coordinates for recorded receptor signals, independent of image sampling.

This is a lamina-column atlas inferred from MaleCNS annotations, not a measured
optical field or an ommatidium-by-ommatidium reconstruction of the eye surface.
"""
from pathlib import Path
import numpy as np
from flyocr.common import identity, read_json, save_json


def make_atlas(ids, retina, uv, sides, graph_id):
    uv = np.asarray(uv, np.float32)
    sides = np.asarray(sides)
    if uv.shape != (len(retina), 2) or sides.shape != (len(retina),):
        raise ValueError("Atlas dimensions differ from retinal inputs")
    if not np.isfinite(uv).all() or not np.isin(sides, ["L", "R"]).all():
        raise ValueError("Invalid retinal atlas geometry")
    atlas = {
        "format": 1, "graph_id": graph_id,
        "retina_body_ids": [str(i) for i in np.asarray(ids)[retina]],
        "original_uv": uv.tolist(), "eye": sides.tolist(),
        "description": "Lamina-column atlas inferred from strongest R1-R6 to L1/L2/L3 contacts and assignedOlHex annotations. Eye outlines and hexagonal facets are schematic. Colors show recorded sampled brightness, not retinal spikes or measured optical fields.",
    }
    atlas["atlas_id"] = identity(atlas)
    return atlas


def validate_atlas(atlas, n=None, graph_id=None):
    if identity({k: v for k, v in atlas.items() if k != "atlas_id"}) != atlas["atlas_id"]:
        raise ValueError("Retinal atlas checksum mismatch")
    uv = np.asarray(atlas["original_uv"])
    n = n if n is not None else len(atlas["eye"])
    if (uv.shape != (n, 2) or not np.isfinite(uv).all() or
            np.any((uv < 0) | (uv > 1)) or len(atlas["eye"]) != n or
            not set(atlas["eye"]) <= {"L", "R"} or
            len(atlas["retina_body_ids"]) != n or len(set(atlas["retina_body_ids"])) != n):
        raise ValueError("Invalid retinal atlas")
    if graph_id is not None and atlas["graph_id"] != graph_id:
        raise ValueError("Atlas belongs to another graph")


def load_atlas(graph):
    path = Path(graph)/"retinal-atlas.json"
    if not path.exists():
        return None
    atlas = read_json(path)
    manifest = read_json(Path(graph)/"manifest.json")
    validate_atlas(atlas, manifest["retinal_inputs"], manifest["graph_id"])
    ids = np.load(Path(graph)/"ids.npy", allow_pickle=False)
    retina = np.load(Path(graph)/"retina.npy", allow_pickle=False)
    if atlas["retina_body_ids"] != [str(i) for i in ids[retina]]:
        raise ValueError("Atlas receptor order mismatch")
    if not np.array_equal(atlas["original_uv"], np.load(Path(graph)/"uv.npy", allow_pickle=False)):
        raise ValueError("Atlas source coordinates mismatch")
    return atlas
