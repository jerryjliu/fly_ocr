"""Resumable, content-identified neural features; no document labels enter Brain."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from flyocr.common import identity, digest_file, save_json, read_json


def extract(brain, corpus, split, neurons, root, shard_size=100):
    corpus, root = Path(corpus), Path(root)
    source = corpus/(split+".npz")
    protocol = {"model_id": brain.model_id, "source_sha256": digest_file(source),
                "neurons": np.asarray(neurons).tolist(), "bins": brain.config.bins,
                "format": 2, "shard_size": shard_size,
                "implementation_hashes": {p.name: digest_file(p) for p in [Path(__file__),
                    Path(__file__).with_name("model.py"), Path(__file__).with_name("native.py"),
                    Path(__file__).parent.parent/"vision"/"retina.py"]}}
    key = identity(protocol)
    directory = root/key; directory.mkdir(parents=True, exist_ok=True)
    save_json(directory/"protocol.json", protocol)
    data = np.load(source, allow_pickle=False)
    features, durations = [], []
    for start in range(0, len(data["images"]), shard_size):
        stop = min(start+shard_size, len(data["images"]))
        path = directory/f"{start:06d}.npz"; meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            m = read_json(meta)
            if m["sha256"] != digest_file(path) or m["start"] != start or m["stop"] != stop:
                raise ValueError("Feature shard checksum or interval mismatch")
            a = np.load(path, allow_pickle=False)
            x, timing = a["features"], a["wall_seconds"]
        else:
            x, timing = [], []
            for im in data["images"][start:stop]:
                begin = time.perf_counter()
                x.append(brain.encode(im, neurons)[0].ravel())
                timing.append(time.perf_counter()-begin)
            x, timing = np.asarray(x, np.int16), np.asarray(timing)
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as f:
                np.savez_compressed(f, features=x, wall_seconds=timing)
            temporary.replace(path)
            save_json(meta, {"sha256": digest_file(path), "start": start, "stop": stop})
        if x.shape != (stop-start, len(neurons)*brain.config.bins) or not np.isfinite(x).all():
            raise ValueError("Malformed feature shard")
        features.append(x); durations.extend(timing.tolist())
        print(f"{split}: {stop}/{len(data['images'])} glyphs", flush=True)
    return np.concatenate(features), {"cache_id": key, "mean_seconds": float(np.mean(durations)),
        "p95_seconds": float(np.percentile(durations, 95)), "total_neural_seconds": float(sum(durations)),
        "n": len(durations), "feature_bytes": sum(x.nbytes for x in features)}
