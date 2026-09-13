"""Bounded local workers with independent circuit state and checked feature shards."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import time
import numpy as np
from flyocr.common import read_json, save_json, digest_file, identity
from flyocr.brain.model import Brain, StimulusConfig

_worker = None


def initialize(graph, config, retina_uv=None):
    global _worker
    _worker = Brain(graph, StimulusConfig(**config), retina_uv=retina_uv)


def encode_job(job):
    image, neurons = job
    start = time.perf_counter()
    counts, _, _ = _worker.encode(image, neurons)
    return counts.astype(np.int16), time.perf_counter()-start


def pool(graph, config, workers=2, retina_uv=None):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize, initargs=(str(graph), asdict(config), retina_uv))


def extract(executor, model_id, corpus, split, neurons, root="data/letter-features", shard_size=100):
    corpus, root = Path(corpus), Path(root)
    source = corpus/(split+".npz")
    protocol = {"format":1, "model_id":model_id, "source_sha256":digest_file(source),
        "neurons":np.asarray(neurons).tolist(), "shard_size":shard_size,
        "implementation_hashes":{str(p.relative_to(Path(__file__).parents[1])):digest_file(p) for p in
            [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("native.py"),
             Path(__file__).parents[1]/"vision/retina.py", Path(__file__).parents[1]/"vision/letterbox.py"]}}
    key = identity(protocol); directory = root/key; directory.mkdir(parents=True, exist_ok=True)
    save_json(directory/"protocol.json", protocol)
    data = np.load(source, allow_pickle=False)
    features, durations = [], []
    for start in range(0, len(data["images"]), shard_size):
        stop = min(len(data["images"]), start+shard_size)
        path = directory/f"{start:06d}.npz"; meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            m = read_json(meta)
            if m["sha256"] != digest_file(path) or m["start"] != start or m["stop"] != stop:
                raise ValueError("Letter feature shard integrity mismatch")
            shard = np.load(path, allow_pickle=False); x, timing = shard["features"], shard["wall_seconds"]
        else:
            results = list(executor.map(encode_job, ((im, neurons) for im in data["images"][start:stop]), chunksize=1))
            x = np.asarray([r[0].ravel() for r in results], np.int16); timing = np.array([r[1] for r in results])
            temporary = path.with_suffix(".partial")
            with temporary.open("wb") as f: np.savez_compressed(f, features=x, wall_seconds=timing)
            temporary.replace(path); save_json(meta, {"start":start,"stop":stop,"sha256":digest_file(path)})
        if x.ndim != 2 or x.shape[0] != stop-start or not np.isfinite(x).all():
            raise ValueError("Malformed letter feature shard")
        features.append(x); durations.extend(timing.tolist())
        print(f"{split}: {stop}/{len(data['images'])} glyphs", flush=True)
    return np.concatenate(features), {"cache_id":key,"n":len(durations),"worker_seconds":sum(durations),"mean_worker_seconds":float(np.mean(durations))}
