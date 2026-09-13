"""Extract a bounded, resumable calibrated-retina corpus on the local Mac."""
import argparse
from pathlib import Path
import time
import numpy as np
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.brain.batch import pool, encode_job, extract
from flyocr.common import read_json, save_json, digest_file, identity
from flyocr.experiments.pilot import balanced_indices
from flyocr.vision.coverage import balanced_retina, sampling_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2: parser.error("This experiment is capped at two local workers")
    artifact = Path("artifacts/letters-v2"); artifact.mkdir(parents=True, exist_ok=True)
    output = Path("data/letters-v2-features"); output.mkdir(parents=True, exist_ok=True)
    corpus = Path("data/glyphs-letters")
    previous = read_json("artifacts/letters/model-card.json")
    config = StimulusConfig(**previous["config"])
    uv, calibration = balanced_retina(np.load("data/graph/uv.npy"))
    projection = artifact/"retina-uv.npy"
    if projection.exists() and not np.array_equal(uv, np.load(projection)):
        raise ValueError("Calibration changed")
    np.save(projection, uv)
    calibration["pixel_support_fraction"] = float(sampling_support(uv).mean())
    save_json(artifact/"input-calibration.json", calibration)
    brain = Brain("data/graph", config, retina_uv=uv)
    model_id = brain.model_id
    candidates = brain.graph["candidates"].copy()
    graph_id = brain.manifest["graph_id"]
    weight_hash = brain.manifest["array_hashes"]["weight"]
    del brain
    protocol = {
        "source_artifact_id": previous["artifact_id"],
        "corpus_id": read_json(corpus/"manifest.json")["corpus_id"],
        "model_id": model_id, "graph_id": graph_id,
        "input_projection_sha256": digest_file(projection),
        "feature_selection": "Same training-only variance rule: 8 glyphs/class, 1024 downstream cells",
        "head_candidates": [
            {"name": "raw64", "count_transform": "identity", "hidden": [64]},
            {"name": "sqrt64", "count_transform": "sqrt", "hidden": [64]},
            {"name": "sqrt64x32", "count_transform": "sqrt", "hidden": [64, 32]},
            {"name": "sqrt128x64", "count_transform": "sqrt", "hidden": [128, 64]},
        ],
        "optimizer": "AdamW", "learning_rate": .002, "weight_decay": .01,
        "dropout": .1, "max_epochs": 120, "patience": 18, "batch_size": 256, "seed": 817,
        "head_selection": "Best validation accuracy; ties prefer fewer parameters",
        "validation_access": True, "test_access_before_checkpoint_freeze": False,
        "fresh_holdout": {"fonts": ["Verdana", "Georgia"], "samples_per_font_per_class": 4,
                           "seed": 20260914, "generated_after_freeze": True},
        "budget": "Two local workers; one 100ms circuit presentation per glyph; max 540000 head parameters",
        "internal_synapses_learned": False,
    }
    protocol_path = artifact/"training-protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Training protocol changed")
    save_json(protocol_path, protocol)
    begin = time.perf_counter()
    train = np.load(corpus/"train.npz")
    with pool("data/graph", config, args.workers, retina_uv=uv) as executor:
        selection_path = artifact/"selection.json"
        if selection_path.exists():
            selection = read_json(selection_path)
            if selection["protocol_id"] != identity(protocol): raise ValueError("Selection mismatch")
            neurons = np.asarray(selection["neurons"], np.int32)
        else:
            indices = balanced_indices(train["labels"], 8)
            mean = np.zeros((config.bins, len(candidates)), np.float64); m2 = mean.copy()
            for n, (counts, _) in enumerate(executor.map(encode_job, ((im,candidates) for im in train["images"][indices]), chunksize=1), 1):
                delta = counts-mean; mean += delta/n; m2 += delta*(counts-mean)
                if n % 50 == 0: print(f"Cell selection: {n}/{len(indices)}", flush=True)
            score = m2.sum(0)/len(indices)
            active = np.flatnonzero(score > 0)
            neurons = candidates[active[np.argsort(-score[active], kind="stable")[:1024]]]
            save_json(selection_path, {"protocol_id": identity(protocol), "neurons": neurons.tolist(), "training_indices": indices.tolist()})
        benchmarks = {}
        for split in ["train", "validation"]:
            x, benchmark = extract(executor, model_id, corpus, split, neurons, root="data/letters-v2-shards")
            path = output/(split+".npz")
            np.savez_compressed(path, features=x, labels=np.load(corpus/(split+".npz"))["labels"])
            save_json(path.with_suffix(".json"), {"sha256": digest_file(path), "model_id": model_id,
                "source_sha256": digest_file(corpus/(split+".npz")), "neurons": neurons.tolist(), "benchmark": benchmark})
            benchmarks[split] = benchmark
    if digest_file(Path("data/graph/weight.npy")) != weight_hash: raise ValueError("Internal weights changed")
    save_json(artifact/"extraction.json", {"benchmarks": benchmarks, "wall_seconds": time.perf_counter()-begin,
        "model_id": model_id, "internal_weight_hash_unchanged": True})
    print("Training and validation responses ready. No held-out images were opened.", flush=True)


if __name__ == "__main__": main()
