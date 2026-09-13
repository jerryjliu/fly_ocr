from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import psutil
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.brain.features import extract
from flyocr.experiments.pilot import balanced_indices, select_neurons
from flyocr.readout.train import fit_readout, predict, probabilities
from flyocr.eval.metrics import classification
from flyocr.vision.retina import sample_retina
from flyocr.common import save_json, read_json, digest_file, identity


def run(graph, corpus, pilot, output, cache="data/features", baselines=True):
    output, corpus = Path(output), Path(corpus); output.mkdir(parents=True, exist_ok=True)
    selected = read_json(Path(pilot)/"selected.json")
    config = StimulusConfig(**selected["config"])
    start = time.perf_counter(); brain = Brain(graph, config)
    cold_seconds = time.perf_counter()-start
    cm = read_json(corpus/"manifest.json")
    train = np.load(corpus/"train.npz"); val = np.load(corpus/"validation.npz")
    selection_path = output/"selection.json"
    selection_identity = {"model_id": brain.model_id, "corpus_id": cm["corpus_id"], "rule": "training-only variance, 12/class, max512"}
    if selection_path.exists():
        selection = read_json(selection_path)
        if selection["identity"] != selection_identity: raise ValueError("Selection provenance changed")
        neurons = np.array(selection["neurons"], np.int32)
    else:
        idx = balanced_indices(train["labels"], 12)
        counts = np.array([brain.encode(im)[0] for im in train["images"][idx]])
        neurons, _ = select_neurons(counts, brain.graph["candidates"])
        if not len(neurons): raise ValueError("No varying downstream activity")
        save_json(selection_path, {"identity": selection_identity, "neurons": neurons.tolist(), "training_indices": idx.tolist()})
    tx, tb = extract(brain, corpus, "train", neurons, cache)
    vx, vb = extract(brain, corpus, "validation", neurons, cache)
    payload, fit = fit_readout(tx, train["labels"], vx, val["labels"])
    checkpoint = output/"readout.npz"
    np.savez_compressed(checkpoint, **payload, neurons=neurons)
    # This frozen record is written BEFORE opening the test split or PDF.
    card = {"format": 1, "characters": cm["characters"], "corpus_id": cm["corpus_id"],
            "model_id": brain.model_id, "graph_id": brain.manifest["graph_id"], "config": asdict(config),
            "readout_sha256": digest_file(checkpoint), "selection_sha256": digest_file(selection_path),
            "feature_cells": len(neurons), "feature_count": tx.shape[1], "fit": fit,
            "reject_threshold": 0.0, "rejection_policy": "closed 16/10-class vocabulary; no calibrated open-set detector",
            "learned": "training standardization and regularized linear readout only",
            "fixed": "retinal projection, cell dynamics, all internal edges and weights"}
    card["artifact_id"] = identity(card); save_json(output/"model-card.json", card)
    test = np.load(corpus/"test.npz")
    ex, eb = extract(brain, corpus, "test", neurons, cache)
    classes = np.arange(len(cm["characters"]))
    result = {"artifact_id": card["artifact_id"], "characters": cm["characters"],
              "neural": classification(test["labels"], predict(payload, ex), classes),
              "validation": fit, "benchmark": {"cold_start_seconds": cold_seconds, "train": tb, "validation": vb, "test": eb,
                  "rss_mib": psutil.Process().memory_info().rss/2**20}, "baselines": {}}
    groups = read_json(corpus/"test.json")
    predicted = predict(payload, ex)
    result["by_test_font"] = {f: classification(test["labels"][ix], predicted[ix], classes)
        for f in sorted({g["family"] for g in groups})
        for ix in [np.array([g["family"] == f for g in groups])]}
    np.savez_compressed(output/"heldout-predictions.npz", labels=test["labels"], predictions=predicted,
                        probabilities=probabilities(payload, ex), features=ex)
    result["baselines"]["majority"] = classification(test["labels"], np.full(len(ex), np.bincount(train["labels"]).argmax()), classes)
    rng = np.random.default_rng(817)
    shuffled, sf = fit_readout(tx, rng.permutation(train["labels"]), vx, val["labels"])
    result["baselines"]["shuffled_training_labels"] = {**classification(test["labels"], predict(shuffled, ex), classes), "fit": sf}
    result["baselines"]["mismatched_test_images"] = classification(test["labels"], predicted[rng.permutation(len(ex))], classes)
    blank = brain.encode(np.full((48, 48), 255, np.uint8), neurons)[0].ravel()
    result["baselines"]["blank_input_original_head"] = classification(test["labels"], predict(payload, np.tile(blank, (len(ex), 1))), classes)
    save_json(output/"recognition.json", result)
    if baselines:
        for name in ["raw_pixels", "retinal_samples"]:
            def transform(images):
                if name == "raw_pixels": return images.reshape(len(images), -1)/255.
                return np.array([sample_retina(im, brain.graph["uv"]) for im in images])
            model, baseline_fit = fit_readout(transform(train["images"]), train["labels"], transform(val["images"]), val["labels"])
            result["baselines"][name] = {**classification(test["labels"], predict(model, transform(test["images"])), classes), "fit": baseline_fit}
            save_json(output/"recognition.json", result)
    # A read-only mmap plus this final on-disk hash check guards the fixed circuit claim.
    if digest_file(Path(graph)/"weight.npy") != brain.manifest["array_hashes"]["weight"]:
        raise ValueError("Internal graph weights changed")
    result["internal_weight_hash_unchanged"] = True
    result["wall_seconds"] = time.perf_counter()-start
    save_json(output/"recognition.json", result)
    print(f"Held-out recognition: {result['neural']['correct']}/{len(ex)} = {result['neural']['accuracy']:.1%}", flush=True)
    return result
