"""Bounded development sweep. Does not load the test split or target PDF."""
from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import psutil
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.common import save_json, read_json, identity
from flyocr.readout.train import fit_readout


def balanced_indices(labels, per_class):
    return np.concatenate([np.flatnonzero(labels == c)[:per_class] for c in np.unique(labels)])


def select_neurons(counts, candidates, count=512):
    # counts: samples x bins x candidates. Only training responses are supplied.
    score = counts.astype(np.float32).var(axis=0).sum(axis=0)
    active = np.flatnonzero(score > 0)
    selected = active[np.argsort(-score[active], kind="stable")[:count]]
    return candidates[selected], selected


def run(graph, corpus, output, configs=None, train_per_class=12, val_per_class=8):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    corpus = Path(corpus)
    train, val = np.load(corpus/"train.npz"), np.load(corpus/"validation.npz")
    ti, vi = balanced_indices(train["labels"], train_per_class), balanced_indices(val["labels"], val_per_class)
    configs = configs or [StimulusConfig(sensory_gain=g, lamina_drive=l) for g in [12., 20., 30., 45.] for l in [8., 12., 20.]]
    protocol = {"configs": [asdict(c) for c in configs], "train_indices": ti.tolist(), "validation_indices": vi.tolist(), "test_access": False,
                "corpus_id": read_json(corpus/"manifest.json")["corpus_id"], "feature_selection": "training-only variance; max 512 downstream cells"}
    save_json(output/"protocol.json", protocol)
    results = []
    base = Brain(graph, configs[0])
    for index, config in enumerate(configs):
        key = identity({"protocol": protocol, "config": asdict(config)})[:16]
        result_path = output/(key+".json")
        if result_path.exists():
            results.append(read_json(result_path)); continue
        brain = base if index == 0 else Brain(base.graph, config)
        begin = time.perf_counter()
        counts = []
        for image in train["images"][ti]:
            x, _, _ = brain.encode(image)
            counts.append(x)
        counts = np.asarray(counts)
        neurons, selected = select_neurons(counts, brain.graph["candidates"])
        if len(neurons):
            tx = counts[:, :, selected].reshape(len(ti), -1)
            vx = np.array([brain.encode(image, neurons)[0].ravel() for image in val["images"][vi]])
            _, fit = fit_readout(tx, train["labels"][ti], vx, val["labels"][vi], strengths=(.001, .01, .1, 1.))
        else:
            fit = {"validation_accuracy": .1, "error": "No varying downstream cells"}
        result = {"key": key, "config": asdict(config), "neurons": neurons.tolist(), "feature_count": len(neurons)*config.bins,
                  "varying_candidate_cells": int(np.count_nonzero(counts.astype(np.float32).var(0).sum(0))),
                  "mean_downstream_spikes": float(counts.sum((1, 2)).mean()), "fit": fit,
                  "wall_seconds": time.perf_counter()-begin, "glyphs": len(ti)+len(vi),
                  "rss_mib": psutil.Process().memory_info().rss/2**20}
        results.append(result); save_json(result_path, result)
        print(f"Pilot {index+1}/{len(configs)}: gain={config.sensory_gain:g}, tonic={config.lamina_drive:g}; val={fit['validation_accuracy']:.1%}; {len(neurons)} cells; {result['wall_seconds']:.1f}s", flush=True)
        save_json(output/"results.json", {"protocol": protocol, "results": results})
    best = max(results, key=lambda r: r["fit"]["validation_accuracy"])
    save_json(output/"selected.json", {"config": best["config"], "pilot_key": best["key"], "validation_accuracy": best["fit"]["validation_accuracy"], "graph_id": base.manifest["graph_id"], "all_results": [r["key"] for r in results]})
    return best
