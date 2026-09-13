"""Retrained degree-preserving randomized-target controls, separate from lesions."""
from pathlib import Path
from dataclasses import asdict
import time
import numpy as np
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.experiments.pilot import balanced_indices, select_neurons
from flyocr.readout.train import fit_readout, predict
from flyocr.eval.metrics import classification
from flyocr.common import read_json, save_json, identity


def run(graph, corpus, artifact, output, seeds=(41, 42, 43)):
    corpus, artifact, output = Path(corpus), Path(artifact), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    card = read_json(artifact/"model-card.json")
    config = StimulusConfig(**card["config"])
    base = Brain(graph, config)
    full = {s: np.load(corpus/(s+".npz")) for s in ["train", "validation", "test"]}
    indices = {s: balanced_indices(a["labels"], {"train":20,"validation":10,"test":20}[s]) for s,a in full.items()}
    arrays = {s: {k:a[k][indices[s]] for k in ["images","labels"]} for s,a in full.items()}
    save_json(output/"protocol.json", {"artifact_id":card["artifact_id"], "seeds":list(seeds), "indices":{s:ix.tolist() for s,ix in indices.items()},
        "budget_note":"200 training / 100 validation / 200 test glyphs, balanced by class; separately fitted intact comparator; fixed before control accuracies observed. Full-size attempt stopped after measured randomized-graph runtime increase."})
    classes = np.arange(len(card["characters"]))
    original = np.load(artifact/"readout.npz")
    zero_graph = {**base.graph, "weight": np.zeros_like(base.graph["weight"])}
    lesion = Brain(zero_graph, config)
    for im in arrays["test"]["images"][:10]:
        assert not lesion.encode(im, original["neurons"])[0].any()
    result = {"artifact_id": card["artifact_id"], "edge_lesion_original_head": classification(arrays["test"]["labels"],
        predict(original, np.zeros((len(arrays["test"]["labels"]), len(original["mean"])))), classes),
        "lesion_caveat": "Original head under distribution shift. Ten inputs verified zero downstream counts; remaining undriven cells stay at rest without edges.",
        "randomized": []}
    intact_features = {s:np.array([base.encode(im,original["neurons"])[0].ravel() for im in a["images"]]) for s,a in arrays.items()}
    intact_head,intact_fit = fit_readout(intact_features["train"], arrays["train"]["labels"], intact_features["validation"], arrays["validation"]["labels"])
    result["intact_matched_subset"] = {**classification(arrays["test"]["labels"], predict(intact_head,intact_features["test"]),classes), "fit":intact_fit,
        "mean_feature_count":float(intact_features["train"].mean()), "n_train":len(arrays["train"]["labels"]), "n_validation":len(arrays["validation"]["labels"])}
    del lesion, zero_graph
    for seed in seeds:
        path = output/f"seed-{seed}.json"
        if path.exists():
            saved = read_json(path)
            if saved["artifact_id"] != card["artifact_id"]: raise ValueError("Control model changed")
            result["randomized"].append(saved); continue
        start = time.perf_counter()
        rng = np.random.default_rng(seed)
        post = rng.permutation(base.graph["post"])
        assert np.array_equal(np.bincount(post, minlength=base.n), np.bincount(base.graph["post"], minlength=base.n))
        randomized = Brain({**base.graph, "post": post}, config)
        selection = balanced_indices(arrays["train"]["labels"], 12)
        counts = np.array([randomized.encode(im)[0] for im in arrays["train"]["images"][selection]])
        neurons, _ = select_neurons(counts, randomized.graph["candidates"])
        if not len(neurons):
            saved = {"artifact_id": card["artifact_id"], "seed": seed, "error": "No varying eligible features"}
        else:
            features = {}
            for split, a in arrays.items():
                features[split] = []
                for i,im in enumerate(a["images"]):
                    features[split].append(randomized.encode(im, neurons)[0].ravel())
                    if (i+1)%100 == 0: print(f"Random graph {seed}: {split} {i+1}/{len(a['images'])}",flush=True)
                features[split] = np.array(features[split])
                print(f"Random graph {seed}: {split} complete", flush=True)
            head, fit = fit_readout(features["train"], arrays["train"]["labels"], features["validation"], arrays["validation"]["labels"])
            saved = {"artifact_id": card["artifact_id"], "seed": seed,
                "neural": classification(arrays["test"]["labels"], predict(head, features["test"]), classes),
                "fit": fit, "feature_cells": len(neurons), "mean_feature_count": float(features["train"].mean()),
                "construction": "Uniform permutation of edge target vector; exact in/out edge degrees and source weight/sign lists preserved. Spatial structure and individual contacts per target are not preserved; multiedges may occur.",
                "verified_exact_in_degree": True, "verified_exact_out_degree": True,
                "verified_source_weight_sign_unchanged": True, "config": asdict(config), "wall_seconds": time.perf_counter()-start}
        save_json(path, saved); result["randomized"].append(saved)
        save_json(output/"controls.json", result)
    save_json(output/"controls.json", result)
    return result
