"""Train an expanded alphabet with fixed circuit settings and line-relative input."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.brain.batch import pool, encode_job, extract
from flyocr.common import read_json, save_json, digest_file, identity
from flyocr.experiments.pilot import balanced_indices
from flyocr.readout.train import fit_readout, predict, probabilities
from flyocr.eval.metrics import classification


def run(graph="data/graph", corpus="data/glyphs-letters", output="artifacts/letters", workers=2):
    corpus, output = Path(corpus), Path(output); output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = StimulusConfig(**read_json("reports/frozen-pilot/selected.json")["config"])
    cm = read_json(corpus/"manifest.json")
    brain = Brain(graph, config)
    model_id, graph_id = brain.model_id, brain.manifest["graph_id"]
    candidates = brain.graph["candidates"].copy()
    weight_hash = brain.manifest["array_hashes"]["weight"]
    del brain
    protocol = {"characters":cm["characters"],"corpus_id":cm["corpus_id"],"model_id":model_id,
        "normalization":cm["normalization"],"feature_selection":"training variance; 8 examples/class; max1024 cells",
        "regularization_strengths":[.001,.01,.1,1.,10.],"test_access_before_freeze":False,
        "workers":workers,"internal_weights_learned":False}
    protocol_path = output/"training-protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Declared letter training protocol changed")
    save_json(protocol_path, protocol)
    train = np.load(corpus/"train.npz"); val = np.load(corpus/"validation.npz")
    selection_path = output/"selection.json"
    with pool(graph, config, workers) as executor:
        if selection_path.exists():
            selection = read_json(selection_path)
            if selection["protocol_id"] != identity(protocol): raise ValueError("Selection identity changed")
            neurons = np.asarray(selection["neurons"], np.int32)
        else:
            indices = balanced_indices(train["labels"], 8)
            mean = np.zeros((config.bins,len(candidates)),np.float64); m2 = mean.copy()
            for n,(counts,_) in enumerate(executor.map(encode_job, ((im,candidates) for im in train["images"][indices]), chunksize=1), 1):
                delta = counts-mean; mean += delta/n; m2 += delta*(counts-mean)
                if n%50==0: print(f"Cell selection: {n}/{len(indices)} glyphs",flush=True)
            score = m2.sum(0)/len(indices); active = np.flatnonzero(score>0)
            neurons = candidates[active[np.argsort(-score[active],kind="stable")[:1024]]]
            save_json(selection_path,{"protocol_id":identity(protocol),"neurons":neurons.tolist(),"training_indices":indices.tolist()})
        tx,tb = extract(executor,model_id,corpus,"train",neurons)
        vx,vb = extract(executor,model_id,corpus,"validation",neurons)
        print("Fitting the letter readout on training features; selecting strength on validation fonts.",flush=True)
        payload, fit = fit_readout(tx,train["labels"],vx,val["labels"])
        checkpoint = output/"readout.npz"; np.savez_compressed(checkpoint,**payload,neurons=neurons)
        card = {"format":1,"characters":cm["characters"],"corpus_id":cm["corpus_id"],"model_id":model_id,
            "graph_id":graph_id,"config":asdict(config),"readout_sha256":digest_file(checkpoint),
            "selection_sha256":digest_file(selection_path),"feature_cells":len(neurons),"feature_count":tx.shape[1],
            "normalization":cm["normalization"],"normalization_sha256":digest_file(Path(__file__).parents[1]/"vision/letterbox.py"),
            "fit":fit,"reject_threshold":0.,"rejection_policy":"closed 68-class vocabulary; no calibrated open-set detector",
            "learned":"training standardization and regularized linear readout only",
            "fixed":"retinal projection, cell dynamics, all internal edges and weights"}
        card["artifact_id"] = identity(card); save_json(output/"model-card.json",card)
        print(f"Checkpoint frozen. Validation accuracy: {fit['validation_accuracy']:.1%}. Opening held-out test split.",flush=True)
        test = np.load(corpus/"test.npz")
        ex,eb = extract(executor,model_id,corpus,"test",neurons)
    pred = predict(payload,ex); chars = cm["characters"]; classes = np.arange(len(chars))
    groups = read_json(corpus/"test.json")
    result = {"artifact_id":card["artifact_id"],"characters":chars,"neural":classification(test["labels"],pred,classes),
        "validation":fit,"benchmark":{"train":tb,"validation":vb,"test":eb},
        "by_group":{},"by_test_font":{}}
    for name, alphabet in {"uppercase":"ABCDEFGHIJKLMNOPQRSTUVWXYZ","lowercase":"abcdefghijklmnopqrstuvwxyz","digits":"0123456789","punctuation":",.-()$"}.items():
        mask = np.isin(test["labels"],[chars.index(c) for c in alphabet])
        result["by_group"][name] = classification(test["labels"][mask],pred[mask],classes)
    for font in sorted({g["family"] for g in groups}):
        mask = np.array([g["family"]==font for g in groups])
        result["by_test_font"][font] = classification(test["labels"][mask],pred[mask],classes)
    np.savez_compressed(output/"heldout-predictions.npz",labels=test["labels"],predictions=pred,
        probabilities=probabilities(payload,ex),features=ex)
    if digest_file(Path(graph)/"weight.npy") != weight_hash: raise ValueError("Internal weights changed")
    result["internal_weight_hash_unchanged"]=True; result["wall_seconds"]=time.perf_counter()-start
    save_json(output/"recognition.json",result)
    print(f"Held-out letters checkpoint: {result['neural']['correct']}/{len(pred)} ({result['neural']['accuracy']:.1%})",flush=True)
    return result
