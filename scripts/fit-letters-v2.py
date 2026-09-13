"""Four compact nonlinear readouts on cached fly activity; validation-only selection."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from flyocr.common import read_json, save_json, digest_file, identity
from flyocr.readout.train import probabilities


def main():
    output = Path("artifacts/letters-v2")
    if (output/"model-card.json").exists():
        print("A validation-selected checkpoint is already frozen."); return
    protocol = read_json(output/"training-protocol.json")
    selection = read_json(output/"selection.json")
    previous = read_json("artifacts/letters/model-card.json")
    data = {}
    for split in ["train", "validation"]:
        path = Path("data/letters-v2-features")/(split+".npz")
        meta = read_json(path.with_suffix(".json"))
        if (meta["sha256"] != digest_file(path) or meta["model_id"] != protocol["model_id"]
                or meta["neurons"] != selection["neurons"]
                or meta["source_sha256"] != digest_file(Path("data/glyphs-letters")/(split+".npz"))):
            raise ValueError("Feature provenance mismatch")
        data[split] = np.load(path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    y = torch.tensor(data["train"]["labels"].astype(np.int64), device=device)
    vy = torch.tensor(data["validation"]["labels"].astype(np.int64), device=device)
    best_key = None; chosen = None; trials = []; begin = time.perf_counter()
    for spec in protocol["head_candidates"]:
        transform = spec["count_transform"]
        tx, vx = (data[s]["features"].astype(np.float32) for s in ["train", "validation"])
        if transform == "sqrt": tx, vx = np.sqrt(tx), np.sqrt(vx)
        scaler = StandardScaler().fit(tx)
        train = np.clip(scaler.transform(tx), -10, 10)
        val = np.clip(scaler.transform(vx), -10, 10)
        x, v = torch.tensor(train, device=device), torch.tensor(val, device=device)
        torch.manual_seed(protocol["seed"]); rng = np.random.default_rng(protocol["seed"])
        widths = [x.shape[1], *spec["hidden"], len(previous["characters"])]
        layers = []; linears = []
        for i, (left, right) in enumerate(zip(widths, widths[1:])):
            layer = nn.Linear(left, right); layers.append(layer); linears.append(layer)
            if i < len(widths)-2: layers.extend([nn.ReLU(), nn.Dropout(protocol["dropout"])])
        net = nn.Sequential(*layers).to(device)
        parameters = sum(p.numel() for p in net.parameters())
        if parameters > 540000: raise ValueError("Head exceeded declared parameter cap")
        optimizer = torch.optim.AdamW(net.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
        best = -1.; best_epoch = 0; best_state = None; history = []
        head_start = time.perf_counter()
        for epoch in range(1, protocol["max_epochs"]+1):
            net.train(); order = rng.permutation(len(train))
            for lo in range(0, len(order), protocol["batch_size"]):
                idx = torch.tensor(order[lo:lo+protocol["batch_size"]], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(net(x[idx]), y[idx]); loss.backward(); optimizer.step()
            net.eval()
            with torch.no_grad(): correct = int((net(v).argmax(1) == vy).sum().cpu())
            accuracy = correct/len(vy); history.append(accuracy)
            if accuracy > best:
                best = accuracy; best_epoch = epoch
                best_state = {k:t.detach().cpu().clone() for k,t in net.state_dict().items()}
            if epoch % 10 == 0: print(spec["name"], "epoch", epoch, "best", round(best, 4), flush=True)
            if epoch-best_epoch >= protocol["patience"]: break
        net.load_state_dict(best_state); net.eval()
        payload = {"mean": scaler.mean_, "scale": scaler.scale_, "count_transform": np.array(transform),
            "neurons": np.asarray(selection["neurons"], np.int32), "classes": np.arange(len(previous["characters"])),
            "hidden_coef": linears[0].weight.detach().cpu().numpy(), "hidden_intercept": linears[0].bias.detach().cpu().numpy(),
            "coef": linears[-1].weight.detach().cpu().numpy(), "intercept": linears[-1].bias.detach().cpu().numpy()}
        if len(spec["hidden"]) == 2:
            payload.update({"hidden2_coef": linears[1].weight.detach().cpu().numpy(), "hidden2_intercept": linears[1].bias.detach().cpu().numpy()})
        path = output/("readout-"+spec["name"]+".npz")
        np.savez_compressed(path, **payload)
        # Check the actual saved NumPy inference path against Torch predictions.
        with torch.no_grad(): expected = net(v).argmax(1).cpu().numpy()
        with threadpool_limits(limits=4):
            restored = dict(np.load(path, allow_pickle=False))
            actual = probabilities(restored, data["validation"]["features"]).argmax(1)
        if not np.array_equal(expected, actual): raise ValueError("Serialized head differs from Torch")
        trial = {**spec, "parameters": parameters, "best_epoch": best_epoch, "epochs": epoch,
                 "validation_accuracy": best, "correct": int((actual == data["validation"]["labels"]).sum()),
                 "n": len(actual), "history": history, "seconds": time.perf_counter()-head_start,
                 "serialization_predictions_exact": True}
        trials.append(trial)
        key = (best, -parameters)
        if best_key is None or key > best_key:
            best_key = key; chosen = trial
            np.savez_compressed(output/"readout.npz", **payload)
        save_json(output/"validation.json", {"trials": trials, "selected": chosen["name"],
            "best_accuracy": best_key[0], "wall_seconds": time.perf_counter()-begin, "device": device, "test_access": False})
        print(spec["name"], f"{best:.1%}", parameters, "parameters", flush=True)
    card = {"format": 1, "characters": previous["characters"], "corpus_id": protocol["corpus_id"],
        "model_id": protocol["model_id"], "graph_id": protocol["graph_id"], "config": previous["config"],
        "readout_sha256": digest_file(output/"readout.npz"), "selection_sha256": digest_file(output/"selection.json"),
        "feature_cells": len(selection["neurons"]), "feature_count": data["train"]["features"].shape[1],
        "normalization": "line-box-v1", "normalization_sha256": previous["normalization_sha256"],
        "readout_type": "mlp", "readout_architecture": {"hidden": chosen["hidden"], "count_transform": chosen["count_transform"], "parameters": chosen["parameters"]},
        "input_projection": {**read_json(output/"input-calibration.json"), "sha256": digest_file(output/"retina-uv.npy")},
        "fit": {"validation_accuracy": chosen["validation_accuracy"], "selected": chosen["name"], "selection": protocol["head_selection"]},
        "reject_threshold": 0., "rejection_policy": previous["rejection_policy"],
        "learned": "Training feature selection/standardization and a compact nonlinear readout of downstream spike counts",
        "fixed": "Label-free input calibration, all internal edges and weights, cell dynamics, stimulus timing",
        "training_protocol_sha256": digest_file(output/"training-protocol.json"),
        "previous_artifact_id": previous["artifact_id"]}
    card["artifact_id"] = identity(card); save_json(output/"model-card.json", card)
    print("FROZEN:", chosen["name"], chosen["validation_accuracy"], card["artifact_id"], flush=True)


if __name__ == "__main__": main()
