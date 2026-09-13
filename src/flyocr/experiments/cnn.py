"""Conventional image CNN reference; completely separate from the fly recognizer."""
from pathlib import Path
import copy
import numpy as np
from flyocr.common import save_json, read_json
from flyocr.eval.metrics import classification


def run(corpus="data/glyphs-digits", output="reports/cnn.json", epochs=20):
    import torch
    from torch import nn
    torch.set_num_threads(1); torch.manual_seed(911)
    arrays = {s: np.load(Path(corpus)/(s+".npz")) for s in ["train", "validation", "test"]}
    data = {s: (torch.from_numpy(1-a["images"].astype(np.float32)/255)[:,None], torch.from_numpy(a["labels"].astype(np.int64))) for s,a in arrays.items()}
    n = len(read_json(Path(corpus)/"manifest.json")["characters"])
    model = nn.Sequential(nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8,16,3,padding=1),
        nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16*12*12,64), nn.ReLU(), nn.Linear(64,n))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data["train"]),batch_size=64,shuffle=True)
    best, chosen, trials = -1., None, []
    for epoch in range(epochs):
        model.train()
        for x,y in loader:
            optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(x),y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): accuracy = float((model(data["validation"][0]).argmax(1) == data["validation"][1]).float().mean())
        trials.append({"epoch": epoch+1, "validation_accuracy": accuracy})
        if accuracy > best: best, chosen = accuracy, copy.deepcopy(model.state_dict())
    model.load_state_dict(chosen); model.eval()
    with torch.no_grad(): predictions = model(data["test"][0]).argmax(1).numpy()
    result = {"architecture": "Conv8-pool-Conv16-pool-FC64-linear, ReLU, 48x48 input", "seed": 911,
        "parameters": sum(p.numel() for p in model.parameters()), "selection": "best validation epoch of fixed 20-epoch budget",
        "trials": trials, "test": classification(arrays["test"]["labels"], predictions, np.arange(n)),
        "corpus_id": read_json(Path(corpus)/"manifest.json")["corpus_id"]}
    save_json(output, result); print(result["test"]["accuracy"])
    return result
