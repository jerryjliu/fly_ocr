from pathlib import Path
import numpy as np
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.common import read_json, digest_file, identity
from flyocr.readout.train import probabilities


class Recognizer:
    def __init__(self, graph, artifact):
        artifact = Path(artifact)
        self.card = read_json(artifact/"model-card.json")
        if identity({k: v for k, v in self.card.items() if k != "artifact_id"}) != self.card["artifact_id"]:
            raise ValueError("Model card identity mismatch")
        if digest_file(artifact/"readout.npz") != self.card["readout_sha256"]:
            raise ValueError("Readout checksum mismatch")
        if self.card.get("normalization") == "line-box-v1":
            normalizer = Path(__file__).parents[1]/"vision/letterbox.py"
            if digest_file(normalizer) != self.card["normalization_sha256"]:
                raise ValueError("Letter normalization differs from the trained checkpoint")
        self.payload = dict(np.load(artifact/"readout.npz", allow_pickle=False))
        retina_uv = None
        if "input_projection" in self.card:
            if digest_file(artifact/"retina-uv.npy") != self.card["input_projection"]["sha256"]:
                raise ValueError("Retinal calibration checksum mismatch")
            retina_uv = np.load(artifact/"retina-uv.npy", allow_pickle=False)
        self.brain = Brain(graph, StimulusConfig(**self.card["config"]), retina_uv=retina_uv)
        if self.brain.model_id != self.card["model_id"]: raise ValueError("Model/graph mismatch")
        self.neurons = self.payload["neurons"]

    def recognize(self, pixels):
        counts, retina, totals = self.brain.encode(pixels, self.neurons)
        p = probabilities(self.payload, counts.reshape(1, -1))[0]
        idx = int(p.argmax()); label = int(self.payload["classes"][idx])
        character = self.card["characters"][label] if p[idx] >= self.card["reject_threshold"] else "?"
        return {"character": character, "confidence": float(p[idx]), "probabilities": p.tolist(),
            "counts": counts.tolist(), "retina": retina.tolist(), "network_spikes_per_bin": totals}
