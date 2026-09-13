from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from flyocr.brain.native import Native
from flyocr.common import read_json, digest_file, identity
from flyocr.vision.retina import sample_retina


@dataclass(frozen=True)
class StimulusConfig:
    sensory_gain: float = 30.0
    lamina_drive: float = 12.0
    warmup_ms: float = 50.0
    duration_ms: float = 100.0
    bins: int = 4
    half_saturation: float = .02
    invert: bool = False

    def validate(self):
        values = [self.sensory_gain, self.lamina_drive, self.warmup_ms, self.duration_ms, self.half_saturation]
        if not all(math.isfinite(v) for v in values) or self.half_saturation <= 0 or not isinstance(self.bins, int) or self.bins < 1 or self.bins > 100:
            raise ValueError("Invalid stimulus configuration")
        if self.sensory_gain < 0 or self.sensory_gain > 100 or self.lamina_drive < 0 or self.lamina_drive > 100 or self.warmup_ms < 0 or self.duration_ms <= 0:
            raise ValueError("Unsafe or invalid stimulus bounds")
        if not np.isclose(self.duration_ms/self.bins/.1, round(self.duration_ms/self.bins/.1)):
            raise ValueError("Temporal bins must align to 0.1 ms grid")


def validate_graph(a):
    required = {"ptr": np.int64, "post": np.int32, "weight": np.float32}
    for key, dtype in required.items():
        if a[key].ndim != 1 or a[key].dtype != dtype or not a[key].flags.c_contiguous:
            raise ValueError(f"Invalid native graph array {key}")
    n = len(a["ptr"])-1
    if n < 1 or a["ptr"][0] != 0 or a["ptr"][-1] != len(a["post"]) or np.any(np.diff(a["ptr"]) < 0) or len(a["post"]) != len(a["weight"]):
        raise ValueError("Invalid CSR structure")
    if np.any(a["post"] < 0) or np.any(a["post"] >= n) or not np.isfinite(a["weight"]).all():
        raise ValueError("Invalid synaptic target or weight")
    return n


class Brain:
    def __init__(self, graph, config=None, verify=True, retina_uv=None):
        self.config = config or StimulusConfig()
        self.config.validate()
        self.manifest = {}
        if isinstance(graph, (str, Path)):
            path = Path(graph)
            self.manifest = read_json(path/"manifest.json")
            if verify:
                for key, expected in self.manifest["array_hashes"].items():
                    if digest_file(path/(key+".npy")) != expected:
                        raise ValueError(f"Graph array changed: {key}")
            self.graph = {key: np.load(path/(key+".npy"), mmap_mode="r", allow_pickle=False) for key in self.manifest["array_hashes"]}
        else:
            self.graph = graph
        self.retina_uv_id = None
        if retina_uv is not None:
            calibrated = np.asarray(retina_uv, np.float32)
            self.retina_uv_id = identity({"retina_uv": calibrated.tolist()})
            self.graph = {**self.graph, "uv": calibrated.copy()}
        self.n = validate_graph(self.graph)
        self.native = Native()
        self.reset()
        self.canonical = None
        if "retina" in self.graph:
            self._validate_sensory()
            if self.config.warmup_ms:
                blank = np.ones(len(self.graph["retina"]), np.float32)
                self.present_samples(blank, self.config.warmup_ms)
            self.canonical = self.snapshot()

    @property
    def model_id(self):
        details = {"graph": self.manifest.get("graph_id", "fixture"), "stimulus": asdict(self.config), "kernel": digest_file(Path(__file__).with_name("kernel.cpp"))}
        if self.retina_uv_id is not None:
            details["input_projection"] = self.retina_uv_id
        return identity(details)

    def _validate_sensory(self):
        for key in ("retina", "lamina"):
            ids = self.graph[key]
            if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.n):
                raise ValueError("Invalid sensory index")
        uv = self.graph["uv"]
        if uv.shape != (len(self.graph["retina"]), 2) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1):
            raise ValueError("Invalid eye coordinates")

    def reset(self):
        n = self.n
        self.state = {
            "v": np.full(n, -52, np.float32), "g": np.zeros(n, np.float32),
            "refractory": np.zeros(n, np.int16), "drive": np.zeros(n, np.float32),
            "previous_drive": np.zeros(n, np.float32), "queue": np.zeros((19, n), np.int32),
            "queue_count": np.zeros(19, np.int32), "clock": np.zeros(1, np.int64),
            "counts": np.zeros(n, np.int32), "active": np.arange(n, dtype=np.int32),
            "flags": np.ones(n, np.uint8), "nactive": np.array([n], np.int32),
            "last": np.full(n, -1, np.int64),
            "luminance": np.zeros(len(self.graph.get("retina", [])), np.float32),
        }

    def snapshot(self):
        return {key: value.copy() for key, value in self.state.items()}

    def restore(self, snapshot):
        if set(snapshot) != set(self.state):
            raise ValueError("Incomplete neural state")
        for key, value in snapshot.items():
            if value.shape != self.state[key].shape or value.dtype != self.state[key].dtype:
                raise ValueError("Incompatible state")
        for key, value in snapshot.items():
            np.copyto(self.state[key], value)

    def advance_drive(self, drive, duration_ms):
        drive = np.asarray(drive, np.float32)
        if not math.isfinite(duration_ms): raise ValueError("Invalid neural duration")
        steps = round(duration_ms/.1)
        if drive.shape != (self.n,) or not np.isfinite(drive).all() or not math.isfinite(duration_ms) or steps < 1 or steps > 1000000 or not np.isclose(steps*.1, duration_ms):
            raise ValueError("Invalid neural step")
        self.state["drive"][:] = drive
        self.state["counts"].fill(0)
        self.native.advance(self.n, self.graph, self.state, steps)
        return self.state["counts"].copy()

    def present_samples(self, brightness, duration_ms):
        values = np.asarray(brightness, np.float32)
        if values.shape != self.state["luminance"].shape or not np.isfinite(values).all():
            raise ValueError("Invalid retinal brightness")
        values = np.clip(values, 0, 1)
        if self.config.invert:
            values = 1-values
        luminance = self.state["luminance"]
        luminance += (1-math.exp(-duration_ms/10))*(values-luminance)
        drive = np.zeros(self.n, np.float32)
        drive[self.graph["lamina"]] = self.config.lamina_drive
        drive[self.graph["retina"]] = self.config.sensory_gain*luminance/(self.config.half_saturation+luminance)
        return self.advance_drive(drive, duration_ms)

    def encode(self, image, neurons=None, reset=True):
        if reset:
            if self.canonical is None:
                raise ValueError("No sensory canonical state")
            self.restore(self.canonical)
        samples = sample_retina(image, self.graph["uv"])
        if neurons is None:
            neurons = self.graph["candidates"]
        neurons = np.asarray(neurons)
        if neurons.ndim != 1 or np.any(neurons < 0) or np.any(neurons >= self.n):
            raise ValueError("Invalid feature neurons")
        counts, totals = [], []
        for _ in range(self.config.bins):
            all_counts = self.present_samples(samples, self.config.duration_ms/self.config.bins)
            counts.append(all_counts[neurons])
            totals.append(int(all_counts.sum()))
        return np.asarray(counts, np.int32), samples, totals
