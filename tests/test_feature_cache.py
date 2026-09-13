from types import SimpleNamespace
import numpy as np
import pytest
from flyocr.brain.features import extract


class FakeBrain:
    model_id = "fixture"
    config = SimpleNamespace(bins=2)
    def encode(self, image, neurons):
        return np.full((2, len(neurons)), int(image[0,0]), np.int32), None, None


def test_resume_checksums_and_image_identity(tmp_path):
    corpus = tmp_path/"corpus"; corpus.mkdir()
    np.savez(corpus/"train.npz", images=np.array([[[3]], [[8]]], np.uint8))
    x, meta = extract(FakeBrain(), corpus, "train", [1], tmp_path/"cache", shard_size=1)
    assert x.tolist() == [[3,3],[8,8]]
    shard = tmp_path/"cache"/meta["cache_id"]/"000000.npz"
    shard.write_bytes(b"partial output")
    with pytest.raises(ValueError, match="checksum"):
        extract(FakeBrain(), corpus, "train", [1], tmp_path/"cache", shard_size=1)
    np.savez(corpus/"train.npz", images=np.array([[[9]]], np.uint8))
    y, new = extract(FakeBrain(), corpus, "train", [1], tmp_path/"cache")
    assert new["cache_id"] != meta["cache_id"] and y.tolist() == [[9,9]]
