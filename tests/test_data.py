from pathlib import Path
import hashlib
import io
import numpy as np
import pytest
from flyocr.data.connectome import exact_ids, index_edges
from flyocr.data.download import fetch
from flyocr.common import read_json


def test_id_integrity_and_edge_accounting():
    ids = exact_ids([2, 5, 9007199254740993])
    assert ids[-1] == 9007199254740993
    i, j, n, keep = index_edges(ids, [2, 5, 7], [5, 5, 2], [1, 9, 2])
    assert i.tolist() == [0, 1] and j.tolist() == [1, 1]
    assert n.tolist() == [1, 9] and keep.tolist() == [True, True, False]
    for bad in [[1.0], [-1], ["12.3"]]:
        with pytest.raises(ValueError): exact_ids(bad)
    with pytest.raises(ValueError): index_edges(np.array([5, 2]), [2], [5], [1])
    with pytest.raises(ValueError): index_edges(ids, [2], [5], [1.5])


def test_checked_download_resume_and_failure(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}
    p = tmp_path/"source.bin"
    p.with_suffix(".bin.partial").write_bytes(b"abc")
    def request(req, timeout):
        assert req.headers["Range"] == "bytes=3-"
        return Response(b"def")
    monkeypatch.setattr("urllib.request.urlopen", request)
    fetch("https://example.org/file", p, hashlib.sha256(b"abcdef").hexdigest(), 6)
    assert p.read_bytes() == b"abcdef"
    assert not p.with_suffix(".bin.partial").exists()
    with pytest.raises(ValueError): fetch("https://example.org/file", p, "bad", 6)


def test_font_split_isolation():
    path = Path("assets/fonts/manifest.json")
    if not path.exists(): pytest.skip("Font assets not downloaded")
    fonts = read_json(path)["fonts"]
    families = {s: {f["family"] for f in fonts if f["split"] == s} for s in ("train", "validation", "test")}
    assert not families["train"] & families["validation"]
    assert not families["train"] & families["test"]
    assert not families["test"] & families["validation"]
