"""Checked resumable downloads; partial files never become accepted inputs."""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from flyocr.common import digest_file


def fetch(url: str, path: Path, sha256: str, expected_bytes: int | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest_file(path) != sha256:
            raise ValueError(f"Checksum mismatch for existing {path.name}; move it aside explicitly.")
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_bytes is None or offset != expected_bytes:
        headers = {"User-Agent": "flyocr-research/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
            resumed = offset > 0 and response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Invalid resume response")
            with partial.open("ab" if resumed else "wb") as stream:
                shutil.copyfileobj(response, stream, length=8 << 20)
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError(f"Incomplete download: {path.name}")
    if digest_file(partial) != sha256:
        partial.unlink()  # Only our invalid temporary download, never a user source.
        raise ValueError(f"Downloaded checksum mismatch: {path.name}")
    os.replace(partial, path)
    return path
