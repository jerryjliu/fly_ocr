from __future__ import annotations
import ctypes as C
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from flyocr.common import digest_file, save_json, read_json


def build(cache=Path(".cache/native")):
    source = Path(__file__).with_name("kernel.cpp")
    key = digest_file(source)[:20] + "-" + platform.machine()
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/(key+(".dylib" if sys.platform == "darwin" else ".so"))
    manifest = path.with_suffix(path.suffix+".json")
    if path.exists() and manifest.exists() and read_json(manifest)["binary_sha256"] == digest_file(path):
        return path
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("Install a C++ compiler (Xcode Command Line Tools on macOS).")
    command = [compiler, "-O3", "-std=c++17", "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared", str(source), "-o", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    save_json(manifest, {"source_sha256": digest_file(source), "binary_sha256": digest_file(path), "architecture": platform.machine(), "flags": command[1:5]})
    return path


class Native:
    def __init__(self, cache=Path(".cache/native")):
        self.library = C.CDLL(str(build(cache).resolve()))
        self.function = self.library.neural_advance
        self.function.argtypes = [C.c_int]+[C.c_void_p]*11+[C.c_int, C.c_float]+[C.c_void_p]*5
        self.function.restype = None

    def advance(self, n, graph, state, steps):
        arrays = [graph[k] for k in ("ptr", "post", "weight")]+[state[k] for k in ("v", "g", "refractory", "drive", "previous_drive", "queue", "queue_count", "clock")]
        tail = [state[k] for k in ("counts", "active", "flags", "nactive", "last")]
        self.function(n, *[a.ctypes.data for a in arrays], steps, .1, *[a.ctypes.data for a in tail])
