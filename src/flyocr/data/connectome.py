"""MaleCNS import adapted from DOOMFLY (MIT; see THIRD_PARTY_NOTICES.md)."""
from __future__ import annotations

import gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from flyocr.common import read_json, save_json, digest_file, identity


def exact_ids(values):
    a = np.asarray(values)
    if a.dtype.kind == "f":
        raise ValueError("Biological IDs cannot be floating-point values")
    if a.dtype.kind in "iu":
        if np.any(a < 0):
            raise ValueError("Negative biological ID")
        return a.astype(np.uint64)
    if any(not str(v).isascii() or not str(v).isdecimal() for v in a):
        raise ValueError("Invalid biological ID")
    return a.astype(np.uint64)


def index_edges(ids, pre, post, counts):
    if not len(ids) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("IDs must be sorted and unique")
    pre, post, counts = exact_ids(pre), exact_ids(post), np.asarray(counts)
    if not len(pre) == len(post) == len(counts):
        raise ValueError("Inconsistent edge columns")
    if not np.all(np.isfinite(counts)) or np.any(counts < 1) or np.any(counts != np.floor(counts)) or np.any(counts > 2**32 - 1):
        raise ValueError("Invalid contact count")
    i, j = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    keep = (i < len(ids)) & (j < len(ids))
    keep &= ids[np.minimum(i, len(ids)-1)] == pre
    keep &= ids[np.minimum(j, len(ids)-1)] == post
    return i[keep].astype(np.int32), j[keep].astype(np.int32), counts[keep].astype(np.uint32), keep


def signs_for(values):
    signs, unknown = [], []
    for v in values:
        tokens = set(str(v).lower().split(","))
        known = ({1} if "acetylcholine" in tokens else set()) | ({-1} if tokens & {"gaba", "glutamate", "histamine"} else set())
        signs.append(next(iter(known)) if len(known) == 1 else 1)
        unknown.append(len(known) != 1)
    return np.array(signs, np.int8), np.array(unknown, bool)


def retinal_mapping(nodes, pre, post, counts):
    receptor = nodes.type.eq("R1-R6").to_numpy()
    anchor = nodes.type.isin(["L1", "L2", "L3"]).to_numpy() & nodes.assignedOlHex1.notna().to_numpy() & nodes.assignedOlHex2.notna().to_numpy()
    keep = receptor[pre] & anchor[post]
    columns = {}
    h1, h2 = nodes.assignedOlHex1.to_numpy(), nodes.assignedOlHex2.to_numpy()
    for i, j, n in zip(pre[keep], post[keep], counts[keep]):
        key = (float(h1[j]), float(h2[j]))
        c = columns.setdefault(int(i), {})
        c[key] = c.get(key, 0) + int(n)
    indices, xy, confidence = [], [], []
    for i, c in sorted(columns.items()):
        h = max(c, key=c.get)
        indices.append(i)
        xy.append((h[0] - .5*h[1], np.sqrt(3)/2*h[1]))
        confidence.append(c[h] / sum(c.values()))
    indices = np.array(indices, np.int32)
    xy = np.array(xy)
    uv = np.full_like(xy, np.nan)
    side = nodes.rootSide.to_numpy()[indices]
    for s in ["L", "R"]:
        mask = side == s
        z = xy[mask]
        if not len(z) or np.any(np.ptp(z, axis=0) == 0):
            raise ValueError("Retinal map has no spatial extent")
        z = (z-z.min(axis=0))/np.ptp(z, axis=0)
        uv[mask, 0] = .60*z[:, 0] if s == "L" else .40+.60*(1-z[:, 0])
        uv[mask, 1] = 1-z[:, 1]
    if not np.isfinite(uv).all():
        raise ValueError("Unmapped retinal side")
    return indices, uv.astype(np.float32), np.array(confidence, np.float32)


def prepare(source: Path, output: Path, manifest: Path):
    source, output = Path(source), Path(output)
    config = read_json(manifest)
    output.mkdir(parents=True, exist_ok=True)
    for name, info in config["files"].items():
        if digest_file(source/name) != info["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    annotations = feather.read_table(source/"annotations.feather").to_pandas()
    if annotations.bodyId.duplicated().any():
        raise ValueError("Duplicate annotation ID")
    keep = annotations.superclass.notna() & annotations.superclass.astype(str).ne("") & ~annotations.status.eq("Glia")
    nodes = annotations.loc[keep].sort_values("bodyId").reset_index(drop=True)
    ids = exact_ids(nodes.bodyId)
    nt = feather.read_table(source/"neurotransmitters.feather").to_pandas().set_index("body")
    if not nt.index.is_unique:
        raise ValueError("Duplicate neurotransmitter ID")
    transmitter = nodes.bodyId.map(nt.consensus_nt).fillna("missing")
    signs, unknown = signs_for(transmitter)
    schema = pa.schema([("pre", pa.int32()), ("post", pa.int32()), ("count", pa.uint32())])
    stats = dict(source_edges=0, edges=0, contacts=0, source_contacts=0, self_edges=0, weight_one_edges=0)
    reader = ipc.open_file(pa.memory_map(str(source/"edges.feather"), "r"))
    with pa.OSFile(str(output/"edges.arrow.partial"), "wb") as sink, ipc.new_file(sink, schema) as writer:
        for batch_number in range(reader.num_record_batches):
            batch = reader.get_batch(batch_number)
            p, q, c = [batch.column(batch.schema.get_field_index(k)).to_numpy() for k in ("body_pre", "body_post", "weight")]
            i, j, n, _ = index_edges(ids, p, q, c)
            stats["source_edges"] += len(p)
            stats["edges"] += len(i)
            stats["contacts"] += int(n.sum(dtype=np.uint64))
            stats["source_contacts"] += int(c.sum(dtype=np.uint64))
            stats["self_edges"] += int((i == j).sum())
            stats["weight_one_edges"] += int((n == 1).sum())
            writer.write_batch(pa.record_batch([pa.array(i), pa.array(j), pa.array(n)], schema=schema))
    (output/"edges.arrow.partial").replace(output/"edges.arrow")
    if {"nodes": len(ids), "edges": stats["edges"], "contacts": stats["contacts"]} != config["expected"]:
        raise ValueError(f"Unexpected graph counts: {len(ids)}, {stats}")
    print(f"Retained {len(ids):,} neurons; {stats['edges']:,} edges", flush=True)
    edges = ipc.open_file(pa.memory_map(str(output/"edges.arrow"), "r")).read_all()
    pre, post, count = [edges.column(k).to_numpy() for k in ("pre", "post", "count")]
    retina, uv, confidence = retinal_mapping(nodes, pre, post, count)
    order = np.argsort(pre, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(pre, minlength=len(ids)))].astype(np.int64)
    weight = (count[order].astype(np.float32)*signs[pre[order]]*.275).astype(np.float32)
    lamina = np.flatnonzero(nodes.type.isin(["L1", "L2", "L3", "L5"])).astype(np.int32)
    candidates = np.flatnonzero(nodes.superclass.isin(["ol_intrinsic", "visual_projection"])).astype(np.int32)
    candidates = np.setdiff1d(candidates, np.r_[retina, lamina]).astype(np.int32)
    arrays = dict(ptr=ptr, post=post[order].astype(np.int32), weight=weight,
                  ids=ids, retina=retina, uv=uv, retina_confidence=confidence,
                  lamina=lamina, candidates=candidates, signs=signs,
                  cell_type=nodes.type.fillna("").to_numpy(dtype="U48"),
                  superclass=nodes.superclass.fillna("").to_numpy(dtype="U32"))
    for key, value in arrays.items():
        np.save(output/(key+".npy"), value, allow_pickle=False)
    hashes = {key: digest_file(output/(key+".npy")) for key in arrays}
    report = {"dataset": "MaleCNS v1.0", "nodes": len(ids), **stats,
              "excluded_objects": int((~keep).sum()), "excluded_edges": stats["source_edges"]-stats["edges"],
              "retinal_inputs": len(retina), "tonic_inputs": len(lamina), "candidate_features": len(candidates),
              "unknown_sign_cells": int(unknown.sum()), "source": config,
              "array_hashes": hashes, "graph_id": identity(hashes),
              "model_assumptions": "LIF proxy, contact-count weights, inferred eye projection; not validated fly vision"}
    save_json(output/"manifest.json", report)
    # A separate, display-only sidecar leaves graph/model identities unchanged.
    from flyocr.vision.atlas import make_atlas
    save_json(output/"retinal-atlas.json", make_atlas(ids, retina, uv,
              nodes.rootSide.to_numpy()[retina], report["graph_id"]))
    del edges, pre, post, count, order, arrays, nt, annotations
    gc.collect()
    return report
