"""Flatten a trained model into the layout the MUSA kernel reads.

The scaler goes with the weights, always. The kernel is fed standardised
features, so a model exported without its mean and standard deviation is a model
that silently mispredicts on any input drawn from a different distribution -- and
nothing about running it would look wrong. Writing them together makes that
mistake impossible to make by omission.

    python tools/export_state.py <state.pt> <out_dir> [mean.npy std.npy]
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from spine import netdef  # noqa: E402


def export(state_path: str, out_dir: str, mean=None, std=None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    arrays = netdef.state_to_arrays(state, 0)
    if not arrays:
        raise ValueError(f"no linear layers found in {state_path}")

    shapes, wflat, bflat = [], [], []
    for a in arrays:
        shapes.append(f"{a['out_features']} {a['in_features']}")
        for row in a["weight"]:
            wflat.extend(row)
        bflat.extend(a["bias"])

    with open(os.path.join(out_dir, "shapes.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(shapes) + "\n")
    with open(os.path.join(out_dir, "weights.f32"), "wb") as f:
        f.write(struct.pack("<%df" % len(wflat), *wflat))
    with open(os.path.join(out_dir, "biases.f32"), "wb") as f:
        f.write(struct.pack("<%df" % len(bflat), *bflat))

    meta = {"layers": shapes, "n_params": len(wflat) + len(bflat),
            "architecture": netdef.describe(arrays[0]["in_features"])}
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        np.save(os.path.join(out_dir, "mean.npy"), mean)
        np.save(os.path.join(out_dir, "std.npy"), std)
        meta["has_scaler"] = True
        meta["in_dim"] = int(mean.shape[0])
    else:
        meta["has_scaler"] = False
        meta["in_dim"] = int(arrays[0]["in_features"])
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    return meta


if __name__ == "__main__":
    mp, od = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    m = export(mp, od, *(np.load(a) for a in args))
    print(json.dumps(m, indent=1))
