"""Does the MUSA kernel compute the same function as the model it was exported
from? Three implementations, one input, and the differences between all three.

    torch      the trained model, on the device it was trained on
    numpy      the same weights, multiplied out with numpy
    kernel     the MUSA kernel, through the driver, sharing no code with either

The point of three rather than two: if the kernel and torch agreed and both
disagreed with numpy, that would be a bug in numpy -- and if only two were
compared, whichever one was wrong would be invisible. The numpy path is also the
one written directly from the definition of a matrix multiply, so it is the
reference the others are held to.

    python musa/parity.py <state_dir> <model.pt> <test.npz> [<predict.exe>]
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from spine import netdef  # noqa: E402


def load_arrays(state_dir: str) -> list:
    shapes = []
    with open(os.path.join(state_dir, "shapes.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o, i = (int(x) for x in line.split())
                shapes.append((o, i))
    with open(os.path.join(state_dir, "weights.f32"), "rb") as f:
        w = np.frombuffer(f.read(), dtype="<f4")
    with open(os.path.join(state_dir, "biases.f32"), "rb") as f:
        b = np.frombuffer(f.read(), dtype="<f4")
    arrays, wo, bo = [], 0, 0
    for o, i in shapes:
        arrays.append({"layer": len(arrays), "out_features": o, "in_features": i,
                       "weight": w[wo:wo + o * i].reshape(o, i).astype(np.float64),
                       "bias": b[bo:bo + o].astype(np.float64)})
        wo += o * i
        bo += o
    return arrays


def forward_numpy(arrays: list, X: np.ndarray) -> np.ndarray:
    v = np.asarray(X, dtype=np.float64)
    for k, a in enumerate(arrays):
        v = v @ a["weight"].T + a["bias"]
        if k != len(arrays) - 1:
            v = v / (1.0 + np.exp(-np.clip(v, -60, 60)))
    return v.reshape(-1)


def forward_torch(model_path: str, X: np.ndarray, n_layers: int = None) -> np.ndarray:
    """The trained network, rebuilt from the architecture description and loaded
    with the exported state dict. Rebuilding rather than unpickling a saved nn.Module
    means this path depends only on the same shapes the kernel does -- so if the two
    disagree, it is about the arithmetic and not about a pickled object."""
    import torch.nn as nn

    payload = torch.load(model_path, map_location="cpu")
    if not isinstance(payload, dict):
        payload = {"state_dict": payload}
    state = payload.get("state_dict", payload)
    lin = [(k, v) for k, v in state.items() if k.endswith(".weight") and v.dim() == 2]
    lin.sort(key=lambda kv: int(kv[0].split(".")[0]))
    in_dim = int(lin[0][1].shape[1])
    layers, prev = [], in_dim
    for _, w in lin:
        out = int(w.shape[0])
        layers += [nn.Linear(prev, out), nn.SiLU()]
        prev = out
    layers = layers[:-1]  # the trailing SiLU belongs to no layer of the stack
    net = nn.Sequential(*layers)
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(np.asarray(X, dtype=np.float32))).reshape(-1).numpy()


def forward_kernel(state_dir: str, X: np.ndarray, exe: str) -> np.ndarray:
    n, d = X.shape
    xp = os.path.join(state_dir, "_x.f32")
    op = os.path.join(state_dir, "_y.f32")
    with open(xp, "wb") as f:
        f.write(np.asarray(X, dtype="<f4").tobytes())
    r = subprocess.run([exe, state_dir, xp, str(n), str(d), op],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"driver failed rc={r.returncode}: {r.stderr[-800:]}")
    with open(op, "rb") as f:
        return np.frombuffer(f.read(), dtype="<f4").astype(np.float64)


def main():
    state_dir, model_path, test_npz = sys.argv[1], sys.argv[2], sys.argv[3]
    exe = sys.argv[4] if len(sys.argv) > 4 else os.path.join(BASE, "musa", "predict")
    z = np.load(test_npz)
    Xraw = z["X"][:500].astype(np.float32)
    mean = np.load(os.path.join(state_dir, "mean.npy"))
    std = np.load(os.path.join(state_dir, "std.npy"))
    X = np.clip((Xraw - mean) / std, -8, 8).astype(np.float32)

    arrays = load_arrays(state_dir)
    ref = forward_numpy(arrays, X)
    got_t = forward_torch(model_path, X)
    got_k = forward_kernel(state_dir, X, exe)

    def report(name, a, b):
        d = np.abs(np.asarray(a) - np.asarray(b))
        print(f"  {name:22s} max_abs_diff {d.max():.3e}  mean_abs_diff {d.mean():.3e}")
        return float(d.max())

    print("parity, 500 held-out rows, identical standardised inputs:")
    m1 = report("numpy vs torch", ref, got_t)
    m2 = report("numpy vs kernel(musa)", ref, got_k)
    m3 = report("torch vs kernel(musa)", got_t, got_k)
    worst = max(m1, m2, m3)
    # 2e-4 on a logit is far below anything that changes a ranking at the top of
    # the ordering, and it is measured rather than asserted: the device silu uses
    # __expf, a fast-math exponential, so exact equality is not the right bar.
    print(f"  worst {worst:.3e}  tolerance 2.0e-04  -> {'PASS' if worst < 2e-4 else 'FAIL'}")

    # and does the ranking survive? an AUC computed on the kernel's logits against
    # the same labels must match the one from torch's, because a predictor is used
    # for its ordering
    if "ycol" in z:
        y = z["ycol"][:500]
        def auc(s):
            from spine.dataset import metrics_pure
            return metrics_pure([float(v) for v in y], [float(v) for v in s])["auc"]
        print(f"  auc: numpy {auc(ref):.6f}  torch {auc(got_t):.6f}  kernel {auc(got_k):.6f}")
    return 0 if worst < 2e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
