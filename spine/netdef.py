"""The predictor's architecture, as data.

Both the numpy reference model and the MUSA/CPU torch model are built from this
description, so "the same model on two backends" is a structural fact rather
than something the correctness test has to hope for. The weights are loaded from
the trained torch state dict; the MUSA kernel is handed the same numbers in the
same order.

Layout: interleaved Linear and SiLU, exactly what learn.train_mlp_torch builds.
Anything that changes the architecture has to change it here, which is the point.
"""

from __future__ import annotations

HIDDEN = (256, 256, 128)
ACTIVATION = "silu"


def describe(in_dim: int) -> dict:
    shapes = []
    prev = in_dim
    for h in HIDDEN:
        shapes.append(("linear", prev, h))
        shapes.append(("silu", h, h))
        prev = h
    shapes.append(("linear", prev, 1))
    return {"in_dim": in_dim, "hidden": list(HIDDEN), "activation": ACTIVATION,
            "layers": shapes, "n_layers": len(shapes)}


def state_to_arrays(state: dict, in_dim: int) -> list:
    """Flatten a torch state dict into a list of (kind, weight, bias) with weight
    as a list of lists in row-major order.

    The ordering comes from the *names* in the state dict (0.weight, 0.bias,
    2.weight, ...) rather than from dict iteration order, because dict order is a
    property of the serialiser and not of the model. Getting this wrong produces a
    model that runs, predicts, and is wrong -- the failure mode this function
    exists to make impossible.
    """
    import re

    idx = {}
    for k, v in state.items():
        m = re.match(r"^(\d+)\.(weight|bias)$", k)
        if not m:
            continue
        idx.setdefault(int(m.group(1)), {})[m.group(2)] = v
    arrays = []
    # The indices are NOT contiguous, and that is expected: nn.Sequential numbers
    # every child, and SiLU has no parameters, so a (Linear, SiLU) stack yields
    # 0, 2, 4, ... The sort is what matters -- the arrays come out in the order the
    # layers are applied, which is the order the kernel walks them in.
    order = sorted(idx)
    for i, k in enumerate(order):
        blk = idx[k]
        if "weight" not in blk or "bias" not in blk:
            raise ValueError(f"layer {k} is missing a weight or a bias: {sorted(blk)}")
        w, b = blk["weight"], blk["bias"]
        rows = [[float(x) for x in row] for row in w.tolist()]
        arrays.append({"layer": i, "out_features": int(b.shape[0]),
                       "in_features": int(w.shape[1]),
                       "weight": rows, "bias": [float(x) for x in b.tolist()]})
    return arrays


def forward_arrays(arrays: list, x: list, activation: str = ACTIVATION) -> float:
    """The reference implementation: plain Python, no numpy, no torch. This is
    what the kernel is checked against, and it is written from the definition of a
    matrix multiply rather than from a library call, so a library bug cannot be
    shared by both sides."""
    v = list(x)
    n_linear = 0
    for a in arrays:
        w, b = a["weight"], a["bias"]
        out = []
        for r in range(a["out_features"]):
            acc = b[r]
            wr = w[r]
            for c in range(a["in_features"]):
                acc += wr[c] * v[c]
            out.append(acc)
        v = out
        n_linear += 1
        if len(v) > 1:
            if activation == "silu":
                v = [t / (1.0 + _exp(-t)) for t in v]
            else:
                v = [max(0.0, t) for t in v]
    return float(v[0])


def _exp(x: float) -> float:
    import math

    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0e26
    return math.exp(x)


def sigmoid(x: float) -> float:
    import math

    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))
