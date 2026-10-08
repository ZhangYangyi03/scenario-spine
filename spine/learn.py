"""Predict the *outcome* of a scenario from its *pre-execution* features.

Why this exists, rather than "we trained a model because the task said GPU":

The physics score in spine/difficulty.py is a hand-built formula. Its weights
were chosen by a person, and nothing in that file can tell you whether the
formula is a good predictor of what the simulator actually does -- it only says
the formula is internally consistent. A learned model supplies the missing
comparison. It sees exactly the same information the formula sees (pre-execution
features only), is trained against the simulated outcome, and the two are then
scored against each other on scenarios drawn from seed ranges neither of them
saw. Three outcomes are all informative:

    * the learned model clearly wins -> the hand-set weights are leaving signal
      on the table, and the honest move is to say so and report both numbers;
    * they agree closely -> the formula is a good cheap surrogate, which is worth
      knowing because it needs no training and no data;
    * the formula wins -> the hand-built terms encode something the feature vector
      does not, and the feature vector is what should be fixed.

Column names are attached to every score, because a number produced without a
name is a number nobody can reproduce.
"""

from __future__ import annotations

import json
import math
import os

from . import dataset as D
from . import features as F

#: Which target a model is trained on. `collision` is binary and gets AUC;
#: `risk_index` is continuous and gets Spearman. Both are reported for both
#: targets, so neither metric can be quoted alone as if it were the result.
TARGETS = ("collision", "risk_index")


def _sigmoid(z):
    if z < -60.0:
        return 0.0
    if z > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------- numpy model --

def train_logistic(X, y, epochs: int = 400, lr: float = 0.35, l2: float = 1e-4,
                   batch: int = 0):
    """Plain logistic regression by full-batch gradient descent, in numpy.

    The reference model. It exists so the numbers have a floor that anybody can
    reproduce without a GPU, a framework, or a random seed: if the neural network
    cannot beat this, the neural network is not the finding.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = float(np.log(max(1e-6, y.mean() / max(1e-6, 1.0 - y.mean()))))
    for ep in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        g = p - y
        gw = X.T @ g / n + l2 * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return {"kind": "logistic-numpy", "w": w.tolist(), "b": b,
            "feature_names": F.feature_names(), "epochs": epochs, "l2": l2}


def predict_logistic(model, X):
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    z = X @ np.asarray(model["w"]) + model["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ------------------------------------------------------------------ the GPU --

def torch_available():
    try:
        import torch
        dev = "cpu"
        try:
            import torch_musa  # noqa: F401
            if torch.musa.is_available():
                dev = "musa"
        except Exception:
            pass
        if dev == "cpu" and torch.cuda.is_available():
            dev = "cuda"
        return True, dev, torch.__version__
    except Exception as e:
        return False, None, str(e)


def train_mlp_torch(X, y, target: str = "collision", hidden=(256, 256, 128), epochs: int = 60,
                    lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 512,
                    device: str = None, seed: int = 0, verbose: bool = False):
    """A small MLP trained with AdamW, on whatever accelerator is present.

    Deliberately small and deliberately regularised: the dataset is tens of
    thousands of rows of 40-odd features, and on that scale a wide network
    memorises the corpus. AdamW with weight decay, early stopping on a held-out
    slice of the training seeds, and a fixed seed -- so the reported number is
    reproducible and not the best of fifty runs.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    if device is None:
        device = torch_available()[1]
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = np.asarray(X, dtype=np.float32)
    yv = np.asarray(y, dtype=np.float32)
    binary = target == "collision"
    out_dim = 1 if binary else 1

    # internal validation split, by row here because the caller has already split
    # by seed into train/test; this slice only drives early stopping
    n = len(X)
    idx = np.random.permutation(n)
    nval = max(64, n // 10)
    vi, ti = idx[:nval], idx[nval:]

    xt = torch.tensor(X[ti]).to(device)
    yt = torch.tensor(yv[ti]).reshape(-1, 1).to(device)
    xv = torch.tensor(X[vi]).to(device)
    yv_t = torch.tensor(yv[vi]).reshape(-1, 1).to(device)

    layers, prev = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    net = nn.Sequential(*layers).to(device)

    # The regression target and the classification target want different losses,
    # and using the wrong one is a silent quality loss rather than an error.
    lossf = nn.BCEWithLogitsLoss() if binary else nn.SmoothL1Loss(beta=0.05)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best = {"val": float("inf"), "state": None, "epoch": -1}
    m = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(m, device=device)
        for i in range(0, m, batch):
            sel = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            out = net(xt[sel])
            loss = lossf(out, yt[sel])
            loss.backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            vout = net(xv)
            vloss = float(lossf(vout, yv_t).item())
        if vloss < best["val"]:
            best = {"val": vloss, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d} val_loss {vloss:.5f} best {best['val']:.5f}")
    if best["state"]:
        net.load_state_dict(best["state"])
    return {"kind": "mlp-torch", "net": net, "device": device, "target": target,
            "hidden": list(hidden), "epochs": epochs, "seed": seed,
            "val_loss": best["val"], "val_epoch": best["epoch"],
            "feature_names": F.feature_names()}


def predict_torch(model, X):
    import numpy as np
    import torch

    net = model["net"]
    net.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype=np.float32)).to(model["device"])
        out = net(t).reshape(-1).float().cpu().numpy()
    if model["target"] == "collision":
        return 1.0 / (1.0 + np.exp(-np.clip(out, -60, 60)))
    return out


# ------------------------------------------------------------- the comparison --

def evaluate_corpus(rows: list, test_from: int, test_to: int, model: dict = None,
                    physics_key: str = "physics", verbose: bool = False) -> dict:
    """Score a learned model and the analytic physics score on test *seeds*, and
    report both against the measured outcome. This function is the only place the
    comparison is made, so the README's table cannot drift from the computation.
    """
    train, test = D.split_by_seed(rows, test_from, test_to)
    if len(test) < 50 or len(train) < 200:
        raise ValueError("not enough rows either side of the split: %d train / %d test"
                         % (len(train), len(test)))
    Xtr = [r["x"] for r in train]
    Xte = [r["x"] for r in test]
    out = {"n_train": len(train), "n_test": len(test),
           "test_seed_range": [test_from, test_to],
           "physics_scale": "1..10 (difficulty.score)"}

    # the analytic score, as a normalised risk predictor
    phys_te = [(r[physics_key] - 1.0) / 9.0 for r in test]
    y_col = [r["y"]["collision"] for r in test]
    y_risk = [r["y"]["risk_index"] for r in test]
    out["physics"] = {
        "auc_collision": D.metrics_pure(y_col, phys_te)["auc"],
        "spearman_risk": D.spearman(phys_te, y_risk),
        "spearman_ttc": D.spearman(phys_te, [-r["y"]["min_ttc_s"] for r in test]),
    }

    if model is not None:
        pred = (predict_torch(model, Xte) if model.get("kind") == "mlp-torch"
                else predict_logistic(model, Xte))
        pred = [float(v) for v in pred]
        out["learned"] = {
            "kind": model["kind"], "device": model.get("device"), "target": model.get("target"),
            "auc_collision": D.metrics_pure(y_col, pred)["auc"],
            "spearman_risk": D.spearman(pred, y_risk),
            "spearman_ttc": D.spearman(pred, [-r["y"]["min_ttc_s"] for r in test]),
            "mae_vs_risk": round(sum(abs(a - b) for a, b in zip(pred, y_risk)) / len(pred), 6),
        }
    return out


def save(model: dict, path: str) -> str:
    import numpy as np

    if model.get("kind") == "mlp-torch":
        import torch

        torch.save({k: v for k, v in model.items() if k != "net"}, path)
        torch.save(model["net"].state_dict(), path + ".state")
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(model, f, indent=1)
    return path
