"""Fit the difficulty score's weights to what the simulator actually did.

Why this module is the important one:

spine/difficulty.py is a hand-built formula. Its weights were chosen by a person
with a stated rationale (see DEFAULT_WEIGHTS), and the file's own docstring says
the formula must not be built from the simulator it is supposed to predict, on
pain of circularity. What the file cannot do is tell you whether the rationale
survived contact with the data. Measured on 30k held-out scenarios, it did not:
the default weights rank scenarios worse than chance against the collision
outcome. That is a fact about the formula, and the response to it is not to
quietly delete the formula but to fit it.

So: the six structural components are treated as *features*, and the weights are
estimated by least squares against the measured risk index, subject to the
constraints that make the number mean what the docstring says it means --

    w_i >= 0        a component that makes a scenario safer is not allowed
    sum w_i = 1     the score stays a convex combination, so score in [0, 1]
                    before the 1..10 rescaling and a weight reads as "share of
                    the total difficulty that this term accounts for"

The fit is done on training seeds and evaluated on test seeds, so what is
reported is generalisation, not fit. The unfitted weights are kept in the repo
and both are reported, because the comparison between them is the finding.

There is one thing this module deliberately does NOT do: it does not fit against
the collision indicator alone by classifying, because collision is rare (11.6%)
and a classifier can score well by predicting the base rate. Least squares
against the continuous risk index, which is itself a measured quantity built
from TTC, braking margin and jerk, keeps the target informative on every row.
"""

from __future__ import annotations

import math

from . import difficulty
from . import dataset as D

ORDER = list(difficulty.COMPONENT_ORDER)


def fit_weights(C, y, w0=None, iters: int = 3000, lr: float = 0.5, l2: float = 1e-4,
                verbose: bool = False) -> dict:
    """Projected gradient descent on ||C w - y||^2 / n + l2 ||w||^2, projected onto
    the simplex {w >= 0, sum w = 1} after every step.

    Written out rather than handed to scipy.optimize so the constraint handling is
    visible: the projection is Duchi's simplex projection, which is exact and
    cheap, and there is no possibility of the optimiser returning a weight of
    -0.03 that the caller then has to decide what to do about.
    """
    import numpy as np

    C = np.asarray(C, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = C.shape
    w = np.asarray(w0 if w0 is not None else [1.0 / d] * d, dtype=np.float64)
    w = _project_simplex(w)
    step = lr
    prev = None
    for it in range(iters):
        grad = C.T @ (C @ w - y) / n + 2.0 * l2 * w
        w = _project_simplex(w - step * grad)
        if it % 250 == 0:
            loss = float(((C @ w - y) ** 2).mean() + l2 * (w ** 2).sum())
            if verbose:
                print(f"  it {it:5d} loss {loss:.6f}")
            if prev is not None and abs(prev - loss) < 1e-12:
                break
            prev = loss
        step *= 0.9995
    resid = float(((C @ w - y) ** 2).mean())
    return {"weights": {k: float(v) for k, v in zip(ORDER, w)},
            "w_vector": [float(v) for v in w], "mse": resid,
            "constraints": "w >= 0, sum w = 1",
            "constraints_hold": bool((w >= -1e-9).all() and abs(float(w.sum()) - 1.0) < 1e-6),
            "iters": iters, "l2": l2}


def _project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1}: sort, find the threshold,
    subtract and clip. The standard exact algorithm; no iteration needed."""
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(v) + 1)
    cond = u - (css - 1.0) / k > 0
    rho = int(k[cond][-1])
    theta = (css[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


def calibrated_score(C, model: dict):
    """Apply the fitted weights to a component matrix, returning 1..10 exactly as
    difficulty.score does, so calibrated and default scores are on one scale."""
    import numpy as np

    w = np.asarray(model["w_vector"], dtype=np.float64)
    total = np.asarray(C, dtype=np.float64) @ w
    return 1.0 + 9.0 * np.clip(total, 0.0, 1.0)


def compare(C_train, y_train, C_test, y_test, y_col_test, y_ttc_test=None) -> dict:
    """Default weights versus fitted weights on held-out seeds. This is the only
    place the two are compared, so the table in the README cannot drift from the
    computation that produced it."""
    import numpy as np

    default_w = np.asarray([difficulty.DEFAULT_WEIGHTS[k] for k in ORDER], dtype=np.float64)
    if abs(default_w.sum() - 1.0) > 1e-9:
        default_w = default_w / default_w.sum()
    model = fit_weights(C_train, y_train, w0=default_w)
    fitted_w = np.asarray(model["w_vector"])

    out = {
        "n_train": int(len(C_train)), "n_test": int(len(C_test)),
        "default_weights": {k: round(float(v), 4) for k, v in zip(ORDER, default_w)},
        "fitted_weights": {k: round(float(v), 4) for k, v in zip(ORDER, fitted_w)},
        "weight_l1_change": round(float(np.abs(fitted_w - default_w).sum()), 4),
        "constraints_hold": bool((fitted_w >= -1e-12).all() and abs(fitted_w.sum() - 1.0) < 1e-6),
    }
    for name, w in (("default", default_w), ("fitted", fitted_w)):
        total = np.clip(C_test @ w, 0.0, 1.0)
        s = 1.0 + 9.0 * total
        # Both quantities are on [0, 1] by construction -- the components are
        # clipped there and the risk index is built from clipped terms -- so this
        # is a plain mean squared error and not a rescaled one.
        out[name] = {
            "score_range": [round(float(s.min()), 4), round(float(s.max()), 4)],
            "mse_vs_risk_index": round(float(((total - y_test) ** 2).mean()), 6),
            "spearman_risk": _r(D.spearman(list(total), list(y_test))),
            "auc_collision": _r(D.metrics_pure(list(y_col_test), list(total))["auc"]),
        }
        if y_ttc_test is not None:
            out[name]["spearman_min_ttc"] = _r(D.spearman(list(total), list(y_ttc_test)))
    out["model"] = model
    return out


def _r(x):
    return None if x is None else round(float(x), 6)
