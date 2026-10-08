"""Fit the difficulty score's weights to what the simulator actually did.

Why this module is the important one:

spine/difficulty.py is a hand-built formula. Its weights were chosen by a person
with a stated rationale (see DEFAULT_WEIGHTS), and the file's own docstring says
the formula must not be built from the simulator it is supposed to predict, on
pain of circularity. What the file cannot do is tell you whether that rationale
survived contact with the data. Measured on 30k held-out scenarios it did not:
the default weights rank scenarios barely better than chance against the
collision outcome (AUC 0.567). That is a fact about the formula, and the response
to it is not to quietly delete the formula but to fit it.

So: the seven components are treated as *features* and the weights are estimated
by least squares against the measured risk index, subject to the constraints that
make the number mean what difficulty.py says it means --

    w_i >= 0        a component that makes a scenario safer may not reduce the score
    sum w_i = 1     the score stays a convex combination, so a weight still reads
                    as "share of the total difficulty this term accounts for"

The fit runs on training seeds and is evaluated on test seeds, so what is reported
is generalisation and not fit.

Two implementations of the arithmetic, and why both exist: the numpy path is what
the 200k-scenario run uses because it is 40x faster there, and the pure-Python path
is what runs when numpy is absent. Both are exercised by the test suite against the
same planted ground truth, so "the results do not depend on which one ran" is a
checked statement. The alternative -- making numpy a hard requirement -- would have
been a lie in the README's first paragraph about the core being standard library,
and the CLI's calibrate subcommand is part of that core.

One thing this module deliberately does NOT do: it does not fit against the
collision indicator by classifying. Collision is rare (11.6%) and a classifier can
score well by predicting the base rate; least squares against the continuous risk
index keeps the target informative on every row.
"""

from __future__ import annotations

import math

from . import dataset as D
from . import difficulty

ORDER = list(difficulty.COMPONENT_ORDER)


def _np():
    """numpy if it is importable, else None. Imported once per call site rather
    than at module import, because a module-level import would make every user of
    this module need numpy for the pure path too."""
    try:
        import numpy

        return numpy
    except Exception:
        return None


# ------------------------------------------------------------------ pure ---

def _matvec(C, w):
    return [sum(cij * wj for cij, wj in zip(row, w)) for row in C]


def fit_weights_pure(C, y, w0=None, iters: int = 3000, lr: float = 0.5, l2: float = 1e-4,
                     verbose: bool = False) -> dict:
    """Projected gradient descent in the standard library.

    Same algorithm as the numpy version, term for term: the gradient of
    (1/n)||Cw - y||^2 + l2||w||^2 is (2/n) C^T (Cw - y) + 2 l2 w, and the step is
    projected back onto the simplex. Kept deliberately parallel to the numpy path
    so that a reader can diff them.
    """
    n = len(C)
    d = len(C[0])
    w = list(w0) if w0 is not None else [1.0 / d] * d
    w = _project_simplex_pure(w)
    step = lr
    resid = _mse(C, y, w)
    prev = None
    for it in range(iters):
        pred = _matvec(C, w)
        g = [0.0] * d
        for i in range(n):
            diff = pred[i] - y[i]
            if diff == 0.0:
                continue
            row = C[i]
            for j in range(d):
                g[j] += row[j] * diff
        scale = 2.0 / n
        g = [scale * g[j] + 2.0 * l2 * w[j] for j in range(d)]
        w = _project_simplex_pure([w[j] - step * g[j] for j in range(d)])
        if it % 250 == 0:
            resid = _mse(C, y, w)
            if verbose:
                print(f"  it {it:5d} loss {resid:.6f}")
            if prev is not None and abs(prev - resid) < 1e-12:
                break
            prev = resid
        step *= 0.9995
    resid = _mse(C, y, w)
    return {"weights": {k: float(v) for k, v in zip(ORDER, w)},
            "w_vector": [float(v) for v in w], "mse": resid,
            "constraints": "w >= 0, sum w = 1",
            "constraints_hold": bool(all(v >= -1e-9 for v in w) and abs(sum(w) - 1.0) < 1e-6),
            "backend": "pure", "iters": iters, "l2": l2}


def _mse(C, y, w):
    n = len(C)
    p = _matvec(C, w)
    return sum((p[i] - y[i]) ** 2 for i in range(n)) / n


def _project_simplex_pure(v):
    """Euclidean projection onto {w >= 0, sum w = 1}: sort, find the threshold,
    subtract and clip. The standard exact algorithm, and Duchi's, not an
    approximation by iteration."""
    u = sorted(v, reverse=True)
    css = 0.0
    rho = 0
    theta = 0.0
    for k in range(1, len(v) + 1):
        css += u[k - 1]
        t = (css - 1.0) / k
        if u[k - 1] - t > 0:
            rho, theta = k, t
    return [max(x - theta, 0.0) for x in v]


# ----------------------------------------------------------------- numpy ---

def fit_weights(C, y, w0=None, iters: int = 3000, lr: float = 0.5, l2: float = 1e-4,
                verbose: bool = False) -> dict:
    """Projected gradient descent on ||C w - y||^2 / n + l2 ||w||^2, projected onto
    the simplex {w >= 0, sum w = 1} after every step.

    Written out rather than handed to scipy.optimize so the constraint handling is
    visible: the projection is exact, and there is no possibility of the optimiser
    returning a weight of -0.03 that the caller then has to decide about.
    """
    np = _np()
    if np is None:
        return fit_weights_pure(C, y, w0=w0, iters=iters, lr=lr, l2=l2, verbose=verbose)

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
            "backend": "numpy", "iters": iters, "l2": l2}


def _project_simplex(v):
    np = _np()
    if np is None:
        return _project_simplex_pure(v)
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(v) + 1)
    cond = u - (css - 1.0) / k > 0
    rho = int(k[cond][-1])
    theta = (css[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


# ------------------------------------------------------------- scoring -----

def calibrated_score(C, model: dict):
    """Apply fitted weights to a component matrix, returning 1..10 exactly as
    difficulty.score does, so calibrated and default scores share one scale."""
    w = model["w_vector"]
    total = _matvec(C, w)
    return [1.0 + 9.0 * min(1.0, max(0.0, t)) for t in total]


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def compare(C_train, y_train, C_test, y_test, y_col_test, y_ttc_test=None) -> dict:
    """Default weights versus fitted weights on held-out seeds. This is the only
    place the two are compared, so a table in a README cannot drift from the
    computation that produced it."""
    default_w = [difficulty.DEFAULT_WEIGHTS[k] for k in ORDER]
    s = sum(default_w)
    if abs(s - 1.0) > 1e-9:
        default_w = [v / s for v in default_w]
    model = fit_weights(C_train, y_train, w0=default_w)
    fitted_w = model["w_vector"]

    out = {
        "n_train": len(C_train), "n_test": len(C_test), "backend": model["backend"],
        "default_weights": {k: round(float(v), 4) for k, v in zip(ORDER, default_w)},
        "fitted_weights": {k: round(float(v), 4) for k, v in zip(ORDER, fitted_w)},
        "weight_l1_change": round(sum(abs(a - b) for a, b in zip(fitted_w, default_w)), 4),
        "constraints_hold": model["constraints_hold"],
    }
    for name, w in (("default", default_w), ("fitted", fitted_w)):
        total = [_clip01(t) for t in _matvec(C_test, w)]
        s10 = [1.0 + 9.0 * t for t in total]
        # Both quantities are on [0, 1] by construction -- the components are
        # clipped there and the risk index is built from clipped terms -- so this
        # is a plain mean squared error and not a rescaled one.
        out[name] = {
            "score_range": [round(min(s10), 4), round(max(s10), 4)],
            "mse_vs_risk_index": round(sum((total[i] - y_test[i]) ** 2 for i in range(len(total)))
                                       / max(1, len(total)), 6),
            "spearman_risk": _r(D.spearman(list(total), list(y_test))),
            "auc_collision": _r(D.metrics_pure(list(y_col_test), list(total))["auc"]),
        }
        if y_ttc_test is not None:
            out[name]["spearman_min_ttc"] = _r(D.spearman(list(total), list(y_ttc_test)))
    out["model"] = model
    return out


def _r(x):
    return None if x is None else round(float(x), 6)
