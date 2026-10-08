"""Build a (features, outcome) dataset, in parallel, reproducibly.

Two things matter here and nothing else does:

    * the same (seed, count) must produce the same dataset, so a trained model
      can be reproduced without shipping the weights; and
    * the train/test split must be over *seeds*, not over rows. A random row
      split leaks: the same parameter vector sampled at seed k and k+1 differs in
      the third decimal, so a model can memorise the neighbourhood and score well
      on apparently held-out data. Splitting by seed range means the test
      scenarios are drawn from a part of the parameter space the model never saw,
      which is the question anyone actually asks of it.

Parallelism uses a process pool over contiguous seed blocks, so each worker's
output depends only on its own block -- the dataset is then reproducible
regardless of how many workers happened to run, which a work-stealing pool would
not guarantee.
"""

from __future__ import annotations

import json
import os
import zlib

from . import difficulty
from . import features as F
from . import generate, lang, sim


def _block(args) -> list:
    query_dict, seed0, n, t_end, dt, comps = args
    q = lang.Query(**query_dict)
    rows = []
    for k in range(n):
        sc = generate.generate(q, seed0 + k)
        try:
            o = sim.simulate(sc, dt=dt, t_end=t_end)
        except Exception:
            continue
        row = {"seed": sc.seed, "scenario": sc.id, "params": sc.params,
               "x": F.features(sc), "y": F.outcome_row(o), "physics": None}
        if comps:
            c = difficulty.components(sc)
            row["c"] = [c[k] for k in difficulty.COMPONENT_ORDER]
        rows.append(row)
    return rows


def build(n: int = 2000, seed0: int = 0, query: lang.Query = None, workers: int = 1,
          t_end: float = 12.0, dt: float = 0.02, physics: bool = True,
          components: bool = False) -> list:
    """Generate and simulate `n` scenarios starting at seed `seed0`.

    `workers` > 1 uses a process pool. The block size is computed so every worker
    gets a whole number of scenarios and the same total regardless of worker
    count.
    """
    q = query or lang.Query(text="mixed odd draw")
    if q.pins is None:
        q.pins = {}
    if workers <= 1 or n < 64:
        rows = _block((q.to_dict(), seed0, n, t_end, dt, components))
        return add_physics(rows) if physics else rows
    import multiprocessing as mp

    per = max(1, n // workers)
    args = []
    done = 0
    for w in range(workers):
        cnt = per if w < workers - 1 else n - done
        args.append((q.to_dict(), seed0 + done, cnt, t_end, dt, components))
        done += cnt
    with mp.Pool(min(workers, max(1, len(args)))) as pool:
        blocks = pool.map(_block, args)
    rows = [r for b in blocks for r in b]
    if physics:
        rows = add_physics(rows)
    return rows


def add_physics(rows: list) -> list:
    """Attach the physics score to each row, so the learned model and the
    analytical score are compared on identical rows rather than on two corpora
    that were assumed to be the same.

    The scenario is rebuilt from the parameter vector, not read back from disk.
    That is exact rather than approximate: scenario_from_params is a deterministic
    function of its parameters, so the rebuild reproduces the scenario that was
    simulated -- and test_dataset.py asserts that by re-simulating a rebuilt row
    and comparing the outcome.
    """
    for r in rows:
        sc = generate.scenario_from_params(r["params"], r["scenario"])
        r["physics"] = difficulty.score(sc)["score"]
    return rows


def write_jsonl(rows: list, path: str, compress: bool = True) -> str:
    data = "\n".join(json.dumps(r) for r in rows)
    raw = data.encode("utf-8")
    if compress:
        with open(path, "wb") as f:
            f.write(zlib.compress(raw, 6))
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
    return path


def read_jsonl(path: str) -> list:
    with open(path, "rb") as f:
        head = f.read(2)
        f.seek(0)
        if head == b"\x78\x9c":
            return [json.loads(l) for l in zlib.decompress(f.read()).decode("utf-8").splitlines() if l]
        return [json.loads(l) for l in f.read().decode("utf-8").splitlines() if l]


def matrices(rows: list) -> tuple:
    """(X, y_collision, y_risk, physics). Requires numpy for the array form; the
    repository's own tests do not call this, so the stdlib-only path stays
    standard library only."""
    import numpy as np

    X = np.asarray([r["x"] for r in rows], dtype=np.float32)
    ycol = np.asarray([r["y"]["collision"] for r in rows], dtype=np.float32)
    yrisk = np.asarray([r["y"]["risk_index"] for r in rows], dtype=np.float32)
    phys = np.asarray([r["physics"] if r["physics"] is not None else float("nan") for r in rows],
                      dtype=np.float32)
    return X, ycol, yrisk, phys


def component_matrix(rows: list, arrays: bool = None):
    """(C, y_risk, y_collision, physics, order) for rows built with
    components=True. Used by calibrate.py, which fits the weight vector of the
    hand-built score to what the simulator actually did.

    Returns lists by default and numpy arrays only when numpy is importable,
    because the pure-Python calibrator is a supported path and a list-versus-array
    return type that depended on the environment would be a trap for the caller.
    Pass arrays=True to require numpy (and get an ImportError if it is absent).
    """
    from . import difficulty

    C = [list(r["c"]) for r in rows]
    yrisk = [float(r["y"]["risk_index"]) for r in rows]
    ycol = [float(r["y"]["collision"]) for r in rows]
    phys = [float(r["physics"]) if r["physics"] is not None else float("nan") for r in rows]
    if arrays is None:
        try:
            import numpy  # noqa: F401

            arrays = True
        except Exception:
            arrays = False
    if arrays:
        import numpy as np

        return (np.asarray(C, dtype=np.float64), np.asarray(yrisk, dtype=np.float64),
                np.asarray(ycol, dtype=np.float64), np.asarray(phys, dtype=np.float64),
                list(difficulty.COMPONENT_ORDER))
    return C, yrisk, ycol, phys, list(difficulty.COMPONENT_ORDER)


def standardise(X, mean=None, std=None):
    """Z-score using train statistics. Returning the statistics is the point: a
    model trained on one corpus must be applied with the same scaling, and a
    scaler refitted on test data would leak the test distribution into it."""
    import numpy as np

    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def split_by_seed(rows: list, test_seed_from: int, test_seed_to: int) -> tuple:
    """The split that matters: by seed range, so no test scenario is a near-copy of
    a training one."""
    train = [r for r in rows if not (test_seed_from <= r["seed"] < test_seed_to)]
    test = [r for r in rows if test_seed_from <= r["seed"] < test_seed_to]
    return train, test


def metrics_pure(y_true: list, y_pred: list) -> dict:
    """Rank correlation, AUC and mean absolute error, in the standard library.

    AUC here is computed by the rank formulation (the Mann-Whitney U statistic
    with average ranks for ties), not by trapezoid over a ROC curve, so a corpus
    with many tied predictions -- which a discretised physics score has -- scores
    correctly instead of depending on the tie order.
    """
    n = len(y_true)
    if n == 0:
        return {"n": 0}
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    rho = spearman(y_true, y_pred)
    auc = None
    labels = sorted(set(y_true))
    if set(labels) <= {0.0, 1.0} and 0 < sum(y_true) < n:
        pos = sum(y_true)
        neg = n - pos
        ranks = _average_ranks(y_pred)
        rank_sum_pos = sum(r for r, t in zip(ranks, y_true) if t == 1)
        u = rank_sum_pos - pos * (pos + 1) / 2.0
        auc = u / (pos * neg)
    return {"n": n, "mae": round(mae, 6), "spearman": None if rho is None else round(rho, 6),
            "auc": None if auc is None else round(auc, 6),
            "positives": int(sum(y_true)) if set(labels) <= {0.0, 1.0} else None}


def _average_ranks(vals: list) -> list:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list, b: list) -> float:
    n = len(a)
    if n < 3:
        return None
    ra, rb = _average_ranks(list(a)), _average_ranks(list(b))
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra)
    db = sum((y - mb) ** 2 for y in rb)
    if da <= 0 or db <= 0:
        return None
    return num / (da * db) ** 0.5
