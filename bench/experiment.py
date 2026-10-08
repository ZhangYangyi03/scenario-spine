"""The benchmark. Every number that goes into the README comes from here.

Four measurements, each with the reference it is measured against, because a
number without a reference is not a result:

  1. CORPUS        build N scenarios, report the wall clock, the per-scenario
                   cost, and the collision rate so the corpus's difficulty is
                   visible rather than implied.
  2. COVERAGE      pairwise interaction coverage of a declared ODD under two
                   samplers, aimed (propose_next) and unaimed, at equal corpus
                   size. The claim is that aiming covers strictly more cells.
  3. CALIBRATION   the hand-set weights against weights fitted to 200k simulated
                   scenarios, scored on held-out seed ranges. Reported as
                   rank correlation, AUC and MSE, all three, because any one of
                   them alone can be made to look good.
  4. LEARNED       an MLP on the same pre-execution features, against the same
                   held-out scenarios, so the analytic score has a second
                   independent comparison and not only a fitted version of itself.

Writes bench/results.json. Every field carries the seed range and sample size it
was computed from.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from spine import calibrate, dataset as D, difficulty, features as F, generate, lang, odd  # noqa: E402

TRAIN_N = int(os.environ.get("SPINE_TRAIN", "200000"))
TEST_N = int(os.environ.get("SPINE_TEST", "30000"))
WORKERS = int(os.environ.get("SPINE_WORKERS", str(max(1, (os.cpu_count() or 2) - 1))))
OUT = os.path.join(HERE, "results.json")
TRAIN_SEED0, TEST_SEED0 = 0, 1_000_000

rem = os.environ.get("SPINE_SKIP_LEARNED") != "1"


def log(*a):
    print(*a, flush=True)


def part1_corpus():
    t0 = time.time()
    tr = D.build(n=TRAIN_N, seed0=TRAIN_SEED0, workers=WORKERS, t_end=12.0, dt=0.02,
                 physics=True, components=True)
    t_tr = time.time() - t0
    t0 = time.time()
    te = D.build(n=TEST_N, seed0=TEST_SEED0, workers=WORKERS, t_end=12.0, dt=0.02,
                 physics=True, components=True)
    t_te = time.time() - t0
    out = {"n_train": len(tr), "n_test": len(te), "workers": WORKERS,
           "train_seed_range": [TRAIN_SEED0, TRAIN_SEED0 + len(tr)],
           "test_seed_range": [TEST_SEED0, TEST_SEED0 + len(te)],
           "seconds_train": round(t_tr, 1), "seconds_test": round(t_te, 1),
           "ms_per_scenario_train": round(1000.0 * t_tr / max(1, len(tr)), 3),
           "ms_per_scenario_test": round(1000.0 * t_te / max(1, len(te)), 3),
           "collision_rate_train": round(sum(r["y"]["collision"] for r in tr) / max(1, len(tr)), 5),
           "collision_rate_test": round(sum(r["y"]["collision"] for r in te) / max(1, len(te)), 5),
           "hard_brake_rate_train": round(sum(r["y"]["hard_brake"] for r in tr) / max(1, len(tr)), 5)}
    D.write_jsonl(tr, os.path.join(BASE, "data", "corpus_train.jsonl"))
    D.write_jsonl(te, os.path.join(BASE, "data", "corpus_test.jsonl"))
    log("1. corpus", out)
    return tr, te, out


def part2_coverage():
    spec = odd.declared_odd_highway()
    pins, ranges = {}, {}
    for name, s in spec["axes"].items():
        if s["kind"] == "cat":
            pins[name] = s["values"][0]
        else:
            ranges[name] = (s["lo"], s["hi"])
    q = lang.Query(text="highway pilot in-ODD", pins=pins, ranges=ranges)
    out = {"odd": spec["name"], "bins_per_axis": 4, "n": 400}
    plain = [generate.generate(q, s) for s in range(400)]
    a = odd.coverage(spec, plain, bins_per_axis=4)
    out["unaimed"] = {"order2": a["order2_coverage"], "order1": a["order1_coverage"],
                      "cells": a["interaction_cells_total"]}
    # The aimed corpus is built from empty, so both corpora are exactly 400
    # scenarios and the comparison measures the aiming rather than the budget.
    # (The first version appended the aimed draws to the unaimed ones and then
    # sliced back to 400 -- which compares the unaimed corpus with itself.)
    aimed = []
    for k in range(400):
        nxt = odd.propose_next(spec, aimed, bins_per_axis=4, seed=k)
        if nxt.get("query") is None:
            break
        aimed.append(generate.generate(nxt["query"], 500000 + k))
    b = odd.coverage(spec, aimed, bins_per_axis=4)
    out["aimed"] = {"order2": b["order2_coverage"], "order1": b["order1_coverage"],
                    "cells": b["interaction_cells_total"], "n": len(aimed)}
    out["equal_size"] = {"n": 400}
    out["unaimed_at_equal_size"] = out["unaimed"]["order2"]
    out["aimed_at_equal_size"] = b["order2_coverage"]
    out["aiming_gain"] = round(b["order2_coverage"] - out["unaimed"]["order2"], 6)
    out["scenarios_to_90pct_aimed"] = _until(spec, "gap")
    u = _until(spec, "random", cap=4000)
    out["scenarios_to_90pct_unaimed"] = u if u is not None else ">4000"
    log("2. coverage", out)
    return out


def _until(spec, strategy, cap: int = 4000):
    pins, ranges = {}, {}
    for name, s in spec["axes"].items():
        if s["kind"] == "cat":
            pins[name] = s["values"][0]
        else:
            ranges[name] = (s["lo"], s["hi"])
    q = lang.Query(text="pilot", pins=pins, ranges=ranges)
    corpus = []
    for k in range(cap):
        if strategy == "gap":
            nxt = odd.propose_next(spec, corpus, bins_per_axis=4, seed=k)
            if nxt.get("query") is None:
                return k
            corpus.append(generate.generate(nxt["query"], 700000 + k))
        else:
            corpus.append(generate.generate(q, 700000 + k))
        if k % 50 == 0 or k == cap - 1:
            if odd.coverage(spec, corpus, bins_per_axis=4)["order2_coverage"] >= 0.9:
                return k + 1
    return None


def part3_calibration(tr, te):
    Ctr, ytr, _, _, order = D.component_matrix(tr)
    Cte, yte, ycol, phys, _ = D.component_matrix(te)
    yttc = [-r["y"]["min_ttc_s"] for r in te]
    rep = calibrate.compare(Ctr, ytr, Cte, yte, ycol, yttc)
    model = rep.pop("model")
    rep["components"] = order
    rep["physics_on_rescaled_scale"] = {
        "spearman_risk": _r(D.spearman(list((phys - 1.0) / 9.0), list(yte))),
        "auc_collision": _r(D.metrics_pure(list(ycol), list((phys - 1.0) / 9.0))["auc"]),
    }
    with open(os.path.join(BASE, "data", "_weights.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(model, f, indent=1)
    log("3. calibration", json.dumps(rep, indent=1))
    return rep, model, (Ctr, ytr, Cte, yte, ycol, yttc, phys)


def _r(x):
    return None if x is None else round(float(x), 6)


def part4_learned(tr, te, calib_pack):
    from spine import learn

    Ctr, ytr, Cte, yte, ycol, yttc, phys = calib_pack
    import numpy as np

    Xtr, mu, sd = D.standardise(np.asarray([r["x"] for r in tr], dtype=np.float32))
    Xte = (np.asarray([r["x"] for r in te], dtype=np.float32) - mu) / sd
    Xtr, Xte = np.clip(Xtr, -8, 8), np.clip(Xte, -8, 8)
    avail, dev, ver = learn.torch_available()
    out = {"torch_available": avail, "device": dev, "torch_version": ver,
           "feature_dim": len(F.feature_names()), "hidden": [256, 256, 128]}
    if not avail:
        return out

    ycol_tr = np.asarray([r["y"]["collision"] for r in tr], dtype=np.float32)
    yrisk_tr = np.asarray([r["y"]["risk_index"] for r in tr], dtype=np.float32)
    yrisk_te = np.asarray(yte, dtype=np.float32)

    t0 = time.time()
    logit = learn.train_logistic(Xtr, ycol_tr, epochs=600, lr=0.5)
    out["logistic_numpy"] = {
        "auc_collision": _r(D.metrics_pure(list(ycol), list(learn.predict_logistic(logit, Xte)))["auc"]),
        "seconds": round(time.time() - t0, 1)}

    for target, ytr_t, label in (("collision", ycol_tr, "mlp_collision"),
                                 ("risk_index", yrisk_tr, "mlp_risk")):
        t0 = time.time()
        m = learn.train_mlp_torch(Xtr, ytr_t, target=target, hidden=(256, 256, 128),
                                  epochs=40, lr=2e-3, batch=1024, device=dev, seed=0)
        p = learn.predict_torch(m, Xte)
        out[label] = {
            "seconds": round(time.time() - t0, 1), "device": m["device"],
            "val_loss": m["val_loss"], "val_epoch": m["val_epoch"],
            "auc_collision": _r(D.metrics_pure(list(ycol), list(p))["auc"]),
            "spearman_risk": _r(D.spearman(list(p), list(yrisk_te))),
            "spearman_min_ttc": _r(D.spearman(list(p), list(yttc))),
        }
        log("4.", label, out[label])

    # a calibration table, because an AUC says nothing about whether the score can
    # be read as a probability of anything
    m = learn.train_mlp_torch(Xtr, ycol_tr, target="collision", hidden=(256, 256, 128),
                              epochs=40, lr=2e-3, batch=1024, device=dev, seed=0)
    p = np.asarray(learn.predict_torch(m, Xte), dtype=np.float64)
    order = np.argsort(-p)
    ycol_arr = np.asarray(ycol)
    table = []
    for i in range(10):
        idx = order[i * len(order) // 10:(i + 1) * len(order) // 10]
        table.append({"decile": i + 1, "n": int(len(idx)),
                      "predicted_mean": round(float(p[idx].mean()), 5),
                      "measured_collision_rate": round(float(ycol_arr[idx].mean()), 5)})
    out["reliability_deciles"] = table
    return out


def main():
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    results = {}
    tr, te, results["corpus"] = part1_corpus()
    results["coverage"] = part2_coverage()
    calib, model, pack = part3_calibration(tr, te)
    results["calibration"] = calib
    if rem:
        results["learned"] = part4_learned(tr, te, pack)
    else:
        results["learned"] = {"skipped": True}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=1)
    log("wrote", OUT)


if __name__ == "__main__":
    main()
