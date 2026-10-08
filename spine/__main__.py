"""One command per thing a reader might want to check.

    python -m spine corpus     build a corpus and write it to data/
    python -m spine calibrate  fit the difficulty weights to it, compare
    python -m spine compare    the dispatch comparison, greedy vs auction vs optimum
    python -m spine odd        generate and score a small ODD corpus
    python -m spine fleet      the fleet demo end to end
    python -m spine doctor     what this machine can and cannot run

Every subcommand prints what it computed, including the things that came out
badly. There is no path through this CLI that reports a number without the seed
range, the sample size and the reference it was measured against.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def _out(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
    return path


def cmd_corpus(args):
    from . import dataset as D

    t0 = time.time()
    n_train, n_test = args.train, args.test
    print(f"building {n_train} train (seeds {args.seed0}..{args.seed0 + n_train}) and "
          f"{n_test} test (seeds {args.test_seed0}..) with {args.workers} workers")
    tr = D.build(n=n_train, seed0=args.seed0, workers=args.workers, t_end=args.t_end,
                 dt=args.dt, physics=True, components=True)
    te = D.build(n=n_test, seed0=args.test_seed0, workers=args.workers, t_end=args.t_end,
                 dt=args.dt, physics=True, components=True)
    os.makedirs(DATA, exist_ok=True)
    D.write_jsonl(tr, os.path.join(DATA, "corpus_train.jsonl"))
    D.write_jsonl(te, os.path.join(DATA, "corpus_test.jsonl"))
    summary = {
        "train": len(tr), "test": len(te), "seed0_train": args.seed0,
        "seed0_test": args.test_seed0, "t_end": args.t_end, "dt": args.dt,
        "collision_rate_train": round(sum(r["y"]["collision"] for r in tr) / max(1, len(tr)), 5),
        "collision_rate_test": round(sum(r["y"]["collision"] for r in te) / max(1, len(te)), 5),
        "physics_mean_train": round(sum(r["physics"] for r in tr) / max(1, len(tr)), 4),
        "seconds": round(time.time() - t0, 1),
    }
    _out(os.path.join(DATA, "corpus_summary.json"), summary)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_calibrate(args):
    from . import calibrate, difficulty
    from . import dataset as D

    tr = D.read_jsonl(os.path.join(DATA, "corpus_train.jsonl"))
    te = D.read_jsonl(os.path.join(DATA, "corpus_test.jsonl"))
    Ctr, ytr, _, _, order = D.component_matrix(tr)
    Cte, yte, ycol, _, _ = D.component_matrix(te)
    yttc = [-r["y"]["min_ttc_s"] for r in te]
    rep = calibrate.compare(Ctr, ytr, Cte, yte, ycol, yttc)
    model = rep.pop("model")
    _out(os.path.join(DATA, "calibration.json"), {**rep, "model": model})
    print(json.dumps(rep, indent=1))
    print("\nfitted weights, as written into the repository:")
    print(json.dumps(model["weights"], indent=1))
    return 0


def cmd_compare(args):
    from .fleet import compare, instance

    out = {}
    for nv, nt in [(8, 9), (12, 14), (16, 20)]:
        veh, trips, st = instance(nv, nt, seed=nv * 100 + nt)
        r = compare(veh, trips, exact_nodes=args.nodes)
        out[f"{nv}v_{nt}t"] = r
        ref = r["reference"]
        print(f"\n{nv} vehicles, {nt} trips -- optimum {ref['sequential_optimum']:.1f} "
              f"({'proven' if ref['proven_optimal'] else 'CAPPED'}, {ref['bnb_nodes']} nodes), "
              f"floor {ref['independent_floor']:.1f}")
        for name in ("greedy", "auction", "static_assignment", "sequential_opt"):
            x = r[name]
            print(f"  {name:18s} served {x['served']:2d}  objective {x['objective']:9.2f}  "
                  f"vs optimum {str(x.get('gap_vs_opt_pct')):>8s}%  violations {x['violations']}")
    _out(os.path.join(DATA, "dispatch_compare.json"), out)
    return 0


def cmd_odd(args):
    from . import difficulty, generate, lang, odd

    q = lang.Query(text="mixed")
    scenarios = [generate.generate(q, s) for s in range(args.n)]
    spec = odd.declared_odd_highway()
    rep = odd.coverage(spec, scenarios, bins_per_axis=4, order=2)
    scored = difficulty.rank(scenarios)
    print(f"corpus {len(scenarios)} scenarios, {rep['axes']} axes, {rep['bins_per_axis']} bins/axis")
    print(f"2-way (pairwise) coverage {rep['interaction_cells_hit']}/"
          f"{rep['interaction_cells_total']} = {rep['order2_coverage']:.4f}   "
          f"missing {rep['interaction_cells_missing']}")
    print(f"1-way coverage            {rep['order1_coverage']:.4f}   "
          f"full-product cells {rep['cells_observed']}/{rep['cells_total']} = "
          f"{rep['product_coverage']:.2e}")
    print(f"outside the declared ODD: {rep['outside_odd']}  verdict: {rep['verdict']}")
    nxt = odd.propose_next(spec, scenarios, bins_per_axis=4)
    print("largest remaining pair gap:", nxt.get("cell") if isinstance(nxt, dict) else nxt)
    print("hardest generated:", [(s["scenario_id"], s["score"]) for s in scored[:3]])
    _out(os.path.join(DATA, "odd_report.json"), rep)
    return 0


def cmd_doctor(args):
    from . import learn

    print("python      ", sys.version.split()[0], platform.machine())
    print("platform    ", platform.platform())
    ok, dev, ver = learn.torch_available()
    print("torch       ", ver, "-> accelerator:", dev if ok else "none")
    try:
        import numpy
        print("numpy       ", numpy.__version__)
    except Exception as e:
        print("numpy        missing:", e)
    try:
        import pytest
        print("pytest      ", pytest.__version__)
    except Exception as e:
        print("pytest       missing:", e)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="build and write a corpus")
    c.add_argument("--train", type=int, default=20000)
    c.add_argument("--test", type=int, default=4000)
    c.add_argument("--seed0", type=int, default=0)
    c.add_argument("--test-seed0", dest="test_seed0", type=int, default=1000000)
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c.add_argument("--t-end", dest="t_end", type=float, default=12.0)
    c.add_argument("--dt", type=float, default=0.02)
    c.set_defaults(func=cmd_corpus)

    k = sub.add_parser("calibrate", help="fit the difficulty weights")
    k.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("compare", help="dispatch comparison")
    p.add_argument("--nodes", type=int, default=200000)
    p.set_defaults(func=cmd_compare)

    o = sub.add_parser("odd", help="ODD coverage")
    o.add_argument("--n", type=int, default=200)
    o.set_defaults(func=cmd_odd)

    d = sub.add_parser("doctor", help="what this machine can run")
    d.set_defaults(func=cmd_doctor)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
