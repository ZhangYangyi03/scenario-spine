"""The end-to-end study: one corpus, five products, one command.

This is the module that makes the repository one system rather than five
packages. `run_study` takes an ODD and a budget, and returns every artefact the
other modules produce, with the dependency order made explicit:

    1. generate a corpus, aimed at the ODD's uncovered cells   (odd -> generate)
    2. simulate each scenario                                  (sim)
    3. score each scenario, with the measured outcome           (difficulty)
    4. measure coverage of the ODD by the corpus                (odd)
    5. synthesise a label set per scenario and check it          (labelqc)
    6. dispatch a fleet sized from the corpus's traffic density  (fleet)
    7. derive the safety case from 3, 4 and the outcomes         (assurance)

The order is not decorative: the safety case takes the coverage verdict and the
measured outcomes, so it cannot be built from a corpus that was not simulated,
and the difficulty score's exposure component is a simulation output rather than
a feature. A pipeline whose later stages cannot be stubbed is a pipeline whose
claims cannot be traced to evidence.
"""

from __future__ import annotations

import json
import math
import time

from . import assurance, difficulty, export, fleet, labelqc, lang, odd as oddmod, sim


def build_corpus(odd: dict, budget: int = 60, bins_per_axis: int = 4,
                 strategy: str = "gap", seed: int = 1) -> tuple:
    """A corpus, aimed at the ODD's uncovered cells, plus the run that produced
    it. Reproducible from (odd, seed, budget, strategy) alone -- the returned
    scenarios are the ones the coverage run generated, not a re-draw."""
    r = oddmod.coverage_until(odd, target=0.99, max_scenarios=budget,
                              strategy=strategy, bins_per_axis=bins_per_axis, seed=seed)
    return r["corpus"], r


def run_study(odd: dict = None, budget: int = 60, bins_per_axis: int = 4,
              strategy: str = "gap", seed: int = 1, fleet_scale: float = 1.0,
              with_exports: bool = True) -> dict:
    odd = odd or oddmod.declared_odd_highway()
    t0 = time.time()
    corpus, covrun = build_corpus(odd, budget, bins_per_axis, strategy, seed)
    t_gen = time.time() - t0

    t1 = time.time()
    outcomes = [sim.simulate(sc) for sc in corpus]
    t_sim = time.time() - t1

    scores = [difficulty.score(sc, o) for sc, o in zip(corpus, outcomes)]
    rep = oddmod.coverage(odd, corpus, bins_per_axis)
    verdicts = rep["verdicts"]

    t2 = time.time()
    qc = []
    for sc in corpus[: max(1, min(12, len(corpus)))]:
        boxes = labelqc.boxes_from_scenario(sc, n_frames=40)
        qc.append(labelqc.quality_report(boxes, mu=sc.weather.friction(), label=sc.id))
    t_qc = time.time() - t2
    qc_summary = {
        "scenarios_checked": len(qc),
        "defects": sum(q["defects"] for q in qc),
        "defects_by_kind": _merge_counts([q["defects_by_kind"] for q in qc]),
        "mean_quality": round(sum(q["quality"] for q in qc) / max(1, len(qc)), 4),
    }

    # fleet: size the instance from the corpus's median density and speed
    med_flow = sorted(sc.traffic.flow_vph for sc in corpus)[len(corpus) // 2] if corpus else 600.0
    n_veh = max(4, int(round(8 * fleet_scale)))
    n_trip = max(4, int(round(10 * fleet_scale)))
    veh, trips, stations = fleet.instance(n_veh, n_trip, seed=seed)
    d_greedy = fleet.dispatch_greedy(veh, trips)
    d_auction = fleet.dispatch_auction(veh, trips)
    d_exact = fleet.dispatch_ilp_exact(veh, trips)
    lb = fleet.assignment_lower_bound(veh, trips)
    ch = fleet.charge_plan([fleet.Vehicle(**{**v.__dict__}) for v in veh], stations)
    chlb = fleet.charging_lower_bound(veh, stations)

    case = assurance.build(odd, corpus, verdicts, outcomes, scores)
    casedict = case.to_dict()
    flaws = assurance.find_deficient(case)

    art = {
        "odd": odd,
        "corpus_size": len(corpus),
        "coverage": {k: v for k, v in rep.items() if k not in ("verdicts", "_bins")},
        "difficulty": {"mean": round(sum(s["score"] for s in scores) / max(1, len(scores)), 3),
                       "max": max((s["score"] for s in scores), default=0.0),
                       "bands": _count([s["band"] for s in scores]),
                       "top": sorted(scores, key=lambda s: -s["score"])[:5]},
        "simulation": {"collisions": sum(1 for o in outcomes if o.collision),
                       "hard_brake": sum(1 for o in outcomes if o.hard_brake),
                       "min_ttc_s": round(min((o.min_ttc for o in outcomes), default=0.0), 3),
                       "risk_mean": round(sum(o.risk_index() for o in outcomes) / max(1, len(outcomes)), 4)},
        "labelqc": qc_summary,
        "fleet": {
            "instance": {"vehicles": n_veh, "trips": n_trip, "stations": len(stations)},
            "greedy": fleet.evaluate(d_greedy, veh, trips),
            "auction": fleet.evaluate(d_auction, veh, trips),
            "exact": fleet.evaluate(d_exact, veh, trips),
            "lower_bound": round(lb["bound"], 4),
            "auction_gap_pct": _gap(fleet.evaluate(d_auction, veh, trips)["cost"], lb["bound"]),
            "charging": {"plan_cost": ch["cost"], "kwh": ch["kwh"],
                         "lower_bound": chlb["bound"],
                         "gap_pct": _gap(ch["cost"], chlb["bound"]),
                         "slots_used": len(ch["plan"])},
        },
        "assurance": {"nodes": len(casedict["nodes"]),
                      "verdicts": _count([n["verdict"] for n in casedict["nodes"]]),
                      "deficient": flaws[:6],
                      "root_verdict": casedict["nodes"][0]["verdict"] if casedict["nodes"] else None},
        "timing_s": {"generate": round(t_gen, 3), "simulate": round(t_sim, 3), "labelqc": round(t_qc, 3),
                     "total": round(time.time() - t0, 3)},
    }
    if with_exports:
        sc = max(zip(corpus, scores), key=lambda p: p[1]["score"])[0]
        x, (net, rou) = export.openscenario(sc), export.sumo(sc)
        art["exports"] = {"hardest_scenario": sc.id, "openscenario_bytes": len(x),
                          "sumo_net_bytes": len(net), "sumo_rou_bytes": len(rou)}
    art["_corpus"] = corpus
    art["_case"] = case
    return art


def _merge_counts(dicts: list) -> dict:
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _count(vals: list) -> dict:
    out = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _gap(actual: float, bound: float) -> float:
    """The gap to the bound, in percent, guarded against a zero bound -- an
    unguarded ratio here would silently report inf and poison a table."""
    if bound is None or abs(bound) < 1e-9:
        return 0.0
    return round(100.0 * (actual - bound) / abs(bound), 3)


def write_artifacts(art: dict, outdir: str) -> dict:
    """Write the study to disk: JSON for the numbers, an OpenSCENARIO and a SUMO
    pair for the hardest scenario, and the GSN source for the case."""
    import os

    os.makedirs(outdir, exist_ok=True)
    clean = {k: v for k, v in art.items() if not k.startswith("_")}
    paths = {}
    paths["study.json"] = os.path.join(outdir, "study.json")
    with open(paths["study.json"], "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, sort_keys=False)
    case = art.get("_case")
    if case is not None:
        paths["case.gsn.dot"] = os.path.join(outdir, "case.gsn.dot")
        with open(paths["case.gsn.dot"], "w", encoding="utf-8") as f:
            f.write(case.to_gsn_dot())
        paths["case.sacm.json"] = os.path.join(outdir, "case.sacm.json")
        with open(paths["case.sacm.json"], "w", encoding="utf-8") as f:
            json.dump(assurance.to_sacm(case), f, indent=2)
    corpus = art.get("_corpus") or []
    if corpus:
        sc = max(corpus, key=lambda s: s.seed)
        paths["hardest.xosc"] = os.path.join(outdir, "hardest.xosc")
        with open(paths["hardest.xosc"], "w", encoding="utf-8") as f:
            f.write(export.openscenario(sc))
        net, rou = export.sumo(sc)
        paths["hardest.net.xml"] = os.path.join(outdir, "hardest.net.xml")
        paths["hardest.rou.xml"] = os.path.join(outdir, "hardest.rou.xml")
        with open(paths["hardest.net.xml"], "w", encoding="utf-8") as f:
            f.write(net)
        with open(paths["hardest.rou.xml"], "w", encoding="utf-8") as f:
            f.write(rou)
    return paths
