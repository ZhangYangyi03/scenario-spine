"""ODD coverage in strong Kleene logic, and the gap-directed sampler that closes it.

Why three values and not two. The question a coverage report has to answer is
not "did every test pass" but "is there a part of the declared operating domain
that nothing has been run in". That question has three honest answers, not two:

    SAT       a cell of the declared ODD that a run has exercised
    VIOLATED  a run that fell *outside* the declared ODD -- the run happened, so
              the claim "we only tested inside our declared domain" is refuted
    UNKNOWN   a cell of the declared ODD that no run has exercised -- not a
              failure, an absence, and the one that a two-valued report would
              have to call either pass (hiding the gap) or fail (crying wolf)

The third value is the whole point: coverage gaps are the thing scene libraries
get wrong, and they get it wrong by having no way to say "I did not look there".

Strong Kleene connectives (the same tables spine/assurance.py reuses when it
turns these verdicts into a safety-case goal):

    AND   T T=T   T F=F   T U=U   F F=F   F U=F   U U=U
    OR    T T=T   T F=T   T U=T   F F=F   F U=U   U U=U
    NOT   T->F  F->T  U->U
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import lang
from . import generate
from .ir import Scenario

T, F, U = "SAT", "VIOLATED", "UNKNOWN"


# --------------------------------------------------- Kleene connectives ----

def k_not(a: str) -> str:
    return {T: F, F: T, U: U}[a]


def k_and(a: str, b: str) -> str:
    if a == F or b == F:
        return F
    if a == U or b == U:
        return U
    return T


def k_or(a: str, b: str) -> str:
    if a == T or b == T:
        return T
    if a == U or b == U:
        return U
    return F


def k_all(vals) -> str:
    out = T
    for v in vals:
        out = k_and(out, v)
    return out


def k_any(vals) -> str:
    out = F
    for v in vals:
        out = k_or(out, v)
    return out


def k_violated_fraction(vals) -> float:
    """How much of a conjunction is *positively refuted*. Reporting the Kleene
    verdict alone loses the count, and the count is what a reviewer asks for."""
    vals = list(vals)
    if not vals:
        return 0.0
    return sum(1 for v in vals if v == F) / len(vals)


# ----------------------------------------------------------- the ODD spec ----

def odd_from_dict(d: dict) -> dict:
    """Declared operating domain: per-axis regions. An axis absent from the spec
    is out of scope entirely, which is different from being unknown."""
    axes = {}
    for name, spec in (d.get("axes") or {}).items():
        if name not in lang.AXES:
            continue
        a = lang.AXES[name]
        if a["kind"] == "cat":
            vals = list(spec.get("values", a["values"]))
            axes[name] = {"kind": "cat", "values": vals}
        else:
            lo = float(spec.get("lo", a["lo"]))
            hi = float(spec.get("hi", a["hi"]))
            axes[name] = {"kind": "range", "lo": min(lo, hi), "hi": max(lo, hi)}
    return {"name": d.get("name", "declared-odd"), "axes": axes}


def declared_odd_highway() -> dict:
    """A concrete ODD: a highway pilot with a rain-limited speed envelope. This
    is the ODD the bench experiment measures coverage against."""
    return odd_from_dict({
        "name": "highway-pilot-v1",
        "axes": {
            "road.kind": {"values": ["motorway", "rural"]},
            "road.radius_m": {"lo": 0.0, "hi": 1500.0},
            "road.grade_pct": {"lo": -6.0, "hi": 6.0},
            "road.lanes": {"values": [2, 3]},
            "road.speed_limit_kph": {"lo": 60.0, "hi": 130.0},
            "weather.condition": {"values": ["dry", "wet", "rain"]},
            "weather.fog_visibility_m": {"values": [200.0, 1000.0, 10000.0]},
            "weather.illuminance_lux": {"values": [50.0, 500.0, 5000.0, 100000.0]},
            "ego.speed_kph": {"lo": 30.0, "hi": 130.0},
            "traffic.flow_vph": {"lo": 0.0, "hi": 2500.0},
            "traffic.mix_truck": {"lo": 0.0, "hi": 0.4},
            "maneuver.type": {"values": ["lane_keep", "cut_in", "lead_brake", "on_ramp_merge",
                                         "off_ramp_diverge", "static_obstacle"]},
            "maneuver.t_gap_s": {"lo": 0.4, "hi": 6.0},
            "maneuver.decel_mps2": {"lo": 1.0, "hi": 8.0},
            "maneuver.occlusion": {"values": ["none", "truck", "vegetation"]},
        },
    })


def axis_verdict(sc: Scenario, odd: dict, name: str) -> str:
    """One scenario against one axis: SAT inside, VIOLATED outside, UNKNOWN when
    the scenario does not carry the axis at all (an honest third answer, not a
    silent pass)."""
    spec = odd["axes"].get(name)
    if spec is None:
        return U
    val = sc.params.get(name)
    if val is None:
        return U
    if spec["kind"] == "cat":
        return T if val in spec["values"] else F
    try:
        v = float(val)
    except (TypeError, ValueError):
        return U
    return T if spec["lo"] - 1e-9 <= v <= spec["hi"] + 1e-9 else F


def scenario_verdict(sc: Scenario, odd: dict) -> dict:
    per = {name: axis_verdict(sc, odd, name) for name in odd["axes"]}
    return {"scenario_id": sc.id, "per_axis": per, "verdict": k_all(per.values()),
            "violated_share": k_violated_fraction(per.values())}


# --------------------------------------------------------- cell coverage ----

def _bins(name: str, spec: dict, n: int) -> list:
    a = lang.AXES[name]
    if spec["kind"] == "cat":
        return [v for v in a["values"] if v in spec["values"]]
    lo, hi = spec["lo"], spec["hi"]
    if n <= 1 or hi <= lo:
        return [(lo, hi)]
    if a.get("log") and lo > 0:
        return [(math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / n),
                 math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * (i + 1) / n)) for i in range(n)]
    return [(lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n) for i in range(n)]


def cell_of(sc: Scenario, odd: dict, bins: dict) -> tuple:
    """(tag, left_the_odd, per-axis indices). The tag is the per-axis bin vector as
    a string -- enough to key a set without ever materialising the product."""
    idx, is_out = _cell_indices(sc, odd, bins)
    tag = ",".join(str(idx[n]) for n in odd["axes"])
    return tag, is_out, idx


# --------------------------------------------------------------- coverage ----

def _pair_keys(names: list, order: int = 2):
    """The interaction cells of order `order` over the ODD's axes: (axis, bin)
    pairs. This is the combinatorial-testing notion of coverage, and the reason
    it is the right one here rather than the full product:

        full product       4 bins over 15 axes -> 1.07e9 cells. A corpus can
                           never cover it, so "coverage" would be a number near
                           zero for every corpus and could not rank two of them.
        order-2 cells      sum over axis pairs of bins_a * bins_b -> a few
                           thousand cells. Coverable, rankable, and it is the
                           measure that actually means something physically: an
                           uncovered pair is a *combination* of conditions --
                           ice at a tight radius, a pedestrian behind a parked
                           row at 45 kph -- that nothing has ever run in.

    Coverage of order 1 is reported alongside it, because the two differ and the
    difference is informative: a corpus can be 100% order-1 and 40% order-2,
    which says the axes have each been visited but not in combination.
    """
    import itertools

    out = []
    for a, b in itertools.combinations(names, 2):
        out.append((a, b))
    return out


def coverage(odd: dict, corpus: list, bins_per_axis: int = 4, order: int = 2) -> dict:
    """The report.

        order1_coverage   mean over axes of the share of that axis's bins hit.
        order2_coverage   share of the (axis_a bin, axis_b bin) pairs hit --
                          the headline number, and the one the sampler drives.
        product_coverage  distinct full cells observed over the product, reported
                          so the fact that the product is out of reach is a
                          number in the report rather than a caveat in prose.
    """
    bins = {name: _bins(name, spec, bins_per_axis) for name, spec in odd["axes"].items()}
    names = list(bins)
    total_product = 1
    for n in names:
        total_product *= max(1, len(bins[n]))

    pairs = _pair_keys(names, order)
    pair_universe = set()
    for a, b in pairs:
        for ia in range(len(bins[a])):
            for ib in range(len(bins[b])):
                pair_universe.add((a, ia, b, ib))
    pair_hit = set()

    hits = {n: [0] * max(1, len(bins[n])) for n in names}
    observed, outside, verdicts, outside_examples = set(), 0, [], []
    for sc in corpus:
        v = scenario_verdict(sc, odd)
        verdicts.append(v)
        idx, is_out = _cell_indices(sc, odd, bins)
        if is_out:
            outside += 1
            if len(outside_examples) < 6:
                outside_examples.append({"scenario_id": sc.id,
                                         "axes_outside": sorted(n for n, i in idx.items() if i == "OUT"),
                                         "params": {n: sc.params.get(n) for n, i in idx.items() if i == "OUT"}})
            continue
        observed.add(",".join(str(idx[n]) for n in names))
        for n in names:
            hits[n][idx[n]] += 1
        for a, b in pairs:
            pair_hit.add((a, idx[a], b, idx[b]))

    per_axis = {}
    for n in names:
        nb = max(1, len(bins[n]))
        per_axis[n] = {"bins": nb, "bins_hit": sum(1 for h in hits[n] if h > 0),
                       "hit_fraction": round(sum(1 for h in hits[n] if h > 0) / nb, 4),
                       "counts": hits[n],
                       "empty_bins": [i for i, h in enumerate(hits[n]) if h == 0]}
    order1 = sum(v["hit_fraction"] for v in per_axis.values()) / max(1, len(names))
    order2 = len(pair_hit) / max(1, len(pair_universe))
    empties = sorted(((n, per_axis[n]["empty_bins"]) for n in names if per_axis[n]["empty_bins"]),
                     key=lambda kv: -len(kv[1]))
    uncovered_pairs = sorted(pair_universe - pair_hit,
                             key=lambda k: (k[0], k[1], k[2], k[3]))
    headline = F if outside else (U if order2 < 1.0 - 1e-12 else T)
    return {
        "odd": odd.get("name", "declared-odd"),
        "bins_per_axis": bins_per_axis,
        "order": order,
        "scenarios": len(corpus),
        "axes": len(names),
        "cells_total": total_product,
        "cells_observed": len(observed),
        "product_coverage": round(len(observed) / max(1, total_product), 12),
        "coverage_fraction": round(order2, 9),
        "order2_coverage": round(order2, 9),
        "order1_coverage": round(order1, 6),
        "interaction_cells_total": len(pair_universe),
        "interaction_cells_hit": len(pair_hit),
        "interaction_cells_missing": len(uncovered_pairs),
        "outside_odd": outside,
        "outside_examples": outside_examples,
        "per_axis": per_axis,
        "empty_bins_by_axis": {n: e for n, e in empties},
        "uncovered_cells": [f"{n}[{i}]" for n, e in empties for i in e][:12],
        "uncovered_pairs": [f"{a}[{ia}]x{b}[{ib}]" for a, ia, b, ib in uncovered_pairs[:12]],
        "verdict": headline,
        "verdict_counts": {T: sum(1 for v in verdicts if v["verdict"] == T),
                           F: sum(1 for v in verdicts if v["verdict"] == F),
                           U: sum(1 for v in verdicts if v["verdict"] == U)},
        "verdicts": verdicts,
        # NOTE: the internal sets are deliberately NOT returned. They were, and a
        # CLI that writes this report to JSON died on "Object of type set is not
        # JSON serializable" after doing all the work. Callers that need the
        # bookkeeping call coverage_detail(), which returns them separately and is
        # never the thing that gets serialised.
        "per_axis_counts": {n: hits[n] for n in names},
    }


def coverage_detail(odd: dict, corpus: list, bins_per_axis: int = 4) -> dict:
    """The un-serialisable half of the report, for the sampler: the pair universe,
    the hit set, the bins and the per-bin counts. Kept out of coverage() so that
    the report that gets written to disk can never be the thing that fails."""
    rep = coverage(odd, corpus, bins_per_axis=bins_per_axis)
    names = list(odd["axes"])
    bins = {name: _bins(name, spec, bins_per_axis) for name, spec in odd["axes"].items()}
    pairs = _pair_keys(names, 2)
    universe, hit = set(), set()
    for a, b in pairs:
        for ia in range(len(bins[a])):
            for ib in range(len(bins[b])):
                universe.add((a, ia, b, ib))
    for sc in corpus:
        idx, is_out = _cell_indices(sc, odd, bins)
        if is_out:
            continue
        for a, b in pairs:
            hit.add((a, idx[a], b, idx[b]))
    return {"report": rep, "bins": bins, "pair_universe": universe, "pair_hit": hit,
            "pairs": pairs, "hits": {n: [0] * max(1, len(bins[n])) for n in names}}


def _cell_indices(sc: Scenario, odd: dict, bins: dict) -> tuple:
    """Per-axis bin index for one scenario, or "OUT" on the axes it leaves. This
    is per-axis rather than a full product tuple on purpose: no code path in this
    module ever materialises the product."""
    out, is_out = {}, False
    for name, spec in odd["axes"].items():
        val = sc.params.get(name)
        if val is None:
            out[name] = "OUT"
            is_out = True
            continue
        got = None
        if spec["kind"] == "cat":
            for i, b in enumerate(bins[name]):
                if val == b:
                    got = i
                    break
        else:
            for i, (lo, hi) in enumerate(bins[name]):
                try:
                    if lo - 1e-9 <= float(val) <= hi + 1e-9:
                        got = i
                        break
                except (TypeError, ValueError):
                    break
        if got is None:
            out[name] = "OUT"
            is_out = True
        else:
            out[name] = got
    return out, is_out


def propose_next(odd: dict, corpus: list, bins_per_axis: int = 4, seed: int = 0,
                 candidates: int = 24) -> dict:
    """One scenario aimed at the interaction cells the corpus has not covered.

    The construction is AETG-style (greedy, seeded), not a weight on the axes:

        for each of up to `candidates` uncovered pairs, seed a scenario with that
        pair, then fill the remaining axes one at a time with whichever value
        covers the most still-uncovered pairs given what is already fixed;
        keep the candidate that covers the most.

    Seeding from an uncovered pair is what makes the sampler *directed* rather
    than *biased*: it cannot wander back into the part of the domain that is
    already dense, because it starts from a cell that is by construction empty.
    """
    import random as _random

    rng = _random.Random(seed)
    bins = {name: _bins(name, spec, bins_per_axis) for name, spec in odd["axes"].items()}
    names = list(bins)
    import itertools

    universe = set()
    for a, b in itertools.combinations(names, 2):
        for ia in range(len(bins[a])):
            for ib in range(len(bins[b])):
                universe.add((a, ia, b, ib))
    hit = set()
    for sc in corpus:
        idx, is_out = _cell_indices(sc, odd, bins)
        if is_out:
            continue
        for a, b in itertools.combinations(names, 2):
            hit.add((a, idx[a], b, idx[b]))
    missing = sorted(universe - hit)
    if not missing:
        return {"query": None, "cell": None,
                "reason": "every order-2 interaction cell of the declared ODD has been covered",
                "targets": {}}

    def covers(assign: dict, cell) -> bool:
        a, ia, b, ib = cell
        return assign.get(a) == ia and assign.get(b) == ib

    def greedy(seed_pair) -> tuple:
        assign = {}
        a, ia, b, ib = seed_pair
        assign[a], assign[b] = ia, ib
        remaining = [n for n in names if n not in assign]
        while remaining:
            best_n, best_v, best_c = None, None, -1
            for n in remaining:
                for v in range(len(bins[n])):
                    trial = dict(assign)
                    trial[n] = v
                    c = 0
                    for m in missing:
                        if m in hit:
                            continue
                        # count only cells that are *fully pinned* by the trial
                        am, iam, bm, ibm = m
                        if am in trial and bm in trial and trial[am] == iam and trial[bm] == ibm:
                            c += 1
                    if c > best_c:
                        best_n, best_v, best_c = n, v, c
            if best_n is None:
                # no remaining uncovered pair touches this axis: pin to the
                # least-used bin so the scenario is at least informative
                best_n = remaining[0]
                best_v = min(range(len(bins[best_n])), key=lambda v: sum(
                    1 for s in corpus if _cell_indices(s, odd, bins)[0].get(best_n) == v))
            assign[best_n] = best_v
            remaining = [n for n in names if n not in assign]
        newly = set()
        for m in missing:
            if covers(assign, m):
                newly.add(m)
        return assign, newly

    seeds = missing if len(missing) <= candidates else rng.sample(missing, candidates)
    best_assign, best_new = None, set()
    for s in seeds:
        assign, newly = greedy(s)
        if len(newly) > len(best_new):
            best_assign, best_new = assign, newly

    pins, ranges = {}, {}
    for name, i in best_assign.items():
        b = bins[name][i]
        if odd["axes"][name]["kind"] == "cat":
            pins[name] = b
        else:
            lo, hi = b
            if hi - lo < 1e-9:
                lo, hi = lo, lo + max(1e-6, abs(lo) * 1e-3)
            ranges[name] = (lo, hi)
    q = lang.Query(text=f"gap-directed order-2 coverage of {odd.get('name')}", pins=pins, ranges=ranges)
    q.weights = {"targets": {n: i for n, i in best_assign.items()}}
    return {"query": q, "cell": ";".join(f"{n}:{i}" for n, i in sorted(best_assign.items())),
            "missing_before": len(missing), "would_cover": len(best_new),
            "targets": {n: _bin_label(bins[n][i]) for n, i in sorted(best_assign.items())}}


def coverage_until(odd: dict, target: float = 0.9, max_scenarios: int = 400,
                   strategy: str = "gap", bins_per_axis: int = 4, seed: int = 1) -> dict:
    """Generate until the target order-2 coverage is reached, by one of two
    strategies:

        gap     each new scenario is constructed to cover as many still-missing
                interaction cells as possible (propose_next)
        random  each new scenario is drawn uniformly from the whole ODD

    Both draw from the same sampler over the same axis grid, so the comparison in
    bench/experiment.py measures the aiming and not two different generators.
    """
    import random as _random

    rng = _random.Random(seed)
    corpus, log, cell = [], [], None
    for k in range(max_scenarios):
        if strategy == "gap":
            nxt = propose_next(odd, corpus, bins_per_axis, seed + k)
            if nxt["query"] is None:
                break
            q = nxt["query"]
            cell = nxt["cell"]
        else:
            pins = {}
            for name, spec in odd["axes"].items():
                if spec["kind"] == "cat":
                    pins[name] = rng.choice(spec["values"])
            q = lang.Query(text="uniform odd draw", pins=pins)
            for name, spec in odd["axes"].items():
                if spec["kind"] == "range":
                    q.ranges[name] = (spec["lo"], spec["hi"])
        corpus.append(generate.scenario_from_query(q, seed * 100003 + k))
        if (k + 1) % 5 == 0 or k == 0:
            cov = coverage(odd, corpus, bins_per_axis)
            log.append({"k": k + 1, "order2": cov["order2_coverage"],
                        "order1": cov["order1_coverage"], "cell": cell})
            if cov["order2_coverage"] >= target:
                break
    cov = coverage(odd, corpus, bins_per_axis)
    log.append({"k": len(corpus), "order2": cov["order2_coverage"],
                "order1": cov["order1_coverage"], "cell": cell})
    return {"strategy": strategy, "scenarios_used": len(corpus), "target": target,
            "reached": cov["order2_coverage"] >= target,
            "coverage": cov["order2_coverage"],
            "order1_coverage": cov["order1_coverage"],
            "log": log, "corpus_ids": [s.id for s in corpus], "report": cov,
            "corpus": corpus}


def gap_score(odd: dict, corpus: list, cell: str, bins: dict) -> float:
    """How far a named (axis:bin, axis:bin) interaction cell is from being covered.

    Kept as a standalone because a caller may want to score a cell it proposes
    itself; the sampler uses the greedy construction in propose_next, which
    optimises the same quantity without scoring one cell at a time.
    """
    parts = [p for p in cell.split(";") if p]
    if len(parts) < 2:
        return 1.0
    try:
        (a, ia), (b, ib) = [(p.split(":")[0], int(p.split(":")[1])) for p in parts[:2]]
    except (ValueError, IndexError):
        return 1.0
    hit = False
    for sc in corpus:
        idx, is_out = _cell_indices(sc, odd, bins)
        if is_out:
            continue
        if idx.get(a) == ia and idx.get(b) == ib:
            return 0.0
        if idx.get(a) == ia or idx.get(b) == ib:
            hit = True
    return 0.5 if hit else 1.0


def _bin_label(b) -> str:
    """A bin as a readable span, for the report and for a person reading it."""
    if isinstance(b, tuple):
        lo, hi = b
        return f"[{lo:.4g}, {hi:.4g}]"
    return str(b)
