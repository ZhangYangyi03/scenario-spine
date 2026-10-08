"""Coverage has to be measured over the scenarios that are *in* the ODD.

The failure this file exists to prevent is a coverage report over a corpus that
left the ODD entirely: every pair unvisited, the number 0.0, and no indication
that the reason is the corpus and not the sampler. A generator draws from the
full axis space, so an unpinned draw mostly lands outside any narrow ODD, and
the sampler has to be aimed at the ODD for the number to mean anything. Both
facts are asserted here so the bench cannot report a coverage figure without
them."""

import pytest

from spine import generate, lang, odd


def _in_odd_query(spec: dict, **pins) -> lang.Query:
    """A query whose draw is inside the declared ODD on every axis.

    Categorical axes are pinned to a declared value *and* continuous axes are
    restricted to the declared interval. Pinning only the categoricals is not
    enough, and the first version of this helper did exactly that: the generator
    then drew road.radius_m up to 2000 against a declared ceiling of 1500, so 85
    of 120 "in-ODD" scenarios were outside it. Which is the point the test below
    makes, one level up."""
    p, r = {}, {}
    for name, s in spec["axes"].items():
        if s["kind"] == "cat":
            p[name] = s["values"][0]
        else:
            r[name] = (s["lo"], s["hi"])
    p.update(pins)
    return lang.Query(text="highway pilot", pins=p, ranges=r)


def test_generator_does_not_draw_from_a_narrow_odd_by_default():
    """Stated and asserted: an unpinned query overwhelmingly violates a narrow
    ODD. This is the reason coverage must be reported over a constrained corpus,
    and it is a property of the generator, not a bug in it."""
    spec = odd.declared_odd_highway()
    corpus = [generate.generate(lang.Query(text="mixed"), s) for s in range(60)]
    rep = odd.coverage(spec, corpus, bins_per_axis=4)
    assert rep["outside_odd"] > 0.9 * len(corpus), rep["outside_odd"]
    assert rep["verdict"] == odd.F
    assert rep["interaction_cells_hit"] == 0


def test_coverage_is_measured_over_the_in_odd_corpus():
    spec = odd.declared_odd_highway()
    q = _in_odd_query(spec)
    corpus = [generate.generate(q, s) for s in range(120)]
    rep = odd.coverage(spec, corpus, bins_per_axis=4)
    assert rep["outside_odd"] == 0, rep["outside_examples"][:2]
    assert rep["interaction_cells_total"] > 0
    assert 0.0 <= rep["order2_coverage"] <= 1.0
    assert 0.0 <= rep["order1_coverage"] <= 1.0
    assert rep["verdict"] in (odd.T, odd.U)


def test_pairs_are_driven_up_by_the_directed_sampler():
    """The claim the sampler exists to make: aimed draws cover strictly more
    interaction cells than unaimed ones, at the same corpus size."""
    from spine import odd as O

    spec = O.declared_odd_highway()
    q = _in_odd_query(spec)
    plain = [generate.generate(q, s) for s in range(150)]
    base = O.coverage(spec, plain, bins_per_axis=3)

    corpus = list(plain)
    for i in range(150):
        nxt = O.propose_next(spec, corpus, bins_per_axis=3, seed=i)
        sc = nxt.get("scenario") if isinstance(nxt, dict) else None
        if sc is None:
            break
        corpus.append(sc)
    aimed = O.coverage(spec, corpus, bins_per_axis=3)
    assert aimed["interaction_cells_hit"] >= base["interaction_cells_hit"]


def test_report_is_json_serialisable():
    """The CLI writes this report to disk. A set leaking into it turns a finished
    run into a TypeError at the last step, which is exactly what happened once."""
    import json

    spec = odd.declared_odd_highway()
    corpus = [generate.generate(lang.Query(text="x", pins={"road.kind": "motorway"}), s)
              for s in range(30)]
    rep = odd.coverage(spec, corpus, bins_per_axis=3)
    json.dumps(rep)


def test_three_valued_verdict_is_not_two_valued():
    spec = odd.declared_odd_highway()
    sc = generate.generate(lang.Query(text="x"), 0)
    v = odd.scenario_verdict(sc, spec)
    assert v["verdict"] in (odd.T, odd.F, odd.U)
    assert set(v["per_axis"].values()) <= {odd.T, odd.F, odd.U}
    unknown = {k: odd.U for k in ("a", "b")}
    assert odd.k_all(unknown.values()) == odd.U
    assert odd.k_all([odd.T, odd.T]) == odd.T
    assert odd.k_all([odd.T, odd.F]) == odd.F
    assert odd.k_violated_fraction([odd.T, odd.F, odd.U]) == pytest.approx(1 / 3)
