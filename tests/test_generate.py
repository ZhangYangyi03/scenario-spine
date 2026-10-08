"""The generator is held to the property that makes the rest of the package
worth anything: the parameters a scenario carries are the ones its geometry
actually realises. A generator that samples offsets and then labels them with a
time gap it did not solve for would make the difficulty score and the coverage
report describe a scenario that does not exist."""

import math

import pytest

from spine import generate, lang


def _p(**kw):
    """A complete parameter vector, so scenario_from_params can be called with a
    single axis overridden without the other axes being absent."""
    base = {name: (a["default"] if a["kind"] == "range" else a["values"][0])
            for name, a in lang.AXES.items()}
    base.update(kw)
    base["seed"] = 0
    return base


def test_every_declared_maneuver_can_be_generated():
    for man in lang.AXES["maneuver.type"]["values"]:
        sc = generate.scenario_from_params(_p(**{"maneuver.type": man}))
        assert len(sc.actors) >= 1 and sc.actors[0].id == "ego"
        ids = [a.id for a in sc.actors]
        assert len(ids) == len(set(ids)), f"{man} produced duplicate actor ids"


def test_time_gap_is_realised_not_sampled():
    """lane_keep places the leader at ego_v * t_gap. The claim is checkable in
    closed form, so it is checked rather than assumed."""
    for gap in (0.5, 1.0, 2.5, 4.0):
        sc = generate.scenario_from_params(_p(**{"maneuver.type": "lane_keep",
                                                 "ego.speed_kph": 72.0,
                                                 "maneuver.t_gap_s": gap}))
        ego = next(a for a in sc.actors if a.id == "ego")
        lead = next(a for a in sc.actors if a.id == "lead")
        # the gap is between *bodies*, so the half-lengths come off, exactly as
        # difficulty.py measures it; measuring centre-to-centre would silently
        # report a headway larger than the one asked for
        half = (lead.length_width()[0] + ego.length_width()[0]) / 2.0
        headway = (lead.s0 - ego.s0 - half) / ego.v0
        assert headway == pytest.approx(gap, rel=1e-6), (gap, headway)


def test_lead_brake_deceleration_is_the_declared_one():
    sc = generate.scenario_from_params(_p(**{"maneuver.type": "lead_brake",
                                            "maneuver.decel_mps2": 6.5}))
    lead = next(a for a in sc.actors if a.id == "lead")
    accels = [b["a"] for b in lead.behavior if "a" in b]
    assert min(accels) == pytest.approx(-6.5)


def test_higher_query_speed_produces_higher_initial_speed():
    slow = generate.scenario_from_params(_p(**{"ego.speed_kph": 20.0}))
    fast = generate.scenario_from_params(_p(**{"ego.speed_kph": 120.0}))
    assert (next(a for a in fast.actors if a.id == "ego").v0 >
            next(a for a in slow.actors if a.id == "ego").v0)


def test_occluded_pedestrian_geometry_is_reachable_by_the_ego():
    """The pedestrian must start ahead of the ego, not behind it, or the scenario
    is not a conflict at all."""
    sc = generate.scenario_from_params(_p(**{"maneuver.type": "pedestrian_crossing",
                                            "ego.speed_kph": 40.0,
                                            "maneuver.t_gap_s": 2.0}))
    ego = next(a for a in sc.actors if a.id == "ego")
    ped = next(a for a in sc.actors if a.id == "ped")
    assert ped.s0 > ego.s0
    assert ped.cls == "pedestrian"


def test_seeding_is_deterministic():
    """Two calls with the same seed must produce byte-identical scenarios, or the
    corpus the coverage report describes cannot be reproduced."""
    a = generate.generate(lang.Query(text="x"), 12345)
    b = generate.generate(lang.Query(text="x"), 12345)
    assert a.to_json() == b.to_json()


def test_different_seeds_produce_different_scenarios():
    """Not a statistical test -- a check that the seed actually enters the draw.
    Twenty seeds yielding one scenario would be a silently pinned generator."""
    seen = {generate.generate(lang.Query(text="x"), s).to_json() for s in range(20)}
    assert len(seen) > 12
    for name in ("weather.condition", "maneuver.type", "road.kind"):
        vals = {generate.generate(lang.Query(text="x"), s).params[name] for s in range(24)}
        assert len(vals) > 1, f"{name} never varied across seeds"


def test_pinned_axes_are_respected_exactly():
    q = lang.Query(text="pinned", pins={"weather.condition": "snow", "road.kind": "rural",
                                       "maneuver.type": "lead_brake"})
    for s in range(8):
        sc = generate.generate(q, s)
        assert sc.weather.condition == "snow"
        assert sc.road["kind"] == "rural"
        assert sc.params["maneuver.type"] == "lead_brake"


def test_query_parses_the_documented_syntax():
    q = lang.parse("road.kind=motorway | weather.condition=rain | maneuver.decel_mps2>5")
    assert q.pins["road.kind"] == "motorway"
    assert q.pins["weather.condition"] == "rain"
    lo, hi = q.ranges["maneuver.decel_mps2"]
    assert lo == 5.0 and hi > 5.0


def test_range_restriction_narrows_the_draw():
    q = lang.Query(text="narrow", pins={"maneuver.type": "lead_brake"},
                   ranges={"ego.speed_kph": (100.0, 110.0)})
    vals = [generate.generate(q, s).params["ego.speed_kph"] for s in range(20)]
    assert all(100.0 <= v <= 110.0 for v in vals)
    assert max(vals) - min(vals) > 0.5, "the range must not collapse to a point"


def test_recipes_all_generate():
    for name, r in lang.recipes().items():
        q = lang.query_from_recipe(r)
        sc = generate.generate(q, 3)
        assert sc.actors, f"recipe {name} generated no actors"
