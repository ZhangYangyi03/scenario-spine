"""The intermediate representation: what it refuses, and whether the road it
describes is the road it says it describes."""

import math

import pytest

from spine import ir


def test_actor_rejects_unknown_class():
    with pytest.raises(ir.ScenarioError):
        ir.Actor(id="x", cls="hoverboard", lane=0.0, s0=0.0, v0=1.0)


def test_actor_rejects_negative_speed_and_position():
    with pytest.raises(ir.ScenarioError):
        ir.Actor(id="x", cls="car", lane=0.0, s0=0.0, v0=-1.0)
    with pytest.raises(ir.ScenarioError):
        ir.Actor(id="x", cls="car", lane=0.0, s0=-1.0, v0=1.0)


def test_road_without_segments_is_refused():
    with pytest.raises(ir.ScenarioError):
        ir.road_from_params({})


def test_segment_lengths_must_be_positive():
    with pytest.raises(ir.ScenarioError):
        ir.road_from_params({"segments": [{"length": 0.0}]})


def test_road_segments_are_contiguous_and_carry_their_radius():
    road = ir.road_from_params({"segments": [
        {"length": 200.0, "curvature_r": 0.0, "speed_limit_kph": 100.0, "lanes": 2},
        {"length": 150.0, "curvature_r": 300.0, "speed_limit_kph": 80.0, "lanes": 2},
    ], "lane_width_m": 3.5})
    segs = road["segments"]
    assert segs[0]["s0"] == 0.0 and segs[0]["s1"] == 200.0
    assert segs[1]["s0"] == 200.0, "segments must tile the road without a gap"
    assert segs[1]["radius_m"] == 300.0 and segs[1]["sign"] == 1
    assert road["total_length_m"] == pytest.approx(350.0)
    assert road["lane_width_m"] == 3.5


def test_weather_friction_ordering():
    """Dry grips better than wet grips better than snow grips better than ice.
    A weather model that did not preserve this ordering would make the difficulty
    score's surface component non-monotone."""
    mus = [ir.Weather(condition=c).friction() for c in ("dry", "wet", "rain", "snow", "ice")]
    assert all(a > b for a, b in zip(mus, mus[1:])), mus
    assert ir.Weather(condition="ice").friction() >= 0.05, "friction must not reach zero"


def test_scenario_roundtrips_through_json():
    road = ir.road_from_params({"segments": [{"length": 100.0}]})
    sc = ir.Scenario(id="s1", seed=7, params={"a": 1}, road=road,
                     actors=[ir.Actor(id="ego", cls="car", lane=0.0, s0=0.0, v0=10.0)],
                     weather=ir.Weather(condition="rain", rain_mmh=4.0),
                     traffic=ir.Traffic(flow_vph=900.0), tags=["x"])
    back = ir.Scenario.from_dict(sc.to_dict())
    assert back.id == sc.id and back.seed == sc.seed
    assert back.actors[0].v0 == 10.0 and back.actors[0].cls == "car"
    assert back.weather.condition == "rain" and back.weather.rain_mmh == 4.0
    assert back.traffic.flow_vph == 900.0
    assert back.road["total_length_m"] == pytest.approx(100.0)


def test_actor_dimensions_are_physically_ordered():
    d = {c: ir.Actor(id="a", cls=c, lane=0.0, s0=0.0, v0=0.0) for c in ir.ACTOR_CLASSES}
    assert d["truck"].length_width()[0] > d["car"].length_width()[0] > d["bicycle"].length_width()[0]
