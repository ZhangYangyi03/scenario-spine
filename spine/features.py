"""Feature extraction: a scenario in, a fixed-length vector out.

The features are split into two classes, and the split is the module's whole
point:

    PRE-EXECUTION features   computable before the scenario is ever run
    POST-EXECUTION outcomes  known only after simulating it

A predictor is only useful if it sees the first kind. That is what makes the
learned model in spine/learn.py a *predictor* of risk rather than a copy of the
simulator's output, and it is why the physics score in spine/difficulty.py is
built from the same first class: the two are then comparable, because they are
answering the same question from the same information.

Encoding choices, stated because they are not neutral:

    * maneuver / weather / road kind are one-hot, not ordinal. Ice is not "more
      weather" than snow, and an ordinal code would assert that it is.
    * distances and radii enter as log1p, not raw. The interesting variation in
      a radius is multiplicative (30 m vs 300 m), and a raw feature would let
      the two large values dominate a distance metric.
    * the derived physical terms (grip fraction, stopping distance over sight
      distance, headway in seconds) are included explicitly rather than left for
      the model to rediscover from raw parameters. A model that has to learn
      s = v^2/(2*a*g) from data needs many more samples to learn it than one that
      is handed the ratio, and the ratio is what a person would compute anyway.
"""

from __future__ import annotations

import math

from . import lang
from .difficulty import required_lateral_accel, stopping_distance, visible_hazard_distance
from .ir import G, Scenario

G = 9.81

CATEGORICAL_BLOCKS = {
    "maneuver.type": lang.AXES["maneuver.type"]["values"],
    "weather.condition": lang.AXES["weather.condition"]["values"],
    "road.kind": lang.AXES["road.kind"]["values"],
    "maneuver.occlusion": lang.AXES["maneuver.occlusion"]["values"],
}

#: Continuous, pre-execution scalar features, in order.
NUMERIC_FEATURES = [
    "log_ego_speed",
    "log_speed_limit",
    "log_radius",
    "grade_over_10",
    "lanes_over_3",
    "mu",
    "log_traffic_flow",
    "mix_truck",
    "mix_bicycle",
    "mix_pedestrian",
    "log_visible_hazard",
    "log_stopping_distance",
    "log_t_gap",
    "decel_over_8",
    "log_trigger_dist",
    "grip_demand_fraction",
    "stopping_over_visible",
    "headway_s",
    "fog_log_ratio",
    "luminance_log",
    "bicycle_flag",
    "pedestrian_flag",
    "truck_flag",
]


def feature_names() -> list:
    names = list(NUMERIC_FEATURES)
    for block, values in CATEGORICAL_BLOCKS.items():
        names.extend(f"{block}={v}" for v in values)
    return names


def _log1p_abs(x: float) -> float:
    return math.log1p(abs(float(x)))


def features(sc: Scenario) -> list:
    """The pre-execution vector. Every term is computable from the scenario
    object alone: no simulation, no outcome, no future information."""
    p = sc.params
    ego = next((a for a in sc.actors if a.id == "ego"), None)
    v = ego.v0 if ego else 0.0
    mu = sc.weather.friction()
    vis = visible_hazard_distance(sc)
    stop = stopping_distance(sc)
    lat = required_lateral_accel(sc)

    # headway to the nearest in-lane actor, in seconds; 0 if none ahead
    headway = 0.0
    for a in sc.actors:
        if a.id == "ego" or abs(a.lane) > 0.6:
            continue
        gap = a.s0 - (ego.s0 if ego else 0.0) - (a.length_width()[0] + (ego.length_width()[0] if ego else 4.6)) / 2.0
        if gap > 0 and v > 1e-6:
            headway = gap / v if headway == 0.0 else min(headway, gap / v)

    row = [
        math.log1p(v),
        math.log1p(float(p.get("road.speed_limit_kph", 90.0))),
        _log1p_abs(float(p.get("road.radius_m") or 0.0)),
        float(p.get("road.grade_pct", 0.0)) / 10.0,
        int(p.get("road.lanes", 2)) / 3.0,
        mu,
        math.log1p(max(0.0, sc.traffic.flow_vph)),
        sc.traffic.mix_truck,
        sc.traffic.mix_bicycle,
        sc.traffic.mix_pedestrian,
        math.log1p(min(vis, 1e6)),
        math.log1p(min(stop, 1e5)),
        math.log1p(float(p.get("maneuver.t_gap_s", 2.0))),
        float(p.get("maneuver.decel_mps2", 4.0)) / 8.0,
        math.log1p(float(p.get("maneuver.trigger_dist_m", 60.0))),
        min(3.0, lat / (G * max(0.05, mu))),
        min(20.0, stop / max(1.0, vis)),
        min(30.0, headway),
        math.log1p(sc.weather.fog_visibility_m) - math.log1p(10000.0),
        math.log1p(max(0.1, sc.weather.illuminance_lux)) / math.log1p(100000.0),
        1.0 if sc.traffic.mix_bicycle > 0.05 else 0.0,
        1.0 if sc.traffic.mix_pedestrian > 0.05 else 0.0,
        1.0 if sc.traffic.mix_truck > 0.25 else 0.0,
    ]
    for block, values in CATEGORICAL_BLOCKS.items():
        cur = p.get(block)
        row.extend(1.0 if cur == val else 0.0 for val in values)
    return row


def outcome_row(outcome) -> dict:
    """The post-execution targets. Kept separate from features() on purpose: a
    single function returning both would make it possible to feed a target into a
    feature by accident, and that accident is undetectable in the results."""
    from .sim import Outcome

    if not isinstance(outcome, Outcome):
        raise TypeError("outcome_row expects a sim.Outcome")
    return {
        "collision": int(bool(outcome.collision)),
        "risk_index": outcome.risk_index(),
        "min_ttc_s": min(outcome.min_ttc, 99.0),
        "min_gap_m": outcome.min_gap,
        "max_decel_mps2": outcome.max_decel,
        "peak_jerk": outcome.peak_jerk,
        "hard_brake": int(bool(outcome.hard_brake)),
    }
