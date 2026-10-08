"""The scenario description language.

A scenario is a sentence in this grammar, and the grammar exists so that three
things are the same object:

    * a *query* a person writes ("motorway on-ramp merge, rain, lead brakes")
    * a *parameter vector* a sampler draws from
    * the *description* attached to a generated scenario

Because the three share one structure, "generate a scenario that is hard" and
"measure how hard this scenario is" and "find a scenario nobody has generated"
are all operations on the same space rather than on three translations of it.

    <axis> := name [values...] (default=v) (weight=w)
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from .ir import ScenarioError

# The ODD axes. Categorical axes are enumerated; continuous axes are sampled
# from a log-uniform grid between lo and hi, which is the right prior for
# distances and radii (a 20 m radius is a different regime from a 2 km one, and
# a linear grid would spend almost all its points in the flat part).
AXES = {
    "road.kind": dict(kind="cat", values=["motorway", "urban", "rural", "junction", "parking"]),
    "road.radius_m": dict(kind="range", lo=15.0, hi=2000.0, log=True, default=0.0),
    "road.lanes": dict(kind="cat", values=[1, 2, 3]),
    "road.grade_pct": dict(kind="range", lo=-8.0, hi=8.0, default=0.0),
    "road.speed_limit_kph": dict(kind="range", lo=10.0, hi=130.0, default=90.0),
    "road.segment_len_m": dict(kind="range", lo=40.0, hi=1200.0, default=300.0),
    "weather.condition": dict(kind="cat", values=["dry", "wet", "rain", "snow", "ice"]),
    "weather.rain_mmh": dict(kind="range", lo=0.0, hi=60.0, default=0.0),
    "weather.fog_visibility_m": dict(kind="cat", values=[30.0, 80.0, 200.0, 1000.0, 10000.0]),
    "weather.illuminance_lux": dict(kind="cat", values=[5.0, 50.0, 500.0, 5000.0, 100000.0]),
    "ego.speed_kph": dict(kind="range", lo=5.0, hi=130.0, default=60.0),
    "traffic.flow_vph": dict(kind="range", lo=0.0, hi=3600.0, default=600.0),
    "traffic.mix_truck": dict(kind="range", lo=0.0, hi=0.5, default=0.1),
    "traffic.mix_bicycle": dict(kind="range", lo=0.0, hi=0.4, default=0.0),
    "traffic.mix_pedestrian": dict(kind="range", lo=0.0, hi=0.4, default=0.0),
    "maneuver.type": dict(
        kind="cat",
        values=[
            "lane_keep",
            "cut_in",
            "lead_brake",
            "on_ramp_merge",
            "off_ramp_diverge",
            "crossing",
            "pedestrian_crossing",
            "static_obstacle",
            "unprotected_left",
            "roundabout",
        ],
    ),
    "maneuver.t_gap_s": dict(kind="range", lo=0.3, hi=6.0, default=2.0),
    "maneuver.decel_mps2": dict(kind="range", lo=1.0, hi=8.0, default=4.0),
    "maneuver.occlusion": dict(kind="cat", values=["none", "parked_row", "truck", "building", "vegetation"]),
    "maneuver.trigger_dist_m": dict(kind="range", lo=5.0, hi=200.0, default=60.0),
}

CATEGORICAL = {k: v for k, v in AXES.items() if v["kind"] == "cat"}
CONTINUOUS = {k: v for k, v in AXES.items() if v["kind"] == "range"}


@dataclass
class Query:
    """A point or a region of the axes. `pins` fixes an axis, `ranges` bounds a
    continuous one (overriding its natural range), `weights` biases sampling."""

    pins: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    text: str = ""

    def to_dict(self) -> dict:
        return {"pins": self.pins, "ranges": self.ranges, "weights": self.weights, "text": self.text}


def axis_bounds(name: str, q: Query) -> tuple:
    a = AXES[name]
    lo, hi = q.ranges.get(name, (a["lo"], a["hi"]))
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def grid(name: str, q: Query, n: int) -> list:
    """n samples of one axis. Continuous axes use a log grid unless they cross
    zero, in which case log is meaningless and a linear grid is used."""
    a = AXES[name]
    if a["kind"] == "cat":
        vals = list(a["values"])
        if name in q.pins:
            vals = [v for v in vals if v == q.pins[name]] or [q.pins[name]]
        if n >= len(vals):
            return vals
        step = len(vals) / n
        return [vals[min(len(vals) - 1, int(i * step))] for i in range(n)]
    lo, hi = axis_bounds(name, q)
    if name in q.pins:
        return [q.pins[name]] * n
    if n == 1:
        return [lo]
    if a.get("log") and lo > 0:
        import math

        return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (n - 1)) for i in range(n)]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def parse(text: str) -> Query:
    """Read a query sentence. Two forms are accepted:

        merge on-ramp | weather.condition=rain | maneuver.decel_mps2>4
        maneuver.type=cut_in, road.kind=motorway, weather.condition in rain,snow

    Anything not recognised as an axis is kept as a free-text tag, because a
    query that names something the grammar does not know is still a request.
    """
    q = Query(text=text)
    if not text:
        return q
    parts = re.split(r"[|;]", text)
    if len(parts) == 1:
        parts = re.split(r",", text)
    for raw in parts:
        tok = raw.strip()
        if not tok:
            continue
        m = re.match(r"^([a-z0-9_.]+)\s*(=|>=|<=|>|<|in)\s*(.+)$", tok)
        if not m:
            continue
        name, op, val = m.group(1), m.group(2), m.group(3).strip()
        if name not in AXES:
            continue
        if op == "in":
            vals = [v.strip() for v in val.split(",") if v.strip()]
            q.ranges[name] = (vals, vals)
            continue
        cast = _cast(name, val)
        if cast is None:
            continue
        if op == "=":
            q.pins[name] = cast
        elif op == ">":
            lo, hi = axis_bounds(name, q)
            q.ranges[name] = (max(lo, cast), hi)
        elif op == ">=":
            lo, hi = axis_bounds(name, q)
            q.ranges[name] = (max(lo, cast), hi)
        elif op == "<":
            lo, hi = axis_bounds(name, q)
            q.ranges[name] = (lo, min(hi, cast))
        elif op == "<=":
            lo, hi = axis_bounds(name, q)
            q.ranges[name] = (lo, min(hi, cast))
    return q


def _cast(name, val):
    a = AXES[name]
    try:
        if a["kind"] == "cat" and a["values"] and isinstance(a["values"][0], str):
            return val
        if a["kind"] == "cat":
            return int(float(val)) if float(val).is_integer() else float(val)
        return float(val)
    except (TypeError, ValueError):
        return None


def query_from_recipe(recipe: dict) -> Query:
    """The structured form of a query, used by the recipe library."""
    q = Query(text=recipe.get("text", ""))
    for k, v in (recipe.get("pins") or {}).items():
        if k in AXES:
            q.pins[k] = v
    for k, v in (recipe.get("ranges") or {}).items():
        if k in AXES and isinstance(v, (list, tuple)) and len(v) == 2:
            q.ranges[k] = (v[0], v[1])
    for k, v in (recipe.get("weights") or {}).items():
        if k in AXES:
            q.weights[k] = v
    return q


def recipes() -> dict:
    """Named queries. These are the entry points a person actually uses, and
    they are also the benchmark's task list."""
    R = {
        "motorway_onramp_merge": {
            "text": "motorway on-ramp merge, moderate traffic",
            "pins": {"road.kind": "motorway", "maneuver.type": "on_ramp_merge"},
            "ranges": {"ego.speed_kph": (70.0, 110.0), "traffic.flow_vph": (600.0, 1800.0)},
        },
        "urban_cutin_rain": {
            "text": "urban cut-in in rain",
            "pins": {"road.kind": "urban", "maneuver.type": "cut_in", "weather.condition": "rain"},
            "ranges": {"ego.speed_kph": (20.0, 50.0), "maneuver.t_gap_s": (0.5, 2.0)},
        },
        "fog_lead_brake": {
            "text": "lead vehicle emergency brake in fog",
            "pins": {"maneuver.type": "lead_brake"},
            "ranges": {"weather.fog_visibility_m": (30.0, 200.0), "maneuver.decel_mps2": (5.0, 8.0)},
        },
        "occluded_pedestrian": {
            "text": "pedestrian from behind a parked row",
            "pins": {"maneuver.type": "pedestrian_crossing", "maneuver.occlusion": "parked_row"},
            "ranges": {"ego.speed_kph": (15.0, 45.0)},
        },
        "night_rural_bicycle": {
            "text": "unlit cyclist at night on a rural road",
            "pins": {"road.kind": "rural", "weather.illuminance_lux": 5.0},
            "ranges": {"traffic.mix_bicycle": (0.1, 0.4)},
        },
        "ice_curve": {
            "text": "tight curve on ice",
            "pins": {"weather.condition": "ice"},
            "ranges": {"road.radius_m": (20.0, 200.0), "ego.speed_kph": (40.0, 90.0)},
        },
        "junction_unprotected_left": {
            "text": "unprotected left turn at a junction",
            "pins": {"road.kind": "junction", "maneuver.type": "unprotected_left"},
            "ranges": {"traffic.flow_vph": (200.0, 1200.0)},
        },
        "roundabout_merge": {
            "text": "roundabout entry with circulating traffic",
            "pins": {"maneuver.type": "roundabout"},
            "ranges": {"traffic.flow_vph": (300.0, 1500.0)},
        },
        "truck_cutin_highway": {
            "text": "truck cuts in at motorway speed",
            "pins": {"maneuver.type": "cut_in", "road.kind": "motorway"},
            "ranges": {"traffic.mix_truck": (0.3, 0.5), "ego.speed_kph": (90.0, 130.0)},
        },
        "snow_offramp": {
            "text": "diverge to an off-ramp in snow",
            "pins": {"maneuver.type": "off_ramp_diverge", "weather.condition": "snow"},
        },
    }
    return R
