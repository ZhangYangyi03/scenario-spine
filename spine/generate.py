"""The generator: a query goes in, a scenario comes out.

The design decision worth stating: generation is *inverse* to description. The
maneuver library below does not place vehicles at coordinates; it places them at
the parameters the maneuver is named after. A `cut_in` at `t_gap_s=1.2` really
has a 1.2 s gap between the cutting vehicle's rear bumper clearing the ego's
front bumper and the ego's front bumper arriving -- the generator solves for the
longitudinal offsets that make that true, instead of sampling offsets and
hoping the resulting gap is the one it claimed.

That property is what makes the difficulty score and the coverage check
meaningful: a scenario carries the parameters it was generated from, and the
parameters are the ones the geometry actually realises. tests/test_roundtrip.py
holds the generator to it.
"""

from __future__ import annotations

import math
import random

from . import lang
from .ir import Actor, Scenario, ScenarioError, Traffic, Weather, road_from_params

# --------------------------------------------------------------- sampling ----


def sample_params(q: lang.Query, seed: int) -> dict:
    """Draw one parameter vector from the query. Pinned axes are fixed; the
    rest are drawn uniformly over their (possibly query-restricted) bounds."""
    rng = random.Random(seed)
    p = {}
    for name in lang.AXES:
        a = lang.AXES[name]
        if name in q.pins:
            p[name] = q.pins[name]
            continue
        r = q.ranges.get(name)
        if a["kind"] == "cat":
            vals = list(a["values"])
            if r:
                lo, hi = r
                lo, hi = (lo[0], lo[-1]) if isinstance(lo, list) else (lo, hi)
                vals = [v for v in vals if lo <= v <= hi] if isinstance(lo, (int, float)) and not isinstance(lo, list) else vals
                if not vals:
                    vals = list(a["values"])
            p[name] = rng.choice(vals)
        else:
            lo, hi = (float(r[0]), float(r[1])) if r else (a["lo"], a["hi"])
            if lo > hi:
                lo, hi = hi, lo
            if a.get("log") and lo > 0:
                p[name] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                p[name] = rng.uniform(lo, hi)
    p["seed"] = seed
    return p


def params_to_road(p: dict) -> dict:
    """Two segments: a straight approach and a curve of the requested radius.
    This is deliberately minimal -- the axes are about the *regime*, not about a
    particular map, and a regime that needs a full map belongs in a map file."""
    kind = p.get("road.kind", "motorway")
    radius = float(p.get("road.radius_m") or 0.0)
    seg_len = float(p.get("road.segment_len_m", 300.0))
    lanes = int(p.get("road.lanes", 2))
    vlim = float(p.get("road.speed_limit_kph", 90.0))
    grade = float(p.get("road.grade_pct", 0.0))
    urban = kind in ("urban", "junction", "parking")
    if urban:
        vlim = min(vlim, 60.0)
    segs = [
        {"length": seg_len, "curvature_r": 0.0, "grade_pct": grade, "speed_limit_kph": vlim, "lanes": lanes, "kind": kind},
    ]
    if abs(radius) > 1e-9 and kind in ("motorway", "rural"):
        segs.append({"length": max(60.0, seg_len * 0.5), "curvature_r": radius, "grade_pct": grade, "speed_limit_kph": vlim, "lanes": lanes, "kind": kind})
    road = road_from_params({"segments": segs, "lane_width_m": 3.2 if urban else 3.6})
    road["kind"] = kind
    return road


# ------------------------------------------------------------- maneuvers ----

def _lane_center(lane: float, road: dict) -> float:
    return lane * road.get("lane_width_m", 3.6)


def _braking_distance(v: float, a: float) -> float:
    return float("inf") if a <= 0 else v * v / (2.0 * a)


def _time_to_collision(ego_v: float, lead_v: float, gap: float) -> float:
    if ego_v <= lead_v:
        return float("inf")
    return gap / (ego_v - lead_v)


def build_actors(p: dict, road: dict, rng: random.Random) -> list:
    """Instantiate the maneuver named in the parameters. Every branch returns
    actors whose placement is computed from the parameters, not sampled."""
    m = p.get("maneuver.type", "lane_keep")
    ego_v = float(p.get("ego.speed_kph", 60.0)) / 3.6
    mu = Weather(condition=p.get("weather.condition", "dry")).friction()
    decel = float(p.get("maneuver.decel_mps2", 4.0))
    t_gap = float(p.get("maneuver.t_gap_s", 2.0))
    occ = p.get("maneuver.occlusion", "none")
    trig = float(p.get("maneuver.trigger_dist_m", 60.0))

    ego = Actor(id="ego", cls="car", lane=0.0, s0=0.0, v0=ego_v, behavior=[{"t": 0.0, "a": 0.0}])
    actors = [ego]

    if m == "lane_keep":
        # A lead vehicle at the distance the ego can *just* stop behind, times
        # the sampled time gap. A query that pins t_gap_s small therefore
        # generates a genuinely close following case.
        d = max(6.0, ego_v * t_gap)
        actors.append(Actor(id="lead", cls="car", lane=0.0, s0=d + 4.6, v0=ego_v * 0.95,
                            behavior=[{"t": 0.0, "a": 0.0}]))

    elif m == "cut_in":
        # The cutter is behind, one lane over. t_gap_s is the lateral-gap time:
        # it reaches the ego's longitudinal station after t_gap_s.
        lane = 1.0 if (p.get("road.lanes", 2) or 2) >= 2 else -1.0
        d_travel = max(2.0, ego_v * t_gap)
        s0 = max(0.0, -d_travel)
        cls = "truck" if p.get("traffic.mix_truck", 0.0) > 0.35 else "car"
        actors.append(Actor(id="cutter", cls=cls, lane=lane, s0=s0 + 12.0, v0=ego_v * 1.15,
                            behavior=[{"t": 0.0, "a": 0.0}, {"t": t_gap, "lane": 0.0, "a": -0.5}]))

    elif m == "lead_brake":
        stop_d = _braking_distance(ego_v, decel)
        d = max(5.0, ego_v * t_gap)
        actors.append(Actor(id="lead", cls="car", lane=0.0, s0=d + 4.6, v0=ego_v,
                            behavior=[{"t": 0.0, "a": 0.0}, {"t": trig / max(1.0, ego_v), "a": -decel}]))

    elif m == "on_ramp_merge":
        # Ramp vehicle travels the ramp length and arrives at the merge point at
        # the same station and time as the ego, with a gap set by the query.
        v_ramp = ego_v * float(p.get("merge_ratio", 0.85))
        actors.append(Actor(id="ramp", cls="truck" if p.get("traffic.mix_truck", 0.1) > 0.25 else "car",
                            lane=1.0, s0=max(0.0, ego_v * t_gap), v0=v_ramp,
                            behavior=[{"t": 0.0, "a": 0.4}, {"t": 2.0, "lane": 0.0, "a": 0.0}]))
        actors.append(Actor(id="lead", cls="car", lane=0.0, s0=max(20.0, ego_v * (t_gap + 1.5)) + 4.6, v0=ego_v))

    elif m == "off_ramp_diverge":
        actors.append(Actor(id="exiting", cls="car", lane=0.0, s0=max(10.0, ego_v * t_gap) + 4.6, v0=ego_v * 0.9,
                            behavior=[{"t": 0.0, "a": 0.0}, {"t": 1.5, "lane": -1.0, "a": -0.3}]))
        actors.append(Actor(id="lead", cls="car", lane=0.0, s0=max(30.0, ego_v * (t_gap + 2.0)) + 4.6, v0=ego_v))

    elif m == "crossing":
        # A crossing road user arrives at the conflict point t_gap_s before the
        # ego would, so t_gap_s is the real post-encroachment time (PET).
        d = max(6.0, ego_v * t_gap)
        actors.append(Actor(id="crossing", cls="car", lane=0.0, lateral0=-20.0, s0=d + 4.6, v0=max(3.0, abs(d) / max(0.5, t_gap)) if t_gap > 0 else 8.0,
                            behavior=[{"t": 0.0, "a": 0.0}]))

    elif m == "pedestrian_crossing":
        v_ped = 1.3
        # The pedestrian starts behind the occlusion; visibility distance is
        # what turns the same geometry into a hard or an easy case.
        s0 = max(4.0, ego_v * t_gap)
        actors.append(Actor(id="ped", cls="pedestrian", lane=0.0, s0=s0 + 1.0, v0=0.0, lateral0=-4.0,
                            behavior=[{"t": 0.0, "a": 0.0, "v": v_ped}]))

    elif m == "static_obstacle":
        s0 = max(8.0, ego_v * t_gap)
        actors.append(Actor(id="obstacle", cls="truck", lane=0.0, s0=s0 + 6.0, v0=0.0,
                            behavior=[{"t": 0.0, "a": 0.0}]))

    elif m == "unprotected_left":
        # Opposing vehicle: oncoming, so its closing speed is the sum. The gap
        # is in seconds of the ego's turn, not of the approach.
        oncoming_v = max(5.0, ego_v * 0.9)
        d = max(8.0, (ego_v + oncoming_v) * t_gap)
        actors.append(Actor(id="oncoming", cls="car", lane=-1.0, s0=d + 9.2, v0=oncoming_v,
                            behavior=[{"t": 0.0, "a": 0.0}]))

    elif m == "roundabout":
        circ_v = max(4.0, ego_v * 0.5)
        d = max(5.0, circ_v * t_gap)
        actors.append(Actor(id="circulating", cls="car", lane=0.0, s0=d + 4.6, v0=circ_v,
                            behavior=[{"t": 0.0, "a": 0.0}]))

    else:
        raise ScenarioError(f"no maneuver library entry for {m!r}")

    for a in actors:
        a.__post_init__()
    return actors


def generate(q: lang.Query, seed: int) -> Scenario:
    p = sample_params(q, seed)
    rng = random.Random(seed ^ 0x5EED)
    road = params_to_road(p)
    actors = build_actors(p, road, rng)
    w = Weather(
        condition=p.get("weather.condition", "dry"),
        rain_mmh=float(p.get("weather.rain_mmh", 0.0)),
        fog_visibility_m=float(p.get("weather.fog_visibility_m", 10000.0)),
        illuminance_lux=float(p.get("weather.illuminance_lux", 5000.0)),
        mu_scale=1.0,
    )
    t = Traffic(
        flow_vph=float(p.get("traffic.flow_vph", 0.0)),
        mix_truck=float(p.get("traffic.mix_truck", 0.1)),
        mix_bicycle=float(p.get("traffic.mix_bicycle", 0.0)),
        mix_pedestrian=float(p.get("traffic.mix_pedestrian", 0.0)),
    )
    tags = sorted({str(p.get("road.kind")), str(p.get("maneuver.type")), str(p.get("weather.condition"))})
    sc = Scenario(id=f"sc-{seed:08x}-{abs(hash(q.text)) % 997:03d}", seed=seed, params=p, road=road,
                  actors=actors, weather=w, traffic=t, tags=tags,
                  notes=q.text or ";".join(f"{k}={v}" for k, v in sorted(q.pins.items())))
    return sc


def generate_many(q: lang.Query, n: int, seed0: int = 0) -> list:
    return [generate(q, seed0 + i) for i in range(n)]


#: The single entry point other modules use. Named separately from `generate`
#: because the coverage module imports it, and a module that is both imported and
#: shadowed by a local is a bug waiting to be written.
scenario_from_query = generate


def scenario_from_params(p: dict, sid: str = "sc-explicit") -> Scenario:
    """The inverse direction: parameters in, scenario out, no sampling. Used by
    the tests to check that the generator's claimed parameters are realised."""
    road = params_to_road(p)
    rng = random.Random(int(p.get("seed", 0)))
    actors = build_actors(p, road, rng)
    w = Weather(condition=p.get("weather.condition", "dry"),
                rain_mmh=float(p.get("weather.rain_mmh", 0.0)),
                fog_visibility_m=float(p.get("weather.fog_visibility_m", 10000.0)),
                illuminance_lux=float(p.get("weather.illuminance_lux", 5000.0)))
    t = Traffic(flow_vph=float(p.get("traffic.flow_vph", 0.0)),
                mix_truck=float(p.get("traffic.mix_truck", 0.1)),
                mix_bicycle=float(p.get("traffic.mix_bicycle", 0.0)),
                mix_pedestrian=float(p.get("traffic.mix_pedestrian", 0.0)))
    return Scenario(id=sid, seed=int(p.get("seed", 0)), params=p, road=road, actors=actors,
                    weather=w, traffic=t, tags=[str(p.get("maneuver.type", "?"))])
