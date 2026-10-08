"""The difficulty score: a 1-10 number with the sources that produced it.

The design claim this module makes is narrow and checkable: every component of
the score is a *ratio of physical quantities*, not a learned weight on a
feature. That is why the score can be audited -- if the tight-curve component
says 7/10, there is a number in the output saying the required lateral
acceleration is 3.9 m/s^2 against 9.81*mu of available grip, and you can check
it by hand.

    score = 1 + 9 * sum_i w_i * c_i          c_i in [0,1], sum w_i = 1

The weights are a stated prior (below), and bench/fit.py exists to show what
happens when they are instead fitted to simulated outcomes -- reported, not
silently adopted, because a score whose weights came from the simulator it is
supposed to predict is circular.
"""

from __future__ import annotations

import math

from .ir import G, Scenario
from .sim import Outcome, analytic_min_gap

#: Stated prior weights. Chosen so that the two components that decide whether a
#: conflict is *reachable* (visibility and surface) weigh as much together as
#: the two that decide how violent it is when reached.
DEFAULT_WEIGHTS = {
    "time_budget": 0.22,     # how little time the stack has
    "visibility": 0.18,      # how late the hazard appears
    "surface": 0.15,         # how little grip is left
    "curvature": 0.12,       # whether the path itself is at the limit
    "occlusion": 0.13,       # whether the hazard was hidden
    "traffic_density": 0.10, # how many other actors constrain the escape
    "exposure": 0.10,        # measured (simulated) conflict severity
}

#: Thresholds. Each is a physical constant with a reason, and they are named so
#: the report can quote them rather than quote a magic number.
THRESHOLDS = dict(
    ttc_hard_s=1.0,        # AEB intervention horizon in the field
    ttc_cautious_s=3.0,    # ISO 15622 (ACC) minimum steady-state following
    lat_accel_limit_frac=1.0,
    vis_critical_m=60.0,   # stopping distance from 80 kph at 0.7 g
    vis_good_m=300.0,
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def required_lateral_accel(sc: Scenario) -> float:
    """v^2/R on the tightest curved segment the ego reaches."""
    v = next((a.v0 for a in sc.actors if a.id == "ego"), 20.0)
    r = None
    for seg in sc.road.get("segments", []):
        if seg.get("radius_m"):
            r = seg["radius_m"] if r is None else min(r, seg["radius_m"])
    if not r:
        return 0.0
    return v * v / r


def stopping_distance(sc: Scenario, b: float = 7.5) -> float:
    ego = next((a for a in sc.actors if a.id == "ego"), None)
    if ego is None:
        return 0.0
    mu = sc.weather.friction()
    return ego.v0 ** 2 / (2.0 * b * max(0.2, mu))


def visible_hazard_distance(sc: Scenario) -> float:
    """The distance at which the first hazard can be *seen*, given fog, light
    and occlusion. Occlusion is modelled as a fixed cut-off: a parked row hides
    the crossing pedestrian until the geometry opens up, which is what makes the
    occluded-pedestrian case the canonical SOTIF example."""
    occ = str(sc.params.get("maneuver.occlusion", "none"))
    base = {
        "none": 1e9,
        "vegetation": 90.0,
        "building": 45.0,
        "parked_row": 22.0,
        "truck": 30.0,
    }.get(occ, 1e9)
    fog = sc.weather.fog_visibility_m
    lux = sc.weather.illuminance_lux
    light = 1.0 if lux >= 200.0 else (0.75 if lux >= 50.0 else (0.45 if lux >= 20.0 else 0.25))
    return min(base, fog * light)


def components(sc: Scenario, outcome: Outcome = None) -> dict:
    """The six structural terms that do not need a simulation, plus the measured
    exposure term when an outcome is supplied. Public because calibrate.py fits
    weights to exactly these, and a calibrator reaching into a private function is
    the kind of coupling that silently breaks.

    Ordered, because an order is what a fitted weight vector needs:
    COMPONENT_ORDER below is the contract between this module and calibrate.py.
    """
    return _components(sc, outcome)


#: The order of the fitted weight vector. Appended to, never reordered.
COMPONENT_ORDER = ("time_budget", "visibility", "surface", "curvature",
                   "occlusion", "traffic_density", "exposure")


def _components(sc: Scenario, outcome: Outcome = None) -> dict:
    ego = next((a for a in sc.actors if a.id == "ego"), None)
    v = ego.v0 if ego else 20.0
    mu = sc.weather.friction()

    # time budget: the smallest time-to-collision the geometry implies, against
    # the ACC/AEB horizons.
    ttc = float("inf")
    for a in sc.actors:
        if a.id == "ego":
            continue
        if abs(a.lane) > 0.6:
            continue
        gap = a.s0 - (ego.s0 if ego else 0.0)
        gap -= (a.length_width()[0] + (ego.length_width()[0] if ego else 4.6)) / 2.0
        if gap <= 0:
            continue
        dv = v - a.v0
        if dv > 1e-6:
            ttc = min(ttc, gap / dv)
    t0, t1 = THRESHOLDS["ttc_hard_s"], THRESHOLDS["ttc_cautious_s"]
    time_budget = 1.0 if not math.isfinite(ttc) else _clamp01((t1 - min(ttc, t1)) / (t1 - t0)) if ttc < t1 else 0.0

    vis = visible_hazard_distance(sc)
    visibility = _clamp01((THRESHOLDS["vis_good_m"] - min(vis, THRESHOLDS["vis_good_m"])) /
                          (THRESHOLDS["vis_good_m"] - THRESHOLDS["vis_critical_m"]))

    surface = _clamp01((0.9 - mu) / 0.75)

    lat = required_lateral_accel(sc)
    curvature = _clamp01(lat / (G * mu * THRESHOLDS["lat_accel_limit_frac"])) if mu > 0 else 0.0

    occlusion = {"none": 0.0, "vegetation": 0.4, "building": 0.6, "truck": 0.8, "parked_row": 1.0}.get(
        str(sc.params.get("maneuver.occlusion", "none")), 0.0)

    flow = max(0.0, sc.traffic.flow_vph)
    density = _clamp01(math.log1p(flow) / math.log1p(3600.0))
    density = min(1.0, density * (1.0 + 0.5 * sc.traffic.mix_bicycle + 0.6 * sc.traffic.mix_pedestrian))

    exposure = outcome.risk_index() if outcome is not None else 0.0

    return dict(time_budget=time_budget, visibility=visibility, surface=surface,
                curvature=curvature, occlusion=occlusion, traffic_density=density,
                exposure=exposure)


def score(sc: Scenario, outcome: Outcome = None, weights: dict = None) -> dict:
    """Return the score with a full attribution trail, so a number in a report
    can always be traced to the terms that produced it."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    if outcome is None:
        # Without a simulation there is no exposure term; renormalise over the
        # terms that are actually available rather than scoring a zero.
        z = sum(v for k, v in w.items() if k != "exposure")
        for k in list(w):
            w[k] = 0.0 if k == "exposure" else w[k] / z
    c = _components(sc, outcome)
    total = sum(w[k] * c[k] for k in w)
    s = 1.0 + 9.0 * _clamp01(total)
    top = sorted(((k, w[k] * c[k]) for k in c), key=lambda kv: -kv[1])[:3]
    return {
        "scenario_id": sc.id,
        "score": round(s, 3),
        "band": band(s),
        "weights": w,
        "components": {k: round(v, 4) for k, v in c.items()},
        "sources": [{"axis": k, "contribution": round(val, 4),
                     "share": round(val / max(1e-9, sum(w[k2] * c[k2] for k2 in c)), 3),
                     "explain": explain(k, sc, c)} for k, val in top],
        "measured": {
            "min_ttc_s": None if outcome is None else round(outcome.min_ttc, 3),
            "min_gap_m": None if outcome is None else round(outcome.min_gap, 3),
            "collision": None if outcome is None else outcome.collision,
            "required_lat_accel_g": round(required_lateral_accel(sc) / G, 3),
            "surface_mu": round(sc.weather.friction(), 3),
            "stopping_distance_m": round(stopping_distance(sc), 1),
            "visible_hazard_distance_m": round(visible_hazard_distance(sc), 1),
        },
    }


def explain(axis: str, sc: Scenario, c: dict) -> str:
    ego = next((a for a in sc.actors if a.id == "ego"), None)
    if axis == "time_budget":
        return f"eta-ego {ego.v0 * 3.6:.0f} kph against the nearest in-lane actor leaves {c['time_budget']:.2f} of the 1.0-3.0 s horizon"
    if axis == "visibility":
        return f"hazard first visible at {visible_hazard_distance(sc):.0f} m ({sc.weather.condition}, {sc.weather.fog_visibility_m:.0f} m fog, {sc.weather.illuminance_lux:.0f} lux)"
    if axis == "surface":
        return f"mu={sc.weather.friction():.2f} on {sc.weather.condition}, so braking distance is {stopping_distance(sc):.0f} m from {ego.v0 * 3.6:.0f} kph"
    if axis == "curvature":
        return f"lateral demand {required_lateral_accel(sc) / G:.2f} g of {sc.weather.friction():.2f} g available"
    if axis == "occlusion":
        return f"maneuver.occlusion={sc.params.get('maneuver.occlusion', 'none')}"
    if axis == "traffic_density":
        return f"{sc.traffic.flow_vph:.0f} veh/h with {sc.traffic.mix_truck:.0%} trucks, {sc.traffic.mix_bicycle:.0%} cycles, {sc.traffic.mix_pedestrian:.0%} pedestrians"
    if axis == "exposure":
        return "measured by simulation, not inferred"
    return ""


def band(s: float) -> str:
    if s < 3.0:
        return "benign"
    if s < 5.0:
        return "routine"
    if s < 7.0:
        return "demanding"
    if s < 8.5:
        return "hard"
    return "critical"


def rank(scenarios: list, outcomes: list = None) -> list:
    """Score a list, hardest first. This is the scene-selection entry point."""
    outcomes = outcomes or [None] * len(scenarios)
    out = [score(s, o) for s, o in zip(scenarios, outcomes)]
    return sorted(out, key=lambda d: -d["score"])
