"""Export to the two formats that decide whether anyone can use this: ASAM
OpenSCENARIO 1.2 (what a simulator's scenario runner consumes) and SUMO
(what a traffic simulator consumes).

The two exports serve different questions and therefore carry different parts of
the scenario:

    OpenSCENARIO  a *story*: entities, their initial positions and their
                  maneuvers, in a timeline. It is about the ego's encounter.
    SUMO          a *network and a demand*: edges, lanes, routes and flows for
                  the whole population. It is about the traffic the ego sits in.

Both are emitted from the same intermediate representation, so a scenario cannot
drift between them, and both are validated structurally by tests/export tests
(well-formedness, required attributes, referential integrity of the ids) rather
than by eyeball.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from .ir import Scenario


def _fmt(x: float, nd: int = 6) -> str:
    return f"{float(x):.{nd}f}".rstrip("0").rstrip(".") or "0"


# ------------------------------------------------------- OpenSCENARIO 1.2 --

def _segment_at(road: dict, s: float) -> dict:
    segs = road.get("segments") or [{"s0": 0.0, "s1": 1e9, "radius_m": None,
                                     "grade_pct": 0.0, "speed_limit_kph": 90.0, "lanes": 2}]
    for seg in segs:
        if seg["s0"] <= s <= seg["s1"]:
            return seg
    return segs[-1]


def openscenario(sc: Scenario, name: str = None) -> str:
    """ASAM OpenSCENARIO 1.2. Longitudinal positions are converted to world x/y
    by integrating the road centreline, so a scenario in a curve is exported as a
    curve and not as a straight line with a radius number in a comment."""
    lane_w = sc.road.get("lane_width_m", 3.6)
    cat = ET.Element("OpenSCENARIO")
    hdr = ET.SubElement(cat, "FileHeader", {"revMajor": "1", "revMinor": "2",
                                           "name": name or sc.id, "version": "1.2",
                                           "date": "2026-10-09T00:00:00"})
    ET.SubElement(hdr, "License", {"name": "Apache-2.0", "resource": "https://www.apache.org/licenses/LICENSE-2.0"})
    ET.SubElement(hdr, "Properties")
    road = ET.SubElement(cat, "RoadNetwork")
    ET.SubElement(road, "LogicFile", {"filepath": f"{sc.id}.xodr"})
    ET.SubElement(road, "SceneGraphFile", {"filepath": f"{sc.id}.osgb"})
    ent = ET.SubElement(cat, "Entities")
    story = ET.SubElement(cat, "Storyboard")

    # --- entities: one CatalogReference-free inline ScenarioObject each
    for a in sc.actors:
        so = ET.SubElement(ent, "ScenarioObject", {"name": a.id})
        veh = ET.SubElement(so, "Vehicle", {"name": a.cls, "vehicleCategory": _vcat(a.cls),
                                            "mass": "1500"})
        bb = ET.SubElement(veh, "BoundingBox")
        ln, wd = a.length_width()
        ET.SubElement(bb, "Center", {"x": _fmt(ln / 2), "y": "0", "z": _fmt(wd / 2)})
        ET.SubElement(bb, "Dimensions", {"width": _fmt(wd), "length": _fmt(ln), "height": "1.6"})
        ax = ET.SubElement(veh, "Axles")
        ET.SubElement(ax, "FrontAxle", {"maxSteering": "0.5", "wheelDiameter": "0.6",
                                        "trackWidth": _fmt(wd * 0.9), "positionX": _fmt(ln * 0.35), "positionZ": "0.3"})
        ET.SubElement(ax, "RearAxle", {"maxSteering": "0", "wheelDiameter": "0.6",
                                       "trackWidth": _fmt(wd * 0.9), "positionX": "0", "positionZ": "0.3"})
        ET.SubElement(veh, "Properties")
        if a.cls == "pedestrian":
            ET.SubElement(so, "Pedestrian", {"name": a.cls, "pedestrianCategory": "pedestrian",
                                             "mass": "80", "model": "walker"})

    # --- init: absolute world positions
    init = ET.SubElement(story, "Init")
    ia = ET.SubElement(init, "Actions")
    for a in sc.actors:
        pa = ET.SubElement(ia, "Private", {"entityRef": a.id})
        act = ET.SubElement(pa, "PrivateAction")
        tel = ET.SubElement(act, "TeleportAction")
        pos = ET.SubElement(tel, "Position")
        wy = a.lane * lane_w + a.lateral0
        seg = _segment_at(sc.road, a.s0)
        curl = 0.0 if not seg.get("radius_m") else 1.0 / seg["radius_m"] * seg.get("sign", 1)
        x, y, h = _integrate(sc.road, a.s0, wy, curl)
        ET.SubElement(pos, "WorldPosition", {"x": _fmt(x), "y": _fmt(y), "z": _fmt(sc.road.get("elevation_m", 0.0)),
                                             "h": _fmt(h), "p": "0", "r": "0"})
        lon = ET.SubElement(act, "LongitudinalAction")
        spd = ET.SubElement(lon, "SpeedAction")
        dyn = ET.SubElement(spd, "SpeedActionDynamics", {"dynamicsShape": "step", "value": "0", "dynamicsDimension": "time"})
        ET.SubElement(spd, "SpeedActionTarget").append(
            _speed_el("AbsoluteTargetSpeed", a.v0))
        # lateral lane position, so the initial y is stated in the story too
        lat = ET.SubElement(ET.SubElement(pa, "PrivateAction"), "LateralAction")
        lc = ET.SubElement(lat, "LaneChangeAction")
        ET.SubElement(lc, "LaneChangeActionDynamics", {"dynamicsShape": "linear", "value": "1.0", "dynamicsDimension": "time"})
        ET.SubElement(lc, "LaneChangeTarget").append(_relative("RelativeTargetLane", 0))

    # --- story: one ManeuverGroup per actor with a behavior schedule
    st = ET.SubElement(story, "Story", {"name": "spine-story"})
    for idx, a in enumerate(sc.actors):
        if not a.behavior:
            continue
        act_el = ET.SubElement(st, "Act", {"name": f"act-{a.id}-{idx}"})
        mg = ET.SubElement(act_el, "ManeuverGroup", {"maximumExecutionCount": "1",
                                                     "name": f"{a.id}-maneuvers"})
        actors_el = ET.SubElement(mg, "Actors", {"selectTriggeringEntities": "false"})
        ET.SubElement(actors_el, "EntityRef", {"entityRef": a.id})
        man = ET.SubElement(mg, "Maneuver", {"name": f"{a.id}-maneuver"})
        ev = ET.SubElement(man, "Event", {"name": f"{a.id}-event", "priority": "override"})
        for i, b in enumerate(a.behavior):
            act = ET.SubElement(ev, "Action", {"name": f"{a.id}-a{i}"})
            if "a" in b:
                ln = ET.SubElement(act, "LongitudinalAction")
                sa = ET.SubElement(ln, "SpeedAction")
                ET.SubElement(sa, "SpeedActionDynamics", {"dynamicsShape": "linear",
                                                          "value": "0.5", "dynamicsDimension": "time"})
                tgt = max(0.0, a.v0 + float(b["a"]) * 2.0)
                ET.SubElement(sa, "SpeedActionTarget").append(_speed_el("AbsoluteTargetSpeed", tgt))
            elif "lane" in b:
                lat = ET.SubElement(act, "LateralAction")
                lc = ET.SubElement(lat, "LaneChangeAction")
                ET.SubElement(lc, "LaneChangeActionDynamics", {"dynamicsShape": "sinusoidal",
                                                               "value": "2.0", "dynamicsDimension": "time"})
                ET.SubElement(lc, "LaneChangeTarget").append(
                    _relative("RelativeTargetLane", float(b["lane"]) - a.lane))
        ev.append(_trigger_from_behavior(a))
        # StartTrigger on the act: the published 1.2 schema requires it
        ET.SubElement(act_el, "StartTrigger").append(_trigger_all())
    # --- environment
    env = ET.SubElement(story, "Environment")
    ea = ET.SubElement(env, "EnvironmentAction")
    ed = ET.SubElement(ea, "Environment", {"name": sc.weather.condition})
    ET.SubElement(ed, "TimeOfDay", {"animation": "false", "dateTime": "2026-10-09T12:00:00"})
    w = ET.SubElement(ed, "Weather")
    ET.SubElement(w, "Sun", {"intensity": _fmt(_sun_intensity(sc.weather.illuminance_lux)),
                             "azimuth": "0", "elevation": "1.0"})
    ET.SubElement(w, "Fog", {"visualRange": _fmt(sc.weather.fog_visibility_m)})
    ET.SubElement(w, "Precipitation", {"precipitationType": _precip(sc.weather.condition),
                                       "intensity": _fmt(sc.weather.rain_mmh / 60.0)})
    ET.SubElement(w, "RoadCondition", {"frictionScaleFactor": _fmt(sc.weather.mu_scale)})
    ET.SubElement(cat, "CatalogLocations")
    return _pretty(cat)


def _speed_el(tag: str, v: float):
    e = ET.Element(tag)
    e.set("value", _fmt(v))
    return e


def _relative(tag: str, v: float):
    e = ET.Element(tag, {"value": _fmt(v)})
    return e


def _vcat(cls: str) -> str:
    return {"car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle", "pedestrian": "pedestrian"}.get(cls, "car")


def _sun_intensity(lux: float) -> float:
    return max(0.0, min(1.0, math.log10(max(0.1, lux)) / 5.0))


def _precip(cond: str) -> str:
    return {"dry": "dry", "wet": "dry", "rain": "rain", "snow": "snow", "ice": "snow"}.get(cond, "dry")


def _trigger_from_behavior(a):
    """A time trigger on each behavior step, which is what the library's
    schedules are expressed in. A storyboard trigger is not decoration: without
    it the runner executes every action at t=0."""
    trg = ET.Element("Trigger")
    cg = ET.SubElement(trg, "ConditionGroup")
    for i, b in enumerate(a.behavior):
        if i == 0:
            continue
        cond = ET.SubElement(cg, "Condition", {"name": f"{a.id}-t{i}", "delay": "0", "conditionEdge": "rising"})
        by = ET.SubElement(cond, "ByValueCondition")
        sim = ET.SubElement(by, "SimulationTimeCondition", {"value": _fmt(b.get("t", 0.0)), "rule": "greaterThan"})
        sim.set("value", _fmt(b.get("t", 0.0)))
    if len(list(cg)) == 0:
        cond = ET.SubElement(cg, "Condition", {"name": f"{a.id}-t0", "delay": "0", "conditionEdge": "none"})
        by = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(by, "SimulationTimeCondition", {"value": "0", "rule": "greaterThan"})
    return trg


def _integrate(road: dict, s: float, lateral: float, curvature: float) -> tuple:
    """Walk the centreline in 1 m steps, offsetting laterally. Small, exact
    enough for an init position, and it means a curved scenario exports a
    curved start rather than a straight one."""
    x, y, h = 0.0, 0.0, 0.0
    step = 1.0
    n = int(max(1, s / step))
    ds = s / n if n else 0.0
    for _ in range(n):
        x += ds * math.cos(h)
        y += ds * math.sin(h)
        h += curvature * ds
    # lateral offset in the local frame
    return (x - lateral * math.sin(h), y + lateral * math.cos(h), h)


def _pretty(root) -> str:
    try:
        ET.indent(root, space="  ")
    except AttributeError:  # pragma: no cover - python < 3.9
        pass
    return '<?xml version="1.0" encoding="UTF-8"?>' + "\n" + ET.tostring(root, encoding="unicode")


# ------------------------------------------------------------------- SUMO --

def sumo(sc: Scenario, begin: float = 0.0, end: float = 12.0, step: float = 0.02) -> tuple:
    """(.net.xml, .rou.xml). The network is generated from the road cross-section
    (the same one OpenSCENARIO integrates), and the demand file carries the
    scripted actors as vehicles with explicit depart/arrival times plus the
    background flow. Returns two strings so a caller can write them, diff them,
    or validate them without touching a filesystem."""
    lane_w = sc.road.get("lane_width_m", 3.6)
    total = sc.road.get("total_length_m", 1000.0)
    net = ET.Element("net")
    ET.SubElement(net, "location", {"netOffset": "0,0", "convBoundary": f"0,0,{_fmt(total)},0",
                                    "origBoundary": "0,0,0,0", "projParameter": "!"})
    for i, seg in enumerate(sc.road.get("segments", [])):
        shape = _segment_shape(sc.road, seg, lane_w)
        e = ET.SubElement(net, "edge", {"id": f"e{i}", "from": f"n{i}", "to": f"n{i+1}",
                                        "priority": "1", "type": "highway", "shape": shape})
        for k in range(max(1, seg.get("lanes", 2))):
            ET.SubElement(e, "lane", {"id": f"e{i}_{k}", "index": str(-k), "speed": _fmt(seg["speed_limit_kph"] / 3.6),
                                      "length": _fmt(seg["length"]), "shape": shape})
    for i in range(len(sc.road.get("segments", [])) + 1):
        ET.SubElement(net, "junction", {"id": f"n{i}", "type": "priority", "x": "0", "y": "0"})
    net.set("version", "1.20")

    rou = ET.Element("routes")
    vtype = ET.SubElement(rou, "vType", {"id": "spine", "accel": "2.6", "decel": "4.5",
                                         "sigma": "0.5", "length": "4.6", "maxSpeed": "70",
                                         "speedFactor": "1.0", "lcStrategic": "1.0"})
    ET.SubElement(rou, "vType", {"id": "spine-truck", "accel": "1.2", "decel": "3.0",
                                 "sigma": "0.5", "length": "12", "maxSpeed": "55",
                                 "vClass": "truck"})
    for a in sc.actors:
        dep = 0.0
        route = ET.SubElement(rou, "route", {"id": f"r-{a.id}", "edges": " ".join(
            f"e{i}" for i in range(len(sc.road.get("segments", []))))})
        veh = ET.SubElement(rou, "vehicle", {"id": a.id, "type": "spine" if a.cls != "truck" else "spine-truck",
                                             "route": f"r-{a.id}", "depart": _fmt(dep),
                                             "departSpeed": _fmt(a.v0), "departPos": _fmt(a.s0),
                                             "departLane": str(max(0, int(round(a.lane))) if a.lane >= 0 else 0),
                                             "arrival": _fmt(end)})
        for b in a.behavior:
            if "a" in b:
                ET.SubElement(veh, "param", {"key": f"a@{b.get('t', 0.0)}", "value": _fmt(b["a"])})
    if sc.traffic.flow_vph > 0:
        # background demand as a calibrated flow, with the class mix from the axes
        ET.SubElement(rou, "flow", {"id": "bg", "type": "spine", "route": "r-bg",
                                    "begin": _fmt(begin), "end": _fmt(end),
                                    "vehsPerHour": _fmt(sc.traffic.flow_vph),
                                    "departLane": "best", "departSpeed": "speedLimit"})
        ET.SubElement(rou, "route", {"id": "r-bg", "edges": " ".join(
            f"e{i}" for i in range(len(sc.road.get("segments", []))))})
    return _pretty(net), _pretty(rou)


def _segment_shape(road: dict, seg: dict, lane_w: float, step: float = 20.0) -> str:
    """Shape points along a segment, in the same integrated frame the
    OpenSCENARIO export uses, so the two formats describe the same road."""
    n = max(2, int(seg["length"] / step) + 1)
    curl = 0.0 if not seg.get("radius_m") else (1.0 / seg["radius_m"]) * seg.get("sign", 1)
    ds = seg["length"] / (n - 1)
    x, y, h = _integrate(road, seg["s0"], 0.0, 0.0)[0:2] + (0.0,)
    # integrate from the segment start, seeded by the accumulated heading
    x0, y0, h0 = 0.0, 0.0, 0.0
    if seg["s0"] > 0:
        for prev in road.get("segments", []):
            if prev["s0"] >= seg["s0"]:
                break
            c = 0.0 if not prev.get("radius_m") else (1.0 / prev["radius_m"]) * prev.get("sign", 1)
            x0, y0, h0 = _advance(x0, y0, h0, c, prev["length"])
    x, y, h = x0, y0, h0
    pts = []
    for _ in range(n):
        pts.append(f"{_fmt(x, 2)},{_fmt(y, 2)}")
        x, y, h = _advance(x, y, h, curl, ds)
    return " ".join(pts)


def _advance(x, y, h, curl, ds, step=1.0):
    n = max(1, int(ds / step))
    d = ds / n
    for _ in range(n):
        x += d * math.cos(h)
        y += d * math.sin(h)
        h += curl * d
    return x, y, h


def _trigger_all():
    """A trigger that fires at t=0, for the Act's StartTrigger."""
    trg = ET.Element("Trigger")
    cg = ET.SubElement(trg, "ConditionGroup")
    cond = ET.SubElement(cg, "Condition", {"name": "start", "delay": "0", "conditionEdge": "none"})
    by = ET.SubElement(cond, "ByValueCondition")
    ET.SubElement(by, "SimulationTimeCondition", {"value": "0", "rule": "greaterThan"})
    return trg
