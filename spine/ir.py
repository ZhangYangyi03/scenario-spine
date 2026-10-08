"""The scenario intermediate representation.

Everything in this package speaks this one structure, in SI units:

    position  m      speed  m/s     accel  m/s^2    angle  rad     time  s

The point of a single IR is that the five products are not five codebases: the
generator writes it, the difficulty scorer reads it, the coverage check asks
whether a set of them covers an ODD, label QC reads the *annotation* of it, and
the safety case reads what the coverage check and the simulation concluded.

A scenario is *parametric*: `params` are the knobs, `elements` are the resolved
world. scenario_from_params() is the only place a knob turns into geometry, so a
sampled scenario can always be described back in the language it was sampled in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# Fields a scenario must carry to be simulable. Kept explicit so a malformed
# scenario is rejected at the door rather than in the middle of a physics step.
REQUIRED_ACTOR_FIELDS = ("id", "cls", "lane", "s0", "v0")
ACTOR_CLASSES = ("car", "truck", "bus", "bicycle", "pedestrian")

G = 9.81

# Dry-road friction by condition; the weather parameter scales this, and the
# scaler is the only place weather touches the physics.
FRICTION = {
    "dry": 0.90,
    "wet": 0.60,
    "rain": 0.55,
    "snow": 0.30,
    "ice": 0.15,
}


class ScenarioError(ValueError):
    """Raised when a parameter set cannot describe a scenario."""


@dataclass
class Actor:
    """One dynamic object. `lane` is signed: 0 is the centreline, +1 one lane
    left of it (the overtaking side), -1 one lane right."""

    id: str
    cls: str
    lane: float
    s0: float
    v0: float
    lateral0: float = 0.0
    a0: float = 0.0
    behavior: list = field(default_factory=list)

    def __post_init__(self):
        if self.cls not in ACTOR_CLASSES:
            raise ScenarioError(f"unknown actor class {self.cls!r}")
        if self.v0 < 0:
            raise ScenarioError("speed must be non-negative")
        if self.s0 < 0:
            raise ScenarioError("initial arc length must be non-negative")

    def length_width(self) -> tuple:
        return {
            "car": (4.6, 1.85),
            "truck": (12.0, 2.55),
            "bus": (11.5, 2.55),
            "bicycle": (1.8, 0.65),
            "pedestrian": (0.6, 0.6),
        }[self.cls]


@dataclass
class Weather:
    condition: str = "dry"
    rain_mmh: float = 0.0
    fog_visibility_m: float = 10000.0
    illuminance_lux: float = 5000.0
    mu_scale: float = 1.0

    def friction(self) -> float:
        base = FRICTION.get(self.condition, 0.9)
        return max(0.05, base * self.mu_scale)


@dataclass
class Traffic:
    flow_vph: float = 0.0
    mix_truck: float = 0.1
    mix_bicycle: float = 0.0
    mix_pedestrian: float = 0.0


@dataclass
class Scenario:
    id: str
    seed: int
    params: dict = field(default_factory=dict)
    road: dict = field(default_factory=dict)
    actors: list = field(default_factory=list)
    weather: Weather = field(default_factory=Weather)
    traffic: Traffic = field(default_factory=Traffic)
    tags: list = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actors"] = [asdict(a) if not isinstance(a, dict) else a for a in self.actors]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        d = dict(d)
        actors = [Actor(**a) if isinstance(a, dict) else a for a in d.get("actors", [])]
        weather = d.get("weather") or {}
        traffic = d.get("traffic") or {}
        return Scenario(
            id=d["id"],
            seed=d.get("seed", 0),
            params=d.get("params", {}),
            road=d.get("road", {}),
            actors=actors,
            weather=Weather(**weather) if isinstance(weather, dict) else weather,
            traffic=Traffic(**traffic) if isinstance(traffic, dict) else traffic,
            tags=d.get("tags", []),
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------- road ----

def road_from_params(params: dict) -> dict:
    """A longitudinal cross-section: a list of segments, each straight or
    curved, with its own speed limit and grade. The geometry is *integrated*
    from curvature rather than drawn, so a generated curve really is the
    radius it claims to be."""
    segs = params.get("segments")
    if not segs:
        raise ScenarioError("road needs at least one segment")
    out, s = [], 0.0
    for i, seg in enumerate(segs):
        length = float(seg.get("length", seg.get("l", 200.0)))
        if length <= 0:
            raise ScenarioError("segment length must be positive")
        cur = float(seg.get("curvature_r") or seg.get("r") or 0.0)
        out.append(
            {
                "i": i,
                "s0": s,
                "s1": s + length,
                "length": length,
                "radius_m": (None if abs(cur) < 1e-9 else abs(cur)),
                "sign": (0 if abs(cur) < 1e-9 else (1 if cur > 0 else -1)),
                "grade_pct": float(seg.get("grade_pct", 0.0)),
                "speed_limit_kph": float(seg.get("speed_limit_kph", seg.get("vlim", 90.0))),
                "lanes": int(seg.get("lanes", 2)),
                "kind": seg.get("kind", "motorway"),
            }
        )
        s += length
    return {
        "total_length_m": s,
        "lane_width_m": float(params.get("lane_width_m", 3.6)),
        "segments": out,
        "surface": params.get("surface", "asphalt"),
    }
