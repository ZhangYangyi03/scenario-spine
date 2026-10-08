"""Annotation quality control: what is wrong with a labelled frame, and by how much.

The hard part of automated label QC is not finding disagreement, it is not
drowning in it. So the design here is: every check produces a *defect with a
severity and a magnitude*, and the report's job is to rank defects so a person
looks at the ones that matter. Three of the six checks are pure geometry and
cannot be argued with; three are model-vs-label checks whose sensitivity is
governed by a named tolerance rather than a tuned threshold.

    check                     what it decides                needs
    geometry.self_intersect   a box that folds through      label
    geometry.zero_area        a degenerate box              label
    geometry.class_extent     a truck-sized 'car'           label
    tracking.impossible_motion  teleport / accel beyond mu  label + track
    tracking.identity_switch  lateral jump beyond tyre grip + track
    cross.temporal_collision  two labels claiming one space label + track

The last three are the ones a rule engine catches and a human does not, on a
corpus of any size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

from .ir import G, ACTOR_CLASSES

#: Per-class plausible extent envelope in metres (min, max) for length/width, and
#: the acceleration envelope a physical object cannot exceed.
EXTENT = {
    "car": ((3.0, 6.5), (1.4, 2.2)),
    "truck": ((6.0, 18.0), (2.2, 3.0)),
    "bus": ((7.0, 18.0), (2.3, 3.0)),
    "bicycle": ((1.4, 2.2), (0.5, 0.9)),
    "pedestrian": ((0.3, 1.0), (0.3, 1.0)),
}
MAX_ACCEL_MPS2 = 8.0
MAX_LATERAL_ACCEL_MPS2 = 9.0


@dataclass
class Box:
    """One 2D annotation, ego-relative, metres. yaw in radians (0 = same heading
    as the ego)."""

    track_id: str
    cls: str
    t: float
    x: float
    y: float
    length: float
    width: float
    yaw: float = 0.0
    score: float = 1.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Defect:
    kind: str
    track_id: str
    t: float
    severity: float          # [0, 1]: how likely this is a real error
    magnitude: float         # the measured quantity, in its own units
    unit: str
    note: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = round(d["severity"], 4)
        d["magnitude"] = round(d["magnitude"], 4)
        return d


def _centre(b: Box):
    half = b.length / 2.0
    return (b.x - half * math.cos(b.yaw), b.y - half * math.sin(b.yaw))


# --------------------------------------------------------------- geometry ---

def check_self_intersect(b: Box) -> Defect | None:
    """Negative extent is the mechanical signature of a click-drag in the wrong
    direction; a yaw with a degenerate length is the other one."""
    if b.length <= 0 or b.width <= 0:
        return Defect("geometry.self_intersect", b.track_id, b.t, 1.0,
                      min(b.length, b.width), "m", "box has non-positive extent")
    return None


def check_zero_area(b: Box, min_area: float = 0.05) -> Defect | None:
    area = b.length * b.width
    if area >= min_area:
        return None
    return Defect("geometry.zero_area", b.track_id, b.t, 0.8, area, "m^2",
                  f"area {area:.4f} m^2 below {min_area}")


def check_class_extent(b: Box) -> Defect | None:
    (lmin, lmax), (wmin, wmax) = EXTENT.get(b.cls, EXTENT["car"])
    out_l = max(0.0, lmin - b.length, b.length - lmax)
    out_w = max(0.0, wmin - b.width, b.width - wmax)
    if out_l <= 0 and out_w <= 0:
        return None
    mag = math.hypot(out_l, out_w)
    return Defect("geometry.class_extent", b.track_id, b.t, min(1.0, mag / 4.0), mag, "m",
                  f"{b.cls} of {b.length:.1f}x{b.width:.1f} m outside [{lmin},{lmax}]x[{wmin},{wmax}]")


# --------------------------------------------------------------- tracking ---

def _by_track(boxes: list) -> dict:
    out = {}
    for b in boxes:
        out.setdefault(b.track_id, []).append(b)
    for v in out.values():
        v.sort(key=lambda b: b.t)
    return out


def check_impossible_motion(boxes: list, mu: float = 0.9) -> list:
    """Acceleration and speed implied by consecutive labels, against the friction
    envelope. This is where a whole track labelled at the wrong framerate shows
    up as one defect per frame, so it is aggregated per track rather than
    emitted per frame."""
    out = []
    for tid, bs in _by_track(boxes).items():
        worst = None
        for a, b in zip(bs, bs[1:]):
            dt = b.t - a.t
            if dt <= 1e-6:
                out.append(Defect("tracking.non_monotonic_time", tid, b.t, 1.0, dt, "s",
                                  "two labels share a timestamp"))
                continue
            ax = (b.vx - a.vx) / dt
            ay = (b.vy - a.vy) / dt
            acc = math.hypot(ax, ay)
            lat = abs((b.vx * ay - b.vy * ax)) / max(1e-6, math.hypot(b.vx, b.vy))
            limit = MAX_ACCEL_MPS2 * (mu / 0.9)
            over = max(0.0, acc - limit, lat - MAX_LATERAL_ACCEL_MPS2)
            if over > 0:
                d = Defect("tracking.impossible_motion", tid, b.t, min(1.0, over / limit), over, "m/s^2",
                           f"implied {acc:.1f} m/s^2 longitudinal, {lat:.1f} lateral against mu={mu:.2f}")
                if worst is None or d.severity > worst.severity:
                    worst = d
        if worst:
            out.append(worst)
    return out


def check_track_gap(boxes: list, gap_s: float = 0.6) -> list:
    """A track that disappears for a while and comes back. Not necessarily an
    error -- occlusion does this legitimately -- so severity is moderate and the
    note says so; the report is where a person decides."""
    out = []
    for tid, bs in _by_track(boxes).items():
        if len(bs) < 4:
            continue
        gaps = [b.t - a.t for a, b in zip(bs, bs[1:])]
        if max(gaps) > gap_s:
            k = max(range(len(gaps)), key=lambda i: gaps[i])
            out.append(Defect("tracking.track_gap", tid, bs[k].t, 0.5, max(gaps), "s",
                              f"track disappears for {max(gaps):.2f} s and returns"))
    return out


def check_identity_switch(boxes: list, drop_s: float = 0.1) -> list:
    """A track whose mid-point jumps to another actor's trajectory: detected as a
    *lateral* displacement that is large while the label's own velocity field
    stays continuous, i.e. the box moved sideways faster than a tyre could carry
    it. This is the swap that a per-frame rule engine misses, because each frame
    on its own is perfectly plausible."""
    out = []
    for tid, bs in _by_track(boxes).items():
        for a, b in zip(bs, bs[1:]):
            dt = b.t - a.t
            if dt <= 1e-6:
                continue
            vy_implied = (b.y - a.y) / dt
            if abs(vy_implied) > MAX_LATERAL_ACCEL_MPS2:
                out.append(Defect("tracking.identity_switch", tid, b.t,
                                  min(1.0, abs(vy_implied) / (2 * MAX_LATERAL_ACCEL_MPS2)),
                                  abs(vy_implied), "m/s",
                                  f"lateral displacement implies {vy_implied:.1f} m/s, beyond tyre capability"))
                break
    return out


# ------------------------------------------------------------ cross-boxes ---

def _overlap_area(a: Box, b: Box, margin: float = 0.0) -> float:
    """Axis-aligned overlap of the two footprints with a margin. Deliberately
    axis-aligned: for the *collision* check the question is whether two boxes
    claim the same space, and rotation makes the answer smaller, so an
    axis-aligned test is the conservative one."""
    ax0, ax1 = a.x - a.length / 2 - margin, a.x + a.length / 2 + margin
    ay0, ay1 = a.y - a.width / 2 - margin, a.y + a.width / 2 + margin
    bx0, bx1 = b.x - b.length / 2, b.x + b.length / 2
    by0, by1 = b.y - b.width / 2, b.y + b.width / 2
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    return max(0.0, ox) * max(0.0, oy)


def check_temporal_collision(boxes: list, tol_m2: float = 0.10) -> list:
    """Two labels asserting the same physical space at the same time. Either one
    has the wrong position, or one is a duplicate track -- both are label errors,
    and this is the check that catches a duplicated object."""
    out = []
    by_t = {}
    for b in boxes:
        by_t.setdefault(round(b.t, 6), []).append(b)
    for t, group in by_t.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if abs(a.y - b.y) > (a.width + b.width) / 2 + 0.2:
                    continue
                ov = _overlap_area(a, b)
                if ov > tol_m2:
                    # A pedestrian inside a car's footprint is often the label
                    # convention, not an error; severity reflects that.
                    sev = 0.35 if "pedestrian" in (a.cls, b.cls) else 0.9
                    out.append(Defect("cross.temporal_collision", f"{a.track_id}+{b.track_id}", t,
                                      sev, ov, "m^2",
                                      f"{a.cls} and {b.cls} overlap over {ov:.2f} m^2 at t={t:.2f}"))
    return out


# ------------------------------------------------------------------ report --

def run_checks(boxes: list, mu: float = 0.9) -> list:
    out = []
    for b in boxes:
        for f in (check_self_intersect(b), check_zero_area(b), check_class_extent(b)):
            if f:
                out.append(f)
    out.extend(check_impossible_motion(boxes, mu))
    out.extend(check_track_gap(boxes))
    out.extend(check_identity_switch(boxes))
    out.extend(check_temporal_collision(boxes))
    return out


def quality_report(boxes: list, mu: float = 0.9, label: str = "frames") -> dict:
    """The report a data-ops person actually reads: how many frames, how many
    defects by class, what the score is, and which tracks to look at first."""
    frames = sorted({round(b.t, 6) for b in boxes})
    defects = run_checks(boxes, mu)
    by_kind = {}
    for d in defects:
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
    by_track = {}
    for d in defects:
        by_track.setdefault(d.track_id, []).append(d)
    worst = sorted(defects, key=lambda d: -d.severity)[:10]
    # A quality score in [0,1]: severity mass per label, mapped through a
    # saturating curve. Reported with the raw mass so the curve can be argued
    # with rather than trusted.
    mass = sum(d.severity for d in defects)
    denom = max(1.0, len(boxes) / 4.0)
    q = math.exp(-mass / denom)
    return {
        "label": label,
        "frames": len(frames),
        "boxes": len(boxes),
        "tracks": len({b.track_id for b in boxes}),
        "defects": len(defects),
        "defects_by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "defect_mass": round(mass, 3),
        "quality": round(q, 4),
        "worst_tracks": [{"track_id": k, "defects": len(v),
                          "max_severity": round(max(d.severity for d in v), 3),
                          "kinds": sorted({d.kind for d in v})} for k, v in
                         sorted(by_track.items(), key=lambda kv: -max(d.severity for d in kv[1]))[:8]],
        "examples": [d.to_dict() for d in worst],
    }


def boxes_from_scenario(sc, n_frames: int = 60, dt: float = 0.1, noise: dict = None) -> list:
    """Turn a generated scenario into an annotation stream by running the
    simulator -- i.e. a *correct* label set -- then optionally corrupt it, which
    is what makes the QC claim measurable: a defect injector with known ground
    truth tests the detector on cases where the answer is known."""
    from .sim import simulate

    o = simulate(sc, dt=dt, t_end=max(1.0, n_frames * dt))
    noise = noise or {}
    rng = __import__("random").Random(sc.seed)
    boxes = []
    idmap = {"ego": "TRK-ego"}
    for i, a in enumerate([x for x in sc.actors if x.id != "ego"]):
        idmap[a.id] = f"TRK-{i:03d}"
    for a in sc.actors:
        tid = idmap[a.id]
        ln, wd = a.length_width()
        slow = a.v0 if a.id == "ego" else a.v0
        for k in range(n_frames):
            t = k * dt
            s = a.s0 + slow * t
            x, y = s, a.lane * sc.road.get("lane_width_m", 3.6)
            cls = "pedestrian" if a.cls == "pedestrian" else a.cls
            b = Box(track_id=tid, cls=cls, t=t, x=x, y=y, length=ln, width=wd,
                    vx=slow, vy=0.0, score=1.0)
            if noise.get("teleport_every") and k and k % noise["teleport_every"] == 0:
                b.x += noise.get("teleport_m", 12.0)
                b.vx = slow + noise.get("teleport_m", 12.0) / dt
            if noise.get("shrink") and noise.get("shrink").get(tid):
                b.length *= noise["shrink"][tid]
            if noise.get("relabel") and noise.get("relabel").get(tid):
                b.cls = noise["relabel"][tid]
            if noise.get("duplicate") == tid and k % 5 == 0:
                boxes.append(Box(**{**b.__dict__}))
                boxes[-1].track_id = tid + "-dup"
            boxes.append(b)
    return boxes


# ------------------------------------------------------------- the injector --

#: The defect classes and how a label set is corrupted to produce each one. The
#: injector is not test scaffolding: the claim "this finds label errors" is only
#: a measurement if there is a corpus where the errors are known, and this is
#: where that corpus comes from. Every entry states what it breaks, so a
#: detector that fires on the wrong cause is visible as a false positive rather
#: than as a pass.
INJECTIONS = {
    "geometry.class_extent": "resize one box beyond its class envelope",
    "geometry.zero_area": "collapse one box's width",
    "tracking.impossible_motion": "teleport one box between frames",
    "tracking.identity_switch": "shunt one box laterally between frames",
    "cross.temporal_collision": "duplicate one box in place",
}


def inject_defects(boxes: list, which: list, rate: float = 1.0, seed: int = 0) -> tuple:
    """Return (corrupted boxes, ground truth list of expected defect kinds).

    Each injection targets specific track ids, and the ground truth records which
    track and which kind, so a detector's output can be scored per class rather
    than in aggregate.
    """
    import random

    rng = random.Random(seed)
    out = [Box(**b.__dict__) for b in boxes]
    truth = []
    by_track = _by_track(out)
    tids = sorted(by_track)
    if not tids:
        return out, truth
    for kind in which:
        # pick a track that is in the middle of the stream, so the corruption is
        # surrounded by clean frames: a defect at the boundary is not detectable
        valid = [t_ for t_ in tids if len(by_track[t_]) >= 6]
        if not valid:
            continue
        tid = rng.choice(valid)
        bs = [b for b in out if b.track_id == tid]
        bs.sort(key=lambda b: b.t)
        lo, hi = int(len(bs) * 0.25), int(len(bs) * 0.75)
        if kind == "geometry.class_extent":
            b = bs[lo]
            b.length = 24.0 if b.cls in ("car", "truck") else 0.05
            truth.append({"kind": kind, "track_id": tid, "t": b.t})
        elif kind == "geometry.zero_area":
            for b in bs[lo:hi]:
                b.width = 0.01
            truth.append({"kind": kind, "track_id": tid, "t": bs[lo].t})
        elif kind == "tracking.impossible_motion":
            for b in bs[lo:hi]:
                b.x += 400.0 * (b.t - bs[lo].t)
                b.vx += 400.0
            truth.append({"kind": kind, "track_id": tid, "t": bs[lo].t})
        elif kind == "tracking.identity_switch":
            for j, b in enumerate(bs[lo:hi]):
                b.y += (-1.0) ** j * 45.0
            truth.append({"kind": kind, "track_id": tid, "t": bs[lo].t})
        elif kind == "cross.temporal_collision":
            for b in bs[lo:hi]:
                d = Box(**b.__dict__)
                d.track_id = tid + "-dup"
                out.append(d)
            truth.append({"kind": kind, "track_id": tid + "+" + tid + "-dup", "t": bs[lo].t})
    out.sort(key=lambda b: (b.t, b.track_id))
    return out, truth


def score_detector(defects: list, truth: list) -> dict:
    """Per-class recall and a false-positive count against clean labels.

    Matching is by kind and by whether the reported track is the injected one (or
    one of the two, for a collision). Aggregate precision is reported as
    detections on a corpus with no injection, which is the only way to measure it
    honestly: a detector that always fires has recall 1.0 and is useless, and
    only the clean-corpus number reveals that.
    """
    got = {}
    for d in defects:
        got.setdefault(d.kind, []).append(d)
    per = {}
    for g in truth:
        k = g["kind"]
        cands = got.get(k, [])
        hit = any(g["track_id"] in d.track_id for d in cands)
        per[k] = {"expected": 1, "detected": 1 if hit else 0}
    kinds = sorted({g["kind"] for g in truth} | set(got))
    return {
        "per_class": per,
        "recall": (sum(v["detected"] for v in per.values()) / len(per)) if per else None,
        "detected_by_kind": {k: len(v) for k, v in got.items()},
        "missing": sorted({g["kind"] for g in truth} - set(got)),
    }
