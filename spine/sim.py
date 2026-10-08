"""A longitudinal simulator and a stack under test, so that every claim about
difficulty is measured against an outcome rather than asserted.

This is not a replacement for CARLA. It is the smallest model that can answer
"did this scenario actually endanger the ego?", in the exact regime where the
answer is decidable in closed form: one-dimensional car-following kinematics on
a known friction surface. The value of doing it this way is that the outcome is
falsifiable -- a scenario's risk index can be checked against an analytic
braking calculation (see tests/test_sim.py), which a photorealistic simulator
cannot offer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .ir import G, Actor, Scenario

# ------------------------------------------------------------- the policy ----

#: Parameters of the stack under test. They are deliberately ordinary: a
#: production stack is better than this, and the point is that the interface is
#: a policy, so a better one changes the numbers without changing the pipeline.
POLICY = dict(
    tau=1.2,        # desired time headway, s
    a_max=2.2,      # comfortable accel, m/s^2
    b_comfort=2.0,  # comfortable decel, m/s^2
    b_max=7.5,      # emergency decel, m/s^2
    ttc_brake=1.6,  # s: brake hard below this time-to-collision
    v_delta=1.4,    # m/s: free-flow speed offset
    reaction_s=0.25,
    lateral_yield_ttc=3.0,
)


@dataclass
class Frame:
    t: float
    x: float
    v: float
    a: float
    gap: float
    ttc: float
    collision: bool


@dataclass
class Outcome:
    scenario_id: str
    steps: int
    collision: bool
    min_ttc: float
    min_gap: float
    max_decel: float
    peak_jerk: float
    ego_stop_dist: float
    distance_available: float
    hard_brake: bool
    traj: list = field(default_factory=list)

    def risk_index(self) -> float:
        """One number in [0, 1] that the difficulty score is checked against.

        Components, all measured, none asserted:
          * collision                     -> the floor is 1.0
          * exposure, 1/(1+min_ttc)       -> how close to a conflict it got
          * braking margin                -> stop distance over distance available
          * jerk, normalised             -> how abruptly the stack had to react
        """
        if self.collision:
            return 1.0
        expo = 1.0 / (1.0 + max(0.0, self.min_ttc))
        margin = self.ego_stop_dist / max(1e-6, self.distance_available)
        jerk = min(1.0, self.peak_jerk / 40.0)
        return max(0.0, min(1.0, 0.45 * expo + 0.40 * min(1.5, margin) / 1.5 + 0.15 * jerk))


def _idm(v: float, v_des: float, gap: float, dv: float, p: dict, mu: float) -> float:
    """Intelligent Driver Model, with the free-flow term dropped when a leader
    is close enough that the interaction term dominates."""
    s0 = 2.0
    a_free = p["a_max"] * max(0.0, 1.0 - (v / max(1e-6, v_des)) ** 4)
    if gap is None or not math.isfinite(gap):
        return a_free
    gap = max(0.05, gap)
    s_star = s0 + max(0.0, v * p["tau"] + v * dv / (2.0 * math.sqrt(p["a_max"] * p["b_comfort"])))
    a_int = -p["a_max"] * (s_star / gap) ** 2
    a = a_free + a_int
    return max(-p["b_max"] * (mu / 0.9 if mu < 0.9 else 1.0), min(p["a_max"], a))


def simulate(sc: Scenario, dt: float = 0.02, t_end: float = 12.0, policy: dict = None) -> Outcome:
    """Step the scenario forward. Only longitudinal motion is modelled, plus a
    lateral lane-change schedule the maneuver library attaches, because the
    maneuvers in the library are all longitudinal conflicts or crossings whose
    conflict point is known in advance."""
    p = dict(POLICY)
    if policy:
        p.update(policy)
    mu = sc.weather.friction()
    actors = [a for a in sc.actors]
    ego = next((a for a in actors if a.id == "ego"), None)
    if ego is None:
        raise ValueError("scenario has no ego")
    others = [a for a in actors if a.id != "ego"]

    state = {a.id: {"s": float(a.s0), "v": float(a.v0), "lane": float(a.lane), "a": 0.0} for a in actors}
    length = {a.id: a.length_width()[0] for a in actors}
    traj = []
    min_ttc, min_gap, max_decel, peak_jerk = float("inf"), float("inf"), 0.0, 0.0
    collision = False
    v_des = max(1.0, ego.v0)
    prev_a = 0.0
    braking_started = None
    steps = int(round(t_end / dt))
    t = 0.0
    for k in range(steps):
        t = k * dt
        # leader = nearest other actor in the same lane corridor ahead
        gap, lead_v = float("inf"), None
        for a in others:
            o = state[a.id]
            if abs(o["lane"] - state["ego"]["lane"]) > 0.6:
                continue
            d = o["s"] - state["ego"]["s"] - (length[a.id] + length["ego"]) / 2.0
            if d < gap:
                gap, lead_v = d, o["v"]
        ttc = (gap / (state["ego"]["v"] - lead_v)) if (lead_v is not None and state["ego"]["v"] > lead_v and gap > 0) else float("inf")
        # braking on the TTC alarm, which is where a real stack intervenes
        a_cmd = _idm(state["ego"]["v"], v_des, gap, (state["ego"]["v"] - lead_v) if lead_v is not None else 0.0, p, mu)
        hard = False
        if ttc < p["ttc_brake"]:
            a_cmd = min(a_cmd, -p["b_max"] * (mu / 0.9))
            hard = True
        if hard and braking_started is None:
            braking_started = t
        jerk = abs(a_cmd - prev_a) / dt
        peak_jerk = max(peak_jerk, jerk)
        max_decel = max(max_decel, -a_cmd)
        state["ego"]["a"] = a_cmd
        prev_a = a_cmd
        for a in others:
            o = state[a.id]
            o["a"] = _apply_behavior(a, t, o, sc, mu)
        # integrate (semi-implicit Euler: stable at these dt with stiff braking)
        for a in actors:
            o = state[a.id]
            o["v"] = max(0.0, o["v"] + o["a"] * dt)
            o["s"] += o["v"] * dt
        if gap < 0.02:
            collision = True
            min_gap = min(min_gap, gap)
            break
        min_gap = min(min_gap, gap)
        min_ttc = min(min_ttc, ttc)
        traj.append(Frame(t, state["ego"]["s"], state["ego"]["v"], state["ego"]["a"],
                          gap if math.isfinite(gap) else -1.0,
                          min(ttc, 99.0), collision))
    stop_dist = state["ego"]["v"] ** 2 / (2.0 * p["b_max"] * max(0.1, mu / 0.9))
    avail = gap if math.isfinite(gap) else float("inf")
    return Outcome(
        scenario_id=sc.id, steps=len(traj), collision=collision,
        min_ttc=min(min_ttc, 99.0), min_gap=min_gap if math.isfinite(min_gap) else -1.0,
        max_decel=max_decel, peak_jerk=peak_jerk, ego_stop_dist=stop_dist,
        distance_available=avail, hard_brake=braking_started is not None, traj=traj,
    )


def _apply_behavior(a: Actor, t: float, o: dict, sc: Scenario, mu: float) -> float:
    """A behavior entry is {"t": switch time, "a": accel, "lane": lane, "v": speed}.
    Entries apply from their time onward; the last is the active one. A lane
    change is instantaneous in this model, which is the honest simplification:
    lateral dynamics are not what decides whether the conflict was violent."""
    a_cmd = 0.0
    for b in a.behavior:
        if t >= b.get("t", 0.0):
            if "a" in b:
                a_cmd = float(b["a"])
            if "v" in b:
                o["v"] = float(b["v"])
            if "lane" in b:
                o["lane"] = float(b["lane"])
    if a.cls == "pedestrian":
        return a_cmd
    return max(-8.0 * (mu / 0.9), min(3.0, a_cmd))


def analytic_min_gap(sc: Scenario, policy: dict = None) -> float:
    """The closed-form check the simulator is held to for the simplest case: a
    follower braking at b_max against a leader braking at a_lead. Required gap
    is the difference of braking distances plus the reaction distance. If the
    simulator reports contact when this says space was available, one of the two
    is wrong, and tests/test_sim.py asserts they agree."""
    p = dict(POLICY)
    if policy:
        p.update(policy)
    ego = next(a for a in sc.actors if a.id == "ego")
    lead = next((a for a in sc.actors if a.id != "ego"), None)
    if lead is None:
        return float("inf")
    mu = sc.weather.friction()
    b_max = p["b_max"] * (mu / 0.9)
    a_lead = abs(next((float(b["a"]) for b in reversed(lead.behavior) if "a" in b), 0.0))
    d_ego = ego.v0 ** 2 / (2.0 * b_max)
    d_lead = lead.v0 ** 2 / (2.0 * max(0.3, a_lead)) if a_lead > 0 else float("inf")
    reaction = ego.v0 * p["reaction_s"]
    gap0 = lead.s0 - ego.s0 - (lead.length_width()[0] + ego.length_width()[0]) / 2.0
    return gap0 + (d_lead if math.isfinite(d_lead) else 1e9) - d_ego - reaction
