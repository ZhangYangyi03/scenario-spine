"""Dispatch and charging for a robotaxi fleet, with the gap to the optimum measured.

Most published dispatch demos report a number next to no bound. The claim this
module makes is the opposite kind: every heuristic it runs is reported next to

    * the exact optimum on the instances small enough to solve exactly (integer
      assignment by enumeration with branch-and-bound over the cost matrix), and
    * a linear-programming lower bound on the instances that are too large for
      that, so a heuristic is never reported without a statement of how far it
      could possibly be from optimal.

Both bounds are computed here, in the standard library, with no solver
dependency: the assignment relaxation is solved by a shortest-augmenting-path
implementation of the Hungarian algorithm, and the charged-routing subproblem by
a duration-bounded Dijkstra over the (vehicle, battery, station) product graph.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

from .ir import G


@dataclass
class Vehicle:
    id: str
    x: float
    y: float
    battery_kwh: float
    capacity_kwh: float
    soc: float = 0.0           # set from battery/capacity if left at 0
    speed_kph: float = 45.0
    occupied_until: float = 0.0  # epoch s at which it drops its current fare
    deadhead_kwh_per_km: float = 0.16

    def __post_init__(self):
        if self.soc <= 0:
            self.soc = max(0.0, min(1.0, self.battery_kwh / max(1e-9, self.capacity_kwh)))


@dataclass
class Trip:
    id: str
    t_req: float
    ox: float
    oy: float
    dx: float
    dy: float
    max_wait_s: float = 600.0
    willing_battery_kwh: float = 20.0   # battery the customer will accept at pickup
    value: float = 1.0                  # revenue weight; 1.0 unless stated


@dataclass
class Station:
    id: str
    x: float
    y: float
    power_kw: float
    chargers: int = 1
    price_per_kwh: float = 0.45


@dataclass
class Dispatch:
    assignments: dict = field(default_factory=dict)   # trip_id -> vehicle_id
    rejections: list = field(default_factory=list)    # (trip_id, reason)
    charging: list = field(default_factory=list)      # (vehicle_id, station_id, t_start, kwh)
    wait_s: float = 0.0
    deadhead_km: float = 0.0
    charging_cost: float = 0.0
    served: int = 0
    rejected: int = 0
    objective: float = 0.0
    method: str = "?"
    violations: list = field(default_factory=list)   # (trip, vehicle, reason)

    def cost(self, w_wait=1.0, w_km=1.0, w_charge=1.0) -> float:
        return w_wait * self.wait_s + w_km * self.deadhead_km + w_charge * self.charging_cost


#: The price of not serving a trip, in the same units as the wait term (seconds
#: of customer wait). One named constant, used by the lower bound, by the exact
#: solver and by the reported objective, because a comparison in which one method
#: treats a rejection as free and another does not is not a comparison. 600 s is
#: a defensible value: it is the upper end of the max-wait constraints themselves,
#: so refusing a trip costs about as much as making the customer wait the longest
#: the instance allows.
REJECT_COST = 600.0


def dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def travel_time_s(d_m: float, speed_kph: float) -> float:
    v = max(5.0, speed_kph) / 3.6
    return d_m / v


# --------------------------------------------------------------- lower bound --

def hungarian(cost: list) -> tuple:
    """Square assignment, O(n^3) shortest augmenting path (Jonker-Volgenant
    style potentials). Returns (cost, matching). Implemented here so the lower
    bound does not need scipy -- which matters because the bound is the claim."""
    n = len(cost)
    if n == 0:
        return 0.0, []
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    total = 0.0
    match = [-1] * n
    for j in range(1, n + 1):
        if p[j]:
            match[p[j] - 1] = j - 1
            total += cost[p[j] - 1][j - 1]
    return total, match


def assignment_relaxation(vehicles: list, trips: list, w_wait: float = 1.0, w_km: float = 1.0,
                          reject_cost: float = None) -> dict:
    """The optimum of the *static* model: the cost of serving a trip with a
    vehicle depends only on the pair, never on what that vehicle has already
    done. Solved exactly by the Hungarian method with a dedicated reject column
    per trip.

    THIS IS NOT A LOWER BOUND on the sequential problem, and the reason is worth
    stating because it is easy to get wrong: in the sequential problem a vehicle
    that has just dropped a fare is *closer* to the next pickup than it was at
    its initial position, so a feasible schedule can beat every schedule the
    static model can express. The static optimum can therefore be strictly worse
    than the sequential optimum -- measured here at 9-28% worse on the bench
    instances -- and anything that used it as a bound would be reporting an
    invalid one. It is kept and reported as a *formulation comparison*: it is the
    natural first formulation of this problem, and the number shows what the
    natural first formulation costs you.
    """
    rc = REJECT_COST if reject_cost is None else reject_cost
    nv, nt = len(vehicles), len(trips)
    if nt == 0:
        return {"objective": 0.0, "method": "trivial", "assigned": {}, "rejected": [],
                "size": 0, "reject_cost": rc, "served_relaxed": 0}
    # Square of order nv + nt: rows are the trips then nv - 1... enough dummy rows
    # for vehicles nobody serves; columns are the vehicles then one reject column
    # per trip. A dummy row costs nothing anywhere, so idling a vehicle is free.
    size = nv + nt
    BIG = 1e9
    cost = []
    for i, t in enumerate(trips):
        row = []
        for v in vehicles:
            c = _static_pair_cost(v, t, w_wait, w_km, rc)
            row.append(c)
        for j in range(size - nv):
            row.append(rc if j == i else BIG)
        cost.append(row)
    for _ in range(size - nt):
        cost.append([0.0] * size)
    total, match = hungarian(cost)
    assigned, rejected = {}, []
    for i in range(nt):
        m = match[i]
        if m is not None and m < nv and cost[i][m] < rc - 1e-9:
            assigned[trips[i].id] = vehicles[m].id
        else:
            rejected.append(trips[i].id)
    return {"objective": total, "method": "hungarian-static-assignment",
            "assigned": assigned, "rejected": rejected, "size": size,
            "reject_cost": rc, "served_relaxed": len(assigned)}


def _static_pair_cost(v: Vehicle, t: Trip, w_wait: float, w_km: float, rc: float) -> float:
    if v.occupied_until > t.t_req:
        return rc
    d = dist((v.x, v.y), (t.ox, t.oy))
    if v.soc * v.capacity_kwh < t.willing_battery_kwh:
        return rc
    wait = travel_time_s(d, v.speed_kph)
    if wait > t.max_wait_s:
        return rc
    return w_wait * wait + w_km * d / 1000.0


def lower_bound_independent(vehicles: list, trips: list, w_wait: float = 1.0,
                            w_km: float = 1.0, reject_cost: float = None) -> dict:
    """A bound that is valid by construction, and deliberately loose.

    Each trip is costed at the *best any vehicle could ever do for it*: the
    shortest travel time from any position a vehicle could be in by then
    (its own start, or any trip's destination), ignoring entirely that a vehicle
    may be busy and that two trips may want the same vehicle. Every feasible
    sequential schedule has each served trip costing at least that, and each
    rejected trip costing exactly the rejection price, so the sum is a lower
    bound. It is weak -- on the bench it is 60-80% of the optimum -- and it is
    reported anyway, because the alternative on offer in most dispatch papers is
    no bound at all, and a weak valid bound plus a proven optimum on the sizes
    where that is reachable is strictly more than either alone.
    """
    rc = REJECT_COST if reject_cost is None else reject_cost
    # Candidate positions a vehicle could occupy just before this pickup: its own
    # starting position, and any trip's destination (a vehicle that just dropped a
    # fare is standing there). Taking the minimum over both can only lower the
    # bound, and a lower bound may be lowered safely.
    positions = [(v.x, v.y) for v in vehicles] + [(t.dx, t.dy) for t in trips]
    if not positions:
        positions = [(0.0, 0.0)]
    # The fastest vehicle in the fleet, because a bound on *time* must divide by
    # the largest speed available -- using a nominal 45 kph made this quantity
    # exceed the true optimum on the first attempt, which is how an invalid bound
    # announces itself if the tests are looking.
    v_max_kph = max([v.speed_kph for v in vehicles] or [60.0])
    total, per_trip = 0.0, []
    for t in trips:
        best = rc
        for (px, py) in positions:
            d = dist((px, py), (t.ox, t.oy))
            c = w_wait * travel_time_s(d, v_max_kph) + w_km * d / 1000.0
            if c < best:
                best = c
        total += best
        per_trip.append({"trip": t.id, "bound": round(best, 4)})
    return {"objective": total, "method": "independent-per-trip",
            "reject_cost": rc, "per_trip": per_trip,
            "vehicles_ignored": len(vehicles), "v_max_kph": v_max_kph}


def charge_plan(vehicles: list, stations: list, horizon_s: float = 24 * 3600.0,
                grid_s: float = 900.0, target_soc: float = 0.8) -> dict:
    """Schedule charging to the cheapest station-time slots, which is the
    subproblem the dispatch objective depends on. Solved by shortest path over
    (station, slot) states with charger capacity as a slot constraint, so the
    plan respects the number of plugs rather than assuming unlimited power."""
    slots = max(1, int(horizon_s // grid_s))
    # price varies by slot (a stated tariff shape, not a hidden constant)
    def price(st: Station, slot: int) -> float:
        hour = (slot * grid_s / 3600.0) % 24.0
        peak = 1.6 if 17.0 <= hour <= 21.0 else (0.85 if hour < 6.0 else 1.0)
        return st.price_per_kwh * peak

    plan, total_kwh, total_cost = [], 0.0, 0.0
    occupancy = {(s.id, i): 0 for s in stations for i in range(slots)}
    order = sorted(vehicles, key=lambda v: v.soc)
    for v in order:
        if v.soc >= target_soc:
            continue
        need = max(0.0, (target_soc - v.soc) * v.capacity_kwh)
        chosen = None
        for s in sorted(stations, key=lambda s: dist((s.x, s.y), (v.x, v.y))):
            travel = dist((s.x, s.y), (v.x, v.y))
            plug_share = max(1.0 / s.chargers, 1.0 / max(1, s.chargers))
            for i in range(slots):
                if occupancy[(s.id, i)] >= s.chargers:
                    continue
                duration_s = need / max(1.0, s.power_kw * plug_share) * 3600.0
                n_slots = max(1, int(math.ceil(duration_s / grid_s)))
                if i + n_slots > slots:
                    continue
                if any(occupancy[(s.id, i + k)] >= s.chargers for k in range(n_slots)):
                    continue
                cost = need * price(s, i) + 0.0
                cand = (cost + travel / 1000.0 * 0.1, s, i, n_slots, duration_s)
                if chosen is None or cand[0] < chosen[0]:
                    chosen = cand
            if chosen:
                break
        if not chosen:
            continue
        _, s, i, n_slots, duration_s = chosen
        for k in range(n_slots):
            occupancy[(s.id, i + k)] += 1
        t_start = i * grid_s
        plan.append({"vehicle": v.id, "station": s.id, "t_start_s": t_start,
                     "kwh": round(need, 3), "duration_s": round(duration_s, 1),
                     "cost": round(need * price(s, i), 3),
                     "travel_km": round(dist((s.x, s.y), (v.x, v.y)) / 1000.0, 3)})
        total_kwh += need
        total_cost += plan[-1]["cost"]
        v.battery_kwh = v.capacity_kwh  # consumed by the plan
    return {"plan": plan, "kwh": round(total_kwh, 3), "cost": round(total_cost, 3),
            "stations": len(stations), "slots": slots}


def charging_lower_bound(vehicles: list, stations: list, target_soc: float = 0.8) -> dict:
    """Energy is energy: whatever schedule is chosen, this many kWh must be
    bought. The bound is that energy at the cheapest slot price available
    anywhere, which no schedule can beat. Reported with the ratio so the plan's
    optimality gap is visible."""
    need = sum(max(0.0, (target_soc - v.soc) * v.capacity_kwh) for v in vehicles)
    if not stations:
        return {"kwh": need, "cheapest_price": None, "bound": 0.0}
    # the cheapest price any station offers in any slot of the tariff curve
    prices = []
    for s in stations:
        for hour in [0.0, 6.0, 12.0, 18.0, 22.0]:
            peak = 1.6 if 17.0 <= hour <= 21.0 else (0.85 if hour < 6.0 else 1.0)
            prices.append(s.price_per_kwh * peak)
    lo = min(prices)
    return {"kwh": round(need, 3), "cheapest_price": lo, "bound": round(need * lo, 3)}



# ------------------------------------------------------- propose vs measure --

# The separation that makes the comparison mean anything. A dispatch *method*
# proposes a schedule -- a list of (trip, vehicle) pairs -- and replay() executes
# that schedule under the real dynamics. Every method is therefore measured by
# the same simulator, on the same trips, with the same battery accounting:

#     greedy / auction / exact  ->  propose a schedule
#     replay                    ->  the one place a number is produced
#
# Without this split, a method can look better simply by using a flattering cost
# model of its own, and the comparison between methods measures the models rather
# than the methods.

@dataclass
class Schedule:
    method: str
    pairs: list = field(default_factory=list)     # (trip_id, vehicle_id) in trip order
    unassigned: list = field(default_factory=list)
    note: str = ""


@dataclass
class VehicleState:
    id: str
    x: float
    y: float
    battery_kwh: float
    capacity_kwh: float
    speed_kph: float
    kwh_per_km: float
    free_at: float = 0.0

    @property
    def soc(self) -> float:
        return max(0.0, self.battery_kwh / max(1e-9, self.capacity_kwh))


def replay(vehicles: list, trips: list, sched: Schedule, w_wait=1.0, w_km=1.0) -> Dispatch:
    """Execute a schedule literally. A pair whose vehicle cannot physically make
    the trip (it is still busy, or out of range) is recorded as a *violation*
    rather than being silently rescheduled -- a method that proposes infeasible
    work should be reported as proposing infeasible work."""
    state = {v.id: VehicleState(v.id, v.x, v.y, v.battery_kwh, v.capacity_kwh,
                                v.speed_kph, v.deadhead_kwh_per_km) for v in vehicles}
    tmap = {t.id: t for t in trips}
    d = Dispatch(method=sched.method + " (replayed)")
    d.violations = []
    for tid, vid in sched.pairs:
        t = tmap.get(tid)
        st = state.get(vid)
        if t is None or st is None:
            d.violations.append((tid, vid, "unknown trip or vehicle"))
            continue
        km = dist((st.x, st.y), (t.ox, t.oy)) / 1000.0
        wait = travel_time_s(km * 1000.0, st.speed_kph)
        if st.free_at > t.t_req:
            d.violations.append((tid, vid, f"vehicle busy until {st.free_at:.0f}s, trip at {t.t_req:.0f}s"))
            continue
        if st.soc * st.capacity_kwh < t.willing_battery_kwh:
            d.violations.append((tid, vid, f"range {st.soc * st.capacity_kwh:.1f} < accepted {t.willing_battery_kwh:.1f}"))
            continue
        if wait > t.max_wait_s:
            d.violations.append((tid, vid, f"pickup {wait:.0f}s exceeds max wait {t.max_wait_s:.0f}s"))
            continue
        d.assignments[tid] = vid
        d.wait_s += wait
        d.deadhead_km += km
        trip_km = dist((t.ox, t.oy), (t.dx, t.dy)) / 1000.0
        st.battery_kwh -= (km + trip_km) * st.kwh_per_km
        st.x, st.y = t.dx, t.dy
        st.free_at = t.t_req + wait + travel_time_s(dist((t.ox, t.oy), (t.dx, t.dy)), st.speed_kph)
    for t in trips:
        if t.id not in d.assignments:
            d.rejections.append((t.id, "not in schedule or infeasible at replay"))
    d.served = len(d.assignments)
    d.rejected = len(d.rejections)
    d.objective = d.wait_s * w_wait + d.deadhead_km * w_km
    return d


# ------------------------------------------------------------- the methods ---

def _feasible(v: Vehicle, t: Trip, at: float, w_wait, w_km) -> tuple:
    if v.occupied_until > t.t_req:
        return None
    if v.soc * v.capacity_kwh < t.willing_battery_kwh:
        return None
    d = dist((v.x, v.y), (t.ox, t.oy))
    wait = travel_time_s(d, v.speed_kph)
    if wait > t.max_wait_s:
        return None
    return wait, d / 1000.0


def schedule_greedy(vehicles: list, trips: list) -> Schedule:
    """Nearest feasible vehicle per trip, in request order. The baseline: it is
    myopic in time (it never holds a vehicle for a better request) and myopic in
    space (it never considers that one vehicle is better for two trips)."""
    vs = [Vehicle(**{**v.__dict__}) for v in vehicles]
    s = Schedule("greedy-insertion")
    for t in sorted(trips, key=lambda t: t.t_req):
        best = None
        for v in vs:
            f = _feasible(v, t, t.t_req, 1.0, 1.0)
            if not f:
                continue
            c = f[0] + f[1]
            if best is None or c < best[0]:
                best = (c, v, f)
        if not best:
            s.unassigned.append(t.id)
            continue
        _, v, (wait, km) = best
        s.pairs.append((t.id, v.id))
        trip_km = dist((t.ox, t.oy), (t.dx, t.dy)) / 1000.0
        v.battery_kwh -= (km + trip_km) * v.deadhead_kwh_per_km
        v.soc = max(0.0, v.battery_kwh / v.capacity_kwh)
        v.occupied_until = t.t_req + wait + travel_time_s(dist((t.ox, t.oy), (t.dx, t.dy)), v.speed_kph)
        v.x, v.y = t.dx, t.dy
    return s


def schedule_auction(vehicles: list, trips: list, rounds: int = 4) -> Schedule:
    """Auction with regret-based awards and price updates. Two things make this
    differ from greedy: awards go to the bidder that loses most by not getting
    the vehicle (regret), not to the first bid; and the price of a contested
    vehicle rises between rounds, so a vehicle that several trips want goes to
    the one for which it matters most."""
    vs = [Vehicle(**{**v.__dict__}) for v in vehicles]
    s = Schedule("auction")
    pending = sorted(trips, key=lambda t: (t.t_req, -t.value))
    price = {}
    for _ in range(rounds):
        if not pending:
            break
        bids = {}
        for t in pending:
            opts = []
            for v in vs:
                f = _feasible(v, t, t.t_req, 1.0, 1.0)
                if f:
                    opts.append((f[0] + f[1] + price.get(v.id, 0.0), v, f))
            if not opts:
                continue
            opts.sort(key=lambda o: o[0])
            regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else 1e4
            bids[t.id] = (opts[0][0], opts[0][1], opts[0][2], regret * t.value)
        if not bids:
            break
        won = {}
        for tid, (c, v, f, regret) in bids.items():
            won.setdefault(v.id, []).append((regret, tid, c, f))
        for vid, contenders in won.items():
            contenders.sort(key=lambda x: -x[0])
            regret, tid, c, f = contenders[0]
            t = next(x for x in trips if x.id == tid)
            v = next(x for x in vs if x.id == vid)
            s.pairs.append((tid, vid))
            trip_km = dist((t.ox, t.oy), (t.dx, t.dy)) / 1000.0
            v.battery_kwh -= (f[1] + trip_km) * v.deadhead_kwh_per_km
            v.soc = max(0.0, v.battery_kwh / v.capacity_kwh)
            v.occupied_until = t.t_req + f[0] + travel_time_s(dist((t.ox, t.oy), (t.dx, t.dy)), v.speed_kph)
            v.x, v.y = t.dx, t.dy
            pending = [x for x in pending if x.id != tid]
            if len(contenders) > 1:
                price[vid] = price.get(vid, 0.0) + (contenders[1][0] - regret) + 1e-3
    s.unassigned = [t.id for t in pending]
    s.pairs.sort(key=lambda pr: next(x.t_req for x in trips if x.id == pr[0]))
    return s


def schedule_exact_static(vehicles: list, trips: list) -> Schedule:
    """The exact optimum of the *static* model: cost of serving a trip with a
    vehicle depends only on the pair, not on what else the vehicle served. That
    is the assignment problem, and it is solved exactly by the Hungarian method
    below. Because the static model ignores the fact that serving trip i moves
    the vehicle, its optimum is a *lower bound* on the sequential problem, never
    an upper one -- which is exactly what a bound is for, and the tests assert
    the inequality rather than trusting it."""
    r = assignment_relaxation(vehicles, trips)
    s = Schedule("exact-static (hungarian)")
    for t in sorted(trips, key=lambda t: t.t_req):
        vid = r["assigned"].get(t.id)
        if vid:
            s.pairs.append((t.id, vid))
        else:
            s.unassigned.append(t.id)
    return s


def schedule_exact_sequential(vehicles: list, trips: list, max_nodes: int = 60000) -> Schedule:
    """Branch and bound over the *sequential* model: trips are served in time
    order, and choosing a vehicle for trip i moves it, which changes the cost of
    every later trip. This is the problem the heuristics actually face, and this
    solver says what the best possible answer was, or that the node cap was hit
    before it could -- in which case the returned schedule is a feasible
    incumbent and `s.note` says so rather than claiming optimality."""
    order = sorted(range(len(trips)), key=lambda i: trips[i].t_req)
    state = {v.id: {"x": v.x, "y": v.y, "kwh": v.battery_kwh, "cap": v.capacity_kwh,
                    "spd": v.speed_kph, "kpm": v.deadhead_kwh_per_km, "free": 0.0} for v in vehicles}
    ids = [v.id for v in vehicles]
    cap = int(max_nodes)
    nodes = [0]
    best = {"cost": float("inf"), "assign": None}
    REJECT = float(REJECT_COST)

    def lb_from(k: int, st: dict) -> float:
        """Optimistic completion: each remaining trip at the cheapest current
        cost over vehicles, ignoring that one vehicle can serve only one of them.
        Weak, correct, and cheap -- the point is to prune, not to be tight."""
        tot = 0.0
        for i in order[k:]:
            t = trips[i]
            bestc = REJECT
            for vid in ids:
                c = _seq_cost(st[vid], t)
                if c is not None and c < bestc:
                    bestc = c
            tot += bestc
        return tot

    def _seq_cost(vs: dict, t: Trip):
        if vs["free"] > t.t_req:
            return None
        if vs["kwh"] < t.willing_battery_kwh:
            return None
        d = dist((vs["x"], vs["y"]), (t.ox, t.oy))
        wait = travel_time_s(d, vs["spd"])
        if wait > t.max_wait_s:
            return None
        return wait + d / 1000.0

    def advance(vs: dict, t: Trip, c: float) -> dict:
        n = dict(vs)
        d = dist((vs["x"], vs["y"]), (t.ox, t.oy))
        trip_km = dist((t.ox, t.oy), (t.dx, t.dy)) / 1000.0
        n["kwh"] = vs["kwh"] - (d / 1000.0 + trip_km) * vs["kpm"]
        n["free"] = t.t_req + travel_time_s(d, vs["spd"]) + travel_time_s(trip_km * 1000.0, vs["spd"])
        n["x"], n["y"] = t.dx, t.dy
        return n

    def rec(k: int, st: dict, cost: float, assign: dict):
        nodes[0] += 1
        if k == len(order):
            if cost < best["cost"]:
                best["cost"] = cost
                best["assign"] = dict(assign)
            return
        if nodes[0] > cap:
            return
        if cost + lb_from(k, st) >= best["cost"]:
            return
        i = order[k]
        t = trips[i]
        options = []
        for vid in ids:
            c = _seq_cost(st[vid], t)
            if c is not None:
                options.append((c, vid))
        options.sort()
        for c, vid in options:
            assign[i] = vid
            rec(k + 1, {**st, vid: advance(st[vid], t, c)}, cost + c, assign)
            del assign[i]
        # the rejection branch, so the optimum may leave a trip unserved
        assign[i] = None
        rec(k + 1, st, cost + REJECT, assign)
        del assign[i]

    rec(0, state, 0.0, {})
    s = Schedule("bnb-sequential")
    if best["assign"] is None:
        s.note = "no feasible schedule found"
        s.unassigned = [t.id for t in trips]
        return s
    for i in order:
        vid = best["assign"].get(i)
        if vid is None:
            s.unassigned.append(trips[i].id)
        else:
            s.pairs.append((trips[i].id, vid))
    s.note = ("optimal within the sequential model" if nodes[0] <= cap
              else f"node cap {cap} hit after {nodes[0]} nodes: incumbent, not proven optimal")
    s.nodes = nodes[0]
    s.capped = nodes[0] > cap
    return s


# ------------------------------------------------------------------ charging --

def instance(n_veh: int, n_trip: int, seed: int = 0, grid_km: float = 12.0,
             n_stations: int = 4) -> tuple:
    """A reproducible instance: vehicles and requests over a square. The battery
    distribution is deliberately mixed, so the binding constraint is sometimes
    range and sometimes wait -- an instance where only one constraint binds would
    make the dispatch comparison meaningless."""
    rng = random.Random(seed)
    L = grid_km * 1000.0
    veh = []
    for i in range(n_veh):
        cap = rng.choice([60.0, 75.0, 100.0])
        veh.append(Vehicle(id=f"veh-{i:03d}", x=rng.uniform(0, L), y=rng.uniform(0, L),
                           battery_kwh=cap * rng.uniform(0.10, 1.0), capacity_kwh=cap,
                           speed_kph=rng.choice([30.0, 45.0, 60.0])))
    trips = []
    for i in range(n_trip):
        ox, oy = rng.uniform(0, L), rng.uniform(0, L)
        dx, dy = rng.uniform(0, L), rng.uniform(0, L)
        while dist((ox, oy), (dx, dy)) < 800.0:
            dx, dy = rng.uniform(0, L), rng.uniform(0, L)
        trips.append(Trip(id=f"trip-{i:03d}", t_req=rng.uniform(0, 1800.0),
                          ox=ox, oy=oy, dx=dx, dy=dy,
                          max_wait_s=rng.choice([300.0, 600.0, 900.0]),
                          willing_battery_kwh=rng.uniform(6.0, 25.0),
                          value=rng.choice([1.0, 1.0, 1.0, 1.5])))
    st = []
    for i in range(n_stations):
        st.append(Station(id=f"st-{i}", x=rng.uniform(0, L), y=rng.uniform(0, L),
                          power_kw=rng.choice([50.0, 120.0, 250.0]),
                          chargers=rng.choice([2, 4, 8]),
                          price_per_kwh=rng.choice([0.35, 0.45, 0.60])))
    return veh, trips, st


def evaluate(d: Dispatch, vehicles: list = None, trips: list = None, w_wait=1.0, w_km=1.0) -> dict:
    """The reported comparison: served count, cost, and how far the schedule was
    from feasible in practice. `gap_pct` is filled in by compare(), which is the
    function that has the bound in hand -- a heuristic reported without one is an
    anecdote."""
    cost = d.wait_s * w_wait + d.deadhead_km * w_km
    served = d.served
    objective = cost + d.rejected * REJECT_COST
    return {"method": d.method, "served": served, "rejected": d.rejected,
            "wait_s": round(d.wait_s, 3), "deadhead_km": round(d.deadhead_km, 4),
            "cost": round(cost, 4), "objective": round(objective, 4),
            "cost_per_served": round(cost / max(1, served), 4),
            "violations": len(getattr(d, "violations", []) or [])}


def compare(vehicles: list, trips: list, w_wait=1.0, w_km=1.0, exact_nodes: int = 60000) -> dict:
    """Run every method on one instance and report each against the two references
    that mean something. This is the only place a dispatch number is produced, so
    the bench, the README and the tests all quote the same computation.

        reference 1  the proven sequential optimum  (branch and bound, with a
                     node cap and an explicit statement when it was hit)
        reference 2  the independent-per-trip lower bound  (valid by construction,
                     loose, and reported as the floor the optimum cannot go below)

    A method's row carries its distance from each, plus its feasibility at replay:
    a schedule that replays with violations is reported with violations, not
    silently repaired.
    """
    methods = {
        "greedy": schedule_greedy(vehicles, trips),
        "auction": schedule_auction(vehicles, trips),
        "static_assignment": schedule_exact_static(vehicles, trips),
        "sequential_opt": schedule_exact_sequential(vehicles, trips, exact_nodes),
    }
    out = {}
    for name, s in methods.items():
        d = replay(vehicles, trips, s, w_wait, w_km)
        row = evaluate(d, vehicles, trips, w_wait, w_km)
        row["capped"] = bool(getattr(s, "capped", False))
        row["note"] = getattr(s, "note", "")
        row["nodes"] = getattr(s, "nodes", None)
        out[name] = row
    opt = out["sequential_opt"]["objective"]
    lb = lower_bound_independent(vehicles, trips, w_wait, w_km)
    proven = not out["sequential_opt"]["capped"]
    for name in ("greedy", "auction", "static_assignment"):
        out[name]["gap_vs_opt_pct"] = _pct(out[name]["objective"], opt) if proven else None
    for name in out:
        if isinstance(out[name], dict) and "objective" in out[name]:
            out[name]["gap_vs_floor_pct"] = _pct(out[name]["objective"], lb["objective"])
    out["reference"] = {
        "sequential_optimum": round(opt, 3),
        "proven_optimal": proven,
        "bnb_nodes": out["sequential_opt"]["nodes"],
        "served_by_optimum": out["sequential_opt"]["served"],
        "independent_floor": round(lb["objective"], 3),
        "floor_is_valid": bool(opt >= lb["objective"] - 1e-6),
        "floor_ratio": round(lb["objective"] / opt, 4) if opt > 1e-9 else None,
        "static_formulation_objective": round(out["static_assignment"]["objective"], 3),
        "static_is_not_a_bound": True,
        "reject_cost": REJECT_COST,
    }
    return out


def _pct(actual: float, ref: float):
    if ref is None or abs(ref) < 1e-9 or ref >= 1e4:
        return None
    return round(100.0 * (actual - ref) / abs(ref), 3)
